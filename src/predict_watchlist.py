"""
拉取富途港股自选股，用最新K线数据预测未来15天走势。
港股持仓（实盘/模拟）单独分组展示；环境变量 PREDICT_WATCHLIST_POSITIONS_ENV=REAL|SIMULATE，
FUTU_SECURITY_FIRM 指定券商（默认 FUTUSECURITIES）。
"""

import json
import os
import sys
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stock_predictor import (
    compute_enhanced_feature_matrix, compute_inflection_signal,
    DualModelPredictor, DiverseEnsemble,
    MODEL_CONFIGS, LOOKBACK, FORWARD, CONFIDENCE_MARGIN,
    UP_P_THRESHOLD, UP_VOTE_MODEL_THRESH, UP_VOTE_AGREE, DN_P_THRESHOLD,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
REPORT_DIR = os.path.join(RESULT_DIR, "report")
DUAL_MODEL_PATH_HK = os.path.join(RESULT_DIR, "dual_model_hk.pkl")
DUAL_MODEL_PATH_CN = os.path.join(RESULT_DIR, "dual_model_cn.pkl")
TRAIN_WINDOWS = [6, 12, 24]
WEIGHT_HALFLIFE = 90


def _to_jsonable(obj):
    """将 numpy 标量等转为 JSON 可序列化类型。"""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    return obj


def _predict_report_path():
    """按日期命名；同日多次运行则加时分秒避免覆盖。目录：results/report/。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    now = datetime.now()
    d = now.strftime("%Y-%m-%d")
    base = os.path.join(REPORT_DIR, f"predict_watchlist_{d}.json")
    if os.path.isfile(base):
        return os.path.join(
            REPORT_DIR, f"predict_watchlist_{d}_{now.strftime('%H%M%S')}.json"
        )
    return base


def get_hk_watchlist():
    """获取富途港股自选股列表，返回 (codes, name_map)"""
    from futu import OpenQuoteContext, RET_OK
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        ret, groups = ctx.get_user_security_group()
        if ret != RET_OK:
            print(f"获取自选分组失败: {groups}")
            return [], {}

        hk_codes = []
        name_map = {}
        seen = set()
        for _, row in groups.iterrows():
            gname = row["group_name"]
            ret2, stocks = ctx.get_user_security(gname)
            if ret2 != RET_OK:
                continue
            for _, s in stocks.iterrows():
                code = s["code"]
                if code.startswith("HK.") and code not in seen:
                    hk_codes.append(code)
                    seen.add(code)
                    if "name" in s.index and s["name"]:
                        name_map[code] = s["name"]

        return hk_codes, name_map
    finally:
        ctx.close()


def get_hk_positions(trd_env):
    """通过富途交易接口查询港股持仓。返回 (rows, err_msg)。

    rows 每项: code, qty, stock_name, pl_ratio(可选 float)。
    需 OpenD 已登录交易；券商可通过环境变量 FUTU_SECURITY_FIRM（默认 FUTUSECURITIES）。
    """
    from futu import OpenSecTradeContext, TrdEnv, TrdMarket, RET_OK, SecurityFirm

    firm_name = os.environ.get("FUTU_SECURITY_FIRM", "FUTUSECURITIES").upper()
    firm = getattr(SecurityFirm, firm_name, SecurityFirm.FUTUSECURITIES)
    ctx = OpenSecTradeContext(
        filter_trdmarket=TrdMarket.NONE,
        host="127.0.0.1",
        port=11111,
        security_firm=firm,
    )
    try:
        ret, accs = ctx.get_acc_list()
        if ret != RET_OK:
            return [], f"get_acc_list 失败: {accs}"
        if accs is None or accs.empty:
            return [], "无交易账户（请确认 OpenD 已登录交易）"
        env_s = "REAL" if trd_env == TrdEnv.REAL else "SIMULATE"
        accs = accs[accs["trd_env"] == env_s]
        if accs.empty:
            return [], f"无 {env_s} 账户"
        merged = {}
        for _, acc in accs.iterrows():
            acc_id = int(acc["acc_id"])
            ret_p, pos = ctx.position_list_query(
                position_market=TrdMarket.HK,
                trd_env=trd_env,
                acc_id=acc_id,
                refresh_cache=True,
            )
            if ret_p != RET_OK or pos is None or pos.empty:
                continue
            for _, p in pos.iterrows():
                code = str(p["code"])
                if not code.startswith("HK."):
                    continue
                try:
                    q = float(p.get("qty", 0) or 0)
                except (TypeError, ValueError):
                    q = 0.0
                if q <= 0:
                    continue
                nm = p.get("stock_name", "") or ""
                plr = p.get("pl_ratio")
                try:
                    if plr is None or (isinstance(plr, float) and np.isnan(plr)):
                        plr_f = None
                    else:
                        plr_f = float(plr)
                except (TypeError, ValueError):
                    plr_f = None
                if code not in merged:
                    merged[code] = {
                        "code": code,
                        "qty": q,
                        "stock_name": nm,
                        "pl_ratio": plr_f,
                    }
                else:
                    merged[code]["qty"] += q
                    if not merged[code]["stock_name"] and nm:
                        merged[code]["stock_name"] = nm
        return list(merged.values()), None
    except Exception as e:
        return [], str(e)
    finally:
        ctx.close()


def _cache_path(code):
    return os.path.join(DATA_DIR, f"{code.replace('.', '_')}.csv")


def _load_cached(code):
    """从本地缓存加载K线数据，返回 DataFrame 或 None"""
    p = _cache_path(code)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["time_key"] = pd.to_datetime(df["time_key"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close", "high", "low", "volume"])
    return df.sort_values("time_key").reset_index(drop=True)


def _save_cache(code, df):
    """保存K线数据到本地缓存"""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(_cache_path(code), index=False)


def fetch_recent_kline(codes, days=200):
    """获取最近N天的日K线数据，优先读缓存，仅增量获取缺失日期"""
    from futu import OpenQuoteContext, KLType, AuType, RET_OK

    today = datetime.now().strftime("%Y-%m-%d")
    full_start = (datetime.now() - timedelta(days=int(days * 1.6))).strftime("%Y-%m-%d")

    result = {}
    need_fetch = []
    now = datetime.now()

    for code in codes:
        cached = _load_cached(code)
        if cached is not None and len(cached) > 0:
            last_ts = cached["time_key"].max()
            days_stale = (now - last_ts).days
            if days_stale <= 3:
                result[code] = cached
            else:
                need_fetch.append((code, cached, last_ts.strftime("%Y-%m-%d")))
        else:
            need_fetch.append((code, None, None))

    cached_count = len(result)
    if cached_count > 0:
        print(f"  ✅ {cached_count} 只股票数据已是最新，从缓存加载")
    if not need_fetch:
        return result

    print(f"  📡 需要从API获取/更新 {len(need_fetch)} 只股票")
    sys.stdout.flush()

    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    api_calls = 0
    try:
        for i, (code, cached, last_date) in enumerate(need_fetch):
            if cached is not None:
                fetch_start = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                print(f"  [{i+1}/{len(need_fetch)}] {code} 增量更新 {last_date} → {today} ...", end=" ")
            else:
                fetch_start = full_start
                print(f"  [{i+1}/{len(need_fetch)}] {code} 全量获取 ...", end=" ")
            sys.stdout.flush()

            rows, pk = [], None
            while True:
                ret, data, pk = ctx.request_history_kline(
                    code, start=fetch_start, end=today,
                    ktype=KLType.K_DAY, autype=AuType.QFQ,
                    max_count=1000, page_req_key=pk)
                if ret != RET_OK:
                    print(f"失败: {data}")
                    break
                rows.append(data)
                if pk is None:
                    break
                time.sleep(0.3)

            if rows:
                new_df = pd.concat(rows, ignore_index=True)
                new_df["time_key"] = pd.to_datetime(new_df["time_key"])
                for c in ["open", "close", "high", "low", "volume"]:
                    new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
                new_df = new_df.dropna(subset=["open", "close", "high", "low", "volume"])

                if cached is not None and len(new_df) > 0:
                    df = pd.concat([cached, new_df], ignore_index=True)
                    df = df.drop_duplicates(subset=["time_key"]).sort_values("time_key").reset_index(drop=True)
                    print(f"+{len(new_df)}天 → 共{len(df)}天")
                elif cached is not None:
                    df = cached
                    print(f"无新数据 → 共{len(df)}天")
                else:
                    df = new_df.sort_values("time_key").reset_index(drop=True)
                    print(f"{len(df)}天")

                _save_cache(code, df)
                result[code] = df
            elif cached is not None:
                result[code] = cached
                print(f"无新数据 → 共{len(cached)}天")

            api_calls += 1
            if api_calls % 55 == 0:
                print("  ⏳ 等待30秒避免频率限制...")
                sys.stdout.flush()
                time.sleep(31)
            else:
                time.sleep(0.5)
    finally:
        ctx.close()
    return result


def compute_market_benchmark(stock_data):
    """计算港股市场基准收益率"""
    rets = []
    for code, df in stock_data.items():
        close = df["close"].astype(float)
        r20 = close / close.shift(20) - 1
        r20.index = df["time_key"]
        rets.append(r20)
    if not rets:
        return None
    return pd.concat(rets, axis=1).mean(axis=1)


def predict_stocks(stock_data, dual_model, name_map=None):
    """对每只股票用双模型进行预测"""
    name_map = name_map or {}
    benchmark = compute_market_benchmark(stock_data)
    results = []

    for code, df in stock_data.items():
        name = name_map.get(code, "")
        if len(df) < LOOKBACK + 10:
            results.append({
                "code": code, "name": name, "prediction": "数据不足",
                "probability": None, "confidence": None,
                "last_close": None, "last_date": None,
                "inflection": None,
            })
            continue

        feat = compute_enhanced_feature_matrix(df)

        stock_ret20 = feat["ret_20d"]
        if benchmark is not None:
            bm_aligned = benchmark.reindex(df["time_key"].values)
            mv = bm_aligned.values if len(bm_aligned) == len(df) else np.zeros(len(df))
            if isinstance(mv, pd.Series):
                mv = mv.values
            feat["rel_strength"] = stock_ret20.values - mv
            feat["mkt_momentum"] = mv
        else:
            feat["rel_strength"] = 0.0
            feat["mkt_momentum"] = 0.0

        last_row = feat.iloc[[-1]]
        if last_row.isna().all(axis=1).iloc[0]:
            results.append({
                "code": code, "name": name, "prediction": "特征计算失败",
                "probability": None, "confidence": None,
                "last_close": None, "last_date": None,
                "inflection": None,
            })
            continue

        inflection = compute_inflection_signal(feat)

        preds, mask, details = dual_model.predict(last_row)
        p_std = details["p_std"][0]
        p_strong = details["p_strong"][0]
        up_ratio = details["up_ratio"][0]
        is_up = details["is_up"][0]
        is_dn = details["is_dn"][0]

        if is_up:
            prediction = "涨"
            signal = "强"
            confidence = up_ratio
        elif is_dn:
            prediction = "跌"
            signal = "强"
            confidence = 1 - p_std
        elif p_std >= 0.60:
            prediction = "偏涨"
            signal = "弱"
            confidence = p_std - 0.5
        elif p_std <= 0.40:
            prediction = "偏跌"
            signal = "弱"
            confidence = 0.5 - p_std
        else:
            prediction = "震荡"
            signal = "无"
            confidence = abs(p_std - 0.5)

        last_close = df.iloc[-1]["close"]
        last_date = df.iloc[-1]["time_key"].strftime("%Y-%m-%d")

        results.append({
            "code": code,
            "name": name,
            "prediction": prediction,
            "signal": signal,
            "probability_std": round(p_std, 4),
            "probability_strong": round(p_strong, 4),
            "vote_ratio": round(up_ratio, 4),
            "confidence": round(confidence, 4),
            "last_close": round(last_close, 2),
            "last_date": last_date,
            "inflection": inflection,
        })

    return results


def main():
    print("=" * 70)
    print("  港股自选股走势预测 — V3 双模型版")
    print("=" * 70)
    print(f"  预测周期: 未来 {FORWARD} 天")
    print(f"  涨信号: 强标签模型P≥{UP_P_THRESHOLD}+子模型投票≥{UP_VOTE_AGREE:.0%}")
    print(f"  跌信号: 标准模型P≤{DN_P_THRESHOLD}")
    print(f"  拐点: 技术面规则(区间/RSI/MACD/均线斜率等)，与双模型独立")
    print("=" * 70)
    sys.stdout.flush()

    # 1. 获取自选股
    print("\n[1] 获取港股自选 ...")
    sys.stdout.flush()
    wl_codes, name_map = get_hk_watchlist()
    if not wl_codes:
        print("  未找到港股自选股。请确认富途自选中有港股。")
        return
    print(f"  找到 {len(wl_codes)} 只港股: {', '.join(wl_codes[:10])}{'...' if len(wl_codes) > 10 else ''}")
    sys.stdout.flush()

    # 1b. 真实/模拟账户港股持仓（用于单独分组；K 线会与自选合并拉取）
    from futu import TrdEnv

    pos_env_raw = os.environ.get("PREDICT_WATCHLIST_POSITIONS_ENV", "REAL").upper()
    trd_env = TrdEnv.SIMULATE if pos_env_raw == "SIMULATE" else TrdEnv.REAL
    pos_env_label = "模拟 SIMULATE" if trd_env == TrdEnv.SIMULATE else "实盘 REAL"
    print(f"\n[1b] 获取港股持仓 ({pos_env_label}) ...")
    sys.stdout.flush()
    position_rows, pos_err = get_hk_positions(trd_env)
    position_by_code = {r["code"]: r for r in position_rows}
    position_codes = list(position_by_code.keys())
    if pos_err:
        print(f"  ⚠ {pos_err}（将仅展示自选预测）")
    elif not position_codes:
        print(f"  当前账户无港股持仓（qty>0）。")
    else:
        print(f"  港股持仓 {len(position_codes)} 只: {', '.join(position_codes[:10])}{'...' if len(position_codes) > 10 else ''}")
    for pr in position_rows:
        c = pr["code"]
        nm = pr.get("stock_name") or ""
        if nm and not name_map.get(c):
            name_map[c] = nm

    codes = list(dict.fromkeys([*position_codes, *wl_codes]))

    # 2. 获取最新K线
    print(f"\n[2] 获取最新 K 线 (最近 {LOOKBACK+50} 天) ...")
    sys.stdout.flush()
    stock_data = fetch_recent_kline(codes, days=LOOKBACK + 50)
    print(f"  成功获取 {len(stock_data)} 只股票数据（自选+持仓去重）")

    # 3. 加载双模型
    print("\n[3] 加载双模型 ...")
    if not os.path.exists(DUAL_MODEL_PATH_HK):
        print(f"  双模型文件不存在: {DUAL_MODEL_PATH_HK}")
        print("  请先运行 run_backtest.py 训练模型。")
        return
    dual_model = DualModelPredictor.load(DUAL_MODEL_PATH_HK)
    n_std = len(dual_model.std_model.models)
    n_strong = len(dual_model.strong_model.models)
    print(f"  双模型已加载 (标准{n_std}+强标签{n_strong}个子模型)")

    # 4. 预测
    print("\n[4] 预测走势 ...")
    sys.stdout.flush()
    results = predict_stocks(stock_data, dual_model, name_map)

    # 5. 输出结果：按模型信号区间分块展示，组内按置信度从高到低
    print("\n" + "=" * 118)
    print("  预测结果 (未来 15 天) + 拐点 — 按信号区间分开展示")
    print("=" * 118)

    def model_tag(r):
        if r is None:
            return "无K线"
        pred = r.get("prediction", "") or ""
        if pred == "数据不足":
            return "数据不足"
        if pred == "特征计算失败":
            return "特征失败"
        if r.get("signal") == "强":
            return "强涨" if pred == "涨" else "强跌"
        if r.get("signal") == "弱":
            return "偏涨" if pred == "偏涨" else "偏跌"
        if r.get("signal") == "无":
            return "震荡"
        return "其他"

    def inf_short(inf):
        if not inf:
            return "—", None, "—", "", ""
        k = inf.get("kind", "")
        sc = float(inf.get("score", 50))
        if k == "暂无明显拐点":
            return "—", sc, "—", k, ""
        if k == "可能向上拐点":
            ks = "向上"
        elif k == "可能向下拐点":
            ks = "向下"
        else:
            ks = "切换"
        return ks, sc, inf.get("strength", "—"), k, inf.get("hint", "") or ""

    hdr = (
        f"  {'代码':12s} {'名称':10s} {'模型':6s} {'置信':>6s} {'标准P':>7s} {'强P':>7s} {'投票':>6s} "
        f"{'拐点分':>6s} {'拐点':4s} {'拐强':4s} {'拐点简述':16s} {'收盘':>10s} {'日期':12s}"
    )

    by_code = {r["code"]: r for r in results}
    strong_up = sum(1 for r in results if r.get("signal") == "强" and r.get("prediction") == "涨")
    strong_dn = sum(1 for r in results if r.get("signal") == "强" and r.get("prediction") == "跌")
    weak_up = sum(1 for r in results if r.get("signal") == "弱" and r.get("prediction") == "偏涨")
    weak_dn = sum(1 for r in results if r.get("signal") == "弱" and r.get("prediction") == "偏跌")
    neutral = sum(1 for r in results if r.get("signal") == "无")

    buckets = defaultdict(list)
    for code in wl_codes:
        r = by_code.get(code)
        if r is None:
            buckets["无K线"].append((code, None))
        else:
            buckets[model_tag(r)].append((code, r))

    def numeric_confidence(cr):
        """用于组内排序：强涨=投票一致度，强跌=1-P标，偏弱=偏离0.5；无则置底"""
        _, r = cr
        if r is None:
            return None
        c = r.get("confidence")
        if c is None:
            return None
        try:
            return float(c)
        except (TypeError, ValueError):
            return None

    def sort_bucket_rows(rows):
        rows.sort(
            key=lambda cr: (
                0 if numeric_confidence(cr) is not None else 1,
                -(numeric_confidence(cr) or 0.0),
                cr[0],
            )
        )

    for _k in buckets:
        sort_bucket_rows(buckets[_k])

    section_order = [
        ("强涨", f"强涨 | 强标签 P≥{UP_P_THRESHOLD:.0%} 且子模型投票≥{UP_VOTE_AGREE:.0%}"),
        ("强跌", f"强跌 | 标准模型 P≤{DN_P_THRESHOLD:.0%}"),
        ("偏涨", "偏涨 | 标准 P≥60%，未达强涨条件"),
        ("偏跌", "偏跌 | 标准 P≤40%，未达强跌条件"),
        ("震荡", "震荡 | 标准 P 在 40%–60%"),
        ("数据不足", "数据不足 | K 线长度不够，未参与预测"),
        ("特征失败", "特征失败 | 特征矩阵异常"),
        ("无K线", "无 K 线 | 本地无缓存或未拉到数据"),
        ("其他", "其他 | 未归类"),
    ]

    def print_merged_row(code, r):
        if r is None:
            print(
                f"  {code:12s} {'(无K线)':10s} {'—':6s} {'—':>6s} {'—':>7s} {'—':>7s} {'—':>6s} "
                f"{'—':>6s} {'—':4s} {'—':^4s} {'':16s} {'—':>10s} {'':12s}"
            )
            return
        inf = r.get("inflection")
        ik, iscore, istr, _fullk, ihint = inf_short(inf)
        ik = ik or "—"
        hint_disp = (ihint[:14] + "…") if len(ihint) > 14 else (ihint or "")
        sc_str = f"{iscore:6.1f}" if iscore is not None else "   —  "
        ist = str(istr) if istr is not None else "—"

        cf = r.get("confidence")
        if cf is not None:
            try:
                cf_str = f"{float(cf):5.0%}"
            except (TypeError, ValueError):
                cf_str = "   —  "
        else:
            cf_str = "   —  "

        ps = r.get("probability_std")
        if ps is not None:
            pss = f"{ps:.1%}"
            pst = f"{r.get('probability_strong', 0):.1%}"
            vr = f"{r.get('vote_ratio', 0):.0%}"
        else:
            pss = pst = vr = "   —  "

        lc = r.get("last_close")
        ld = r.get("last_date") or ""
        if lc is None:
            lc = "—"
        else:
            lc = str(lc)

        nm = r.get("name")
        name = ("" if nm is None or (isinstance(nm, float) and np.isnan(nm)) else str(nm))[:10]
        mt = model_tag(r)
        disp_tag = {"强涨": "强涨", "强跌": "强跌", "偏涨": "偏涨", "偏跌": "偏跌", "震荡": "震荡",
                    "数据不足": "数据不足", "特征失败": "特征失败"}.get(mt, "—")
        print(
            f"  {code:12s} {name:10s} {disp_tag:6s} {cf_str:>6s} {pss:>7s} {pst:>7s} {vr:>6s} "
            f"{sc_str} {ik:4s} {ist:^4s} {hint_disp:16s} {lc:>10s} {ld:12s}"
        )

    hdr_holdings = (
        f"  {'代码':12s} {'名称':10s} {'持仓':>8s} {'盈亏%':>7s} {'模型':6s} {'置信':>6s} {'标准P':>7s} {'强P':>7s} {'投票':>6s} "
        f"{'拐点分':>6s} {'拐点':4s} {'拐强':4s} {'拐点简述':16s} {'收盘':>10s} {'日期':12s}"
    )

    def print_holdings_row(code, r, pos_meta):
        qty = pos_meta.get("qty")
        if qty is None:
            qty_s = "       —"
        elif abs(qty - round(qty)) < 1e-6:
            qty_s = f"{int(round(qty)):>8d}"
        else:
            qty_s = f"{qty:>8.2f}"
        plr = pos_meta.get("pl_ratio")
        if plr is not None:
            pls = f"{plr:>+6.1f}%"
        else:
            pls = "     —"
        if r is None:
            print(
                f"  {code:12s} {'(无K线)':10s} {qty_s} {pls:>7s} {'—':6s} {'—':>6s} {'—':>7s} {'—':>7s} {'—':>6s} "
                f"{'—':>6s} {'—':4s} {'—':^4s} {'':16s} {'—':>10s} {'':12s}"
            )
            return
        inf = r.get("inflection")
        ik, iscore, istr, _fullk, ihint = inf_short(inf)
        ik = ik or "—"
        hint_disp = (ihint[:14] + "…") if len(ihint) > 14 else (ihint or "")
        sc_str = f"{iscore:6.1f}" if iscore is not None else "   —  "
        ist = str(istr) if istr is not None else "—"
        cf = r.get("confidence")
        if cf is not None:
            try:
                cf_str = f"{float(cf):5.0%}"
            except (TypeError, ValueError):
                cf_str = "   —  "
        else:
            cf_str = "   —  "
        ps = r.get("probability_std")
        if ps is not None:
            pss = f"{ps:.1%}"
            pst = f"{r.get('probability_strong', 0):.1%}"
            vr = f"{r.get('vote_ratio', 0):.0%}"
        else:
            pss = pst = vr = "   —  "
        lc = r.get("last_close")
        ld = r.get("last_date") or ""
        if lc is None:
            lc = "—"
        else:
            lc = str(lc)
        nm = r.get("name")
        name = ("" if nm is None or (isinstance(nm, float) and np.isnan(nm)) else str(nm))[:10]
        mt = model_tag(r)
        disp_tag = {"强涨": "强涨", "强跌": "强跌", "偏涨": "偏涨", "偏跌": "偏跌", "震荡": "震荡",
                    "数据不足": "数据不足", "特征失败": "特征失败"}.get(mt, "—")
        print(
            f"  {code:12s} {name:10s} {qty_s} {pls:>7s} {disp_tag:6s} {cf_str:>6s} {pss:>7s} {pst:>7s} {vr:>6s} "
            f"{sc_str} {ik:4s} {ist:^4s} {hint_disp:16s} {lc:>10s} {ld:12s}"
        )

    if position_codes:
        ph_list = [(c, by_code.get(c), position_by_code[c]) for c in position_codes]
        ph_list.sort(
            key=lambda t: (
                0 if numeric_confidence((t[0], t[1])) is not None else 1,
                -(numeric_confidence((t[0], t[1])) or 0.0),
                t[0],
            )
        )
        print(f"\n  {'─' * 56}")
        print(f"  【我的港股持仓 | {pos_env_label} · 共 {len(position_codes)} 只】（组内按置信度↓）")
        print(f"  {'─' * 56}")
        print(hdr_holdings)
        print("  " + "-" * 128)
        for code, r, meta in ph_list:
            print_holdings_row(code, r, meta)
    elif not pos_err:
        print(f"\n  {'─' * 56}")
        print(f"  【我的港股持仓 | {pos_env_label}】当前无港股持仓")
        print(f"  {'─' * 56}")

    for key, title in section_order:
        rows = buckets.get(key, [])
        if not rows:
            continue
        print(f"\n  {'─' * 56}")
        print(f"  【{title}】 共 {len(rows)} 只（组内按置信度↓）")
        print(f"  {'─' * 56}")
        print(hdr)
        print("  " + "-" * 118)
        for code, r in rows:
            print_merged_row(code, r)

    print(f"\n" + "-" * 118)
    total = len([r for r in results if r.get("probability_std") is not None])
    report_path = _predict_report_path()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "predict_forward_days": FORWARD,
        "dual_model_path": os.path.basename(DUAL_MODEL_PATH_HK),
        "thresholds": {
            "up_p_strong": UP_P_THRESHOLD,
            "up_vote_agree": UP_VOTE_AGREE,
            "up_vote_model": UP_VOTE_MODEL_THRESH,
            "dn_p_strong": DN_P_THRESHOLD,
        },
        "summary": {
            "watchlist_count": len(wl_codes),
            "codes_fetched_count": len(codes),
            "positions_hk_count": len(position_codes),
            "predictable_count": total,
            "strong_up": strong_up,
            "strong_dn": strong_dn,
            "weak_up": weak_up,
            "weak_dn": weak_dn,
            "neutral": neutral,
        },
        "hk_positions_env": pos_env_label,
        "hk_positions_error": pos_err,
        "hk_positions": _to_jsonable(position_rows),
        "results": _to_jsonable(results),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  总计 {total} 只可预测 / 自选 {len(wl_codes)} 只", end="")
    if len(codes) > len(wl_codes):
        print(f"（拉取 K 线含持仓-only 共 {len(codes)} 只）")
    else:
        print()
    print(f"  强涨:{strong_up} 强跌:{strong_dn} 偏涨:{weak_up} 偏跌:{weak_dn} 震荡:{neutral}")
    print(f"\n  已保存报告: {report_path}")
    print(f"\n  回测精度: 涨信号≈66%  跌信号≈76%")
    print(f"  仅供参考，不构成投资建议。")


if __name__ == "__main__":
    main()
