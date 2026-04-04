"""
拉取富途港股自选股，用最新K线数据预测未来15天走势。
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stock_predictor import (
    compute_feature_matrix, EnsemblePredictor, MODEL_CONFIGS,
    LOOKBACK, FORWARD, CONFIDENCE_MARGIN,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULT_DIR, "model.pkl")
TRAIN_WINDOWS = [9, 12, 18]
WEIGHT_HALFLIFE = 90


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


def predict_stocks(stock_data, model, name_map=None):
    """对每只股票用模型进行预测"""
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
            })
            continue

        feat = compute_feature_matrix(df)

        # 添加市场相对强弱特征
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
            })
            continue

        p_up = model.predict_proba_batch(last_row)[0]
        confidence = abs(p_up - 0.5)

        if p_up >= 0.5 + CONFIDENCE_MARGIN:
            prediction = "涨"
            signal = "强"
        elif p_up <= 0.5 - CONFIDENCE_MARGIN:
            prediction = "跌"
            signal = "强"
        elif p_up >= 0.60:
            prediction = "偏涨"
            signal = "弱"
        elif p_up <= 0.40:
            prediction = "偏跌"
            signal = "弱"
        else:
            prediction = "震荡"
            signal = "无"

        last_close = df.iloc[-1]["close"]
        last_date = df.iloc[-1]["time_key"].strftime("%Y-%m-%d")

        results.append({
            "code": code,
            "name": name,
            "prediction": prediction,
            "signal": signal,
            "probability": round(p_up, 4),
            "confidence": round(confidence, 4),
            "last_close": round(last_close, 2),
            "last_date": last_date,
        })

    return results


def main():
    print("=" * 70)
    print("  港股自选股走势预测")
    print("=" * 70)
    print(f"  预测周期: 未来 {FORWARD} 天")
    print(f"  模型: 15模型集成 + 置信度过滤")
    print(f"  高置信阈值: P ≥ {0.5+CONFIDENCE_MARGIN:.0%} 或 P ≤ {0.5-CONFIDENCE_MARGIN:.0%}")
    print("=" * 70)
    sys.stdout.flush()

    # 1. 获取自选股
    print("\n[1] 获取港股自选 ...")
    sys.stdout.flush()
    codes, name_map = get_hk_watchlist()
    if not codes:
        print("  未找到港股自选股。请确认富途自选中有港股。")
        return
    print(f"  找到 {len(codes)} 只港股: {', '.join(codes[:10])}{'...' if len(codes) > 10 else ''}")
    sys.stdout.flush()

    # 2. 获取最新K线
    print(f"\n[2] 获取最新 K 线 (最近 {LOOKBACK+50} 天) ...")
    sys.stdout.flush()
    stock_data = fetch_recent_kline(codes, days=LOOKBACK + 50)
    print(f"  成功获取 {len(stock_data)} 只股票数据")

    # 3. 加载模型
    print("\n[3] 加载预测模型 ...")
    if not os.path.exists(MODEL_PATH):
        print(f"  模型文件不存在: {MODEL_PATH}")
        print("  请先运行 run_backtest.py 训练模型。")
        return
    model = EnsemblePredictor.load(MODEL_PATH)
    print(f"  模型已加载 ({len(model.models)} 个子模型)")

    # 4. 预测
    print("\n[4] 预测走势 ...")
    sys.stdout.flush()
    results = predict_stocks(stock_data, model, name_map)

    # 5. 输出结果
    print("\n" + "=" * 70)
    print("  预测结果 (未来 15 天)")
    print("=" * 70)

    strong_up = [r for r in results if r.get("signal") == "强" and r["prediction"] == "涨"]
    strong_dn = [r for r in results if r.get("signal") == "强" and r["prediction"] == "跌"]
    weak_up = [r for r in results if r.get("signal") == "弱" and r["prediction"] == "偏涨"]
    weak_dn = [r for r in results if r.get("signal") == "弱" and r["prediction"] == "偏跌"]
    neutral = [r for r in results if r.get("signal") == "无"]
    other = [r for r in results if r.get("signal") is None]

    def print_group(title, items, emoji):
        if not items:
            return
        print(f"\n  {emoji} {title}:")
        for r in sorted(items, key=lambda x: x.get("confidence", 0) or 0, reverse=True):
            p = r["probability"]
            c = r["confidence"]
            label = f"{r['code']} {r.get('name', '')}"
            print(f"    {label:24s}  P(涨)={p:.1%}  置信={c:.2f}  "
                  f"收盘={r['last_close']}  ({r['last_date']})")

    print_group("强烈看涨 (P≥80%)", strong_up, "🟢")
    print_group("强烈看跌 (P≤20%)", strong_dn, "🔴")
    print_group("偏看涨 (60%≤P<80%)", weak_up, "🟡")
    print_group("偏看跌 (20%<P≤40%)", weak_dn, "🟠")
    print_group("震荡 (40%<P<60%)", neutral, "⚪")

    if other:
        print(f"\n  ⚠️ 无法预测:")
        for r in other:
            label = f"{r['code']} {r.get('name', '')}"
            print(f"    {label:24s}  {r['prediction']}")

    # 统计
    print(f"\n" + "-" * 70)
    total = len([r for r in results if r.get("probability") is not None])
    print(f"  总计 {total} 只股票")
    print(f"  强烈看涨: {len(strong_up)}  强烈看跌: {len(strong_dn)}")
    print(f"  偏看涨: {len(weak_up)}  偏看跌: {len(weak_dn)}  震荡: {len(neutral)}")
    print(f"\n  注意: 仅 '强烈看涨/看跌' 信号经回测验证准确率 ≥60%")
    print(f"  其余信号仅供参考，不构成投资建议。")


if __name__ == "__main__":
    main()
