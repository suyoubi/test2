"""
回测框架 — V3 双模型版

算法:
  分市场独立建模 (港股/A股分开训练)
  双模型策略:
    - 跌信号: 标准模型(涨>0%标签), P≤0.20
    - 涨信号: 强标签模型(涨>3%标签) + 子模型超多数投票
  多样性集成: GBM(5配置×3窗口) + RF(3) + ExtraTrees(3) + LR(1)
  增强特征: 基础技术指标 + 日历/波动率状态/缺口/加速度
  Walk-forward 季度滚动训练 + 指数衰减样本权重（按日历切分，禁止随机打乱训练/测试）
"""

import os, sys, time, json, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from stock_predictor import (
    compute_enhanced_feature_matrix, DiverseEnsemble, DualModelPredictor,
    MODEL_CONFIGS, LOOKBACK, FORWARD, CONFIDENCE_MARGIN,
    UP_LABEL_THRESHOLD, UP_P_THRESHOLD, UP_VOTE_MODEL_THRESH,
    UP_VOTE_AGREE, DN_P_THRESHOLD,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

DATA_START = "2019-06-01"
DATA_END   = "2026-04-01"
WEIGHT_HALFLIFE = 90
PRED_THRESHOLD = 0.35

TEST_QUARTERS = [
    ("2021Q3", "2021-07-01", "2021-09-30"),
    ("2022Q2", "2022-04-01", "2022-06-30"),
    ("2023Q1", "2023-01-01", "2023-03-31"),
    ("2023Q4", "2023-10-01", "2023-12-31"),
    ("2024Q3", "2024-07-01", "2024-09-30"),
    ("2025Q2", "2025-04-01", "2025-06-30"),
    ("2026Q1", "2026-01-01", "2026-03-31"),
]

TRAIN_WINDOWS = [6, 12, 24]

STOCK_LIST = [
    "HK.00700","HK.09988","HK.03690","HK.01810","HK.09618",
    "HK.09888","HK.09999","HK.01024","HK.01211","HK.00981",
    "HK.00005","HK.00388","HK.01299","HK.02318","HK.00941",
    "HK.02800","HK.01347","HK.02015","HK.09866","HK.09868",
    "HK.00027","HK.00066","HK.00175","HK.00241","HK.00267",
    "HK.00285","HK.00288","HK.00386","HK.00669","HK.00688",
    "HK.00762","HK.00883","HK.00939","HK.00960","HK.01038",
    "HK.01088","HK.01177","HK.01398","HK.01928","HK.02007",
    "SH.600519","SZ.000001","SH.601318","SH.600036","SZ.300750",
    "SZ.000858","SH.600276","SH.601012","SZ.000333","SH.601888",
]


def fetch_all_data(sl):
    from futu import OpenQuoteContext, KLType, AuType, RET_OK
    os.makedirs(DATA_DIR, exist_ok=True)
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        for i, code in enumerate(sl):
            c = os.path.join(DATA_DIR, f"{code.replace('.','_')}.csv")
            if os.path.exists(c):
                print(f"  [{i+1}/{len(sl)}] {code} — 已缓存"); continue
            print(f"  [{i+1}/{len(sl)}] 获取 {code} ...")
            rows, pk = [], None
            while True:
                ret, data, pk = ctx.request_history_kline(
                    code, start=DATA_START, end=DATA_END,
                    ktype=KLType.K_DAY, autype=AuType.QFQ, max_count=1000, page_req_key=pk)
                if ret != RET_OK: print(f"    ✗ {data}"); break
                rows.append(data)
                if pk is None: break
                time.sleep(0.3)
            if rows:
                pd.concat(rows, ignore_index=True).to_csv(c, index=False)
                print(f"    ✓ {len(pd.concat(rows))} 天")
            time.sleep(0.5)
    finally:
        ctx.close()


def load_stock(code):
    p = os.path.join(DATA_DIR, f"{code.replace('.','_')}.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    df["time_key"] = pd.to_datetime(df["time_key"])
    for c in ["open","close","high","low","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","close","high","low","volume"]).sort_values("time_key").reset_index(drop=True)


def precompute(sl, up_threshold=0.0):
    """up_threshold: 涨幅超过该值才算涨 (0.0=任何正收益, 0.03=涨3%)"""
    r = {}
    orig_labels = {}
    for code in sl:
        df = load_stock(code)
        if df is None or len(df) < LOOKBACK+FORWARD+30: continue
        feat = compute_enhanced_feature_matrix(df)
        future_ret = df["close"].shift(-FORWARD) / df["close"] - 1
        lab = (future_ret > up_threshold).astype(int)
        lab_orig = (future_ret > 0).astype(int)
        dt = df["time_key"]
        v = pd.Series(False, index=df.index)
        v.iloc[LOOKBACK-1 : len(df)-FORWARD] = True
        v = v & feat.notna().all(axis=1) & lab.notna()
        r[code] = (feat, lab, dt, v)
        orig_labels[code] = lab_orig
    return r, orig_labels


def add_market_features(sd):
    mkt = {"HK": [], "CN": []}
    for code, (f, _, d, _) in sd.items():
        m = "HK" if code.startswith("HK") else "CN"
        r = f["ret_20d"].copy(); r.index = d
        mkt[m].append(r)
    bm = {m: pd.concat(rl, axis=1).mean(axis=1) for m, rl in mkt.items() if rl}
    for code in list(sd.keys()):
        f, l, d, v = sd[code]
        fc = f.copy()
        m = "HK" if code.startswith("HK") else "CN"
        if m in bm:
            al = bm[m].reindex(d.values)
            mv = al.values if len(al) == len(f) else np.zeros(len(f))
            if isinstance(mv, pd.Series): mv = mv.values
            fc["rel_strength"] = f["ret_20d"].values - mv
            fc["mkt_momentum"] = mv
        else:
            fc["rel_strength"] = 0.0; fc["mkt_momentum"] = 0.0
        sd[code] = (fc, l, d, v & fc.notna().all(axis=1))
    return sd


def gather(sd, d0, d1, w_ref=None):
    Xs, ys, ws, cs = [], [], [], []
    for code, (f, l, d, v) in sd.items():
        m = v & (d >= d0) & (d <= d1)
        if m.sum() == 0: continue
        Xs.append(f.loc[m]); ys.append(l.loc[m].values); cs.extend([code]*m.sum())
        if w_ref is not None:
            da = (w_ref - d.loc[m]).dt.days.values.astype(float)
            ws.append(np.power(0.5, da / WEIGHT_HALFLIFE))
        else:
            ws.append(np.ones(m.sum()))
    if not Xs: return None, None, None, None
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws), cs


def train_diverse_ensemble(sd, q_start_str):
    q = pd.Timestamp(q_start_str)
    ens = DiverseEnsemble()
    last_data = None
    for wm in TRAIN_WINDOWS:
        ts, te = q - relativedelta(months=wm), q - pd.Timedelta(days=1)
        X, y, w, _ = gather(sd, ts, te, w_ref=te)
        if X is None or len(X) < 200: continue
        for cfg in MODEL_CONFIGS:
            ens.add_gbm(X, y, sample_weight=w, config=cfg)
        ens.add_rf(X, y, sample_weight=w)
        ens.add_et(X, y, sample_weight=w)
        last_data = (X, y, w)
    if last_data:
        ens.add_lr(*last_data)
    return ens if ens.models else None


def gather_orig_labels(sd, orig_map, d0, d1):
    """收集原始标签 (涨>0%), 与gather对应顺序"""
    ys = []
    for code, (f, l, d, v) in sd.items():
        m = v & (d >= d0) & (d <= d1)
        if m.sum() == 0: continue
        ys.append(orig_map[code].loc[m].values)
    if not ys: return None
    return np.concatenate(ys)


def run_backtest(skip_fetch=False):
    print("=" * 70)
    print("  股票走势预测 — V3 双模型版回测")
    print("=" * 70)
    print(f"  算法: 双模型 (标准模型预测跌 + 强标签模型+投票预测涨)")
    print(f"  涨信号: 强{UP_LABEL_THRESHOLD:.0%}标签, P≥{UP_P_THRESHOLD}, 子模型>{UP_VOTE_MODEL_THRESH}需≥{UP_VOTE_AGREE:.0%}同意")
    print(f"  跌信号: 标准模型 P≤{DN_P_THRESHOLD}")
    print(f"  集成: GBM(5cfg×3win) + RF(3) + ET(3) + LR(1) ≈ 22模型/市场")
    print(f"  K线窗口={LOOKBACK}天  预测周期={FORWARD}天")
    print(f"  测试: {TEST_QUARTERS[0][0]}~{TEST_QUARTERS[-1][0]}  股票={len(STOCK_LIST)}只")
    print("=" * 70); sys.stdout.flush()

    if not skip_fetch:
        print("\n[1] 获取数据 ..."); sys.stdout.flush()
        fetch_all_data(STOCK_LIST)
    else:
        print("\n[1] 使用缓存数据")

    # 预处理两套数据: 标准标签 + 强涨标签
    print("\n[2] 预计算增强特征 ..."); sys.stdout.flush()
    sd_std, orig_std = precompute(STOCK_LIST, up_threshold=0.0)
    sd_std = add_market_features(sd_std)
    sd_strong, orig_strong = precompute(STOCK_LIST, up_threshold=UP_LABEL_THRESHOLD)
    sd_strong = add_market_features(sd_strong)

    def split_mkt(sd):
        hk = {k: v for k, v in sd.items() if k.startswith("HK.")}
        cn = {k: v for k, v in sd.items() if not k.startswith("HK.")}
        return hk, cn

    sd_std_hk, sd_std_cn = split_mkt(sd_std)
    sd_str_hk, sd_str_cn = split_mkt(sd_strong)
    orig_std_hk = {k: v for k, v in orig_std.items() if k.startswith("HK.")}
    orig_std_cn = {k: v for k, v in orig_std.items() if not k.startswith("HK.")}
    print(f"  {len(sd_std)} 只股票就绪 (港股 {len(sd_std_hk)}, A股 {len(sd_std_cn)})"); sys.stdout.flush()

    print(f"\n[3] Walk-forward 双模型分市场回测 ({len(TEST_QUARTERS)} 季度) ..."); sys.stdout.flush()

    all_preds, all_labels, all_codes, all_details = [], [], [], []
    last_dual_predictors = {}

    for qname, qs, qe in TEST_QUARTERS:
        q_preds, q_labels, q_codes = [], [], []

        for mkt_name, sub_std, sub_str, sub_orig in [
            ("HK", sd_std_hk, sd_str_hk, orig_std_hk),
            ("CN", sd_std_cn, sd_str_cn, orig_std_cn),
        ]:
            if not sub_std: continue

            std_model = train_diverse_ensemble(sub_std, qs)
            strong_model = train_diverse_ensemble(sub_str, qs)
            if std_model is None or strong_model is None: continue

            dual = DualModelPredictor(std_model, strong_model)
            last_dual_predictors[mkt_name] = dual

            # 测试集: 用标准数据的特征和原始标签评估
            X_te, _, _, te_codes = gather(sub_std, pd.Timestamp(qs), pd.Timestamp(qe))
            y_orig = gather_orig_labels(sub_std, sub_orig, pd.Timestamp(qs), pd.Timestamp(qe))
            if X_te is None or y_orig is None: continue

            preds, mask, details = dual.predict(X_te)
            for i in range(len(X_te)):
                if mask[i]:
                    q_preds.append(preds[i])
                    q_labels.append(y_orig[i])
                    q_codes.append(te_codes[i])

        if not q_preds:
            print(f"  {qname}: 无信号"); continue

        p_arr = np.array(q_preds)
        y_arr = np.array(q_labels)
        acc = float((p_arr == y_arr).mean())
        n_up = int((p_arr == 1).sum())
        n_dn = int((p_arr == 0).sum())
        up_acc = float((y_arr[p_arr == 1] == 1).mean()) if n_up > 0 else 0
        dn_acc = float((y_arr[p_arr == 0] == 0).mean()) if n_dn > 0 else 0

        all_preds.extend(q_preds)
        all_labels.extend(q_labels)
        all_codes.extend(q_codes)

        mk = "★" if acc >= 0.6 else " "
        print(f"  {mk} {qname} 总={acc:.1%}({len(q_preds)}) "
              f"涨={up_acc:.1%}({n_up}) 跌={dn_acc:.1%}({n_dn})")
        sys.stdout.flush()

    PREDS = np.array(all_preds)
    Y = np.array(all_labels)
    C = np.array(all_codes)
    N = len(Y)

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("  V3 双模型结果汇总")
    print("=" * 70)

    total_acc = float((PREDS == Y).mean())
    n_up_total = int((PREDS == 1).sum())
    n_dn_total = int((PREDS == 0).sum())
    up_acc = float((Y[PREDS == 1] == 1).mean()) if n_up_total > 0 else 0
    dn_acc = float((Y[PREDS == 0] == 0).mean()) if n_dn_total > 0 else 0
    baseline = float(Y.mean())

    print(f"\n  总精度: {total_acc:.1%} ({N}次信号)")
    print(f"  涨精度: {up_acc:.1%} ({n_up_total}次)")
    print(f"  跌精度: {dn_acc:.1%} ({n_dn_total}次)")
    print(f"  基线 (始终涨): {baseline:.1%}")
    print(f"  提升: {total_acc - baseline:+.1%}")

    # ── 分市场统计 ──
    for mkt_label, check_fn in [("港股", lambda c: c.startswith("HK.")),
                                  ("A股", lambda c: c.startswith("SH.") or c.startswith("SZ."))]:
        mm = np.array([check_fn(c) for c in C])
        if mm.sum() == 0: continue
        Pm, Ym = PREDS[mm], Y[mm]
        up_m = Pm == 1; dn_m = Pm == 0
        mkt_acc = float((Pm == Ym).mean())
        mkt_up = float((Ym[up_m] == 1).mean()) if up_m.sum() > 0 else 0
        mkt_dn = float((Ym[dn_m] == 0).mean()) if dn_m.sum() > 0 else 0
        print(f"\n  {mkt_label}: 总={mkt_acc:.1%}({mm.sum()}) "
              f"涨={mkt_up:.1%}({up_m.sum()}) 跌={mkt_dn:.1%}({dn_m.sum()}) "
              f"基线={Ym.mean():.1%}")

    # ── 按股票 ──
    print(f"\n  按股票:")
    sr = {}
    for code in sorted(set(C)):
        m = C == code
        pm, ym = PREDS[m], Y[m]
        sc = int((pm == ym).sum())
        sn = int(m.sum())
        up_m = pm == 1
        sr[code] = {
            "accuracy": sc/sn if sn else 0, "correct": sc, "total": sn,
            "up_count": int(up_m.sum()),
            "up_acc": float((ym[up_m]==1).mean()) if up_m.sum() > 0 else 0,
        }

    o60 = 0
    for code in sorted(sr, key=lambda c: sr[c]["accuracy"], reverse=True):
        r = sr[code]
        mk = "✓" if r["accuracy"] >= 0.6 else " "
        if r["accuracy"] >= 0.6: o60 += 1
        print(f"  {mk} {code:12s} 总={r['accuracy']:5.1%}({r['total']:>2d}) "
              f"涨={r['up_acc']:5.1%}({r['up_count']})")
    print(f"\n  ≥60% 的股票: {o60}/{len(sr)}")

    # ── 特征重要性 ──
    for mkt_name, dual in last_dual_predictors.items():
        imp = dual.strong_model.feature_importances()
        if len(imp) > 0:
            print(f"\n  {mkt_name} 强标签模型 Top-10 特征:")
            for fname, score in imp.head(10).items():
                print(f"    {fname:25s} {score:.4f}")

    # ── 保存模型 ──
    os.makedirs(RESULT_DIR, exist_ok=True)
    for mkt_name, dual in last_dual_predictors.items():
        model_path = os.path.join(RESULT_DIR, f"dual_model_{mkt_name.lower()}.pkl")
        dual.save(model_path)
        n_std = len(dual.std_model.models)
        n_strong = len(dual.strong_model.models)
        print(f"\n  双模型已保存: {model_path} (标准{n_std}+强标签{n_strong}个子模型)")

    result_obj = {
        "algorithm": f"V3 dual-model: std(dn P<={DN_P_THRESHOLD}) + strong{UP_LABEL_THRESHOLD:.0%}(up P>={UP_P_THRESHOLD} vote>={UP_VOTE_AGREE:.0%})",
        "total_accuracy": round(total_acc, 4),
        "up_accuracy": round(up_acc, 4),
        "dn_accuracy": round(dn_acc, 4),
        "up_count": n_up_total,
        "dn_count": n_dn_total,
        "total_signals": N,
        "baseline": round(baseline, 4),
        "per_stock": {k: {kk: round(vv,4) if isinstance(vv,float) else vv for kk,vv in v.items()} for k,v in sr.items()},
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(RESULT_DIR, "backtest_result.json"), "w") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {RESULT_DIR}/backtest_result.json")

    return total_acc, result_obj


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="股票走势预测回测 V3")
    p.add_argument("--skip-fetch", action="store_true", help="跳过数据获取")
    run_backtest(skip_fetch=p.parse_args().skip_fetch)
