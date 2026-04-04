"""
回测框架 — 最终版

算法:
  15模型时间维度集成 (5种GBM配置 × 3个训练窗口)
  置信度过滤: P(UP)≥0.80 → 涨, P(UP)≤0.20 → 跌, 否则不预测
  Walk-forward 季度滚动训练 + 指数衰减样本权重
"""

import os, sys, time, json, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from stock_predictor import (
    compute_feature_matrix, EnsemblePredictor, MODEL_CONFIGS,
    LOOKBACK, FORWARD, CONFIDENCE_MARGIN,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

DATA_START = "2019-06-01"
DATA_END   = "2026-04-01"
WEIGHT_HALFLIFE = 90
PRED_THRESHOLD = 0.35

TEST_QUARTERS = [
    ("2024Q3", "2024-07-01", "2024-09-30"),
    ("2024Q4", "2024-10-01", "2024-12-31"),
    ("2025Q1", "2025-01-01", "2025-03-31"),
    ("2025Q2", "2025-04-01", "2025-06-30"),
    ("2025Q3", "2025-07-01", "2025-09-30"),
    ("2025Q4", "2025-10-01", "2025-12-31"),
    ("2026Q1", "2026-01-01", "2026-03-31"),
]

TRAIN_WINDOWS = [9, 12, 18]

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


def precompute(sl):
    r = {}
    for code in sl:
        df = load_stock(code)
        if df is None or len(df) < LOOKBACK+FORWARD+30: continue
        feat = compute_feature_matrix(df)
        lab = (df["close"].shift(-FORWARD) > df["close"]).astype(int)
        dt = df["time_key"]
        v = pd.Series(False, index=df.index)
        v.iloc[LOOKBACK-1 : len(df)-FORWARD] = True
        v = v & feat.notna().all(axis=1) & lab.notna()
        r[code] = (feat, lab, dt, v)
    return r


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


def train_ensemble(sd, q_start_str):
    q = pd.Timestamp(q_start_str)
    predictor = EnsemblePredictor()
    for wm in TRAIN_WINDOWS:
        ts, te = q - relativedelta(months=wm), q - pd.Timedelta(days=1)
        X, y, w, _ = gather(sd, ts, te, w_ref=te)
        if X is None or len(X) < 200: continue
        for cfg in MODEL_CONFIGS:
            predictor.add_model(X, y, sample_weight=w, config=cfg)
    return predictor if predictor.models else None


def run_backtest(skip_fetch=False):
    n_models = len(TRAIN_WINDOWS) * len(MODEL_CONFIGS)
    print("=" * 70)
    print("  股票走势预测 — 最终回测")
    print("=" * 70)
    print(f"  算法: {n_models}模型集成 + 置信度过滤 (|P-0.5|≥{CONFIDENCE_MARGIN})")
    print(f"  K线窗口={LOOKBACK}天  预测周期={FORWARD}天")
    print(f"  预测规则: P≥{0.5+CONFIDENCE_MARGIN:.2f}→涨  P≤{0.5-CONFIDENCE_MARGIN:.2f}→跌  否则不预测")
    print(f"  测试: {TEST_QUARTERS[0][0]}~{TEST_QUARTERS[-1][0]}  股票={len(STOCK_LIST)}只")
    print("=" * 70); sys.stdout.flush()

    if not skip_fetch:
        print("\n[1] 获取数据 ..."); sys.stdout.flush()
        fetch_all_data(STOCK_LIST)
    else:
        print("\n[1] 使用缓存数据")

    print("\n[2] 预计算特征 ..."); sys.stdout.flush()
    sd = precompute(STOCK_LIST)
    sd = add_market_features(sd)
    print(f"  {len(sd)} 只股票就绪"); sys.stdout.flush()

    print(f"\n[3] Walk-forward 回测 ({len(TEST_QUARTERS)} 季度) ..."); sys.stdout.flush()

    all_proba, all_labels, all_codes = [], [], []

    for qname, qs, qe in TEST_QUARTERS:
        predictor = train_ensemble(sd, qs)
        if predictor is None:
            print(f"  {qname}: 失败"); continue

        X_te, y_te, _, te_codes = gather(sd, pd.Timestamp(qs), pd.Timestamp(qe))
        if X_te is None: continue

        p_up = predictor.predict_proba_batch(X_te)
        all_proba.extend(p_up.tolist())
        all_labels.extend(y_te.tolist())
        all_codes.extend(te_codes)

        # 本季度各策略准确率
        conf_mask = np.abs(p_up - 0.5) >= CONFIDENCE_MARGIN
        n_conf = int(conf_mask.sum())
        n_total = len(y_te)

        acc_full = float(((p_up >= PRED_THRESHOLD).astype(int) == y_te).mean())
        acc_conf = float(((p_up[conf_mask] >= PRED_THRESHOLD).astype(int) == y_te[conf_mask]).mean()) if n_conf > 0 else 0

        mk = "★" if acc_conf >= 0.6 else " "
        print(f"  {mk} {qname} UP={y_te.mean():.0%} | 全量={acc_full:.1%}({n_total}) "
              f"置信过滤={acc_conf:.1%}({n_conf}/{n_total}) [{len(predictor.models)}模型]")
        sys.stdout.flush()

    P = np.array(all_proba)
    Y = np.array(all_labels)
    C = np.array(all_codes)
    N = len(Y)

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)

    # 全量 (固定阈值)
    full_preds = (P >= PRED_THRESHOLD).astype(int)
    full_acc = float((full_preds == Y).mean())
    print(f"\n  全量预测 (t={PRED_THRESHOLD}): {full_acc:.1%} ({N}次)")

    # 置信度过滤
    conf_mask = np.abs(P - 0.5) >= CONFIDENCE_MARGIN
    n_conf = int(conf_mask.sum())
    conf_preds = (P[conf_mask] >= PRED_THRESHOLD).astype(int)
    conf_acc = float((conf_preds == Y[conf_mask]).mean()) if n_conf > 0 else 0
    coverage = n_conf / N
    n_stocks = len(set(C[conf_mask]))

    marker = "★★★" if conf_acc >= 0.6 else ""
    print(f"\n  置信度过滤 (|P-0.5|≥{CONFIDENCE_MARGIN}): {conf_acc:.1%} ({n_conf}次) {marker}")
    print(f"  覆盖率: {coverage:.1%} ({n_conf}/{N})")
    print(f"  股票覆盖: {n_stocks}/{len(set(C))}")

    # 不同置信度阈值对比
    print(f"\n  不同置信度阈值:")
    for mc in [0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.32]:
        m = np.abs(P - 0.5) >= mc
        cnt = int(m.sum())
        if cnt < 50: continue
        a = float(((P[m] >= PRED_THRESHOLD).astype(int) == Y[m]).mean())
        mk = "★" if a >= 0.6 else " "
        print(f"    {mk} |P-0.5|≥{mc:.2f}: {a:.1%} ({cnt}次, {cnt/N:.0%}覆盖, {len(set(C[m]))}股)")

    baseline = float(Y.mean())
    print(f"\n  基线 (始终预测涨): {baseline:.1%}")
    print(f"  模型全量提升: {full_acc - baseline:+.1%}")
    print(f"  模型过滤提升: {conf_acc - baseline:+.1%}")

    # ── UP / DOWN 细分 ──
    if conf_mask.sum() > 0:
        cp = conf_preds
        cy = Y[conf_mask]
        up_m = cp == 1
        dn_m = cp == 0
        if up_m.sum() > 0:
            print(f"\n  过滤预测 涨 的精度: {(cy[up_m]==1).mean():.1%} ({up_m.sum()}次)")
        if dn_m.sum() > 0:
            print(f"  过滤预测 跌 的精度: {(cy[dn_m]==0).mean():.1%} ({dn_m.sum()}次)")

    # ── 按股票 ──
    print(f"\n  按股票 (置信过滤):")
    sr = {}
    cc = C[conf_mask]
    for code in sorted(set(cc)):
        m = cc == code
        sc, sn = int((conf_preds[m]==Y[conf_mask][m]).sum()), int(m.sum())
        sr[code] = {"accuracy": sc/sn if sn else 0, "correct": sc, "total": sn}

    o60 = 0
    for code in sorted(sr, key=lambda c: sr[c]["accuracy"], reverse=True):
        r = sr[code]
        mk = "✓" if r["accuracy"] >= 0.6 else " "
        if r["accuracy"] >= 0.6: o60 += 1
        print(f"  {mk} {code:12s} {r['accuracy']:6.1%}  ({r['correct']}/{r['total']})")
    print(f"\n  ≥60% 的股票: {o60}/{len(sr)}")

    # ── 特征重要性 ──
    imp = predictor.feature_importances()
    if len(imp) > 0:
        print(f"\n  Top-15 特征:")
        for fname, score in imp.head(15).items():
            print(f"    {fname:25s} {score:.4f}")

    # 保存
    os.makedirs(RESULT_DIR, exist_ok=True)
    predictor.save(os.path.join(RESULT_DIR, "model.pkl"))
    result_obj = {
        "algorithm": f"{n_models}-model ensemble + confidence filter (|P-0.5|>={CONFIDENCE_MARGIN})",
        "full_accuracy": round(full_acc, 4),
        "filtered_accuracy": round(conf_acc, 4),
        "filtered_count": n_conf,
        "filtered_coverage": round(coverage, 4),
        "total_predictions": N,
        "stocks_covered": n_stocks,
        "baseline": round(baseline, 4),
        "per_stock": {k: {kk: round(vv,4) if isinstance(vv,float) else vv for kk,vv in v.items()} for k,v in sr.items()},
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(RESULT_DIR, "backtest_result.json"), "w") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {RESULT_DIR}/backtest_result.json")

    return conf_acc, result_obj


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="股票走势预测回测")
    p.add_argument("--skip-fetch", action="store_true", help="跳过数据获取")
    run_backtest(skip_fetch=p.parse_args().skip_fetch)
