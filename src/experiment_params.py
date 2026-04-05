"""
参数对比实验：测试不同 TRAIN_WINDOWS 和 WEIGHT_HALFLIFE 组合
"""

import os, sys, time
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_predictor import (
    compute_feature_matrix, EnsemblePredictor, MODEL_CONFIGS,
    LOOKBACK, FORWARD, CONFIDENCE_MARGIN,
)
from run_backtest import (
    STOCK_LIST, TEST_QUARTERS, PRED_THRESHOLD,
    load_stock, add_market_features,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

CONFIGS = [
    {"name": "基线 [9,12,18] hl=90",   "windows": [9, 12, 18], "halflife": 90},
    {"name": "短窗口 [6,9,12] hl=90",   "windows": [6, 9, 12],  "halflife": 90},
    {"name": "宽跨度 [6,12,24] hl=90",  "windows": [6, 12, 24], "halflife": 90},
    {"name": "基线窗口 hl=60",          "windows": [9, 12, 18], "halflife": 60},
    {"name": "基线窗口 hl=120",         "windows": [9, 12, 18], "halflife": 120},
    {"name": "短窗口 [6,9,12] hl=60",   "windows": [6, 9, 12],  "halflife": 60},
]


def precompute(sl):
    r = {}
    for code in sl:
        df = load_stock(code)
        if df is None or len(df) < LOOKBACK + FORWARD + 30:
            continue
        feat = compute_feature_matrix(df)
        lab = (df["close"].shift(-FORWARD) > df["close"]).astype(int)
        dt = df["time_key"]
        v = pd.Series(False, index=df.index)
        v.iloc[LOOKBACK - 1 : len(df) - FORWARD] = True
        v = v & feat.notna().all(axis=1) & lab.notna()
        r[code] = (feat, lab, dt, v)
    return r


def gather(sd, d0, d1, w_ref=None, halflife=90):
    Xs, ys, ws = [], [], []
    for code, (f, l, d, v) in sd.items():
        m = v & (d >= d0) & (d <= d1)
        if m.sum() == 0:
            continue
        Xs.append(f.loc[m])
        ys.append(l.loc[m].values)
        if w_ref is not None:
            da = (w_ref - d.loc[m]).dt.days.values.astype(float)
            ws.append(np.power(0.5, da / halflife))
        else:
            ws.append(np.ones(m.sum()))
    if not Xs:
        return None, None, None
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


def train_ensemble(sd, q_start_str, windows, halflife):
    q = pd.Timestamp(q_start_str)
    predictor = EnsemblePredictor()
    for wm in windows:
        ts = q - relativedelta(months=wm)
        te = q - pd.Timedelta(days=1)
        X, y, w = gather(sd, ts, te, w_ref=te, halflife=halflife)
        if X is None or len(X) < 200:
            continue
        for cfg in MODEL_CONFIGS:
            predictor.add_model(X, y, sample_weight=w, config=cfg)
    return predictor if predictor.models else None


def evaluate_config(sd, cfg):
    windows = cfg["windows"]
    halflife = cfg["halflife"]

    all_proba, all_labels = [], []
    quarter_results = []

    for qname, qs, qe in TEST_QUARTERS:
        predictor = train_ensemble(sd, qs, windows, halflife)
        if predictor is None:
            continue

        X_te, y_te, _ = gather(sd, pd.Timestamp(qs), pd.Timestamp(qe))
        if X_te is None:
            continue

        p_up = predictor.predict_proba_batch(X_te)
        all_proba.extend(p_up.tolist())
        all_labels.extend(y_te.tolist())

        conf_mask = np.abs(p_up - 0.5) >= CONFIDENCE_MARGIN
        n_conf = int(conf_mask.sum())
        acc_conf = float(((p_up[conf_mask] >= PRED_THRESHOLD).astype(int) == y_te[conf_mask]).mean()) if n_conf > 0 else 0
        quarter_results.append((qname, acc_conf, n_conf, len(y_te)))

    P = np.array(all_proba)
    Y = np.array(all_labels)
    N = len(Y)

    full_acc = float(((P >= PRED_THRESHOLD).astype(int) == Y).mean())
    conf_mask = np.abs(P - 0.5) >= CONFIDENCE_MARGIN
    n_conf = int(conf_mask.sum())
    conf_acc = float(((P[conf_mask] >= PRED_THRESHOLD).astype(int) == Y[conf_mask]).mean()) if n_conf > 0 else 0
    coverage = n_conf / N if N > 0 else 0

    return {
        "full_acc": full_acc,
        "conf_acc": conf_acc,
        "n_conf": n_conf,
        "total": N,
        "coverage": coverage,
        "quarters": quarter_results,
        "n_models": len(windows) * len(MODEL_CONFIGS),
    }


def main():
    print("=" * 70)
    print("  参数对比实验")
    print("=" * 70)
    print(f"  测试 {len(CONFIGS)} 种配置 × {len(TEST_QUARTERS)} 季度")
    print(f"  股票数: {len(STOCK_LIST)}")
    print()

    print("[1] 预计算特征 ...")
    t0 = time.time()
    sd = precompute(STOCK_LIST)
    sd = add_market_features(sd)
    print(f"  {len(sd)} 只股票就绪 ({time.time()-t0:.1f}s)")

    results = {}
    for i, cfg in enumerate(CONFIGS):
        print(f"\n[{i+2}] 测试: {cfg['name']}  (windows={cfg['windows']}, halflife={cfg['halflife']})")
        t0 = time.time()
        r = evaluate_config(sd, cfg)
        elapsed = time.time() - t0

        results[cfg["name"]] = r

        for qname, qacc, qn, qtotal in r["quarters"]:
            mk = "★" if qacc >= 0.6 else " "
            print(f"    {mk} {qname}: {qacc:.1%} ({qn}/{qtotal})")
        print(f"  → 过滤准确率={r['conf_acc']:.1%}  全量={r['full_acc']:.1%}  "
              f"覆盖={r['coverage']:.0%} ({r['n_conf']}/{r['total']})  "
              f"[{r['n_models']}模型, {elapsed:.0f}s]")

    print("\n" + "=" * 70)
    print("  对比结果汇总")
    print("=" * 70)
    print(f"\n  {'配置':<28s} {'过滤准确率':>10s} {'全量准确率':>10s} {'覆盖率':>8s} {'过滤次数':>8s}")
    print("  " + "-" * 68)

    best_name, best_acc = "", 0
    for name, r in sorted(results.items(), key=lambda x: x[1]["conf_acc"], reverse=True):
        mk = "★" if r["conf_acc"] > best_acc or r["conf_acc"] == max(rr["conf_acc"] for rr in results.values()) else " "
        if r["conf_acc"] > best_acc:
            best_acc = r["conf_acc"]
            best_name = name
        print(f"  {mk} {name:<26s} {r['conf_acc']:>9.1%} {r['full_acc']:>9.1%} "
              f"{r['coverage']:>7.0%} {r['n_conf']:>7d}")

    print(f"\n  最佳配置: {best_name} (过滤准确率 {best_acc:.1%})")

    # 按季度细分对比
    print(f"\n  按季度过滤准确率对比:")
    header = f"  {'季度':<8s}"
    for name in results:
        short = name[:12]
        header += f" {short:>12s}"
    print(header)
    print("  " + "-" * (8 + 13 * len(results)))

    for qi, (qname, _, _) in enumerate(TEST_QUARTERS):
        row = f"  {qname:<8s}"
        for name, r in results.items():
            found = False
            for qn, qacc, qcnt, _ in r["quarters"]:
                if qn == qname:
                    row += f" {qacc:>11.1%}" if qcnt > 0 else f" {'N/A':>12s}"
                    found = True
                    break
            if not found:
                row += f" {'N/A':>12s}"
        print(row)

    print("\n  实验完成！")


if __name__ == "__main__":
    main()
