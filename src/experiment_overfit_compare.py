"""
V3.1 抗过拟合优化效果对比

直接运行当前代码(已含V3.1优化), 与之前的诊断结果对比:
  - 训练集 vs 测试集准确率差距
  - 季度间稳定性
  - 时间衰减
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_predictor import (
    compute_enhanced_feature_matrix, DiverseEnsemble, MODEL_CONFIGS,
    LOOKBACK, FORWARD, UP_LABEL_THRESHOLD, UP_P_THRESHOLD,
    UP_VOTE_MODEL_THRESH, UP_VOTE_AGREE, DN_P_THRESHOLD,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

WEIGHT_HALFLIFE = 90
TRAIN_WINDOWS = [6, 12, 24]

TEST_QUARTERS = [
    ("2021Q3", "2021-07-01", "2021-09-30"),
    ("2022Q2", "2022-04-01", "2022-06-30"),
    ("2023Q1", "2023-01-01", "2023-03-31"),
    ("2023Q4", "2023-10-01", "2023-12-31"),
    ("2024Q3", "2024-07-01", "2024-09-30"),
    ("2025Q2", "2025-04-01", "2025-06-30"),
    ("2026Q1", "2026-01-01", "2026-03-31"),
]

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

OLD_RESULTS = {
    "2021Q3": {"train_acc": 0.891, "test_acc": 0.889, "gap": 0.003, "n_test": 18},
    "2022Q2": {"train_acc": 0.828, "test_acc": 0.768, "gap": 0.060, "n_test": 272},
    "2023Q1": {"train_acc": 0.893, "test_acc": 0.739, "gap": 0.153, "n_test": 188},
    "2023Q4": {"train_acc": 0.894, "test_acc": 0.636, "gap": 0.258, "n_test": 22},
    "2024Q3": {"train_acc": 0.815, "test_acc": 0.704, "gap": 0.111, "n_test": 27},
    "2025Q2": {"train_acc": 0.932, "test_acc": 0.735, "gap": 0.198, "n_test": 49},
    "2026Q1": {"train_acc": 0.920, "test_acc": 0.375, "gap": 0.545, "n_test": 8},
}


def load_stock(code):
    p = os.path.join(DATA_DIR, f"{code.replace('.','_')}.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    df["time_key"] = pd.to_datetime(df["time_key"])
    for c in ["open","close","high","low","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","close","high","low","volume"]).sort_values("time_key").reset_index(drop=True)


def precompute(sl, up_threshold=0.0):
    r, orig = {}, {}
    for code in sl:
        df = load_stock(code)
        if df is None or len(df) < LOOKBACK + FORWARD + 30: continue
        feat = compute_enhanced_feature_matrix(df)
        future_ret = df["close"].shift(-FORWARD) / df["close"] - 1
        lab = (future_ret > up_threshold).astype(int)
        lab_orig = (future_ret > 0).astype(int)
        dt = df["time_key"]
        v = pd.Series(False, index=df.index)
        v.iloc[LOOKBACK - 1: len(df) - FORWARD] = True
        v = v & feat.notna().all(axis=1) & lab.notna()
        r[code] = (feat, lab, dt, v)
        orig[code] = lab_orig
    return r, orig


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
    Xs, ys, ws = [], [], []
    for code, (f, l, d, v) in sd.items():
        m = v & (d >= d0) & (d <= d1)
        if m.sum() == 0: continue
        Xs.append(f.loc[m]); ys.append(l.loc[m].values)
        if w_ref is not None:
            da = (w_ref - d.loc[m]).dt.days.values.astype(float)
            ws.append(np.power(0.5, da / WEIGHT_HALFLIFE))
        else:
            ws.append(np.ones(m.sum()))
    if not Xs: return None, None, None
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


def gather_orig(sd, orig_map, d0, d1):
    ys = []
    for code, (f, l, d, v) in sd.items():
        m = v & (d >= d0) & (d <= d1)
        if m.sum() == 0: continue
        ys.append(orig_map[code].loc[m].values)
    if not ys: return None
    return np.concatenate(ys)


def train_diverse(sd, qs):
    q = pd.Timestamp(qs)
    ens = DiverseEnsemble()
    last_data = None
    for wm in TRAIN_WINDOWS:
        ts, te = q - relativedelta(months=wm), q - pd.Timedelta(days=1)
        X, y, w = gather(sd, ts, te, w_ref=te)
        if X is None or len(X) < 200: continue
        for cfg in MODEL_CONFIGS:
            ens.add_gbm(X, y, sample_weight=w, config=cfg)
        ens.add_rf(X, y, sample_weight=w)
        ens.add_et(X, y, sample_weight=w)
        last_data = (X, y, w)
    if last_data:
        ens.add_lr(*last_data)
    return ens if ens.models else None


def get_individual_probs(ens, X):
    Xf = ens._prep(X)
    ps = []
    for t, m in ens.models:
        if t == "lr" and ens._scaler:
            ps.append(m.predict_proba(ens._scaler.transform(Xf))[:, 1])
        else:
            ps.append(m.predict_proba(Xf)[:, 1])
    return np.array(ps)


def eval_dual(std_m, str_m, X, y_orig):
    p_std = std_m.predict_proba_batch(X)
    indiv = get_individual_probs(str_m, X)
    p_strong = indiv.mean(axis=0)
    up_ratio = (indiv > UP_VOTE_MODEL_THRESH).mean(axis=0)

    is_up = (p_strong >= UP_P_THRESHOLD) & (up_ratio >= UP_VOTE_AGREE)
    is_dn = p_std <= DN_P_THRESHOLD
    valid = is_up | is_dn

    preds, ys = [], []
    for i in range(len(X)):
        if valid[i]:
            preds.append(1 if is_up[i] else 0)
            ys.append(y_orig[i])

    if not preds: return None
    preds, ys = np.array(preds), np.array(ys)
    up_m, dn_m = preds == 1, preds == 0
    return {
        "total_acc": float((preds == ys).mean()),
        "up_acc": float((ys[up_m] == 1).mean()) if up_m.sum() > 0 else 0,
        "dn_acc": float((ys[dn_m] == 0).mean()) if dn_m.sum() > 0 else 0,
        "n_up": int(up_m.sum()), "n_dn": int(dn_m.sum()), "n": len(ys),
    }


def main():
    t_start = time.time()
    print("=" * 80)
    print("  V3.1 抗过拟合优化效果验证")
    print("=" * 80)

    print(f"\n  GBM 正则化变化:")
    print(f"    max_depth: 2-5 → 2-4")
    print(f"    min_samples_leaf: 40-100 → 80-150")
    print(f"    l2_regularization: 1.0-4.0 → 4.0-10.0")
    print(f"    learning_rate: 0.02-0.08 → 0.015-0.05")
    print(f"  早停增强:")
    print(f"    validation_fraction: 0.12 → 0.20")
    print(f"    n_iter_no_change: 30 → 10")
    print(f"  RF/ET:")
    print(f"    max_depth: 8 → 6, min_samples_leaf: 50 → 80")
    print(f"  LR:")
    print(f"    C: 0.1 → 0.05")
    sys.stdout.flush()

    print("\n[预处理] 计算特征 ...", flush=True)
    sd_std, orig_std = precompute(STOCK_LIST, up_threshold=0.0)
    sd_std = add_market_features(sd_std)
    sd_str, orig_str = precompute(STOCK_LIST, up_threshold=UP_LABEL_THRESHOLD)
    sd_str = add_market_features(sd_str)

    def split_mkt(sd):
        hk = {k: v for k, v in sd.items() if k.startswith("HK.")}
        cn = {k: v for k, v in sd.items() if not k.startswith("HK.")}
        return hk, cn

    sd_std_hk, sd_std_cn = split_mkt(sd_std)
    sd_str_hk, sd_str_cn = split_mkt(sd_str)
    orig_hk = {k: v for k, v in orig_std.items() if k.startswith("HK.")}
    orig_cn = {k: v for k, v in orig_std.items() if not k.startswith("HK.")}
    print(f"  {len(sd_std)} 只股票就绪\n")

    # ═══════════════════════════════════════════════════════════════
    # 逐季度训练-测试对比
    # ═══════════════════════════════════════════════════════════════
    print("=" * 80)
    print("  训练集 vs 测试集准确率: V3 旧 → V3.1 新")
    print("=" * 80)
    print(f"  {'季度':>8s}  {'旧训练':>7s}  {'旧测试':>7s}  {'旧差距':>7s}  {'新训练':>7s}  {'新测试':>7s}  {'新差距':>7s}  {'改善':>7s}")
    print("  " + "-" * 70)
    sys.stdout.flush()

    new_results = []
    all_test_preds, all_test_labels = [], []

    for qn, qs, qe in TEST_QUARTERS:
        train_accs, train_ns = [], []
        test_accs, test_ns = [], []

        for sub_std, sub_str, sub_orig in [
            (sd_std_hk, sd_str_hk, orig_hk),
            (sd_std_cn, sd_str_cn, orig_cn),
        ]:
            if not sub_std: continue
            std_m = train_diverse(sub_std, qs)
            str_m = train_diverse(sub_str, qs)
            if std_m is None or str_m is None: continue

            # 训练集准确率
            q = pd.Timestamp(qs)
            ts = q - relativedelta(months=max(TRAIN_WINDOWS))
            te = q - pd.Timedelta(days=1)
            X_tr, _, _ = gather(sub_std, ts, te)
            y_tr_orig = gather_orig(sub_std, sub_orig, ts, te)
            if X_tr is not None and y_tr_orig is not None:
                r_tr = eval_dual(std_m, str_m, X_tr, y_tr_orig)
                if r_tr:
                    train_accs.append(r_tr["total_acc"] * r_tr["n"])
                    train_ns.append(r_tr["n"])

            # 测试集准确率
            X_te, _, _ = gather(sub_std, pd.Timestamp(qs), pd.Timestamp(qe))
            y_te_orig = gather_orig(sub_std, sub_orig, pd.Timestamp(qs), pd.Timestamp(qe))
            if X_te is not None and y_te_orig is not None:
                r_te = eval_dual(std_m, str_m, X_te, y_te_orig)
                if r_te:
                    test_accs.append(r_te["total_acc"] * r_te["n"])
                    test_ns.append(r_te["n"])

                    p_std = std_m.predict_proba_batch(X_te)
                    indiv = get_individual_probs(str_m, X_te)
                    p_strong = indiv.mean(axis=0)
                    ur = (indiv > UP_VOTE_MODEL_THRESH).mean(axis=0)
                    is_up = (p_strong >= UP_P_THRESHOLD) & (ur >= UP_VOTE_AGREE)
                    is_dn = p_std <= DN_P_THRESHOLD
                    for i in range(len(X_te)):
                        if is_up[i] or is_dn[i]:
                            all_test_preds.append(1 if is_up[i] else 0)
                            all_test_labels.append(y_te_orig[i])

        tr_n = sum(train_ns) if train_ns else 0
        te_n = sum(test_ns) if test_ns else 0
        new_tr = sum(train_accs) / tr_n if tr_n > 0 else 0
        new_te = sum(test_accs) / te_n if te_n > 0 else 0
        new_gap = new_tr - new_te

        old = OLD_RESULTS.get(qn, {})
        old_tr = old.get("train_acc", 0)
        old_te = old.get("test_acc", 0)
        old_gap = old.get("gap", 0)
        gap_improve = old_gap - new_gap

        new_results.append({
            "q": qn, "train_acc": new_tr, "test_acc": new_te,
            "gap": new_gap, "n_test": te_n,
        })

        print(f"  {qn:>8s}  {old_tr:>6.1%}  {old_te:>6.1%}  {old_gap:>+6.1%}  "
              f"{new_tr:>6.1%}  {new_te:>6.1%}  {new_gap:>+6.1%}  {gap_improve:>+6.1%}")
        sys.stdout.flush()

    # 汇总
    old_avg_gap = np.mean([r["gap"] for r in OLD_RESULTS.values()])
    new_avg_gap = np.mean([r["gap"] for r in new_results])
    old_avg_test = np.mean([r["test_acc"] for r in OLD_RESULTS.values()])
    new_avg_test = np.mean([r["test_acc"] for r in new_results])

    print(f"\n  {'指标':>16s}  {'V3 旧':>8s}  {'V3.1 新':>8s}  {'变化':>8s}")
    print("  " + "-" * 42)
    print(f"  {'平均训练-测试差距':>16s}  {old_avg_gap:>+7.1%}  {new_avg_gap:>+7.1%}  {new_avg_gap-old_avg_gap:>+7.1%}")
    print(f"  {'平均测试精度':>16s}  {old_avg_test:>7.1%}  {new_avg_test:>7.1%}  {new_avg_test-old_avg_test:>+7.1%}")

    # 时间衰减
    new_accs = [r["test_acc"] for r in new_results]
    if len(new_accs) >= 4:
        slope = np.polyfit(np.arange(len(new_accs)), new_accs, 1)[0]
        old_slope = -0.059
        print(f"  {'时间衰减斜率':>16s}  {old_slope:>+7.3f}  {slope:>+7.3f}  {slope-old_slope:>+7.3f}")

    # 涨跌分类
    if all_test_preds:
        P = np.array(all_test_preds)
        Y = np.array(all_test_labels)
        total_acc = float((P == Y).mean())
        up_m = P == 1; dn_m = P == 0
        up_acc = float((Y[up_m] == 1).mean()) if up_m.sum() > 0 else 0
        dn_acc = float((Y[dn_m] == 0).mean()) if dn_m.sum() > 0 else 0
        print(f"\n  V3.1 总体: 精度={total_acc:.1%}({len(P)}次信号) "
              f"涨={up_acc:.1%}({up_m.sum()}) 跌={dn_acc:.1%}({dn_m.sum()})")

    # 结论
    print(f"\n{'='*80}")
    print("  结论")
    print("=" * 80)

    if new_avg_gap < old_avg_gap * 0.7:
        print(f"\n  ✓ 过拟合显著缓解: 训练-测试差距从 {old_avg_gap:+.1%} 降至 {new_avg_gap:+.1%}")
    elif new_avg_gap < old_avg_gap * 0.9:
        print(f"\n  ✓ 过拟合有所缓解: 训练-测试差距从 {old_avg_gap:+.1%} 降至 {new_avg_gap:+.1%}")
    else:
        print(f"\n  差距变化不大: {old_avg_gap:+.1%} → {new_avg_gap:+.1%}")

    if new_avg_test > old_avg_test + 0.02:
        print(f"  ✓ 测试精度提升: {old_avg_test:.1%} → {new_avg_test:.1%}")
    elif new_avg_test < old_avg_test - 0.02:
        print(f"  ⚠️ 测试精度下降: {old_avg_test:.1%} → {new_avg_test:.1%} (正则化可能过强)")
    else:
        print(f"  测试精度基本持平: {old_avg_test:.1%} → {new_avg_test:.1%}")

    print(f"\n  总耗时: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")


if __name__ == "__main__":
    main()
