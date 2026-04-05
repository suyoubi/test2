"""
实验: 训练股票数量对准确率的影响

测试不同规模的股票列表:
  1. 20只 (核心蓝筹)
  2. 50只 (当前配置)
  3. 80只
  4. 全部 (~130只)
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

# 50只 (当前配置)
STOCKS_50 = [
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

# 20只核心蓝筹
STOCKS_20 = [
    "HK.00700","HK.09988","HK.01810","HK.09618","HK.09888",
    "HK.00005","HK.00388","HK.02318","HK.00941","HK.02800",
    "HK.01299","HK.03690","HK.01024","HK.01211","HK.00981",
    "SH.600519","SZ.000001","SH.601318","SH.600036","SZ.300750",
]


def discover_all_stocks():
    """从data目录发现所有可用股票"""
    stocks = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith(".csv"):
            continue
        code = f.replace(".csv", "").replace("_", ".")
        if code.startswith("HK.LIST") or code.startswith("HK.800"):
            continue
        stocks.append(code)
    return stocks


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


def gather_orig_labels(sd, orig_map, d0, d1):
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


def get_individual_probs(ens, X):
    Xf = ens._prep(X)
    ps = []
    for t, m in ens.models:
        if t == "lr" and ens._scaler:
            ps.append(m.predict_proba(ens._scaler.transform(Xf))[:, 1])
        else:
            ps.append(m.predict_proba(Xf)[:, 1])
    return np.array(ps)


def run_v3_backtest(stock_list, label):
    """运行V3双模型回测，返回结果"""
    t0 = time.time()
    print(f"\n  [{label}] 预计算 ({len(stock_list)}只) ...", end="", flush=True)

    sd_std, orig_std = precompute(stock_list, up_threshold=0.0)
    sd_std = add_market_features(sd_std)
    sd_str, orig_str = precompute(stock_list, up_threshold=UP_LABEL_THRESHOLD)
    sd_str = add_market_features(sd_str)
    n_valid = len(sd_std)
    print(f" {n_valid}只有效 ({time.time()-t0:.0f}s)")

    def split_mkt(sd):
        hk = {k: v for k, v in sd.items() if k.startswith("HK.")}
        cn = {k: v for k, v in sd.items() if not k.startswith("HK.")}
        return hk, cn

    sd_std_hk, sd_std_cn = split_mkt(sd_std)
    sd_str_hk, sd_str_cn = split_mkt(sd_str)
    orig_hk = {k: v for k, v in orig_std.items() if k.startswith("HK.")}
    orig_cn = {k: v for k, v in orig_std.items() if not k.startswith("HK.")}

    all_preds, all_labels, all_codes = [], [], []

    for qn, qs, qe in TEST_QUARTERS:
        for sub_std, sub_str, sub_orig in [
            (sd_std_hk, sd_str_hk, orig_hk),
            (sd_std_cn, sd_str_cn, orig_cn),
        ]:
            if not sub_std: continue
            std_m = train_diverse(sub_std, qs)
            str_m = train_diverse(sub_str, qs)
            if std_m is None or str_m is None: continue

            X, _, _, codes = gather(sub_std, pd.Timestamp(qs), pd.Timestamp(qe))
            y_orig = gather_orig_labels(sub_std, sub_orig, pd.Timestamp(qs), pd.Timestamp(qe))
            if X is None or y_orig is None: continue

            p_std = std_m.predict_proba_batch(X)
            indiv = get_individual_probs(str_m, X)
            p_strong = indiv.mean(axis=0)
            up_ratio = (indiv > UP_VOTE_MODEL_THRESH).mean(axis=0)

            is_up = (p_strong >= UP_P_THRESHOLD) & (up_ratio >= UP_VOTE_AGREE)
            is_dn = p_std <= DN_P_THRESHOLD
            valid = is_up | is_dn

            for i in range(len(X)):
                if valid[i]:
                    all_preds.append(1 if is_up[i] else 0)
                    all_labels.append(y_orig[i])
                    all_codes.append(codes[i])

        print(f"    {qn}", end="", flush=True)

    elapsed = time.time() - t0
    print(f" ({elapsed:.0f}s)")

    if not all_preds:
        return None

    PREDS = np.array(all_preds)
    Y = np.array(all_labels)
    C = np.array(all_codes)
    N = len(Y)

    up_m = PREDS == 1; dn_m = PREDS == 0
    n_up, n_dn = int(up_m.sum()), int(dn_m.sum())
    total_acc = float((PREDS == Y).mean())
    up_acc = float((Y[up_m] == 1).mean()) if n_up > 0 else 0
    dn_acc = float((Y[dn_m] == 0).mean()) if n_dn > 0 else 0
    baseline = float(Y.mean())

    # 训练样本统计
    total_train = 0
    for _, qs, _ in TEST_QUARTERS:
        q = pd.Timestamp(qs)
        for wm in TRAIN_WINDOWS:
            ts, te = q - relativedelta(months=wm), q - pd.Timedelta(days=1)
            _, y, _, _ = gather(sd_std, ts, te)
            if y is not None:
                total_train += len(y)
                break

    return {
        "label": label,
        "n_stocks_input": len(stock_list),
        "n_stocks_valid": n_valid,
        "n_signals": N,
        "total_acc": total_acc,
        "up_acc": up_acc, "n_up": n_up,
        "dn_acc": dn_acc, "n_dn": n_dn,
        "baseline": baseline,
        "n_stocks_signaled": len(set(C)),
        "train_samples_per_q": total_train,
        "elapsed": elapsed,
    }


def main():
    t_start = time.time()
    print("=" * 80)
    print("  实验: 训练股票数量对准确率的影响")
    print("=" * 80)

    all_stocks = discover_all_stocks()
    hk_all = [s for s in all_stocks if s.startswith("HK.")]
    cn_all = [s for s in all_stocks if s.startswith("SH.") or s.startswith("SZ.")]
    print(f"  可用数据: {len(all_stocks)}只 (港股{len(hk_all)} A股{len(cn_all)})")

    # 80只: 50只 + 额外30只港股
    extra_hk = [s for s in hk_all if s not in STOCKS_50][:30]
    stocks_80 = STOCKS_50 + extra_hk

    # 全部
    stocks_all = all_stocks

    configs = [
        (STOCKS_20, "20只(核心蓝筹)"),
        (STOCKS_50, "50只(当前配置)"),
        (stocks_80, "80只(+30港股)"),
        (stocks_all, f"全部({len(stocks_all)}只)"),
    ]

    results = []
    for sl, label in configs:
        r = run_v3_backtest(sl, label)
        if r:
            results.append(r)

    # ── 结果对比 ──
    print(f"\n\n{'='*80}")
    print("  结果对比")
    print("=" * 80)
    print(f"  {'配置':<20s} {'有效股票':>6s} {'训练样本':>8s} {'总精度':>7s} {'涨精度':>7s} {'涨次':>5s} "
          f"{'跌精度':>7s} {'跌次':>5s} {'信号数':>6s} {'基线':>5s} {'耗时':>5s}")
    print("  " + "-" * 95)

    for r in results:
        mk = "★" if r["total_acc"] >= 0.70 else " "
        print(f"  {mk}{r['label']:<19s} {r['n_stocks_valid']:>6d} {r['train_samples_per_q']:>8d} "
              f"{r['total_acc']:>6.1%} {r['up_acc']:>6.1%} {r['n_up']:>5d} "
              f"{r['dn_acc']:>6.1%} {r['n_dn']:>5d} {r['n_signals']:>6d} "
              f"{r['baseline']:>4.0%} {r['elapsed']:>4.0f}s")

    # ── 分析 ──
    print(f"\n  分析:")
    if len(results) >= 2:
        for i in range(1, len(results)):
            r0, r1 = results[0], results[i]
            d_total = r1["total_acc"] - r0["total_acc"]
            d_up = r1["up_acc"] - r0["up_acc"]
            d_dn = r1["dn_acc"] - r0["dn_acc"]
            print(f"    {r0['label']} → {r1['label']}: "
                  f"总{d_total:+.1%}  涨{d_up:+.1%}  跌{d_dn:+.1%}")

    # 50 vs 全量 直接对比
    r50 = next((r for r in results if "50" in r["label"]), None)
    rmax = results[-1] if results else None
    if r50 and rmax and r50 != rmax:
        d = rmax["total_acc"] - r50["total_acc"]
        print(f"\n  结论: 从50只增加到{rmax['n_stocks_valid']}只, 总精度变化 {d:+.1%}")
        if abs(d) < 0.02:
            print(f"    → 影响不大 (±2%以内), 当前50只配置已足够")
        elif d > 0:
            print(f"    → 有正面影响, 建议增加训练股票数量")
        else:
            print(f"    → 有负面影响, 更多股票可能引入噪音")

    print(f"\n  总耗时: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")


if __name__ == "__main__":
    main()
