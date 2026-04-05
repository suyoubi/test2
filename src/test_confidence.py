"""
V3.1 双模型置信度全面测试（与 stock_predictor 早停/正则化一致）
分析双模型系统在不同参数下的准确率、覆盖率、涨跌细分、季度稳定性。
"""

import os, sys, time
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from stock_predictor import (
    compute_enhanced_feature_matrix, DiverseEnsemble, MODEL_CONFIGS,
    LOOKBACK, FORWARD, CONFIDENCE_MARGIN,
    UP_LABEL_THRESHOLD, UP_P_THRESHOLD, UP_VOTE_MODEL_THRESH,
    UP_VOTE_AGREE, DN_P_THRESHOLD,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

WEIGHT_HALFLIFE = 90

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


def load_stock(code):
    p = os.path.join(DATA_DIR, f"{code.replace('.','_')}.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    df["time_key"] = pd.to_datetime(df["time_key"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "close", "high", "low", "volume"]).sort_values("time_key").reset_index(drop=True)


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
        Xs.append(f.loc[m]); ys.append(l.loc[m].values); cs.extend([code] * m.sum())
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


def main():
    t_start = time.time()
    print("=" * 90)
    print("  V3.1 双模型置信度全面测试")
    print("=" * 90)
    print(f"  当前V3.1参数:")
    print(f"    涨: 强{UP_LABEL_THRESHOLD:.0%}标签 P≥{UP_P_THRESHOLD} 子模型>{UP_VOTE_MODEL_THRESH}需≥{UP_VOTE_AGREE:.0%}同意")
    print(f"    跌: 标准模型 P≤{DN_P_THRESHOLD}")
    print(f"  测试季度: {TEST_QUARTERS[0][0]}~{TEST_QUARTERS[-1][0]} ({len(TEST_QUARTERS)}个)")
    print(f"  股票: {len(STOCK_LIST)}只")
    print("=" * 90)
    sys.stdout.flush()

    # ── 预计算 ──
    print("\n[1] 预计算特征 ...")
    sys.stdout.flush()
    sd_std, orig_std = precompute(STOCK_LIST, up_threshold=0.0)
    sd_std = add_market_features(sd_std)
    sd_strong, orig_strong = precompute(STOCK_LIST, up_threshold=UP_LABEL_THRESHOLD)
    sd_strong = add_market_features(sd_strong)
    print(f"  {len(sd_std)} 只股票就绪")

    def split_mkt(sd):
        hk = {k: v for k, v in sd.items() if k.startswith("HK.")}
        cn = {k: v for k, v in sd.items() if not k.startswith("HK.")}
        return hk, cn

    sd_std_hk, sd_std_cn = split_mkt(sd_std)
    sd_str_hk, sd_str_cn = split_mkt(sd_strong)
    orig_std_hk = {k: v for k, v in orig_std.items() if k.startswith("HK.")}
    orig_std_cn = {k: v for k, v in orig_std.items() if not k.startswith("HK.")}

    # ── Walk-forward 收集所有原始数据 ──
    print(f"\n[2] Walk-forward 双模型训练 ({len(TEST_QUARTERS)} 季度) ...")
    sys.stdout.flush()

    all_p_std, all_p_strong = [], []
    all_indiv_strong = []
    all_y_orig, all_codes, all_quarters = [], [], []

    for qname, qs, qe in TEST_QUARTERS:
        t0 = time.time()
        for mkt_name, sub_std, sub_str, sub_orig in [
            ("HK", sd_std_hk, sd_str_hk, orig_std_hk),
            ("CN", sd_std_cn, sd_str_cn, orig_std_cn),
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

            all_p_std.extend(p_std.tolist())
            all_p_strong.extend(p_strong.tolist())
            all_indiv_strong.append(indiv)
            all_y_orig.extend(y_orig.tolist())
            all_codes.extend(codes)
            all_quarters.extend([qname] * len(y_orig))

        print(f"  {qname} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

    P_STD = np.array(all_p_std)
    P_STR = np.array(all_p_strong)
    INDIV = np.concatenate(all_indiv_strong, axis=1)
    Y = np.array(all_y_orig)
    C = np.array(all_codes)
    Q = np.array(all_quarters)
    N = len(Y)
    baseline = float(Y.mean())

    print(f"\n  总样本: {N}, 基线(涨比例): {baseline:.1%}")
    print(f"  训练完成 ({time.time()-t_start:.0f}s)")

    # ═══════════════════════════════════════════════════════════════
    # 测试一：V3 双模型不同参数组合
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  测试一：V3.1 双模型参数扫描")
    print("=" * 90)
    print(f"  {'配置':<50s} {'涨精度':>7s} {'涨次':>5s} {'跌精度':>7s} {'跌次':>5s} {'总精度':>7s} {'总次':>5s}")
    print("  " + "-" * 86)

    best_results = []

    for up_p in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for mt in [0.55, 0.60, 0.65, 0.70]:
            for agree in [0.60, 0.70, 0.80, 0.90]:
                for dn_p in [0.15, 0.20, 0.25]:
                    up_ratio = (INDIV > mt).mean(axis=0)
                    is_up = (P_STR >= up_p) & (up_ratio >= agree)
                    is_dn = P_STD <= dn_p
                    valid = is_up | is_dn
                    n = int(valid.sum())
                    if n < 30: continue

                    preds = np.where(is_up[valid], 1, 0)
                    y_f = Y[valid]
                    up_m = preds == 1; dn_m = preds == 0
                    n_up = int(up_m.sum()); n_dn = int(dn_m.sum())
                    total_acc = float((preds == y_f).mean())
                    up_acc = float((y_f[up_m] == 1).mean()) if n_up > 0 else 0
                    dn_acc = float((y_f[dn_m] == 0).mean()) if n_dn > 0 else 0

                    best_results.append({
                        "name": f"涨P≥{up_p}+m>{mt}×{agree:.0%} 跌P≤{dn_p}",
                        "up_p": up_p, "mt": mt, "agree": agree, "dn_p": dn_p,
                        "total_acc": total_acc, "up_acc": up_acc, "dn_acc": dn_acc,
                        "n_up": n_up, "n_dn": n_dn, "n": n,
                    })

    # 当前V3配置标记
    v3_name = f"涨P≥{UP_P_THRESHOLD}+m>{UP_VOTE_MODEL_THRESH}×{UP_VOTE_AGREE:.0%} 跌P≤{DN_P_THRESHOLD}"

    # 按总精度排序显示 Top-20
    top_total = sorted(best_results, key=lambda x: x["total_acc"], reverse=True)
    for i, r in enumerate(top_total[:20]):
        v3 = "→" if r["name"] == v3_name else " "
        mk = "★" if r["total_acc"] >= 0.70 else " "
        print(f" {v3}{mk}{r['name']:<50s} {r['up_acc']:>6.1%} {r['n_up']:>5d} "
              f"{r['dn_acc']:>6.1%} {r['n_dn']:>5d} {r['total_acc']:>6.1%} {r['n']:>5d}")

    # ═══════════════════════════════════════════════════════════════
    # 测试二：不同跌信号阈值 (固定当前涨参数)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  测试二：跌信号阈值扫描 (涨参数固定为V3默认)")
    print("=" * 90)

    up_ratio_v3 = (INDIV > UP_VOTE_MODEL_THRESH).mean(axis=0)
    is_up_v3 = (P_STR >= UP_P_THRESHOLD) & (up_ratio_v3 >= UP_VOTE_AGREE)

    print(f"  {'跌P≤':>7s}  {'总精度':>7s}  {'涨精度':>7s}  {'涨次':>5s}  {'跌精度':>7s}  {'跌次':>5s}  {'总次':>5s}  {'覆盖率':>7s}")
    print("  " + "-" * 60)

    for dn_t in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]:
        is_dn = P_STD <= dn_t
        valid = is_up_v3 | is_dn
        n = int(valid.sum())
        if n < 10: continue
        preds = np.where(is_up_v3[valid], 1, 0)
        y_f = Y[valid]
        up_m = preds == 1; dn_m = preds == 0
        n_up, n_dn = int(up_m.sum()), int(dn_m.sum())
        total_acc = float((preds == y_f).mean())
        up_acc = float((y_f[up_m] == 1).mean()) if n_up > 0 else 0
        dn_acc = float((y_f[dn_m] == 0).mean()) if n_dn > 0 else 0
        mk = "→" if abs(dn_t - DN_P_THRESHOLD) < 0.001 else " "
        print(f" {mk} {dn_t:5.2f}  {total_acc:>6.1%}  {up_acc:>6.1%}  {n_up:>5d}  "
              f"{dn_acc:>6.1%}  {n_dn:>5d}  {n:>5d}  {n/N:>6.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 测试三：不同涨信号参数 (固定跌阈值=0.20)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  测试三：涨信号参数扫描 (跌P≤0.20固定)")
    print("=" * 90)
    print(f"  {'配置':<40s} {'涨精度':>7s} {'涨次':>5s} {'总精度':>7s} {'总次':>5s}")
    print("  " + "-" * 62)

    is_dn_fixed = P_STD <= 0.20
    for up_p in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for mt in [0.55, 0.60, 0.65, 0.70]:
            for agree in [0.60, 0.70, 0.80, 0.90]:
                up_ratio = (INDIV > mt).mean(axis=0)
                is_up = (P_STR >= up_p) & (up_ratio >= agree)
                valid = is_up | is_dn_fixed
                n = int(valid.sum())
                if n < 30: continue
                preds = np.where(is_up[valid], 1, 0)
                y_f = Y[valid]
                up_m = preds == 1
                n_up = int(up_m.sum())
                if n_up < 5: continue
                total_acc = float((preds == y_f).mean())
                up_acc = float((y_f[up_m] == 1).mean()) if n_up > 0 else 0
                mk = "★" if up_acc >= 0.60 else " "
                v3 = "→" if (abs(up_p - UP_P_THRESHOLD) < 0.001 and
                             abs(mt - UP_VOTE_MODEL_THRESH) < 0.001 and
                             abs(agree - UP_VOTE_AGREE) < 0.001) else " "
                print(f" {v3}{mk} P≥{up_p}+m>{mt}×{agree:.0%}{'':<15s} "
                      f"{up_acc:>6.1%} {n_up:>5d} {total_acc:>6.1%} {n:>5d}")

    # ═══════════════════════════════════════════════════════════════
    # 测试四：季度稳定性 (V3当前参数)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  测试四：V3.1 各季度详情")
    print("=" * 90)
    print(f"  {'季度':>8s}  {'总精度':>7s}  {'涨精度':>7s}  {'涨次':>5s}  {'跌精度':>7s}  {'跌次':>5s}  {'总次':>5s}  {'基线':>5s}")
    print("  " + "-" * 62)

    q_accs = []
    for qn in sorted(set(Q)):
        qm = Q == qn
        ps, pstr, yo, indv = P_STD[qm], P_STR[qm], Y[qm], INDIV[:, qm]
        ur = (indv > UP_VOTE_MODEL_THRESH).mean(axis=0)
        is_up = (pstr >= UP_P_THRESHOLD) & (ur >= UP_VOTE_AGREE)
        is_dn = ps <= DN_P_THRESHOLD
        valid = is_up | is_dn
        n = int(valid.sum())
        if n == 0:
            print(f"  {qn:>8s}  {'无信号':>7s}"); continue

        preds = np.where(is_up[valid], 1, 0)
        y_f = yo[valid]
        up_m = preds == 1; dn_m = preds == 0
        n_up, n_dn = int(up_m.sum()), int(dn_m.sum())
        total_acc = float((preds == y_f).mean())
        up_acc = float((y_f[up_m] == 1).mean()) if n_up > 0 else 0
        dn_acc = float((y_f[dn_m] == 0).mean()) if n_dn > 0 else 0
        bl = float(yo.mean())
        mk = "★" if total_acc >= 0.70 else " "
        print(f"  {mk}{qn:>7s}  {total_acc:>6.1%}  {up_acc:>6.1%}  {n_up:>5d}  "
              f"{dn_acc:>6.1%}  {n_dn:>5d}  {n:>5d}  {bl:>4.0%}")
        q_accs.append(total_acc)

    if q_accs:
        print(f"\n  季度精度: 均值={np.mean(q_accs):.1%}  标准差={np.std(q_accs):.1%}  "
              f"最高={max(q_accs):.1%}  最低={min(q_accs):.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 测试五：分市场 (港股 vs A股)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  测试五：港股 vs A股")
    print("=" * 90)

    for mkt_label, check_fn in [("港股", lambda c: c.startswith("HK.")),
                                  ("A股", lambda c: c.startswith("SH.") or c.startswith("SZ."))]:
        mm = np.array([check_fn(c) for c in C])
        if mm.sum() == 0: continue
        ps, pstr, yo, indv = P_STD[mm], P_STR[mm], Y[mm], INDIV[:, mm]
        bl = float(yo.mean())

        print(f"\n  {mkt_label} — 样本={mm.sum()}, 基线={bl:.1%}")
        print(f"  {'跌P≤':>7s}  {'总精度':>7s}  {'涨精度':>7s}  {'涨次':>5s}  {'跌精度':>7s}  {'跌次':>5s}  {'总次':>5s}")
        print("  " + "-" * 50)

        ur = (indv > UP_VOTE_MODEL_THRESH).mean(axis=0)
        is_up = (pstr >= UP_P_THRESHOLD) & (ur >= UP_VOTE_AGREE)

        for dn_t in [0.15, 0.20, 0.25, 0.30]:
            is_dn = ps <= dn_t
            valid = is_up | is_dn
            n = int(valid.sum())
            if n < 5: continue
            preds = np.where(is_up[valid], 1, 0)
            y_f = yo[valid]
            up_m, dn_m = preds == 1, preds == 0
            n_up, n_dn = int(up_m.sum()), int(dn_m.sum())
            total_acc = float((preds == y_f).mean())
            up_acc = float((y_f[up_m] == 1).mean()) if n_up > 0 else 0
            dn_acc = float((y_f[dn_m] == 0).mean()) if n_dn > 0 else 0
            mk = "→" if abs(dn_t - DN_P_THRESHOLD) < 0.001 else " "
            print(f" {mk} {dn_t:5.2f}  {total_acc:>6.1%}  {up_acc:>6.1%}  {n_up:>5d}  "
                  f"{dn_acc:>6.1%}  {n_dn:>5d}  {n:>5d}")

    # ═══════════════════════════════════════════════════════════════
    # 测试六：标准模型概率分布
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  测试六：标准模型概率分布 (P_STD)")
    print("=" * 90)

    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    print(f"  {'概率区间':>12s}  {'样本数':>7s}  {'占比':>6s}  {'实际涨%':>8s}")
    print("  " + "-" * 38)
    for lo, hi in bins:
        bm = (P_STD >= lo) & (P_STD < hi)
        cnt = int(bm.sum())
        if cnt == 0: continue
        actual_up = float(Y[bm].mean())
        bar = "#" * int(cnt / N * 150)
        print(f"  [{lo:.1f}, {hi:.1f})  {cnt:7d}  {cnt/N:5.1%}  {actual_up:7.1%}  {bar}")

    print(f"\n  强标签模型概率分布 (P_STRONG):")
    print(f"  {'概率区间':>12s}  {'样本数':>7s}  {'占比':>6s}  {'实际涨%':>8s}")
    print("  " + "-" * 38)
    for lo, hi in bins:
        bm = (P_STR >= lo) & (P_STR < hi)
        cnt = int(bm.sum())
        if cnt == 0: continue
        actual_up = float(Y[bm].mean())
        bar = "#" * int(cnt / N * 150)
        print(f"  [{lo:.1f}, {hi:.1f})  {cnt:7d}  {cnt/N:5.1%}  {actual_up:7.1%}  {bar}")

    # ═══════════════════════════════════════════════════════════════
    # 测试七：按股票统计 (V3默认参数)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  测试七：V3.1 按股票准确率")
    print("=" * 90)

    up_ratio_all = (INDIV > UP_VOTE_MODEL_THRESH).mean(axis=0)
    is_up_all = (P_STR >= UP_P_THRESHOLD) & (up_ratio_all >= UP_VOTE_AGREE)
    is_dn_all = P_STD <= DN_P_THRESHOLD
    valid_all = is_up_all | is_dn_all
    preds_all = np.where(is_up_all, 1, 0)

    sr = {}
    for code in sorted(set(C)):
        cm = (C == code) & valid_all
        n = int(cm.sum())
        if n == 0: continue
        p, y = preds_all[cm], Y[cm]
        up_m = p == 1
        sr[code] = {
            "acc": float((p == y).mean()),
            "n": n,
            "n_up": int(up_m.sum()),
            "up_acc": float((y[up_m] == 1).mean()) if up_m.sum() > 0 else 0,
        }

    print(f"  {'股票':>12s}  {'总精度':>7s}  {'总次':>5s}  {'涨精度':>7s}  {'涨次':>5s}")
    print("  " + "-" * 42)
    o60 = 0
    for code in sorted(sr, key=lambda c: sr[c]["acc"], reverse=True):
        r = sr[code]
        mk = "✓" if r["acc"] >= 0.60 else " "
        if r["acc"] >= 0.60: o60 += 1
        print(f"  {mk}{code:>11s}  {r['acc']:>6.1%}  {r['n']:>5d}  "
              f"{r['up_acc']:>6.1%}  {r['n_up']:>5d}")
    print(f"\n  ≥60% 的股票: {o60}/{len(sr)}")

    # ═══════════════════════════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  总结")
    print("=" * 90)

    # V3 当前参数的结果
    preds_v3 = preds_all[valid_all]
    y_v3 = Y[valid_all]
    n_v3 = len(y_v3)
    up_m_v3 = preds_v3 == 1; dn_m_v3 = preds_v3 == 0
    n_up_v3, n_dn_v3 = int(up_m_v3.sum()), int(dn_m_v3.sum())
    total_v3 = float((preds_v3 == y_v3).mean())
    up_v3 = float((y_v3[up_m_v3] == 1).mean()) if n_up_v3 > 0 else 0
    dn_v3 = float((y_v3[dn_m_v3] == 0).mean()) if n_dn_v3 > 0 else 0

    print(f"\n  V3.1 当前参数:")
    print(f"    涨: 强{UP_LABEL_THRESHOLD:.0%}标签 P≥{UP_P_THRESHOLD} 子模型>{UP_VOTE_MODEL_THRESH}需≥{UP_VOTE_AGREE:.0%}")
    print(f"    跌: 标准模型 P≤{DN_P_THRESHOLD}")
    print(f"\n  总精度: {total_v3:.1%} ({n_v3}次)")
    print(f"  涨精度: {up_v3:.1%} ({n_up_v3}次)")
    print(f"  跌精度: {dn_v3:.1%} ({n_dn_v3}次)")
    print(f"  基线: {baseline:.1%}")
    print(f"  提升: {total_v3 - baseline:+.1%}")

    # 推荐的最优配置
    balanced = [r for r in best_results if r["n_up"] >= 10 and r["up_acc"] >= 0.60 and r["dn_acc"] >= 0.60]
    if balanced:
        best_bal = max(balanced, key=lambda x: x["total_acc"])
        print(f"\n  推荐配置 (涨跌双≥60%, 总精度最高):")
        print(f"    {best_bal['name']}")
        print(f"    总={best_bal['total_acc']:.1%}({best_bal['n']}) "
              f"涨={best_bal['up_acc']:.1%}({best_bal['n_up']}) "
              f"跌={best_bal['dn_acc']:.1%}({best_bal['n_dn']})")

    print(f"\n  总耗时: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")


if __name__ == "__main__":
    main()
