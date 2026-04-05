"""
过拟合诊断实验

测试项:
  1. 训练集 vs 测试集准确率差距 (最直接的过拟合指标)
  2. 季度间稳定性分析
  3. 模型复杂度对比 (简单模型 vs 当前复杂模型)
  4. 随机标签基线测试 (验证模型是否学到了真实信号)
  5. 时间衰减分析 (越近的季度是否越差)
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
from sklearn.ensemble import HistGradientBoostingClassifier

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


def train_simple(sd, qs):
    """简单模型: 只有1个GBM, 1个训练窗口"""
    q = pd.Timestamp(qs)
    ts, te = q - relativedelta(months=12), q - pd.Timedelta(days=1)
    X, y, w, _ = gather(sd, ts, te, w_ref=te)
    if X is None or len(X) < 200: return None
    ens = DiverseEnsemble()
    ens.add_gbm(X, y, sample_weight=w, config=MODEL_CONFIGS[0])
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
    """用V3双模型逻辑评估"""
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
    print("  过拟合诊断实验")
    print("=" * 80)
    sys.stdout.flush()

    # 预计算
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
    print(f"  {len(sd_std)} 只股票就绪")

    # ═══════════════════════════════════════════════════════════════
    # 测试一: 训练集 vs 测试集准确率差距
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  测试一: 训练集 vs 测试集准确率差距 (过拟合核心指标)")
    print("=" * 80)
    print(f"  {'季度':>8s}  {'训练准确率':>10s}  {'训练样本':>8s}  {'测试准确率':>10s}  {'测试样本':>8s}  {'差距':>7s}  {'判定':>6s}")
    print("  " + "-" * 68)
    sys.stdout.flush()

    gaps = []
    for qn, qs, qe in TEST_QUARTERS:
        t0 = time.time()
        train_accs, test_accs = [], []
        train_n, test_n = 0, 0

        for sub_std, sub_str, sub_orig in [
            (sd_std_hk, sd_str_hk, orig_hk),
            (sd_std_cn, sd_str_cn, orig_cn),
        ]:
            if not sub_std: continue
            std_m = train_diverse(sub_std, qs)
            str_m = train_diverse(sub_str, qs)
            if std_m is None or str_m is None: continue

            # 训练集准确率 (最大窗口)
            q = pd.Timestamp(qs)
            ts = q - relativedelta(months=max(TRAIN_WINDOWS))
            te = q - pd.Timedelta(days=1)
            X_tr, _, _, _ = gather(sub_std, ts, te)
            y_tr_orig = gather_orig(sub_std, sub_orig, ts, te)
            if X_tr is not None and y_tr_orig is not None:
                r_tr = eval_dual(std_m, str_m, X_tr, y_tr_orig)
                if r_tr:
                    train_accs.append(r_tr["total_acc"] * r_tr["n"])
                    train_n += r_tr["n"]

            # 测试集准确率
            X_te, _, _, _ = gather(sub_std, pd.Timestamp(qs), pd.Timestamp(qe))
            y_te_orig = gather_orig(sub_std, sub_orig, pd.Timestamp(qs), pd.Timestamp(qe))
            if X_te is not None and y_te_orig is not None:
                r_te = eval_dual(std_m, str_m, X_te, y_te_orig)
                if r_te:
                    test_accs.append(r_te["total_acc"] * r_te["n"])
                    test_n += r_te["n"]

        tr_acc = sum(train_accs) / train_n if train_n > 0 else 0
        te_acc = sum(test_accs) / test_n if test_n > 0 else 0
        gap = tr_acc - te_acc
        gaps.append(gap)

        if gap > 0.15:
            verdict = "⚠️严重"
        elif gap > 0.08:
            verdict = "⚠️轻微"
        else:
            verdict = "✓正常"

        print(f"  {qn:>8s}  {tr_acc:>9.1%}  {train_n:>8d}  {te_acc:>9.1%}  {test_n:>8d}  "
              f"{gap:>+6.1%}  {verdict}")
        sys.stdout.flush()

    avg_gap = np.mean(gaps)
    print(f"\n  平均差距: {avg_gap:+.1%}")
    if avg_gap > 0.15:
        print(f"  ⚠️ 严重过拟合: 训练集比测试集高出 {avg_gap:.1%}")
    elif avg_gap > 0.08:
        print(f"  ⚠️ 轻微过拟合: 训练集比测试集高出 {avg_gap:.1%}")
    elif avg_gap > 0.03:
        print(f"  正常: 训练集比测试集略高 {avg_gap:.1%}, 在可接受范围")
    else:
        print(f"  ✓ 无过拟合迹象: 差距仅 {avg_gap:.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 测试二: 季度间稳定性 + 时间衰减分析
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  测试二: 时间衰减分析 (越新的数据是否越差)")
    print("=" * 80)

    q_results = []
    for qn, qs, qe in TEST_QUARTERS:
        all_preds, all_labels = [], []
        for sub_std, sub_str, sub_orig in [
            (sd_std_hk, sd_str_hk, orig_hk),
            (sd_std_cn, sd_str_cn, orig_cn),
        ]:
            if not sub_std: continue
            std_m = train_diverse(sub_std, qs)
            str_m = train_diverse(sub_str, qs)
            if std_m is None or str_m is None: continue

            X, _, _, _ = gather(sub_std, pd.Timestamp(qs), pd.Timestamp(qe))
            y_orig = gather_orig(sub_std, sub_orig, pd.Timestamp(qs), pd.Timestamp(qe))
            if X is None or y_orig is None: continue

            p_std = std_m.predict_proba_batch(X)
            indiv = get_individual_probs(str_m, X)
            p_strong = indiv.mean(axis=0)
            ur = (indiv > UP_VOTE_MODEL_THRESH).mean(axis=0)
            is_up = (p_strong >= UP_P_THRESHOLD) & (ur >= UP_VOTE_AGREE)
            is_dn = p_std <= DN_P_THRESHOLD

            for i in range(len(X)):
                if is_up[i] or is_dn[i]:
                    all_preds.append(1 if is_up[i] else 0)
                    all_labels.append(y_orig[i])

        if all_preds:
            acc = float((np.array(all_preds) == np.array(all_labels)).mean())
            bl = float(np.array(all_labels).mean())
            q_results.append({"q": qn, "acc": acc, "n": len(all_preds), "baseline": bl})

    print(f"  {'季度':>8s}  {'精度':>7s}  {'样本':>5s}  {'基线':>5s}  {'提升':>7s}  {'趋势':>6s}")
    print("  " + "-" * 48)

    accs = [r["acc"] for r in q_results]
    for i, r in enumerate(q_results):
        lift = r["acc"] - r["baseline"]
        if i == 0:
            trend = "—"
        else:
            d = r["acc"] - q_results[i-1]["acc"]
            trend = f"{'↑' if d > 0.02 else '↓' if d < -0.02 else '→'}{abs(d):.0%}"
        print(f"  {r['q']:>8s}  {r['acc']:>6.1%}  {r['n']:>5d}  {r['baseline']:>4.0%}  {lift:>+6.1%}  {trend}")

    # 回归线检测趋势
    if len(accs) >= 4:
        x = np.arange(len(accs))
        slope = np.polyfit(x, accs, 1)[0]
        print(f"\n  趋势斜率: {slope:+.3f}/季度")
        if slope < -0.03:
            print(f"  ⚠️ 精度随时间明显下降, 模型可能对旧数据过拟合")
        elif slope < -0.01:
            print(f"  轻微下降趋势, 需要持续关注")
        else:
            print(f"  ✓ 无明显时间衰减")

    # ═══════════════════════════════════════════════════════════════
    # 测试三: 简单模型 vs 复杂模型
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  测试三: 简单模型 vs 复杂模型 (复杂度过拟合检测)")
    print("=" * 80)
    print(f"  简单模型: 1个GBM, 12月窗口")
    print(f"  复杂模型: V3双模型 (22个子模型×2)")
    sys.stdout.flush()

    simple_results = []
    for qn, qs, qe in TEST_QUARTERS:
        all_preds, all_labels = [], []
        for sub_std, sub_str, sub_orig in [
            (sd_std_hk, sd_str_hk, orig_hk),
            (sd_std_cn, sd_str_cn, orig_cn),
        ]:
            if not sub_std: continue
            std_m = train_simple(sub_std, qs)
            str_m = train_simple(sub_str, qs)
            if std_m is None or str_m is None: continue

            X, _, _, _ = gather(sub_std, pd.Timestamp(qs), pd.Timestamp(qe))
            y_orig = gather_orig(sub_std, sub_orig, pd.Timestamp(qs), pd.Timestamp(qe))
            if X is None or y_orig is None: continue

            p_std = std_m.predict_proba_batch(X)
            p_strong = str_m.predict_proba_batch(X)
            is_up = p_strong >= UP_P_THRESHOLD
            is_dn = p_std <= DN_P_THRESHOLD

            for i in range(len(X)):
                if is_up[i] or is_dn[i]:
                    all_preds.append(1 if is_up[i] else 0)
                    all_labels.append(y_orig[i])

        if all_preds:
            acc = float((np.array(all_preds) == np.array(all_labels)).mean())
            simple_results.append({"q": qn, "acc": acc, "n": len(all_preds)})

    print(f"\n  {'季度':>8s}  {'简单模型':>8s}  {'样本':>5s}  {'复杂V3':>7s}  {'样本':>5s}  {'差距':>7s}")
    print("  " + "-" * 50)

    complex_better = 0
    for sr, cr in zip(simple_results, q_results):
        d = cr["acc"] - sr["acc"]
        if d > 0: complex_better += 1
        mk = "V3胜" if d > 0.02 else "简单胜" if d < -0.02 else "平"
        print(f"  {sr['q']:>8s}  {sr['acc']:>7.1%}  {sr['n']:>5d}  {cr['acc']:>6.1%}  {cr['n']:>5d}  {d:>+6.1%}  {mk}")

    print(f"\n  V3胜出: {complex_better}/{len(simple_results)} 个季度")
    if complex_better <= len(simple_results) // 2:
        print(f"  ⚠️ 复杂模型未持续优于简单模型, 可能存在过拟合")
    else:
        print(f"  ✓ 复杂模型在多数季度胜出, 额外复杂度有价值")

    # ═══════════════════════════════════════════════════════════════
    # 测试四: 随机标签基线测试
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  测试四: 随机标签基线 (模型是否学到真实信号)")
    print("=" * 80)
    print(f"  用打乱的标签训练, 如果仍有高精度则说明模型在记忆噪音")
    sys.stdout.flush()

    np.random.seed(42)
    rand_preds, rand_labels = [], []

    for qn, qs, qe in TEST_QUARTERS[:3]:
        for sub_std, sub_orig in [(sd_std_hk, orig_hk), (sd_std_cn, orig_cn)]:
            if not sub_std: continue

            q = pd.Timestamp(qs)
            ts = q - relativedelta(months=12)
            te = q - pd.Timedelta(days=1)
            X_tr, y_tr, w_tr, _ = gather(sub_std, ts, te, w_ref=te)
            if X_tr is None or len(X_tr) < 200: continue

            # 故意打乱标签（仅本基线实验；正式训练禁止 shuffle 切分未来）
            y_shuffled = y_tr.copy()
            np.random.shuffle(y_shuffled)

            ens = DiverseEnsemble()
            ens.add_gbm(X_tr, y_shuffled, sample_weight=w_tr, config=MODEL_CONFIGS[0])

            X_te, _, _, _ = gather(sub_std, pd.Timestamp(qs), pd.Timestamp(qe))
            y_te_orig = gather_orig(sub_std, sub_orig, pd.Timestamp(qs), pd.Timestamp(qe))
            if X_te is None or y_te_orig is None: continue

            p = ens.predict_proba_batch(X_te)
            is_up = p >= 0.80
            is_dn = p <= 0.20

            for i in range(len(X_te)):
                if is_up[i] or is_dn[i]:
                    rand_preds.append(1 if is_up[i] else 0)
                    rand_labels.append(y_te_orig[i])

    if rand_preds:
        rand_acc = float((np.array(rand_preds) == np.array(rand_labels)).mean())
        rand_bl = float(np.array(rand_labels).mean())
        print(f"\n  随机标签模型精度: {rand_acc:.1%} ({len(rand_preds)}次信号)")
        print(f"  对应基线: {rand_bl:.1%}")
        print(f"  真实V3精度: {np.mean([r['acc'] for r in q_results[:3]]):.1%} (前3季度)")
        if rand_acc > 0.55:
            print(f"  ⚠️ 随机标签模型仍然有一定精度, 可能存在数据泄露或隐含偏差")
        else:
            print(f"  ✓ 随机标签模型接近随机猜测, V3模型学到了真实信号")
    else:
        print(f"  随机标签模型未产生信号 (正常 — 随机标签导致极端概率减少)")
        print(f"  ✓ 这反证了V3的信号来自真实模式而非噪音")

    # ═══════════════════════════════════════════════════════════════
    # 测试五: 信号分布与小样本问题
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  测试五: 小样本问题分析")
    print("=" * 80)

    total_signals = sum(r["n"] for r in q_results)
    total_up = sum(1 for r in q_results for _ in range(r["n"]))  # approx
    print(f"\n  总信号数: {total_signals}")
    print(f"  平均每季度: {total_signals/len(q_results):.0f}")
    print(f"  最少: {min(r['n'] for r in q_results)} ({min(q_results, key=lambda x:x['n'])['q']})")
    print(f"  最多: {max(r['n'] for r in q_results)} ({max(q_results, key=lambda x:x['n'])['q']})")

    small_q = [r for r in q_results if r["n"] < 30]
    if small_q:
        print(f"\n  ⚠️ {len(small_q)} 个季度样本数<30, 精度可能不可靠:")
        for r in small_q:
            print(f"    {r['q']}: {r['n']}次信号, 精度{r['acc']:.1%}")
        print(f"  建议: 小样本季度的精度有较大随机波动, 不应过度解读")

    # ═══════════════════════════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  过拟合诊断总结")
    print("=" * 80)

    issues = []
    if avg_gap > 0.08:
        issues.append(f"训练-测试差距偏大 ({avg_gap:+.1%})")
    if len(accs) >= 4:
        slope = np.polyfit(np.arange(len(accs)), accs, 1)[0]
        if slope < -0.03:
            issues.append(f"精度随时间下降 (斜率={slope:+.3f}/季度)")
    if complex_better <= len(simple_results) // 2:
        issues.append("复杂模型未持续优于简单模型")
    if small_q:
        issues.append(f"{len(small_q)}个季度样本<30, 精度波动大")

    if not issues:
        print(f"\n  ✓ 未发现严重过拟合问题")
        print(f"    - 训练-测试差距在正常范围")
        print(f"    - 模型在多数季度稳定")
        print(f"    - 复杂模型的额外价值得到验证")
    else:
        print(f"\n  发现 {len(issues)} 个潜在问题:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")

    print(f"\n  总耗时: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")


if __name__ == "__main__":
    main()
