"""
提升"预测涨"精度实验 V2

策略:
  A. 超多数投票 (per-model threshold): 要求>=N%子模型预测P>T才看涨
  B. 更强涨标签训练 + 原始标签评估: 训练目标涨>X%, 预测时看实际是否涨>0%
  C. A+B组合
  D. 不对称阈值组合
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_predictor import (
    compute_enhanced_feature_matrix, DiverseEnsemble, MODEL_CONFIGS,
    LOOKBACK, FORWARD, CONFIDENCE_MARGIN,
)
from run_backtest import (
    STOCK_LIST, TEST_QUARTERS, PRED_THRESHOLD, WEIGHT_HALFLIFE,
    TRAIN_WINDOWS, load_stock, add_market_features,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ═══════════════════════════════════════════════
# Infrastructure
# ═══════════════════════════════════════════════

def precompute_dual(sl, forward=FORWARD, up_threshold=0.0):
    """返回 (strong_label_data, original_labels_map)
    strong_label_data: 用于训练 (标签=涨>up_threshold%)
    original_labels_map: {code: original_label_series} 用于评估 (标签=涨>0%)
    """
    r = {}
    orig = {}
    for code in sl:
        df = load_stock(code)
        if df is None or len(df) < LOOKBACK + forward + 30:
            continue
        feat = compute_enhanced_feature_matrix(df)
        future_ret = df["close"].shift(-forward) / df["close"] - 1
        lab_strong = (future_ret > up_threshold).astype(int)
        lab_orig = (future_ret > 0).astype(int)
        dt = df["time_key"]
        v = pd.Series(False, index=df.index)
        v.iloc[LOOKBACK - 1 : len(df) - forward] = True
        v = v & feat.notna().all(axis=1) & lab_strong.notna()
        r[code] = (feat, lab_strong, dt, v)
        orig[code] = lab_orig
    return r, orig


def gather(sd, d0, d1, w_ref=None):
    Xs, ys, ws, cs = [], [], [], []
    for code, (f, l, d, v) in sd.items():
        m = v & (d >= d0) & (d <= d1)
        if m.sum() == 0:
            continue
        Xs.append(f.loc[m])
        ys.append(l.loc[m].values)
        cs.extend([code] * m.sum())
        if w_ref is not None:
            da = (w_ref - d.loc[m]).dt.days.values.astype(float)
            ws.append(np.power(0.5, da / WEIGHT_HALFLIFE))
        else:
            ws.append(np.ones(m.sum()))
    if not Xs:
        return None, None, None, None
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws), cs


def gather_orig_labels(sd, orig_map, d0, d1):
    """收集原始标签 (涨>0%)，与gather对应"""
    ys = []
    for code, (f, l, d, v) in sd.items():
        m = v & (d >= d0) & (d <= d1)
        if m.sum() == 0:
            continue
        ys.append(orig_map[code].loc[m].values)
    if not ys:
        return None
    return np.concatenate(ys)


def train_diverse(sd, qs):
    q = pd.Timestamp(qs)
    ens = DiverseEnsemble()
    last_data = None
    for wm in TRAIN_WINDOWS:
        ts, te = q - relativedelta(months=wm), q - pd.Timedelta(days=1)
        X, y, w, _ = gather(sd, ts, te, w_ref=te)
        if X is None or len(X) < 200:
            continue
        for cfg in MODEL_CONFIGS:
            ens.add_gbm(X, y, sample_weight=w, config=cfg)
        ens.add_rf(X, y, sample_weight=w)
        ens.add_et(X, y, sample_weight=w)
        last_data = (X, y, w)
    if last_data:
        ens.add_lr(*last_data)
    return ens if ens.models else None


# ═══════════════════════════════════════════════
# Per-model voting (improved)
# ═══════════════════════════════════════════════

def get_individual_probs(ens, X):
    """获取每个子模型的预测概率"""
    Xf = ens._prep(X)
    probs = []
    for t, m in ens.models:
        if t == "lr" and ens._scaler:
            probs.append(m.predict_proba(ens._scaler.transform(Xf))[:, 1])
        else:
            probs.append(m.predict_proba(Xf)[:, 1])
    return np.array(probs)  # (n_models, n_samples)


# ═══════════════════════════════════════════════
# Walk-forward with market split
# ═══════════════════════════════════════════════

def wf_full(sd, orig_map, train_fn):
    """分市场walk-forward，返回概率、各子模型概率、训练标签、原始标签"""
    sd_hk = {k: v for k, v in sd.items() if k.startswith("HK.")}
    sd_cn = {k: v for k, v in sd.items() if not k.startswith("HK.")}
    orig_hk = {k: v for k, v in orig_map.items() if k.startswith("HK.")}
    orig_cn = {k: v for k, v in orig_map.items() if not k.startswith("HK.")}

    a_pavg, a_ytrain, a_yorig, a_codes = [], [], [], []
    a_indiv = []

    for qn, qs, qe in TEST_QUARTERS:
        for sub_sd, sub_orig in [(sd_hk, orig_hk), (sd_cn, orig_cn)]:
            if not sub_sd:
                continue
            m = train_fn(sub_sd, qs)
            if m is None:
                continue
            X, y_tr, _, codes = gather(sub_sd, pd.Timestamp(qs), pd.Timestamp(qe))
            y_orig = gather_orig_labels(sub_sd, sub_orig, pd.Timestamp(qs), pd.Timestamp(qe))
            if X is None:
                continue

            indiv = get_individual_probs(m, X)  # (n_models, n_samples)
            p_avg = indiv.mean(axis=0)

            a_pavg.extend(p_avg.tolist())
            a_ytrain.extend(y_tr.tolist())
            a_yorig.extend(y_orig.tolist())
            a_codes.extend(codes)
            a_indiv.append(indiv)

    if not a_pavg:
        return None
    all_indiv = np.concatenate(a_indiv, axis=1)  # (n_models_max, total_samples)
    return {
        "P": np.array(a_pavg),
        "Y_train": np.array(a_ytrain),
        "Y_orig": np.array(a_yorig),
        "codes": np.array(a_codes),
        "indiv": all_indiv,
    }


# ═══════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════

def eval_config(P, Y, indiv, up_p_thresh, per_model_thresh, up_agree, dn_p_thresh):
    """
    涨条件: P >= up_p_thresh AND (子模型P > per_model_thresh的比例 >= up_agree)
    跌条件: P <= dn_p_thresh
    """
    N = len(Y)
    if N == 0:
        return None

    vote_up = (indiv > per_model_thresh).mean(axis=0)  # 每个样本的看涨模型比例
    is_up = (P >= up_p_thresh) & (vote_up >= up_agree)
    is_dn = P <= dn_p_thresh
    valid = is_up | is_dn
    n = int(valid.sum())
    if n == 0:
        return dict(total_acc=0, up_acc=0, dn_acc=0, n_up=0, n_dn=0, n_conf=n, total=N, coverage=0)

    preds = np.where(is_up[valid], 1, 0)
    y_f = Y[valid]
    up_m = preds == 1
    dn_m = preds == 0
    total_acc = float((preds == y_f).mean())
    up_acc = float((y_f[up_m] == 1).mean()) if up_m.sum() > 0 else 0
    dn_acc = float((y_f[dn_m] == 0).mean()) if dn_m.sum() > 0 else 0
    return dict(
        total_acc=total_acc, up_acc=up_acc, dn_acc=dn_acc,
        n_up=int(up_m.sum()), n_dn=int(dn_m.sum()),
        n_conf=n, total=N, coverage=n / N,
    )


def pr(name, r):
    if r is None:
        print(f"    {name:<56s}  (无数据)")
        return
    mk = "★" if r["up_acc"] >= 0.60 else " "
    print(f"  {mk} {name:<56s} "
          f"涨={r['up_acc']:.1%}({r['n_up']:>3d}) 跌={r['dn_acc']:.1%}({r['n_dn']:>3d}) "
          f"总={r['total_acc']:.1%}({r['n_conf']:>4d}) 覆盖={r['coverage']:.1%}")
    sys.stdout.flush()


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 110)
    print("  提升'预测涨'精度实验 V2")
    print("  修正: 子模型投票用更高阈值; 强标签训练→原始标签评估")
    print("=" * 110)
    sys.stdout.flush()

    results = {}

    # ── Phase 1: Precompute ──
    print("\n[Phase 1] 数据预处理")

    datasets = {}
    for thr_pct, thr_val in [("0%", 0.0), ("2%", 0.02), ("3%", 0.03), ("5%", 0.05)]:
        t0 = time.time()
        print(f"  涨>{thr_pct} ...", end="", flush=True)
        sd, orig = precompute_dual(STOCK_LIST, up_threshold=thr_val)
        sd = add_market_features(sd)
        datasets[thr_pct] = (sd, orig)
        print(f" {len(sd)}只 ({time.time()-t0:.1f}s)")

    # ═══════════════════════════════════════════════
    # Phase 2: 基线 + 超多数投票
    # ═══════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("[Phase 2] 标准模型(涨>0%): 基线 + 超多数投票")
    print("=" * 110)

    t0 = time.time()
    print("  训练中...", flush=True)
    sd0, orig0 = datasets["0%"]
    data0 = wf_full(sd0, orig0, train_diverse)
    P0, Y0, indiv0 = data0["P"], data0["Y_orig"], data0["indiv"]
    print(f"  训练完成 ({time.time()-t0:.0f}s), {len(P0)}个预测, {indiv0.shape[0]}个子模型\n")

    # 基线 (标准阈值)
    r = eval_config(P0, Y0, indiv0, 0.80, 0.5, 0.0, 0.20)
    results["基线 V2 (P≥0.80/≤0.20)"] = r
    pr("基线 V2 (P≥0.80/≤0.20)", r)

    # A: 改进投票——要求子模型P > higher threshold (0.55, 0.60, 0.65, 0.70)
    print("\n  [A] 超多数投票 (子模型阈值提高)")
    for pm_t in [0.55, 0.60, 0.65, 0.70]:
        for agree in [0.70, 0.80, 0.90, 0.95]:
            r = eval_config(P0, Y0, indiv0, 0.80, pm_t, agree, 0.20)
            name = f"子模型>{pm_t:.2f}需≥{agree:.0%}同意 (P≥0.80/≤0.20)"
            if r and (r["n_up"] >= 5 or r["n_dn"] >= 5):
                results[name] = r
                pr(name, r)

    # A+不对称: 投票 + 更严格的涨阈值
    print("\n  [A+不对称] 投票 + 不对称阈值")
    for up_t in [0.80, 0.82, 0.85]:
        for pm_t in [0.55, 0.60, 0.65, 0.70]:
            for agree in [0.70, 0.80, 0.90]:
                for dn_t in [0.15, 0.20]:
                    r = eval_config(P0, Y0, indiv0, up_t, pm_t, agree, dn_t)
                    if r and r["n_up"] >= 5:
                        name = f"P≥{up_t}+模型>{pm_t}×{agree:.0%} 跌≤{dn_t}"
                        results[name] = r
                        pr(name, r)

    # ═══════════════════════════════════════════════
    # Phase 3: 强涨标签训练 → 原始标签评估
    # ═══════════════════════════════════════════════
    for thr_pct in ["2%", "3%", "5%"]:
        print(f"\n{'='*110}")
        print(f"[Phase 3-{thr_pct}] 强标签(涨>{thr_pct})训练 → 原始标签(涨>0%)评估")
        print("=" * 110)

        t0 = time.time()
        print("  训练中...", flush=True)
        sd_s, orig_s = datasets[thr_pct]
        data_s = wf_full(sd_s, orig_s, train_diverse)
        if data_s is None:
            print("  训练失败"); continue
        Ps, Ys_orig, indiv_s = data_s["P"], data_s["Y_orig"], data_s["indiv"]
        Ys_train = data_s["Y_train"]
        print(f"  完成 ({time.time()-t0:.0f}s), {len(Ps)}个预测")

        # 注意: Ps是模型输出概率 (预测"涨>X%"的概率)
        # Ys_orig是原始标签 (实际是否涨>0%)
        # 当模型预测"涨>X%"(高置信)时，检查实际是否涨>0%

        # 标准阈值
        r = eval_config(Ps, Ys_orig, indiv_s, 0.80, 0.5, 0.0, 0.20)
        name = f"强标签涨>{thr_pct} (标准P≥0.80/≤0.20) →原始评估"
        results[name] = r
        pr(name, r)

        # 不同P阈值
        for up_t in [0.60, 0.65, 0.70, 0.75, 0.80]:
            for dn_t in [0.20, 0.25, 0.30, 0.35, 0.40]:
                r = eval_config(Ps, Ys_orig, indiv_s, up_t, 0.5, 0.0, dn_t)
                if r and r["n_up"] >= 10 and r["n_dn"] >= 10:
                    name = f"强{thr_pct} P≥{up_t}/≤{dn_t} →原始"
                    results[name] = r
                    pr(name, r)

        # 投票组合
        print(f"\n  [强{thr_pct}+投票]")
        for up_t in [0.60, 0.65, 0.70, 0.75]:
            for pm_t in [0.55, 0.60, 0.65]:
                for agree in [0.70, 0.80, 0.90]:
                    for dn_t in [0.20, 0.30, 0.40]:
                        r = eval_config(Ps, Ys_orig, indiv_s, up_t, pm_t, agree, dn_t)
                        if r and r["n_up"] >= 10 and r["up_acc"] >= 0.55:
                            name = f"强{thr_pct} P≥{up_t}+m>{pm_t}×{agree:.0%} ≤{dn_t} →原始"
                            results[name] = r
                            pr(name, r)

    # ═══════════════════════════════════════════════
    # Phase 4: 最终排名
    # ═══════════════════════════════════════════════
    print(f"\n\n{'='*110}")
    print("  最终排名")
    print("=" * 110)

    # 按涨精度排序
    valid_up = {k: v for k, v in results.items() if v and v["n_up"] >= 10}
    ranked_up = sorted(valid_up.items(), key=lambda x: x[1]["up_acc"], reverse=True)

    print(f"\n  Top-20 涨精度 (涨次≥10)")
    print(f"  {'#':>3s}  {'配置':<60s} {'涨精度':>7s} {'涨次':>5s} {'跌精度':>7s} {'跌次':>5s} {'总精度':>7s} {'总次':>5s}")
    print("  " + "-" * 110)
    for i, (name, r) in enumerate(ranked_up[:20]):
        mk = "★" if r["up_acc"] >= 0.60 else " "
        print(f"  {mk}{i+1:>2d}  {name:<60s} {r['up_acc']:>6.1%} {r['n_up']:>5d} "
              f"{r['dn_acc']:>6.1%} {r['n_dn']:>5d} {r['total_acc']:>6.1%} {r['n_conf']:>5d}")

    # 按总精度排序 (总次≥50)
    valid_all = {k: v for k, v in results.items() if v and v["n_conf"] >= 50}
    ranked_all = sorted(valid_all.items(), key=lambda x: x[1]["total_acc"], reverse=True)

    print(f"\n  Top-15 总精度 (总次≥50)")
    print(f"  {'#':>3s}  {'配置':<60s} {'涨精度':>7s} {'涨次':>5s} {'跌精度':>7s} {'跌次':>5s} {'总精度':>7s} {'总次':>5s}")
    print("  " + "-" * 110)
    for i, (name, r) in enumerate(ranked_all[:15]):
        mk = "★" if r["total_acc"] >= 0.68 else " "
        print(f"  {mk}{i+1:>2d}  {name:<60s} {r['up_acc']:>6.1%} {r['n_up']:>5d} "
              f"{r['dn_acc']:>6.1%} {r['n_dn']:>5d} {r['total_acc']:>6.1%} {r['n_conf']:>5d}")

    # 涨跌都>=55%, 总次>=30
    print(f"\n  涨跌双≥55%, 总次≥30 的方案 (按涨精度排序):")
    print("  " + "-" * 110)
    balanced = {k: v for k, v in results.items()
                if v and v["up_acc"] >= 0.55 and v["dn_acc"] >= 0.55 and v["n_conf"] >= 30}
    if balanced:
        for name, r in sorted(balanced.items(), key=lambda x: x[1]["up_acc"], reverse=True):
            mk = "★" if r["up_acc"] >= 0.60 else " "
            print(f"  {mk} {name:<60s} 涨={r['up_acc']:.1%}({r['n_up']}) "
                  f"跌={r['dn_acc']:.1%}({r['n_dn']}) 总={r['total_acc']:.1%}({r['n_conf']})")
    else:
        print("  (无)")

    # 涨≥60%, 涨次>=10
    print(f"\n  涨≥60% 且 涨次≥10 的方案:")
    print("  " + "-" * 110)
    high_up = {k: v for k, v in results.items()
               if v and v["up_acc"] >= 0.60 and v["n_up"] >= 10}
    if high_up:
        for name, r in sorted(high_up.items(), key=lambda x: (-x[1]["up_acc"], -x[1]["n_up"])):
            print(f"  ★ {name:<60s} 涨={r['up_acc']:.1%}({r['n_up']}) "
                  f"跌={r['dn_acc']:.1%}({r['n_dn']}) 总={r['total_acc']:.1%}({r['n_conf']})")
    else:
        print("  (无)")

    print(f"\n  总耗时: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")


if __name__ == "__main__":
    main()
