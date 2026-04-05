"""
综合优化实验：测试7种优化方向并找出最佳算法

1. 分市场独立建模 (HK/CN分开训练)
2. 涨跌不对称阈值 (不同涨跌判断标准)
3. 市场环境自适应 (波动率驱动置信度)
4. 模型多样性 (GBM + RF + ExtraTrees + LR)
5. 动态预测周期 (多周期共识)
6. 特征工程增强 (日历/波动率/缺口/加速度)
7. 概率校准 (Isotonic Regression)
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_predictor import (
    compute_feature_matrix, EnsemblePredictor, MODEL_CONFIGS,
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

def precompute_base(sl, forward=FORWARD):
    r = {}
    for code in sl:
        df = load_stock(code)
        if df is None or len(df) < LOOKBACK + forward + 30:
            continue
        feat = compute_feature_matrix(df)
        lab = (df["close"].shift(-forward) > df["close"]).astype(int)
        dt = df["time_key"]
        v = pd.Series(False, index=df.index)
        v.iloc[LOOKBACK - 1 : len(df) - forward] = True
        v = v & feat.notna().all(axis=1) & lab.notna()
        r[code] = (feat, lab, dt, v)
    return r


def _enhanced_features(df, feat):
    f = feat.copy()
    close = df["close"].astype(float)
    high, low = df["high"].astype(float), df["low"].astype(float)
    vol = df["volume"].astype(float)
    dt = df["time_key"]

    m, dw = dt.dt.month, dt.dt.dayofweek
    f["month_sin"] = np.sin(2 * np.pi * m.values / 12)
    f["month_cos"] = np.cos(2 * np.pi * m.values / 12)
    f["dow_sin"] = np.sin(2 * np.pi * dw.values / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dw.values / 5)

    v20 = close.pct_change().rolling(20).std()
    v60 = close.pct_change().rolling(60).std()
    f["vol_regime"] = (v20 / v60.replace(0, np.nan)).values

    gap = (df["open"].astype(float) / close.shift(1) - 1).abs()
    f["gap_freq_20"] = (gap > 0.02).astype(float).rolling(20).mean().values
    f["avg_gap_20"] = gap.rolling(20).mean().values

    r5 = close / close.shift(5) - 1
    f["price_accel"] = (r5 - r5.shift(5)).values

    f["vol_trend"] = (vol.rolling(5).mean() / vol.rolling(20).mean().replace(0, np.nan)).values
    f["vol_accel"] = ((vol.rolling(5).mean() / vol.rolling(10).mean().replace(0, np.nan)) - 1).values

    hl = (high - low) / close
    f["range_trend"] = (hl.rolling(5).mean() / hl.rolling(20).mean().replace(0, np.nan)).values

    up = (close > close.shift(1)).astype(float)
    f["consec_up"] = up.rolling(5).sum().values
    f["consec_dn"] = (1 - up).rolling(5).sum().values

    return f


def precompute_enhanced(sl, forward=FORWARD):
    r = {}
    for code in sl:
        df = load_stock(code)
        if df is None or len(df) < LOOKBACK + forward + 30:
            continue
        feat = _enhanced_features(df, compute_feature_matrix(df))
        lab = (df["close"].shift(-forward) > df["close"]).astype(int)
        dt = df["time_key"]
        v = pd.Series(False, index=df.index)
        v.iloc[LOOKBACK - 1 : len(df) - forward] = True
        v = v & feat.notna().all(axis=1) & lab.notna()
        r[code] = (feat, lab, dt, v)
    return r


def precompute_multi_horizon(sl, forwards):
    raw = {}
    for code in sl:
        df = load_stock(code)
        if df is None or len(df) < LOOKBACK + max(forwards) + 30:
            continue
        raw[code] = (df, compute_feature_matrix(df))

    result = {}
    for fwd in forwards:
        sd = {}
        for code, (df, feat) in raw.items():
            lab = (df["close"].shift(-fwd) > df["close"]).astype(int)
            dt = df["time_key"]
            v = pd.Series(False, index=df.index)
            v.iloc[LOOKBACK - 1 : len(df) - fwd] = True
            v = v & feat.notna().all(axis=1) & lab.notna()
            sd[code] = (feat.copy(), lab, dt, v)
        sd = add_market_features(sd)
        result[fwd] = sd
    return result


def gather(sd, d0, d1, w_ref=None, halflife=WEIGHT_HALFLIFE):
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
            ws.append(np.power(0.5, da / halflife))
        else:
            ws.append(np.ones(m.sum()))
    if not Xs:
        return None, None, None, None
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws), cs


def evaluate(P, Y, cm=CONFIDENCE_MARGIN, pt=PRED_THRESHOLD):
    N = len(Y)
    if N == 0:
        return dict(conf_acc=0, full_acc=0, n_conf=0, total=0, coverage=0, baseline=0, lift=0)
    bl = float(Y.mean())
    fa = float(((P >= pt).astype(int) == Y).mean())
    mask = np.abs(P - 0.5) >= cm
    n = int(mask.sum())
    ca = float(((P[mask] >= pt).astype(int) == Y[mask]).mean()) if n > 0 else 0
    return dict(conf_acc=ca, full_acc=fa, n_conf=n, total=N, coverage=n / N, baseline=bl, lift=ca - bl)


def pr(name, r):
    mk = "★" if r["conf_acc"] >= 0.60 else " "
    print(f"  {mk} {name:<48s} 过滤={r['conf_acc']:.1%}({r['n_conf']:>4d})  "
          f"全量={r['full_acc']:.1%}  覆盖={r['coverage']:.0%}  提升={r['lift']:+.1%}")
    sys.stdout.flush()


# ═══════════════════════════════════════════════
# Training Functions
# ═══════════════════════════════════════════════

def train_baseline(sd, qs):
    q = pd.Timestamp(qs)
    p = EnsemblePredictor()
    for wm in TRAIN_WINDOWS:
        ts, te = q - relativedelta(months=wm), q - pd.Timedelta(days=1)
        X, y, w, _ = gather(sd, ts, te, w_ref=te)
        if X is None or len(X) < 200:
            continue
        for cfg in MODEL_CONFIGS:
            p.add_model(X, y, sample_weight=w, config=cfg)
    return p if p.models else None


class DiverseEnsemble:
    def __init__(self):
        self.models = []
        self.feature_names = None
        self._medians = None
        self._scaler = None

    def _prep(self, X):
        if self.feature_names is None:
            self.feature_names = list(X.columns)
            self._medians = X.median()
        Xf = X.reindex(columns=self.feature_names).fillna(self._medians)
        for c in Xf.columns:
            Xf[c] = pd.to_numeric(Xf[c], errors="coerce")
        return Xf.fillna(self._medians)

    def add_gbm(self, X, y, w=None, cfg=None):
        Xf = self._prep(X)
        m = HistGradientBoostingClassifier(
            max_bins=128, early_stopping=False, **(cfg or MODEL_CONFIGS[0]))
        m.fit(Xf, y, sample_weight=w)
        self.models.append(("gbm", m))

    def add_rf(self, X, y, w=None):
        Xf = self._prep(X)
        m = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            max_features="sqrt", random_state=42, n_jobs=-1)
        m.fit(Xf, y, sample_weight=w)
        self.models.append(("rf", m))

    def add_et(self, X, y, w=None):
        Xf = self._prep(X)
        m = ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            max_features="sqrt", random_state=42, n_jobs=-1)
        m.fit(Xf, y, sample_weight=w)
        self.models.append(("et", m))

    def add_lr(self, X, y, w=None):
        Xf = self._prep(X)
        if self._scaler is None:
            self._scaler = StandardScaler()
            Xs = self._scaler.fit_transform(Xf)
        else:
            Xs = self._scaler.transform(Xf)
        m = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        m.fit(Xs, y, sample_weight=w)
        self.models.append(("lr", m))

    def predict_proba_batch(self, X):
        Xf = self._prep(X)
        ps = []
        for t, m in self.models:
            if t == "lr" and self._scaler:
                ps.append(m.predict_proba(self._scaler.transform(Xf))[:, 1])
            else:
                ps.append(m.predict_proba(Xf)[:, 1])
        return np.mean(ps, axis=0)


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
            ens.add_gbm(X, y, w, cfg)
        ens.add_rf(X, y, w)
        ens.add_et(X, y, w)
        last_data = (X, y, w)
    if last_data:
        ens.add_lr(*last_data)
    return ens if ens.models else None


# ═══════════════════════════════════════════════
# Walk-forward Variants
# ═══════════════════════════════════════════════

def _wf_core(sd, train_fn, quarters=TEST_QUARTERS):
    ap, ay, ac, aq = [], [], [], []
    for qn, qs, qe in quarters:
        m = train_fn(sd, qs)
        if m is None:
            continue
        X, y, _, codes = gather(sd, pd.Timestamp(qs), pd.Timestamp(qe))
        if X is None:
            continue
        p = m.predict_proba_batch(X)
        ap.extend(p); ay.extend(y); ac.extend(codes); aq.extend([qn] * len(y))
    return np.array(ap), np.array(ay), np.array(ac), np.array(aq)


def wf(sd, train_fn):
    return _wf_core(sd, train_fn)


def wf_mkt(sd, train_fn):
    sd_hk = {k: v for k, v in sd.items() if k.startswith("HK.")}
    sd_cn = {k: v for k, v in sd.items() if not k.startswith("HK.")}
    ap, ay, ac, aq = [], [], [], []
    for qn, qs, qe in TEST_QUARTERS:
        for sub in [sd_hk, sd_cn]:
            if not sub:
                continue
            m = train_fn(sub, qs)
            if m is None:
                continue
            X, y, _, codes = gather(sub, pd.Timestamp(qs), pd.Timestamp(qe))
            if X is None:
                continue
            p = m.predict_proba_batch(X)
            ap.extend(p); ay.extend(y); ac.extend(codes); aq.extend([qn] * len(y))
    return np.array(ap), np.array(ay), np.array(ac), np.array(aq)


def wf_calibrated(sd, train_fn):
    ap, ay, ac, aq = [], [], [], []
    for qn, qs, qe in TEST_QUARTERS:
        q = pd.Timestamp(qs)
        m = train_fn(sd, qs)
        if m is None:
            continue
        cal_e = q - pd.Timedelta(days=FORWARD + 1)
        cal_s = cal_e - pd.Timedelta(days=90)
        Xc, yc, _, _ = gather(sd, cal_s, cal_e)
        do_cal = False
        if Xc is not None and len(Xc) >= 30:
            pc = m.predict_proba_batch(Xc)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(pc, yc)
            do_cal = True
        X, y, _, codes = gather(sd, pd.Timestamp(qs), pd.Timestamp(qe))
        if X is None:
            continue
        p = m.predict_proba_batch(X)
        if do_cal:
            p = iso.transform(p)
        ap.extend(p); ay.extend(y); ac.extend(codes); aq.extend([qn] * len(y))
    return np.array(ap), np.array(ay), np.array(ac), np.array(aq)


def wf_mkt_calibrated(sd, train_fn):
    sd_hk = {k: v for k, v in sd.items() if k.startswith("HK.")}
    sd_cn = {k: v for k, v in sd.items() if not k.startswith("HK.")}
    ap, ay, ac, aq = [], [], [], []
    for qn, qs, qe in TEST_QUARTERS:
        q = pd.Timestamp(qs)
        for sub in [sd_hk, sd_cn]:
            if not sub:
                continue
            m = train_fn(sub, qs)
            if m is None:
                continue
            cal_e = q - pd.Timedelta(days=FORWARD + 1)
            cal_s = cal_e - pd.Timedelta(days=90)
            Xc, yc, _, _ = gather(sub, cal_s, cal_e)
            do_cal = False
            if Xc is not None and len(Xc) >= 30:
                pc = m.predict_proba_batch(Xc)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(pc, yc)
                do_cal = True
            X, y, _, codes = gather(sub, pd.Timestamp(qs), pd.Timestamp(qe))
            if X is None:
                continue
            p = m.predict_proba_batch(X)
            if do_cal:
                p = iso.transform(p)
            ap.extend(p); ay.extend(y); ac.extend(codes); aq.extend([qn] * len(y))
    return np.array(ap), np.array(ay), np.array(ac), np.array(aq)


def wf_multi_horizon(sd_dict):
    horizons = sorted(sd_dict.keys())
    primary = 15
    ap, ay, ac, aq = [], [], [], []
    for qn, qs, qe in TEST_QUARTERS:
        models = {}
        for fwd in horizons:
            m = train_baseline(sd_dict[fwd], qs)
            if m is not None:
                models[fwd] = m
        if len(models) < 2:
            continue
        X, y, _, codes = gather(sd_dict[primary], pd.Timestamp(qs), pd.Timestamp(qe))
        if X is None:
            continue
        ps = [m.predict_proba_batch(X) for m in models.values()]
        p = np.mean(ps, axis=0)
        ap.extend(p); ay.extend(y); ac.extend(codes); aq.extend([qn] * len(y))
    return np.array(ap), np.array(ay), np.array(ac), np.array(aq)


# ═══════════════════════════════════════════════
# Post-processing
# ═══════════════════════════════════════════════

def best_asymmetric(P, Y):
    best = dict(acc=0, up_t=0.8, dn_t=0.2, n=0)
    for up_t in np.arange(0.55, 0.92, 0.05):
        for dn_t in np.arange(0.08, 0.45, 0.05):
            valid = (P >= up_t) | (P <= dn_t)
            n = int(valid.sum())
            if n < 50:
                continue
            preds = np.where(P[valid] >= up_t, 1, 0)
            acc = float((preds == Y[valid]).mean())
            if acc > best["acc"]:
                best = dict(acc=acc, up_t=up_t, dn_t=dn_t, n=n)
    return best


def eval_asymmetric(P, Y, up_t, dn_t):
    valid = (P >= up_t) | (P <= dn_t)
    n = int(valid.sum())
    N = len(Y)
    if n == 0:
        return dict(conf_acc=0, full_acc=0, n_conf=0, total=N, coverage=0, baseline=float(Y.mean()), lift=0)
    preds = np.where(P[valid] >= up_t, 1, 0)
    acc = float((preds == Y[valid]).mean())
    bl = float(Y.mean())
    fa = float(((P >= PRED_THRESHOLD).astype(int) == Y).mean())
    return dict(conf_acc=acc, full_acc=fa, n_conf=n, total=N, coverage=n / N, baseline=bl, lift=acc - bl)


def adaptive_eval(P, Y, Q, sd):
    q_margins = {}
    for qn, qs, _ in TEST_QUARTERS:
        q = pd.Timestamp(qs)
        vols = []
        for code, (f, _, d, v) in sd.items():
            recent = (d >= q - pd.Timedelta(days=60)) & (d < q)
            if recent.sum() > 0:
                vv = f.loc[recent, "vol_20d"].dropna()
                if len(vv) > 0:
                    vols.append(float(vv.mean()))
        avg = np.mean(vols) if vols else 0.02
        q_margins[qn] = 0.35 if avg > 0.025 else (0.30 if avg > 0.018 else 0.25)

    margins = np.array([q_margins.get(q, 0.30) for q in Q])
    mask = np.abs(P - 0.5) >= margins
    n = int(mask.sum())
    N = len(Y)
    bl = float(Y.mean())
    fa = float(((P >= PRED_THRESHOLD).astype(int) == Y).mean())
    ca = float(((P[mask] >= PRED_THRESHOLD).astype(int) == Y[mask]).mean()) if n > 0 else 0
    return dict(conf_acc=ca, full_acc=fa, n_conf=n, total=N, coverage=n / N, baseline=bl, lift=ca - bl)


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 80)
    print("  综合优化实验：7种优化 + 组合测试")
    print("=" * 80)
    print(f"  季度: {TEST_QUARTERS[0][0]}~{TEST_QUARTERS[-1][0]} ({len(TEST_QUARTERS)}个)")
    print(f"  股票: {len(STOCK_LIST)}只  置信度: |P-0.5|≥{CONFIDENCE_MARGIN}")
    print("=" * 80)
    sys.stdout.flush()

    results = {}

    # ── Phase 1: Precompute ──
    print("\n[Phase 1] 数据预处理")

    t0 = time.time()
    print("  基础特征 ...", end="", flush=True)
    sd = precompute_base(STOCK_LIST)
    sd = add_market_features(sd)
    print(f" {len(sd)}只 ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("  增强特征 ...", end="", flush=True)
    sd_enh = precompute_enhanced(STOCK_LIST)
    sd_enh = add_market_features(sd_enh)
    print(f" {len(sd_enh)}只 ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("  多周期标签 ...", end="", flush=True)
    sd_multi = precompute_multi_horizon(STOCK_LIST, [10, 15, 20])
    print(f" {len(sd_multi[15])}只×{len(sd_multi)}周期 ({time.time()-t0:.1f}s)")

    # ── Phase 2: Individual tests ──
    print(f"\n{'='*80}")
    print(f"[Phase 2] 单项优化")
    print(f"{'='*80}")

    print("\n  [0] 基线 ...", flush=True)
    t0 = time.time()
    P0, Y0, C0, Q0 = wf(sd, train_baseline)
    results["基线"] = evaluate(P0, Y0)
    pr("基线 (当前算法)", results["基线"])
    print(f"      ({time.time()-t0:.0f}s)")

    print("\n  [1] 分市场独立建模 ...", flush=True)
    t0 = time.time()
    P1, Y1, C1, Q1 = wf_mkt(sd, train_baseline)
    results["① 分市场"] = evaluate(P1, Y1)
    pr("① 分市场独立建模", results["① 分市场"])
    print(f"      ({time.time()-t0:.0f}s)")

    print("\n  [2] 涨跌不对称阈值 ...", flush=True)
    ba = best_asymmetric(P0, Y0)
    results["② 不对称阈值"] = eval_asymmetric(P0, Y0, ba["up_t"], ba["dn_t"])
    pr(f"② 不对称阈值 (涨≥{ba['up_t']:.2f} 跌≤{ba['dn_t']:.2f})", results["② 不对称阈值"])

    print("\n  [3] 市场环境自适应 ...", flush=True)
    results["③ 自适应"] = adaptive_eval(P0, Y0, Q0, sd)
    pr("③ 市场环境自适应", results["③ 自适应"])

    print("\n  [4] 模型多样性 (GBM+RF+ET+LR) ...", flush=True)
    t0 = time.time()
    P4, Y4, C4, Q4 = wf(sd, train_diverse)
    results["④ 多样性"] = evaluate(P4, Y4)
    pr("④ 模型多样性", results["④ 多样性"])
    print(f"      ({time.time()-t0:.0f}s)")

    print("\n  [5] 动态预测周期 (10/15/20天) ...", flush=True)
    t0 = time.time()
    P5, Y5, C5, Q5 = wf_multi_horizon(sd_multi)
    results["⑤ 多周期"] = evaluate(P5, Y5)
    pr("⑤ 多周期共识 (10/15/20天)", results["⑤ 多周期"])
    print(f"      ({time.time()-t0:.0f}s)")

    print("\n  [6] 特征工程增强 ...", flush=True)
    t0 = time.time()
    P6, Y6, C6, Q6 = wf(sd_enh, train_baseline)
    results["⑥ 增强特征"] = evaluate(P6, Y6)
    pr("⑥ 特征工程增强", results["⑥ 增强特征"])
    print(f"      ({time.time()-t0:.0f}s)")

    print("\n  [7] 概率校准 (Isotonic) ...", flush=True)
    t0 = time.time()
    P7, Y7, C7, Q7 = wf_calibrated(sd, train_baseline)
    results["⑦ 概率校准"] = evaluate(P7, Y7)
    pr("⑦ 概率校准", results["⑦ 概率校准"])
    print(f"      ({time.time()-t0:.0f}s)")

    # ── Phase 3: Combinations ──
    print(f"\n{'='*80}")
    print(f"[Phase 3] 组合优化")
    print(f"{'='*80}")

    print("\n  [A] 分市场 + 多样性 ...", flush=True)
    t0 = time.time()
    Pa, Ya, Ca, Qa = wf_mkt(sd, train_diverse)
    results["A: 分市场+多样性"] = evaluate(Pa, Ya)
    pr("A: 分市场 + 多样性", results["A: 分市场+多样性"])
    print(f"      ({time.time()-t0:.0f}s)")

    print("\n  [B] 分市场 + 增强特征 ...", flush=True)
    t0 = time.time()
    Pb, Yb, Cb, Qb = wf_mkt(sd_enh, train_baseline)
    results["B: 分市场+增强"] = evaluate(Pb, Yb)
    pr("B: 分市场 + 增强特征", results["B: 分市场+增强"])
    print(f"      ({time.time()-t0:.0f}s)")

    print("\n  [C] 分市场 + 多样性 + 增强 ...", flush=True)
    t0 = time.time()
    Pc, Yc, Cc, Qc = wf_mkt(sd_enh, train_diverse)
    results["C: 分市场+多样性+增强"] = evaluate(Pc, Yc)
    pr("C: 分市场 + 多样性 + 增强", results["C: 分市场+多样性+增强"])
    print(f"      ({time.time()-t0:.0f}s)")

    print("\n  [D] 分市场 + 多样性 + 校准 ...", flush=True)
    t0 = time.time()
    Pd, Yd, Cd, Qd = wf_mkt_calibrated(sd, train_diverse)
    results["D: 分市场+多样性+校准"] = evaluate(Pd, Yd)
    pr("D: 分市场 + 多样性 + 校准", results["D: 分市场+多样性+校准"])
    print(f"      ({time.time()-t0:.0f}s)")

    print("\n  [E] 分市场 + 多样性 + 增强 + 校准 ...", flush=True)
    t0 = time.time()
    Pe, Ye, Ce, Qe = wf_mkt_calibrated(sd_enh, train_diverse)
    results["E: 分市场+多样性+增强+校准"] = evaluate(Pe, Ye)
    pr("E: 分市场 + 多样性 + 增强 + 校准", results["E: 分市场+多样性+增强+校准"])
    print(f"      ({time.time()-t0:.0f}s)")

    print("\n  [F] 分市场 + 增强 + 校准 ...", flush=True)
    t0 = time.time()
    Pf, Yf, Cf, Qf = wf_mkt_calibrated(sd_enh, train_baseline)
    results["F: 分市场+增强+校准"] = evaluate(Pf, Yf)
    pr("F: 分市场 + 增强 + 校准", results["F: 分市场+增强+校准"])
    print(f"      ({time.time()-t0:.0f}s)")

    # ── Phase 4: Post-processing on top results ──
    print(f"\n{'='*80}")
    print(f"[Phase 4] 后处理 (不对称阈值 + 自适应)")
    print(f"{'='*80}")

    combo_data = [
        ("C: 分市场+多样性+增强", Pc, Yc, Cc, Qc),
        ("D: 分市场+多样性+校准", Pd, Yd, Cd, Qd),
        ("E: 分市场+多样性+增强+校准", Pe, Ye, Ce, Qe),
        ("A: 分市场+多样性", Pa, Ya, Ca, Qa),
    ]
    for name, Px, Yx, Cx, Qx in combo_data:
        if len(Px) == 0:
            continue
        ba2 = best_asymmetric(Px, Yx)
        k = f"{name}+不对称(↑{ba2['up_t']:.2f}↓{ba2['dn_t']:.2f})"
        results[k] = eval_asymmetric(Px, Yx, ba2["up_t"], ba2["dn_t"])
        pr(k, results[k])

        k2 = f"{name}+自适应"
        results[k2] = adaptive_eval(Px, Yx, Qx, sd)
        pr(k2, results[k2])
        print()

    # ── Phase 5: Summary ──
    print(f"\n{'='*80}")
    print("  最终排名")
    print("=" * 80)
    print(f"\n  {'#':>3s}  {'配置':<55s} {'过滤准确率':>8s} {'覆盖':>5s} {'提升':>6s} {'次数':>6s}")
    print("  " + "-" * 88)

    ranked = sorted(results.items(), key=lambda x: x[1]["conf_acc"], reverse=True)
    for i, (name, r) in enumerate(ranked):
        mk = "★" if i == 0 else " "
        print(f"  {mk}{i+1:>2d}  {name:<55s} {r['conf_acc']:>7.1%} {r['coverage']:>4.0%} "
              f"{r['lift']:>+5.1%} {r['n_conf']:>5d}")

    best_name, best_r = ranked[0]
    bl_r = results["基线"]
    print(f"\n  {'='*60}")
    print(f"  最佳: {best_name}")
    print(f"  过滤准确率: {best_r['conf_acc']:.1%} (基线 {bl_r['conf_acc']:.1%}, 提升 {best_r['conf_acc']-bl_r['conf_acc']:+.1%})")
    print(f"  覆盖率: {best_r['coverage']:.0%} ({best_r['n_conf']}/{best_r['total']})")
    print(f"  超越基线: {best_r['lift']:+.1%}")
    print(f"  总耗时: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")


if __name__ == "__main__":
    main()
