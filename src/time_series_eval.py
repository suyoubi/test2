"""
时间序列评估示例：禁止 train_test_split(shuffle=True)。

要点:
  - 样本必须按交易日期全局排序后，再使用 TimeSeriesSplit 或固定年份切分；
    否则多股拼接顺序会破坏「过去→未来」语义。
  - 示例: 2020-01-01～2022-12-31 训练，2023 年全年测试（与 Walk-forward 回测互补）。
  - 生产训练见 run_backtest.py（按季度滚动）；GBM 已 early_stopping=False，避免 sklearn
    内部 stratified 随机验证切分。

用法:
  python3 src/time_series_eval.py
  python3 src/time_series_eval.py --holdout-only
"""

import os, sys, argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_predictor import (
    compute_enhanced_feature_matrix, MODEL_CONFIGS, LOOKBACK, FORWARD,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 与 run_backtest.py 一致，便于对比
STOCK_LIST = [
    "HK.00700", "HK.09988", "HK.03690", "HK.01810", "HK.09618",
    "HK.09888", "HK.09999", "HK.01024", "HK.01211", "HK.00981",
    "HK.00005", "HK.00388", "HK.01299", "HK.02318", "HK.00941",
    "HK.02800", "HK.01347", "HK.02015", "HK.09866", "HK.09868",
    "HK.00027", "HK.00066", "HK.00175", "HK.00241", "HK.00267",
    "HK.00285", "HK.00288", "HK.00386", "HK.00669", "HK.00688",
    "HK.00762", "HK.00883", "HK.00939", "HK.00960", "HK.01038",
    "HK.01088", "HK.01177", "HK.01398", "HK.01928", "HK.02007",
    "SH.600519", "SZ.000001", "SH.601318", "SH.600036", "SZ.300750",
    "SZ.000858", "SH.600276", "SH.601012", "SZ.000333", "SH.601888",
]


def load_stock(code):
    p = os.path.join(DATA_DIR, f"{code.replace('.', '_')}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["time_key"] = pd.to_datetime(df["time_key"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "close", "high", "low", "volume"]).sort_values(
        "time_key"
    ).reset_index(drop=True)


def precompute(sl, up_threshold=0.0):
    r = {}
    for code in sl:
        df = load_stock(code)
        if df is None or len(df) < LOOKBACK + FORWARD + 30:
            continue
        feat = compute_enhanced_feature_matrix(df)
        future_ret = df["close"].shift(-FORWARD) / df["close"] - 1
        lab = (future_ret > up_threshold).astype(int)
        dt = df["time_key"]
        v = pd.Series(False, index=df.index)
        v.iloc[LOOKBACK - 1: len(df) - FORWARD] = True
        v = v & feat.notna().all(axis=1) & lab.notna()
        r[code] = (feat, lab, dt, v)
    return r


def add_market_features(sd):
    mkt = {"HK": [], "CN": []}
    for code, (f, _, d, _) in sd.items():
        m = "HK" if code.startswith("HK") else "CN"
        r = f["ret_20d"].copy()
        r.index = d
        mkt[m].append(r)
    bm = {m: pd.concat(rl, axis=1).mean(axis=1) for m, rl in mkt.items() if rl}
    for code in list(sd.keys()):
        f, l, d, v = sd[code]
        fc = f.copy()
        m = "HK" if code.startswith("HK") else "CN"
        if m in bm:
            al = bm[m].reindex(d.values)
            mv = al.values if len(al) == len(f) else np.zeros(len(f))
            if isinstance(mv, pd.Series):
                mv = mv.values
            fc["rel_strength"] = f["ret_20d"].values - mv
            fc["mkt_momentum"] = mv
        else:
            fc["rel_strength"] = 0.0
            fc["mkt_momentum"] = 0.0
        sd[code] = (fc, l, d, v & fc.notna().all(axis=1))
    return sd


def flatten_chronological(sd, d0, d1):
    """多股样本按 (日期, 代码) 排序，保证索引随时间单调。"""
    d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
    parts = []
    for code, (f, l, d, v) in sd.items():
        m = v & (d >= d0) & (d <= d1)
        if not m.any():
            continue
        sub = f.loc[m].copy()
        sub["_dt"] = d.loc[m].values
        sub["_code"] = code
        sub["_y"] = l.loc[m].values
        parts.append(sub)
    if not parts:
        return None, None, None
    big = pd.concat(parts, ignore_index=True)
    big = big.sort_values(["_dt", "_code"]).reset_index(drop=True)
    y = big["_y"].values.astype(int)
    dates = pd.to_datetime(big["_dt"])
    X = big.drop(columns=["_dt", "_code", "_y"])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    med = X.median()
    X = X.fillna(med)
    return X, y, dates


def eval_holdout(X, y, dates, train_start, train_end, test_start, test_end):
    tr = (dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))
    te = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
    if tr.sum() < 500 or te.sum() < 100:
        print(f"  样本不足: 训练 {tr.sum()} 测试 {te.sum()}，跳过 hold-out")
        return
    X_tr, y_tr = X.loc[tr].reset_index(drop=True), y[tr.values]
    X_te, y_te = X.loc[te].reset_index(drop=True), y[te.values]
    cfg = {k: v for k, v in MODEL_CONFIGS[0].items()}
    m = HistGradientBoostingClassifier(max_bins=128, early_stopping=False, **cfg)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    acc = float((pred == y_te).mean())
    print(f"\n  固定区间 hold-out（按日期切分，无 shuffle）")
    print(f"  训练 {train_start} ~ {train_end}: {tr.sum()} 条")
    print(f"  测试 {test_start} ~ {test_end}: {te.sum()} 条")
    print(f"  测试集准确率: {acc:.1%}  基线(涨占比): {y_te.mean():.1%}")


def eval_timeseries_split(X, y, n_splits=4):
    if len(X) < 800:
        print(f"  总样本 {len(X)} 过少，跳过 TimeSeriesSplit")
        return
    cfg = {k: v for k, v in MODEL_CONFIGS[0].items()}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"\n  TimeSeriesSplit(n_splits={n_splits})，样本已按日期排序")
    fold = 0
    accs = []
    for tr_idx, te_idx in tscv.split(X):
        fold += 1
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        m = HistGradientBoostingClassifier(max_bins=128, early_stopping=False, **cfg)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        acc = float((pred == y_te).mean())
        accs.append(acc)
        print(f"    Fold {fold}: 训练 {len(tr_idx)} → 测试 {len(te_idx)}  准确率 {acc:.1%}")
    print(f"  各折平均准确率: {float(np.mean(accs)):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-only", action="store_true", help="只做 2020-2022 / 2023 切分")
    ap.add_argument("--cv-only", action="store_true", help="只做 TimeSeriesSplit")
    args = ap.parse_args()

    print("=" * 70)
    print("  时间序列评估（无 train_test_split shuffle）")
    print("=" * 70)

    sd = precompute(STOCK_LIST, up_threshold=0.0)
    sd = add_market_features(sd)
    print(f"  股票数: {len(sd)}")

    d_min, d_max = "2019-01-01", "2025-12-31"
    X, y, dates = flatten_chronological(sd, d_min, d_max)
    if X is None:
        print("无数据"); return
    print(f"  按时间排序后总样本: {len(X)}  日期 {dates.min().date()} ~ {dates.max().date()}")

    do_hold = not args.cv_only
    do_cv = not args.holdout_only

    if do_hold:
        eval_holdout(
            X, y, dates,
            train_start="2020-01-01", train_end="2022-12-31",
            test_start="2023-01-01", test_end="2023-12-31",
        )
    if do_cv:
        eval_timeseries_split(X, y, n_splits=4)


if __name__ == "__main__":
    main()
