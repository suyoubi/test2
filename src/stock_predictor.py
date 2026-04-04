"""
股票走势预测算法 — 最终版

算法概述:
  1. 从120天K线数据提取~50个技术指标特征
  2. 添加市场基准相对强弱特征
  3. 用15模型集成 (5个训练窗口配置 × 3个时间窗口) 预测 P(UP)
  4. 置信度过滤: 仅在高置信时输出预测
     - P(UP) ≥ 0.80 → 预测 涨
     - P(UP) ≤ 0.20 → 预测 跌
     - 否则 → 不确定 (不输出预测)

回测准确率: ~60%+ (50只港股/A股, 7个季度, walk-forward训练)
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

LOOKBACK = 120
FORWARD = 15
CONFIDENCE_MARGIN = 0.30  # |P-0.5| >= 0.30 才输出预测


# ── 技术指标 ──────────────────────────────────────────────────

def _sma(s, w):
    return s.rolling(window=w, min_periods=w).mean()

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def _macd(close, fast=12, slow=26, sig=9):
    m = _ema(close, fast) - _ema(close, slow)
    return m, _ema(m, sig), m - _ema(m, sig)

def _bollinger(close, w=20, nstd=2):
    mid = _sma(close, w)
    std = close.rolling(w, min_periods=w).std()
    rng = 2 * nstd * std
    return (close - (mid - nstd*std)) / rng.replace(0, np.nan), rng / mid.replace(0, np.nan)

def _atr(high, low, close, period=14):
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def _stochastic(high, low, close, k=14, d=3):
    lo = low.rolling(k, min_periods=k).min()
    hi = high.rolling(k, min_periods=k).max()
    sk = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    return sk, sk.rolling(d, min_periods=d).mean()

def _adx(high, low, close, period=14):
    pdm = high.diff().clip(lower=0)
    mdm = (-low.diff()).clip(lower=0)
    mask = pdm < mdm; pdm[mask] = 0; mdm[~mask] = 0
    a = _atr(high, low, close, period)
    pdi = 100 * _ema(pdm, period) / a.replace(0, np.nan)
    mdi = 100 * _ema(mdm, period) / a.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _ema(dx, period), pdi, mdi

def _mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(1), 0).rolling(period, min_periods=period).sum()
    neg = mf.where(tp <= tp.shift(1), 0).rolling(period, min_periods=period).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


# ── 特征矩阵 ─────────────────────────────────────────────────

def compute_feature_matrix(df):
    """对整段K线一次性计算所有特征。index >= LOOKBACK-1 的行有完整数据。"""
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)
    opn   = df["open"].astype(float)

    f = pd.DataFrame(index=df.index)

    # 多尺度收益率
    for d in [3, 5, 10, 20, 40, 60, 90]:
        f[f"ret_{d}d"] = close / close.shift(d) - 1

    # EMA 交叉
    f["ema12_26"] = _ema(close, 12) / _ema(close, 26)

    # 均线体系
    s5, s20, s60 = _sma(close, 5), _sma(close, 20), _sma(close, 60)
    f["price_sma5"]  = close / s5
    f["price_sma20"] = close / s20
    f["price_sma60"] = close / s60
    f["sma5_sma20"]  = s5 / s20
    f["sma20_sma60"] = s20 / s60
    f["sma5_slope"]  = s5 / s5.shift(5) - 1
    f["sma20_slope"] = s20 / s20.shift(10) - 1

    # RSI + 滞后
    rsi = _rsi(close, 14)
    f["rsi_14"]      = rsi
    f["rsi_chg5"]    = rsi - rsi.shift(5)
    f["rsi_lag5"]    = rsi.shift(5)

    # MACD + 滞后
    _, _, hist = _macd(close)
    hn = hist / close
    f["macd_hist_norm"] = hn
    f["macd_hist_chg"]  = (hist - hist.shift(5)) / close
    f["macd_hist_lag5"] = hn.shift(5)

    # Stochastic + 滞后
    sk, sd = _stochastic(high, low, close)
    f["stoch_k"]     = sk
    f["stoch_d"]     = sd
    f["stoch_k_lag5"] = sk.shift(5)

    # ADX
    adx_v, pdi, mdi = _adx(high, low, close)
    f["adx"]     = adx_v
    f["di_diff"] = pdi - mdi

    # MFI
    f["mfi_14"] = _mfi(high, low, close, vol, 14)

    # Bollinger
    bb_pctb, bb_bw = _bollinger(close)
    f["bb_pctb"]  = bb_pctb
    f["bb_width"] = bb_bw

    # ATR
    f["atr_ratio"] = _atr(high, low, close) / close

    # 波动率
    dret = close.pct_change()
    f["vol_20d"] = dret.rolling(20).std()

    # 成交量
    f["vol_ratio"] = _sma(vol, 5) / _sma(vol, 20).replace(0, np.nan)

    # 涨跌比
    f["up_ratio_20"] = (dret > 0).astype(float).rolling(20).mean()

    # 区间位置
    rhi = high.rolling(LOOKBACK, min_periods=LOOKBACK).max()
    rlo = low.rolling(LOOKBACK, min_periods=LOOKBACK).min()
    f["pos_120d"]  = (close - rlo) / (rhi - rlo).replace(0, np.nan)
    f["dist_hi20"] = close / high.rolling(20, min_periods=20).max() - 1

    # 趋势一致性
    up_seg = pd.Series(0.0, index=df.index)
    for step in [0, 10, 20, 30, 40, 50]:
        c1, c2 = close.shift(step), close.shift(step+10)
        up_seg += (c1 > c2).astype(float).where(c2.notna() & (c2 > 0), 0)
    f["trend_consistency"] = up_seg / 6

    # 趋势斜率
    def _slope(series, w):
        x = np.arange(w, dtype=float); xm = x.mean(); xv = ((x-xm)**2).sum()
        def s(y):
            if len(y)<w: return np.nan
            if y[0]==0: return 0.0
            yn = y/y[0]; return ((x-xm)*(yn-yn.mean())).sum()/xv
        return series.rolling(w, min_periods=w).apply(s, raw=True)
    f["trend_slope_20"] = _slope(close, 20)

    # K线形态
    body = close - opn
    rng  = (high - low).replace(0, np.nan)
    f["candle_body"]  = body / close
    f["candle_range"] = rng / close

    us = high - pd.concat([close, opn], axis=1).max(axis=1)
    ls = pd.concat([close, opn], axis=1).min(axis=1) - low
    f["upper_shadow_ratio"] = (us / rng).rolling(5).mean()
    f["lower_shadow_ratio"] = (ls / rng).rolling(5).mean()

    eb = ((body>0) & (body.shift(1)<0) & (close>opn.shift(1)) & (opn<close.shift(1))).astype(float)
    er = ((body<0) & (body.shift(1)>0) & (opn>close.shift(1)) & (close<opn.shift(1))).astype(float)
    f["engulf_signal"] = eb.rolling(10).sum() - er.rolling(10).sum()

    return f


def extract_features(df):
    """兼容接口: 从K线DataFrame提取最后一行特征dict"""
    return compute_feature_matrix(df).iloc[-1].to_dict()


# ── 集成预测模型 ──────────────────────────────────────────────

MODEL_CONFIGS = [
    dict(max_iter=400, max_depth=3, learning_rate=0.05, min_samples_leaf=60,
         l2_regularization=2.0, random_state=42),
    dict(max_iter=500, max_depth=4, learning_rate=0.03, min_samples_leaf=80,
         l2_regularization=3.0, random_state=123),
    dict(max_iter=350, max_depth=3, learning_rate=0.07, min_samples_leaf=50,
         l2_regularization=1.5, random_state=456),
    dict(max_iter=300, max_depth=2, learning_rate=0.08, min_samples_leaf=40,
         l2_regularization=1.0, random_state=789),
    dict(max_iter=600, max_depth=5, learning_rate=0.02, min_samples_leaf=100,
         l2_regularization=4.0, random_state=321),
]


class EnsemblePredictor:
    """多模型集成预测器"""

    def __init__(self):
        self.models = []
        self.feature_names = None
        self._medians = None

    def add_model(self, X, y, sample_weight=None, config=None):
        if self.feature_names is None:
            self.feature_names = list(X.columns)
            self._medians = X.median()
        Xf = X.reindex(columns=self.feature_names).fillna(self._medians)
        for col in Xf.columns:
            Xf[col] = pd.to_numeric(Xf[col], errors="coerce")
        Xf = Xf.fillna(self._medians)
        cfg = config or MODEL_CONFIGS[0]
        m = HistGradientBoostingClassifier(
            max_bins=128, early_stopping=True,
            n_iter_no_change=30, validation_fraction=0.12, **cfg)
        m.fit(Xf, y, sample_weight=sample_weight)
        self.models.append(m)

    def predict_proba_batch(self, X):
        Xf = X.reindex(columns=self.feature_names).fillna(self._medians)
        for col in Xf.columns:
            Xf[col] = pd.to_numeric(Xf[col], errors="coerce")
        Xf = Xf.fillna(self._medians)
        return np.mean([m.predict_proba(Xf)[:, 1] for m in self.models], axis=0)

    def predict(self, X, threshold=0.5, min_confidence=CONFIDENCE_MARGIN):
        """
        返回 (predictions, mask):
          predictions: 1=涨, 0=跌
          mask: True=有预测(置信度足够), False=不确定
        """
        p_up = self.predict_proba_batch(X) if len(X.shape) > 1 else self.predict_proba_batch(X.to_frame().T)
        mask = np.abs(p_up - 0.5) >= min_confidence
        preds = (p_up >= threshold).astype(int)
        return preds, mask, p_up

    def feature_importances(self):
        if not self.models or not self.feature_names:
            return pd.Series(dtype=float)
        imps = []
        for m in self.models:
            try: imps.append(m.feature_importances_)
            except: pass
        if not imps: return pd.Series(dtype=float)
        return pd.Series(np.mean(imps, axis=0), index=self.feature_names).sort_values(ascending=False)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "features": self.feature_names, "medians": self._medians}, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as fp:
            d = pickle.load(fp)
        obj.models = d["models"]
        obj.feature_names = d["features"]
        obj._medians = d["medians"]
        return obj
