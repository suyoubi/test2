"""
股票走势预测算法 — V3.1 双模型版 (抗过拟合优化)

算法概述:
  1. 从120天K线数据提取~60个技术指标+增强特征
  2. 添加市场基准相对强弱特征
  3. 分市场独立建模 (港股/A股各自训练)
  4. 多样性集成: GBM(5配置×3窗口) + RF(3) + ET(3) + LR(1) ≈ 22模型
  5. 双模型策略:
     - 跌信号: 标准模型(涨>0%标签), P≤0.20
     - 涨信号: 强标签模型(涨>3%标签) + 子模型投票
  6. 子模型超多数投票: 要求≥70%子模型预测P>0.65才输出涨信号

V3.1 抗过拟合优化:
  - GBM 正则化: 降低 max_depth, 提高 min_samples_leaf 和 l2_regularization

时间序列纪律（禁止「偷看未来」）:
  - 禁止对时序样本使用 train_test_split(shuffle=True)。
  - HistGradientBoostingClassifier(early_stopping=True) 在 sklearn 内部会对分类任务做
    stratified train_test_split（shuffle=True），验证集会混入未来统计结构，故生产路径统一
    early_stopping=False，靠 max_iter + 强正则控制复杂度。
  - 主回测使用 Walk-forward（按季度训练窗→未来测试窗）；另见 time_series_eval.py 的
    TimeSeriesSplit / 固定年份切分示例。
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

LOOKBACK = 120
FORWARD = 15
CONFIDENCE_MARGIN = 0.30  # |P-0.5| >= 0.30 才输出预测
UP_LABEL_THRESHOLD = 0.03  # 强涨标签: 涨幅>3%才算涨 (用于涨信号模型)
UP_P_THRESHOLD = 0.70      # 涨信号: 强标签模型P >= 此值
UP_VOTE_MODEL_THRESH = 0.65  # 投票: 子模型P > 此值算看涨
UP_VOTE_AGREE = 0.70        # 投票: 需≥此比例子模型同意
DN_P_THRESHOLD = 0.20       # 跌信号: 标准模型P <= 此值


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


def compute_enhanced_feature_matrix(df):
    """计算基础特征 + 增强特征 (日历/波动率状态/缺口/价格加速度等)"""
    f = compute_feature_matrix(df)
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


def extract_features(df):
    """兼容接口: 从K线DataFrame提取最后一行特征dict"""
    return compute_enhanced_feature_matrix(df).iloc[-1].to_dict()


def compute_inflection_signal(feat: pd.DataFrame) -> dict:
    """用技术面启发式判断短期是否接近走势拐点（非机器学习输出）。

    综合：120 日区间位置、RSI 与变化、价格加速度、MACD 柱变化、短期均线斜率翻转、
    连涨连跌与短期收益。bull_pts / bear_pts 为内部投票分，用于强度分级。

    返回:
        kind: 可能向上拐点 | 可能向下拐点 | 动能切换观察 | 暂无明显拐点
        strength: 强 | 中 | 弱
        score: 0–100，>50 偏向上拐点证据，<50 偏向下拐点证据；用 tanh+证据强度拉开梯度，避免大量挤在 100
        hint: 一条简短说明
    """
    empty = {
        "kind": "暂无明显拐点", "strength": "—", "score": 50.0, "hint": "",
        "bull_pts": 0, "bear_pts": 0,
    }
    if feat is None or len(feat) < 1:
        return empty

    def gv(row, key, default=np.nan):
        v = row[key] if key in row.index else np.nan
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    last = feat.iloc[-1]
    bull_pts, bear_pts = 0, 0
    hints_bull, hints_bear = [], []

    pos = gv(last, "pos_120d")
    ret5 = gv(last, "ret_5d")
    ret20 = gv(last, "ret_20d")
    ret3 = gv(last, "ret_3d")
    rsi = gv(last, "rsi_14")
    rsi5 = gv(last, "rsi_chg5")
    accel = gv(last, "price_accel")
    macd_n = gv(last, "macd_hist_norm")
    macd_c = gv(last, "macd_hist_chg")
    dist_hi = gv(last, "dist_hi20")
    consec_up = gv(last, "consec_up")
    consec_dn = gv(last, "consec_dn")

    # --- 向下拐点（见顶/滞涨）证据 ---
    if np.isfinite(pos) and pos >= 0.88 and np.isfinite(ret5) and ret5 < 0:
        bear_pts += 2
        hints_bear.append("高位回撤")
    if np.isfinite(rsi) and rsi >= 68.0 and np.isfinite(rsi5) and rsi5 < 0:
        bear_pts += 2
        hints_bear.append("RSI高位回落")
    if np.isfinite(accel) and accel < 0 and np.isfinite(ret20) and ret20 > 0.03:
        bear_pts += 1
        hints_bear.append("涨势减速")
    if np.isfinite(macd_n) and np.isfinite(macd_c) and macd_n > 0 and macd_c < 0:
        bear_pts += 1
        hints_bear.append("MACD柱见顶")
    if np.isfinite(dist_hi) and dist_hi > -0.02 and np.isfinite(ret5) and ret5 < 0:
        bear_pts += 1
        hints_bear.append("贴近20日高")
    if np.isfinite(consec_up) and consec_up >= 4.0 and np.isfinite(ret5) and ret5 < 0:
        bear_pts += 1
        hints_bear.append("连涨后走弱")

    # --- 向上拐点（见底/企稳）证据 ---
    if np.isfinite(pos) and pos <= 0.15 and np.isfinite(ret5) and ret5 > 0:
        bull_pts += 2
        hints_bull.append("低位反弹")
    if np.isfinite(rsi) and rsi <= 35.0 and np.isfinite(rsi5) and rsi5 > 0:
        bull_pts += 2
        hints_bull.append("RSI低位拐头")
    if np.isfinite(accel) and accel > 0 and np.isfinite(ret20) and ret20 < -0.03:
        bull_pts += 1
        hints_bull.append("跌势减速")
    if np.isfinite(macd_n) and np.isfinite(macd_c) and macd_n < 0 and macd_c > 0:
        bull_pts += 1
        hints_bull.append("MACD柱见底")
    if np.isfinite(consec_dn) and consec_dn >= 4.0 and np.isfinite(ret5) and ret5 > 0:
        bull_pts += 1
        hints_bull.append("连跌后企稳")
    if np.isfinite(ret3) and ret3 > 0 and np.isfinite(ret20) and ret20 < -0.05:
        bull_pts += 1
        hints_bull.append("深跌后短反")

    # --- 均线斜率翻转（需至少 2 根）---
    if len(feat) >= 2 and "sma5_slope" in feat.columns:
        s0 = float(feat["sma5_slope"].iloc[-2])
        s1 = float(feat["sma5_slope"].iloc[-1])
        if np.isfinite(s0) and np.isfinite(s1):
            if s0 > 0 and s1 <= 0:
                bear_pts += 2
                hints_bear.append("5日均线斜率转负")
            elif s0 < 0 and s1 >= 0:
                bull_pts += 2
                hints_bull.append("5日均线斜率转正")

    diff = bull_pts - bear_pts
    tot = bull_pts + bear_pts
    if tot == 0:
        score = 50.0
    else:
        # 方向分量 tanh 饱和慢，避免 diff 稍大就顶格；再乘证据强度使「规则少」时更接近 50
        dir_u = float(np.tanh(diff / 5.0))
        mag = min(1.0, tot / 7.0)
        blend = 0.22 + 0.78 * mag
        score = 50.0 + 48.0 * dir_u * blend
        score = float(np.clip(score, 0.0, 100.0))

    if bull_pts >= 4 and diff >= 2:
        kind = "可能向上拐点"
        strength = "强" if bull_pts >= 6 else "中"
        hint = "、".join(hints_bull[:3]) if hints_bull else "多因子偏多转折"
    elif bear_pts >= 4 and diff <= -2:
        kind = "可能向下拐点"
        strength = "强" if bear_pts >= 6 else "中"
        hint = "、".join(hints_bear[:3]) if hints_bear else "多因子偏空转折"
    elif bull_pts >= 2 and diff >= 1:
        kind = "可能向上拐点"
        strength = "弱"
        hint = "、".join(hints_bull[:2]) if hints_bull else "弱反弹迹象"
    elif bear_pts >= 2 and diff <= -1:
        kind = "可能向下拐点"
        strength = "弱"
        hint = "、".join(hints_bear[:2]) if hints_bear else "弱滞涨迹象"
    elif max(bull_pts, bear_pts) >= 2 and abs(diff) <= 1:
        kind = "动能切换观察"
        strength = "弱"
        hint = "多空证据并存，观察确认"
    else:
        kind = "暂无明显拐点"
        strength = "—"
        hint = ""

    return {
        "kind": kind,
        "strength": strength,
        "score": round(score, 1),
        "hint": hint,
        "bull_pts": bull_pts,
        "bear_pts": bear_pts,
    }


# ── 集成预测模型 ──────────────────────────────────────────────

MODEL_CONFIGS = [
    dict(max_iter=500, max_depth=3, learning_rate=0.03, min_samples_leaf=100,
         l2_regularization=5.0, random_state=42),
    dict(max_iter=600, max_depth=3, learning_rate=0.02, min_samples_leaf=120,
         l2_regularization=6.0, random_state=123),
    dict(max_iter=400, max_depth=2, learning_rate=0.05, min_samples_leaf=80,
         l2_regularization=4.0, random_state=456),
    dict(max_iter=350, max_depth=2, learning_rate=0.04, min_samples_leaf=100,
         l2_regularization=5.0, random_state=789),
    dict(max_iter=700, max_depth=4, learning_rate=0.015, min_samples_leaf=150,
         l2_regularization=10.0, random_state=321),
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
            max_bins=128, early_stopping=False, **cfg)
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


class DiverseEnsemble:
    """多模型多样性集成: GBM + RandomForest + ExtraTrees + LogisticRegression"""

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

    def add_gbm(self, X, y, sample_weight=None, config=None):
        Xf = self._prep(X)
        cfg = config or MODEL_CONFIGS[0]
        m = HistGradientBoostingClassifier(
            max_bins=128, early_stopping=False, **cfg)
        m.fit(Xf, y, sample_weight=sample_weight)
        self.models.append(("gbm", m))

    def add_rf(self, X, y, sample_weight=None):
        Xf = self._prep(X)
        m = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=80,
            max_features="sqrt", random_state=42, n_jobs=-1)
        m.fit(Xf, y, sample_weight=sample_weight)
        self.models.append(("rf", m))

    def add_et(self, X, y, sample_weight=None):
        Xf = self._prep(X)
        m = ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=80,
            max_features="sqrt", random_state=42, n_jobs=-1)
        m.fit(Xf, y, sample_weight=sample_weight)
        self.models.append(("et", m))

    def add_lr(self, X, y, sample_weight=None):
        Xf = self._prep(X)
        if self._scaler is None:
            self._scaler = StandardScaler()
            Xs = self._scaler.fit_transform(Xf)
        else:
            Xs = self._scaler.transform(Xf)
        m = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
        m.fit(Xs, y, sample_weight=sample_weight)
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

    def predict_individual(self, X):
        """返回每个子模型的概率, shape: (n_models, n_samples)"""
        Xf = self._prep(X)
        ps = []
        for t, m in self.models:
            if t == "lr" and self._scaler:
                ps.append(m.predict_proba(self._scaler.transform(Xf))[:, 1])
            else:
                ps.append(m.predict_proba(Xf)[:, 1])
        return np.array(ps)

    def predict_with_voting(self, X, model_thresh=UP_VOTE_MODEL_THRESH, agree=UP_VOTE_AGREE):
        """带投票的预测: 返回 (p_avg, up_ratio)
        up_ratio: 每个样本中看涨(P>model_thresh)的子模型占比
        """
        indiv = self.predict_individual(X)
        p_avg = indiv.mean(axis=0)
        up_ratio = (indiv > model_thresh).mean(axis=0)
        return p_avg, up_ratio

    def predict(self, X, threshold=0.5, min_confidence=CONFIDENCE_MARGIN):
        p_up = self.predict_proba_batch(X) if len(X.shape) > 1 else self.predict_proba_batch(X.to_frame().T)
        mask = np.abs(p_up - 0.5) >= min_confidence
        preds = (p_up >= threshold).astype(int)
        return preds, mask, p_up

    def feature_importances(self):
        if not self.models or not self.feature_names:
            return pd.Series(dtype=float)
        imps = []
        for t, m in self.models:
            try:
                imps.append(m.feature_importances_)
            except Exception:
                pass
        if not imps:
            return pd.Series(dtype=float)
        return pd.Series(np.mean(imps, axis=0), index=self.feature_names).sort_values(ascending=False)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models, "features": self.feature_names,
                "medians": self._medians, "scaler": self._scaler,
            }, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as fp:
            d = pickle.load(fp)
        obj.models = d["models"]
        obj.feature_names = d["features"]
        obj._medians = d["medians"]
        obj._scaler = d.get("scaler")
        return obj


class DualModelPredictor:
    """V3 双模型预测: 标准模型负责跌信号, 强标签模型+投票负责涨信号

    涨: 强标签模型(涨>3%) P >= UP_P_THRESHOLD 且 ≥UP_VOTE_AGREE子模型P>UP_VOTE_MODEL_THRESH
    跌: 标准模型 P <= DN_P_THRESHOLD
    """

    def __init__(self, std_model, strong_model):
        self.std_model = std_model
        self.strong_model = strong_model

    def predict(self, X):
        """返回 (predictions, signal_mask, details)
        predictions: 1=涨, 0=跌
        signal_mask: True=有信号
        details: dict with p_std, p_strong, up_ratio
        """
        p_std = self.std_model.predict_proba_batch(X)
        p_strong, up_ratio = self.strong_model.predict_with_voting(
            X, model_thresh=UP_VOTE_MODEL_THRESH, agree=UP_VOTE_AGREE
        )

        is_up = (p_strong >= UP_P_THRESHOLD) & (up_ratio >= UP_VOTE_AGREE)
        is_dn = p_std <= DN_P_THRESHOLD

        signal_mask = is_up | is_dn
        predictions = np.where(is_up, 1, np.where(is_dn, 0, -1))

        return predictions, signal_mask, {
            "p_std": p_std,
            "p_strong": p_strong,
            "up_ratio": up_ratio,
            "is_up": is_up,
            "is_dn": is_dn,
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"std": self.std_model, "strong": self.strong_model}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fp:
            d = pickle.load(fp)
        return cls(d["std"], d["strong"])
