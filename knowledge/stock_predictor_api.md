# stock_predictor.py 使用文档

`src/stock_predictor.py` 是股票走势预测的核心模块，提供特征提取和集成模型预测能力。

## 模块常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `LOOKBACK` | 120 | 特征计算所需的最少 K 线天数 |
| `FORWARD` | 15 | 预测未来的天数 |
| `CONFIDENCE_MARGIN` | 0.30 | 置信度过滤阈值，\|P-0.5\| ≥ 0.30 才输出预测 |
| `MODEL_CONFIGS` | list[dict] × 5 | 5 组 HistGradientBoostingClassifier 超参数配置 |

## 函数：`compute_feature_matrix(df)`

从整段 K 线一次性计算所有技术指标特征，返回完整的特征 DataFrame。

### 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | `pd.DataFrame` | K 线数据，必须包含 `open`, `close`, `high`, `low`, `volume` 列 |

### 输出

| 类型 | 说明 |
|------|------|
| `pd.DataFrame` | 与输入同 index，包含 42 列技术指标特征。index ≥ `LOOKBACK-1` 的行有完整数据，之前的行部分值为 NaN |

### 示例

```python
from stock_predictor import compute_feature_matrix

df = pd.read_csv("data/HK_00700.csv")
features = compute_feature_matrix(df)
# features.shape → (n_rows, 42)
# 有效行从 index=119 开始
```

### 输出特征列表（42 列）

| 类别 | 特征名 | 数量 |
|------|--------|------|
| 多尺度收益率 | `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_40d`, `ret_60d`, `ret_90d` | 7 |
| EMA 交叉 | `ema12_26` | 1 |
| 均线体系 | `price_sma5`, `price_sma20`, `price_sma60`, `sma5_sma20`, `sma20_sma60`, `sma5_slope`, `sma20_slope` | 7 |
| RSI | `rsi_14`, `rsi_chg5`, `rsi_lag5` | 3 |
| MACD | `macd_hist_norm`, `macd_hist_chg`, `macd_hist_lag5` | 3 |
| 随机指标 | `stoch_k`, `stoch_d`, `stoch_k_lag5` | 3 |
| ADX | `adx`, `di_diff` | 2 |
| MFI | `mfi_14` | 1 |
| 布林带 | `bb_pctb`, `bb_width` | 2 |
| ATR | `atr_ratio` | 1 |
| 波动率 | `vol_20d` | 1 |
| 成交量 | `vol_ratio` | 1 |
| 涨跌比 | `up_ratio_20` | 1 |
| 区间位置 | `pos_120d`, `dist_hi20` | 2 |
| 趋势 | `trend_consistency`, `trend_slope_20` | 2 |
| K 线形态 | `candle_body`, `candle_range`, `upper_shadow_ratio`, `lower_shadow_ratio`, `engulf_signal` | 5 |

> 注意：`rel_strength` 和 `mkt_momentum` 两个市场相对强弱特征由调用方在外部计算后添加，不在此函数内生成。加上这两个后共 44 列。

## 函数：`extract_features(df)`

`compute_feature_matrix` 的便捷封装，仅返回最后一行特征。

### 输入 / 输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | `pd.DataFrame` | K 线数据，需至少 `LOOKBACK` 行 |
| 返回值 | `dict` | `{特征名: 特征值}` 字典，共 42 个键 |

### 示例

```python
from stock_predictor import extract_features, LOOKBACK

df = pd.read_csv("data/HK_00700.csv").tail(LOOKBACK + 10)
feat_dict = extract_features(df)
# feat_dict["rsi_14"] → 55.32
```

## 类：`EnsemblePredictor`

15 模型集成预测器，负责训练、预测、序列化。

### 创建与训练

```python
from stock_predictor import EnsemblePredictor, MODEL_CONFIGS

predictor = EnsemblePredictor()

# 添加子模型（每次调用训练一个 HistGradientBoostingClassifier）
for config in MODEL_CONFIGS:
    predictor.add_model(X_train, y_train, sample_weight=weights, config=config)

# 训练后包含 len(MODEL_CONFIGS) 个子模型
print(len(predictor.models))  # → 5
```

#### `add_model(X, y, sample_weight=None, config=None)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `X` | `pd.DataFrame` | 特征矩阵，列名应与 `compute_feature_matrix` 输出一致（加上 `rel_strength`, `mkt_momentum`） |
| `y` | `array-like` | 标签，`1` = 未来 15 天上涨，`0` = 下跌 |
| `sample_weight` | `array-like`, 可选 | 样本权重（推荐使用指数衰减权重） |
| `config` | `dict`, 可选 | 模型超参数，默认使用 `MODEL_CONFIGS[0]` |

> 首次调用 `add_model` 时会记录特征列名和各列中位数（用于后续缺失值填充），后续调用会自动对齐列。

### 预测

#### `predict_proba_batch(X)` — 批量概率预测

```python
p_up = predictor.predict_proba_batch(X_test)
# p_up → np.array, shape=(n_samples,), 值域 [0, 1]
# 含义：每个样本未来 15 天上涨的概率
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `X` | `pd.DataFrame` | 特征矩阵（一行或多行） |
| 返回值 | `np.ndarray` | 15 个子模型输出概率的算术平均 |

#### `predict(X, threshold=0.5, min_confidence=CONFIDENCE_MARGIN)` — 带置信度过滤的预测

```python
preds, mask, p_up = predictor.predict(X_test)
# preds → array, 1=涨 0=跌
# mask  → array, True=高置信（值得参考）, False=不确定
# p_up  → array, 原始概率
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `X` | `pd.DataFrame` | 特征矩阵 |
| `threshold` | `float` | 涨跌判断阈值，默认 0.5 |
| `min_confidence` | `float` | 最低置信度，默认 0.30 |
| 返回值 | `(preds, mask, p_up)` | 预测标签、置信度掩码、原始概率 |

### 特征重要性

```python
imp = predictor.feature_importances()
# → pd.Series, index=特征名, 值=平均重要性, 降序排列
print(imp.head(5))
```

### 序列化

```python
# 保存模型（包含 models + feature_names + medians）
predictor.save("results/model.pkl")

# 加载模型
predictor = EnsemblePredictor.load("results/model.pkl")
print(len(predictor.models))  # → 15
```

`model.pkl` 内部结构为一个 dict：

```python
{
    "models": [HistGradientBoostingClassifier, ...],   # 15 个子模型
    "features": ["ret_3d", "ret_5d", ...],              # 44 个特征名
    "medians": pd.Series(...)                            # 44 个特征中位数
}
```

## 完整使用示例

### 场景 1：加载已训练模型，预测单只股票

```python
import pandas as pd
from stock_predictor import (
    compute_feature_matrix, EnsemblePredictor, LOOKBACK
)

# 1. 加载 K 线数据（至少 LOOKBACK=120 行）
df = pd.read_csv("data/HK_00700.csv")
df["time_key"] = pd.to_datetime(df["time_key"])

# 2. 计算特征
feat = compute_feature_matrix(df)
feat["rel_strength"] = 0.0  # 单只股票时可置 0 或自行计算
feat["mkt_momentum"] = 0.0

# 3. 加载模型
model = EnsemblePredictor.load("results/model.pkl")

# 4. 预测最新一行
last_row = feat.iloc[[-1]]
p_up = model.predict_proba_batch(last_row)[0]

print(f"上涨概率: {p_up:.1%}")
if p_up >= 0.80:
    print("→ 强烈看涨")
elif p_up <= 0.20:
    print("→ 强烈看跌")
else:
    print("→ 不确定")
```

### 场景 2：从零训练模型

```python
import numpy as np
import pandas as pd
from stock_predictor import (
    compute_feature_matrix, EnsemblePredictor, MODEL_CONFIGS,
    LOOKBACK, FORWARD
)

# 1. 准备数据
df = pd.read_csv("data/HK_00700.csv")
df["time_key"] = pd.to_datetime(df["time_key"])

# 2. 计算特征 + 标签
feat = compute_feature_matrix(df)
feat["rel_strength"] = 0.0
feat["mkt_momentum"] = 0.0
label = (df["close"].shift(-FORWARD) > df["close"]).astype(int)

# 3. 筛选有效行
valid = feat.notna().all(axis=1) & label.notna()
X = feat.loc[valid]
y = label.loc[valid].values

# 4. 训练
predictor = EnsemblePredictor()
for config in MODEL_CONFIGS:
    predictor.add_model(X, y, config=config)

# 5. 保存
predictor.save("results/model.pkl")
print(f"训练完成，共 {len(predictor.models)} 个子模型")
```

### 场景 3：批量预测 + 置信度过滤

```python
preds, mask, p_up = model.predict(X_test)

# 只看高置信的预测
confident_indices = np.where(mask)[0]
for i in confident_indices:
    direction = "涨" if preds[i] == 1 else "跌"
    print(f"样本 {i}: 预测={direction}, P(涨)={p_up[i]:.1%}")
```
