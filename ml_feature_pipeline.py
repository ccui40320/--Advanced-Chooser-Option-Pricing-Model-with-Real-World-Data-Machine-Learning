"""
Week 5 — ML Feature Pipeline
==============================
Prepares features for both ML approaches:
  - Loads Dynamic_Cleaned_Dataset.csv
  - Engineers lag/rolling features
  - Creates time-series split (70/15/15)
  - Saves train/val/test sets
  - Prevents look-ahead bias
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 1. 加载数据
# ============================================================

DATA_PATH = 'Dynamic_Cleaned_Dataset.csv'
if not os.path.exists(DATA_PATH):
    DATA_PATH = '../ChooserOptionProject/Dynamic_Cleaned_Dataset.csv'
if not os.path.exists(DATA_PATH):
    DATA_PATH = '../week2_ Deliverables/Dynamic_Cleaned_Dataset.csv'

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
print(f"[OK] 加载数据: {len(df)} 行 x {len(df.columns)} 列")
print(f"  时间: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"  列: {list(df.columns)}")

# ============================================================
# 2. 生成目标变量 (Targets)
# ============================================================

print("\n[1/4] 生成目标变量...")

# 目标 1: Chooser 价格 (用于 Approach 2)
from bsm_model import chooser_price

K = 150.0
T1 = 0.5
T2 = 1.0

rate_col = 'RiskFreeRate_3M_Decimal' if 'RiskFreeRate_3M_Decimal' in df.columns else 'RiskFreeRate_Decimal'
vol_col = 'Volatility_21D' if 'Volatility_21D' in df.columns else 'Rolling_Vol_20D'
div_col = 'Dividend_Yield' if 'Dividend_Yield' in df.columns else None

chooser_prices = []
for idx, row in df.iterrows():
    S = row['JPM_Close']
    r = row[rate_col]
    sigma = row[vol_col]
    q = row[div_col] if div_col else (row['JPM_Dividend'] / S if S > 0 else 0)

    if np.isnan(sigma) or sigma <= 0:
        chooser_prices.append(np.nan)
        continue
    price, _ = chooser_price(S, K, T1, T2, r, sigma, q)
    chooser_prices.append(price)

df['Chooser_Price'] = chooser_prices

# 目标 2: 内在价值 (基线对比)
df['Intrinsic_Value'] = np.maximum(df['JPM_Close'] - K, 0)

# 目标 3: 定价偏差 (BSM vs 内在价值)
df['Pricing_Bias'] = df['Chooser_Price'] - df['Intrinsic_Value']

# 目标 4: 未来波动率 (用于 Approach 1)
df['Vol_t+1'] = df[vol_col].shift(-1)

print(f"  Chooser_Price: {df['Chooser_Price'].notna().sum()} 个有效值")
print(f"  Pricing_Bias:  {df['Pricing_Bias'].notna().sum()} 个有效值")

# ============================================================
# 3. 特征工程
# ============================================================

print("\n[2/4] 特征工程...")

feature_df = df.copy()

# 收益率滞后
for lag in [1, 2, 5, 21]:
    feature_df[f'Return_lag{lag}'] = feature_df['Daily_Return'].shift(lag)

# 波动率滞后
for lag in [1, 5, 21]:
    feature_df[f'Vol_lag{lag}'] = feature_df[vol_col].shift(lag)

# VIX 滞后
feature_df['VIX_lag1'] = feature_df['VIX_Close'].shift(1)
feature_df['VIX_MA20'] = feature_df['VIX_Close'].rolling(20).mean()
feature_df['VIX_ratio'] = feature_df['VIX_Close'] / feature_df['VIX_MA20']

# 波动率极值
feature_df['Vol_max_21D'] = feature_df[vol_col].rolling(21).max()
feature_df['Vol_min_21D'] = feature_df[vol_col].rolling(21).min()

# 股价水平特征
feature_df['Price_MA20'] = feature_df['JPM_Close'].rolling(20).mean()
feature_df['Price_ratio'] = feature_df['JPM_Close'] / feature_df['Price_MA20']

# VIX-JPM 领先滞后关系
feature_df['VIX_JPM_Corr_lag1'] = feature_df['VIX_JPM_Corr'].shift(1)

# 情绪滞后
feature_df['Sentiment_lag1'] = feature_df['Sentiment_Score'].shift(1)

# 利率特征
feature_df['Rate_Spread_1Y_3M'] = (feature_df['RiskFreeRate_1Y_Decimal'] -
                                    feature_df['RiskFreeRate_3M_Decimal'])
feature_df['Rate_Spread_10Y_3M'] = (feature_df['RiskFreeRate_10Y_Decimal'] -
                                     feature_df['RiskFreeRate_3M_Decimal'])

# 波动率变化率
feature_df['Vol_change_5D'] = feature_df[vol_col].pct_change(periods=5)
feature_df['VIX_change_5D'] = feature_df['VIX_Close'].pct_change(periods=5)

print(f"  原始特征: {len(df.columns)} 列")
print(f"  工程后特征: {len(feature_df.columns)} 列")

# ============================================================
# 4. 时间序列分割 (70/15/15)
# ============================================================

print("\n[3/4] 时间序列分割...")

# 按时间排序
feature_df = feature_df.sort_index()

# 分割点
total = len(feature_df)
train_end = int(total * 0.70)
val_end = int(total * 0.85)

train_cutoff = feature_df.index[train_end]
val_cutoff = feature_df.index[val_end]

print(f"  分割点: Train ≤ {train_cutoff.date()}, "
      f"Val ≤ {val_cutoff.date()}, Test > {val_cutoff.date()}")

# 定义特征列和目标列
base_features = [
    'Daily_Return', vol_col, 'VIX_Close', 'VIX_Return',
    'VIX_JPM_Corr', 'Sentiment_Score', 'Dividend_Yield',
    'Interest_Rate_Momentum', 'Price_ratio',
]

engineered_features = [c for c in feature_df.columns if any(
    k in c for k in ['_lag', '_MA', '_ratio', '_change', 'Spread', '_max', '_min',
                     'Return_lag', 'Vol_lag', 'VIX_lag', 'Sentiment_lag'])]

# 合并特征
all_features = base_features + engineered_features
# 只保留存在于 DataFrame 中的特征
all_features = [c for c in all_features if c in feature_df.columns]
print(f"  总特征数: {len(all_features)}")

# 目标变量
targets = {
    'approach1_vol': 'Vol_t+1',
    'approach2_price': 'Chooser_Price',
    'approach2_bias': 'Pricing_Bias',
}

# 创建训练/验证/测试集
train_df = feature_df.iloc[:train_end].copy()
val_df = feature_df.iloc[train_end:val_end].copy()
test_df = feature_df.iloc[val_end:].copy()

print(f"\n  Train: {len(train_df)} 行 ({train_df.index[0].date()} ~ {train_df.index[-1].date()})")
print(f"  Val:   {len(val_df)} 行 ({val_df.index[0].date()} ~ {val_df.index[-1].date()})")
print(f"  Test:  {len(test_df)} 行 ({test_df.index[0].date()} ~ {test_df.index[-1].date()})")

# ============================================================
# 5. 保存
# ============================================================

print("\n[4/4] 保存...")

# 保存全量数据
feature_df.to_csv('ML_Full_Features.csv')
print(f"  -> ML_Full_Features.csv ({len(feature_df)} 行 x {len(feature_df.columns)} 列)")

# 保存分割后的数据集 (带特征列表)
split_data = {
    'train': train_df,
    'val': val_df,
    'test': test_df,
    'features': all_features,
    'targets': targets,
}

import joblib
joblib.dump(split_data, 'ML_Dataset_Split.pkl')
print(f"  -> ML_Dataset_Split.pkl (Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)})")
print(f"  -> 特征列表已保存 ({len(all_features)} 个特征)")
print("\n[OK] 特征流水线完成!")
