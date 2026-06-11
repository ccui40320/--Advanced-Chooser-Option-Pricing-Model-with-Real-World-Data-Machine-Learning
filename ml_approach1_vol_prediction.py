"""
Week 5 — Approach 1: ML Volatility Prediction + BSM Pricing
=============================================================
Pipeline:
  1. Load features & split from ML_Dataset_Split.pkl
  2. Train XGBoost/RF/LR to predict Vol_t+1 (next-day volatility)
  3. Feed predicted vol into BSM Chooser formula
  4. Compare against baseline (historical vol + BSM)
  5. Evaluate: MAE, RMSE, R², improvement over baseline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys, os, warnings, joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120

sys.stdout.reconfigure(encoding='utf-8')

from bsm_model import chooser_price, bsm_call, bsm_put

# ============================================================
# 1. 加载数据
# ============================================================

print("=" * 55)
print("  Approach 1: ML Volatility Prediction + BSM")
print("=" * 55)

split_data = joblib.load('ML_Dataset_Split.pkl')
train_df = split_data['train']
val_df = split_data['val']
test_df = split_data['test']
features = split_data['features']

# 目标: 预测未来波动率
target_col = 'Vol_t+1'

# 只保留有目标值的行
train_ml = train_df.dropna(subset=[target_col]).copy()
val_ml = val_df.dropna(subset=[target_col]).copy()
test_ml = test_df.dropna(subset=[target_col]).copy()

X_train = train_ml[features].fillna(0).values
y_train = train_ml[target_col].values
X_val = val_ml[features].fillna(0).values
y_val = val_ml[target_col].values
X_test = test_ml[features].fillna(0).values
y_test = test_ml[target_col].values

print(f"\n[1/5] 数据准备:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

# ============================================================
# 2. 训练多个模型
# ============================================================

print("\n[2/5] 训练波动率预测模型...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12,
                                            n_jobs=-1, random_state=42),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=20, verbosity=0
    ),
}

trained = {}
for name, model in models.items():
    print(f"  训练 {name}...")

    if name == 'XGBoost':
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    trained[name] = model

# ============================================================
# 3. 评估波动率预测
# ============================================================

print("\n[3/5] 波动率预测评估 (Test Set):")
vol_results = []
for name, model in trained.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    vol_results.append({
        'Model': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2
    })
    print(f"  {name:20s} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

# 基线: 直接用历史波动率 (naive forecast)
vol_col_name = 'Volatility_21D' if 'Volatility_21D' in test_ml.columns else 'Rolling_Vol_20D'
naive_pred = test_ml[vol_col_name].fillna(0).values
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
print(f"  {'Naive(hist vol)':20s} MAE={naive_mae:.4f}  RMSE={naive_rmse:.4f}")

# ============================================================
# 4. 用预测的波动率进行 Chooser 定价
# ============================================================

print("\n[4/5] Chooser 定价对比 (Test Set)…")

K = 150.0
T1 = 0.5
T2 = 1.0

# 获取测试集的 BSM 输入参数
rate_col = 'RiskFreeRate_3M_Decimal' if 'RiskFreeRate_3M_Decimal' in test_df.columns else 'RiskFreeRate_Decimal'
vol_col = 'Volatility_21D' if 'Volatility_21D' in test_df.columns else 'Rolling_Vol_20D'
div_col = 'Dividend_Yield' if 'Dividend_Yield' in test_df.columns else None

results = []
best_model_name = min(vol_results, key=lambda x: x['RMSE'])['Model']

for idx in test_df.index:
    row = test_df.loc[idx]
    S = row['JPM_Close']
    r = row[rate_col]
    sigma_hist = row[vol_col]
    q = row[div_col] if div_col and not pd.isna(row.get(div_col, np.nan)) else 0

    if pd.isna(sigma_hist) or sigma_hist <= 0:
        continue

    # 基线: 历史波动率
    price_baseline, _ = chooser_price(S, K, T1, T2, r, sigma_hist, q)

    # ML 预测波动率
    row_features = test_df.loc[idx, features].fillna(0).values.reshape(1, -1)
    sigma_ml = trained[best_model_name].predict(row_features)[0]
    sigma_ml = max(sigma_ml, 0.05)  # 最低 5%

    price_ml, _ = chooser_price(S, K, T1, T2, r, sigma_ml, q)

    results.append({
        'Date': idx,
        'S': S, 'r': r, 'q': q,
        'sigma_hist': sigma_hist, 'sigma_ml': sigma_ml,
        'price_baseline': price_baseline,
        'price_ml': price_ml,
    })

result_df = pd.DataFrame(results)

# 计算 Chooser 定价误差 (以内在价值为参考基准)
result_df['IV'] = np.maximum(result_df['S'] - K, 0)
result_df['baseline_error'] = abs(result_df['price_baseline'] - result_df['IV'])
result_df['ml_error'] = abs(result_df['price_ml'] - result_df['IV'])

print(f"\n  Chooser 定价结果 ({best_model_name}):")
print(f"  Baseline MAE (hist vol): {result_df['baseline_error'].mean():.4f}")
print(f"  ML-Vol   MAE (pred vol): {result_df['ml_error'].mean():.4f}")
improvement = ((result_df['baseline_error'].mean() - result_df['ml_error'].mean())
               / result_df['baseline_error'].mean() * 100)
print(f"  改进: {improvement:.2f}%")

# ============================================================
# 5. 可视化
# ============================================================

print("\n[5/5] 生成可视化…")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Approach 1: ML Volatility Prediction + BSM Pricing', fontsize=13)

# (a) 波动率预测: 真实 vs 预测 (取最后500天)
ax = axes[0, 0]
subset = result_df.tail(500)
ax.plot(subset['Date'], subset['sigma_hist'].values * 100, 'b-', linewidth=1, alpha=0.7, label='Real Vol')
ax.plot(subset['Date'], subset['sigma_ml'].values * 100, 'r--', linewidth=1, alpha=0.7, label='ML Pred Vol')
ax.set_ylabel('Volatility (%)')
ax.set_title(f'Vol Prediction — {best_model_name}')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) 定价散点: 基线 vs ML-vol
ax = axes[0, 1]
max_price = max(result_df['price_baseline'].max(), result_df['price_ml'].max())
ax.scatter(result_df['price_baseline'], result_df['price_ml'], alpha=0.3, s=5, c='blue')
ax.plot([0, max_price], [0, max_price], 'r--', linewidth=1)
ax.set_xlabel('Baseline Price (hist vol)')
ax.set_ylabel('ML-Vol Price (pred vol)')
ax.set_title(f'Price Comparison (Baseline vs ML-Vol)')
ax.grid(True, alpha=0.3)

# (c) 误差改进时间序列
ax = axes[1, 0]
ax.plot(result_df['Date'], result_df['baseline_error'], 'b-', linewidth=0.8, alpha=0.5, label='Baseline Error')
ax.plot(result_df['Date'], result_df['ml_error'], 'r-', linewidth=0.8, alpha=0.5, label='ML-Vol Error')
ax.set_ylabel('Absolute Error ($)')
ax.set_title('Pricing Error: Baseline vs ML-Vol')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) 模型特征重要性 (XGBoost)
ax = axes[1, 1]
if best_model_name == 'XGBoost' and hasattr(trained['XGBoost'], 'feature_importances_'):
    importances = trained['XGBoost'].feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(15)
    feat_imp.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Features (XGBoost)')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('ML_Approach1_Vol_Prediction.png', dpi=150, bbox_inches='tight')
print(f"  -> 图保存: ML_Approach1_Vol_Prediction.png")

# 保存结果
result_df.to_csv('ML_Approach1_Results.csv', index=False)
print(f"  -> 数据保存: ML_Approach1_Results.csv")

print(f"\n{'='*55}")
print(f"  Approach 1 完成! 改进率: {improvement:.2f}%")
print(f"{'='*55}")
