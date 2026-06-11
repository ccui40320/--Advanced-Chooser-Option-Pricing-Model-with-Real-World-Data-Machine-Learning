"""
Week 5 - Approach 2: End-to-End ML Pricing
=============================================
Directly predict Chooser option price from market features.
Skip BSM entirely - let ML learn the pricing function from data.

Two targets:
  a) Chooser_Price (absolute pricing)
  b) Pricing_Bias (BSM error correction)

Models: XGBoost, GBDT, ElasticNet, Linear Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys, os, warnings, joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 1. 加载数据
# ============================================================

print("=" * 55)
print("  Approach 2: End-to-End ML Pricing")
print("=" * 55)

split_data = joblib.load('ML_Dataset_Split.pkl')
train_df = split_data['train']
val_df = split_data['val']
test_df = split_data['test']
features = split_data['features']

X_train_all = train_df[features].fillna(0).values
X_val_all = val_df[features].fillna(0).values
X_test_all = test_df[features].fillna(0).values

# 保留 DataFrame 用于列名引用
train_df_orig = train_df.copy()
val_df_orig = val_df.copy()
test_df_orig = test_df.copy()

# ============================================================
# 2. 目标 2a: 直接预测 Chooser Price
# ============================================================

print("\n[1/5] 目标 2a: 直接预测 Chooser_Price")

target = 'Chooser_Price'
train_ml = train_df.dropna(subset=[target])
val_ml = val_df.dropna(subset=[target])
test_ml = test_df.dropna(subset=[target])

X_train = train_ml[features].fillna(0).values
y_train = train_ml[target].values
X_val = val_ml[features].fillna(0).values
y_val = val_ml[target].values
X_test = test_ml[features].fillna(0).values
y_test = test_ml[target].values

print("  Train: {}  Val: {}  Test: {}".format(X_train.shape, X_val.shape, X_test.shape))

models_price = {
    'Linear Regression': LinearRegression(),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000, random_state=42),
    'GBDT': GradientBoostingRegressor(n_estimators=200, max_depth=6,
                                       learning_rate=0.05, random_state=42),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=20, verbosity=0
    ),
}

results_price = {}
for name, model in models_price.items():
    if name == 'XGBoost':
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results_price[name] = {
        'y_pred': y_pred,
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
    }

print("\n  {:20s} {:>8} {:>8} {:>8}".format('Model', 'MAE', 'RMSE', 'R2'))
print("  " + "-" * 44)
for name, r in results_price.items():
    print("  {:20s} {:>8.2f} {:>8.2f} {:>8.4f}".format(name, r['mae'], r['rmse'], r['r2']))

# 基线: BSM 模型价格 (对比)
baseline_mae = mean_absolute_error(y_test, test_ml.get('Intrinsic_Value', 0))
print("  {:20s} {:>8.2f}".format('BSM Intrinsic(基线)', baseline_mae))

best_price_model = max(results_price, key=lambda k: results_price[k]['r2'])
print("\n  Best model: {} (R2={:.4f})".format(best_price_model, results_price[best_price_model]['r2']))

# ============================================================
# 3. 目标 2b: 预测 Pricing_Bias (BSM 误差修正)
# ============================================================

print("\n\n[2/5] 目标 2b: 预测 Pricing_Bias (BSM 误差修正)")

target_bias = 'Pricing_Bias'
train_bias = train_df.dropna(subset=[target_bias])
val_bias = val_df.dropna(subset=[target_bias])
test_bias = test_df.dropna(subset=[target_bias])

X_train_b = train_bias[features].fillna(0).values
y_train_b = train_bias[target_bias].values
X_val_b = val_bias[features].fillna(0).values
y_val_b = val_bias[target_bias].values
X_test_b = test_bias[features].fillna(0).values
y_test_b = test_bias[target_bias].values

print("  Train: {}  Val: {}  Test: {}".format(X_train_b.shape, X_val_b.shape, X_test_b.shape))

models_bias = {
    'Linear Regression': LinearRegression(),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000, random_state=42),
    'GBDT': GradientBoostingRegressor(n_estimators=200, max_depth=6,
                                       learning_rate=0.05, random_state=42),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=20, verbosity=0
    ),
}

results_bias = {}
for name, model in models_bias.items():
    if name == 'XGBoost':
        model.fit(X_train_b, y_train_b, eval_set=[(X_val_b, y_val_b)], verbose=False)
    else:
        model.fit(X_train_b, y_train_b)
    y_pred = model.predict(X_test_b)
    results_bias[name] = {
        'y_pred': y_pred,
        'mae': mean_absolute_error(y_test_b, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_b, y_pred)),
        'r2': r2_score(y_test_b, y_pred),
    }

print("\n  {:20s} {:>8} {:>8} {:>8}".format('Model', 'MAE', 'RMSE', 'R2'))
print("  " + "-" * 44)
for name, r in results_bias.items():
    print("  {:20s} {:>8.2f} {:>8.2f} {:>8.4f}".format(name, r['mae'], r['rmse'], r['r2']))

# 基线: 直接预测 0 (即不做修正)
bias_baseline = mean_absolute_error(y_test_b, np.zeros_like(y_test_b))
print("  {:20s} {:>8.2f}".format('Zero (无修正)', bias_baseline))

best_bias_model = max(results_bias, key=lambda k: results_bias[k]['r2'])
print("  Best model: {} (R2={:.4f})".format(best_bias_model, results_bias[best_bias_model]['r2']))

# ============================================================
# 4. 综合定价: BSM + ML Bias Correction
# ============================================================

print("\n\n[3/5] 综合定价: BSM + ML Bias Correction")

test_combined = test_bias.copy()
test_combined['BSM_Price'] = test_combined['Chooser_Price']
test_combined['ML_Bias'] = results_bias[best_bias_model]['y_pred']
test_combined['Corrected_Price'] = (test_combined['BSM_Price'] +
                                    test_combined['ML_Bias'])
test_combined['IV'] = np.maximum(test_combined['JPM_Close'] - 150.0, 0)

bsm_mae = mean_absolute_error(test_combined['IV'], test_combined['BSM_Price'])
corrected_mae = mean_absolute_error(test_combined['IV'], test_combined['Corrected_Price'])

print("  BSM Only MAE:       {:.4f}".format(bsm_mae))
print("  BSM + ML Bias MAE:  {:.4f}".format(corrected_mae))
improvement = (bsm_mae - corrected_mae) / bsm_mae * 100
print("  改进率: {:.2f}%".format(improvement))

# ============================================================
# 5. 可视化
# ============================================================

print("\n\n[4/5] 生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Approach 2: End-to-End ML Pricing', fontsize=13)

# (a) 目标2a: 预测 vs 实际 Chooser 价格
ax = axes[0, 0]
r2_a = results_price[best_price_model]['r2']
ax.scatter(y_test, results_price[best_price_model]['y_pred'], alpha=0.3, s=5, c='blue')
max_val = max(y_test.max(), results_price[best_price_model]['y_pred'].max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
ax.set_xlabel('Real Price')
ax.set_ylabel('Predicted Price')
ax.set_title('Direct Price Prediction - {} (R2={:.3f})'.format(best_price_model, r2_a))
ax.grid(True, alpha=0.3)

# (b) Bias 预测 vs 实际
ax = axes[0, 1]
r2_b = results_bias[best_bias_model]['r2']
ax.scatter(y_test_b, results_bias[best_bias_model]['y_pred'], alpha=0.3, s=5, c='green')
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('Real Bias')
ax.set_ylabel('Predicted Bias')
ax.set_title('Bias Prediction - {} (R2={:.3f})'.format(best_bias_model, r2_b))
ax.grid(True, alpha=0.3)

# (c) 各模型 MAE 对比 (目标2a)
ax = axes[1, 0]
models_n = list(results_price.keys())
maes = [results_price[m]['mae'] for m in models_n]
bars = ax.barh(models_n, maes, color='steelblue')
ax.axvline(baseline_mae, color='red', linestyle='--', label='BSM Baseline ({:.1f})'.format(baseline_mae))
ax.set_xlabel('MAE')
ax.set_title('Model Comparison - Direct Price')
ax.legend(fontsize=8)
for bar, val in zip(bars, maes):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, '{:.1f}'.format(val),
            va='center', fontsize=8)

# (d) 特征重要性 (XGBoost for bias)
ax = axes[1, 1]
xgb_bias = models_bias['XGBoost']
if hasattr(xgb_bias, 'feature_importances_'):
    importances = xgb_bias.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(15)
    feat_imp.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Features - Bias Prediction (XGBoost)')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('ML_Approach2_EndToEnd.png', dpi=150, bbox_inches='tight')
print("  -> 图保存: ML_Approach2_EndToEnd.png")

# ============================================================
# 6. 保存模型
# ============================================================

print("\n[5/5] 保存模型...")

joblib.dump(models_price[best_price_model], 'ML_Model_DirectPrice.pkl')
joblib.dump(models_bias[best_bias_model], 'ML_Model_BiasCorrection.pkl')
print("  -> ML_Model_DirectPrice.pkl ({})".format(best_price_model))
print("  -> ML_Model_BiasCorrection.pkl ({})".format(best_bias_model))

# 保存结果对比
comparison = pd.DataFrame({
    'Approach': ['BSM Baseline', 'Approach1 (ML Vol)', 'Approach2a (Direct Price)', 'Approach2b (Bias Corr)'],
    'MAE': [baseline_mae, 0, results_price[best_price_model]['mae'], corrected_mae],
    'Model': ['BSM', '-', best_price_model, 'BSM + ' + best_bias_model],
})
comparison.to_csv('ML_Model_Comparison.csv', index=False)
print("  -> ML_Model_Comparison.csv")

print("\n{}".format('='*55))
print("  Approach 2 完成!")
print("{}".format('='*55))
