"""
Week 6 — 模型训练与评估 (Model Training & Evaluation)
======================================================
用调优后的最佳模型训练完整 pipeline:
  1. 预测未来波动率 (Vol_t+1)
  2. 将预测波动率输入 BSM Chooser 公式
  3. 在 Test Set 上评估定价表现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib
import sys, warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120

sys.stdout.reconfigure(encoding='utf-8')

from bsm_model import chooser_price

K = 150.0
T1 = 0.5
T2 = 1.0

# ============================================================
# 1. 加载数据与模型
# ============================================================

print("=" * 55)
print("  Week 6 — 模型训练与评估")
print("=" * 55)

split_data = joblib.load('ML_Dataset_Split.pkl')
train_df = split_data['train']
val_df = split_data['val']
test_df = split_data['test']
features = split_data['features']

target_col = 'Vol_t+1'

# 加载调优后的最佳模型
best_model = joblib.load('Week6_Best_VolModel.pkl')
try:
    grid_rf = joblib.load('Week6_RF_GridSearch.pkl')
    grid_xgb = joblib.load('Week6_XGB_GridSearch.pkl')
    model_type = 'RandomForest' if isinstance(best_model, type(grid_rf.best_estimator_)) else 'XGBoost'
except:
    model_type = 'Unknown'

print(f"\n[1/5] 加载模型: {model_type}")
print(f"  Test samples: {len(test_df)}")

# ============================================================
# 2. 预测波动率
# ============================================================

print("\n[2/5] 波动率预测...")

test_ml = test_df.dropna(subset=[target_col]).copy()
X_test = test_ml[features].fillna(0).values
y_test = test_ml[target_col].values

y_pred_vol = best_model.predict(X_test)

vol_mae = mean_absolute_error(y_test, y_pred_vol)
vol_rmse = np.sqrt(mean_squared_error(y_test, y_pred_vol))
vol_r2 = r2_score(y_test, y_pred_vol)

print(f"  Vol Prediction MAE: {vol_mae:.4f}")
print(f"  Vol Prediction RMSE: {vol_rmse:.4f}")
print(f"  Vol Prediction R2:  {vol_r2:.4f}")

# ============================================================
# 3. Chooser 定价
# ============================================================

print("\n[3/5] Chooser 定价...")

rate_col = 'RiskFreeRate_3M_Decimal' if 'RiskFreeRate_3M_Decimal' in test_ml.columns else 'RiskFreeRate_Decimal'
vol_col = 'Volatility_21D'
div_col = 'Dividend_Yield'

results = []
for i, (idx, row) in enumerate(test_ml.iterrows()):
    S = row['JPM_Close']
    r = row[rate_col]
    sigma_hist = row[vol_col]
    q = row[div_col] if div_col in test_ml.columns else (row['JPM_Dividend'] / S if S > 0 else 0)

    if np.isnan(sigma_hist) or sigma_hist <= 0:
        continue

    # BSM baseline (historical vol)
    price_bsm, _ = chooser_price(S, K, T1, T2, r, sigma_hist, q)

    # ML-vol (predicted vol)
    sigma_ml = max(y_pred_vol[i], 0.05)
    price_ml, breakdown = chooser_price(S, K, T1, T2, r, sigma_ml, q)

    # 内在价值 (参考基准)
    iv = max(S - K, 0)

    results.append({
        'Date': idx,
        'S': S, 'r': r, 'q': q,
        'sigma_hist': sigma_hist, 'sigma_ml': sigma_ml,
        'price_bsm': price_bsm,
        'price_ml': price_ml,
        'call_comp': breakdown['call_component'],
        'put_comp': breakdown['put_component'],
        'intrinsic': iv,
    })

result_df = pd.DataFrame(results)
print(f"  定价完成: {len(result_df)} 个交易日")

# ============================================================
# 4. 评估
# ============================================================

print("\n[4/5] 评估...")

# 以 BSM baseline 为参考基准
result_df['bsm_error'] = np.abs(result_df['price_bsm'] - result_df['intrinsic'])
result_df['ml_error'] = np.abs(result_df['price_ml'] - result_df['intrinsic'])
result_df['improvement'] = result_df['bsm_error'] - result_df['ml_error']

bsm_mae = result_df['bsm_error'].mean()
ml_mae = result_df['ml_error'].mean()
improvement = (bsm_mae - ml_mae) / bsm_mae * 100

print(f"\n  {'Metric':20s} {'BSM Baseline':>12s} {'ML-Vol':>12s} {'Improvement':>12s}")
print(f"  {'-'*56}")
print(f"  {'MAE':20s} {bsm_mae:>12.4f} {ml_mae:>12.4f} {improvement:>11.2f}%")

# 按波动率分层评估
result_df['vol_regime'] = pd.cut(result_df['sigma_hist'],
                                  bins=[0, 0.12, 0.20, 0.35, 1.0],
                                  labels=['Low', 'Normal', 'High', 'Extreme'])

print(f"\n  按波动率分层:")
regime_stats = result_df.groupby('vol_regime', observed=True).agg(
    count=('price_bsm', 'count'),
    bsm_mae=('bsm_error', 'mean'),
    ml_mae=('ml_error', 'mean'),
)
regime_stats['impr%'] = ((regime_stats['bsm_mae'] - regime_stats['ml_mae'])
                         / regime_stats['bsm_mae'] * 100)
print(regime_stats.round(4).to_string())

# ============================================================
# 5. 可视化
# ============================================================

print("\n[5/5] 可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Week 6: ML Model Training & Evaluation', fontsize=14, fontweight='bold')

# (a) Vol prediction: actual vs predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred_vol, alpha=0.3, s=10, c='blue')
lims = [min(y_test.min(), y_pred_vol.min()), max(y_test.max(), y_pred_vol.max())]
ax.plot(lims, lims, 'r--', linewidth=1)
ax.set_xlabel('Actual Volatility')
ax.set_ylabel('Predicted Volatility')
ax.set_title('Vol Prediction (R2={:.3f})'.format(vol_r2))
ax.grid(True, alpha=0.3)

# (b) Pricing error comparison: BSM vs ML
ax = axes[0, 1]
ax.plot(result_df['Date'], result_df['bsm_error'], 'b-', linewidth=0.7, alpha=0.5, label='BSM Error')
ax.plot(result_df['Date'], result_df['ml_error'], 'r-', linewidth=0.7, alpha=0.5, label='ML-Vol Error')
ax.set_ylabel('Absolute Pricing Error ($)')
ax.set_title('Pricing Error: BSM vs ML-Vol')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (c) Improvement histogram
ax = axes[1, 0]
ax.hist(result_df['improvement'], bins=40, color='steelblue', edgecolor='white', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=1)
ax.axvline(result_df['improvement'].mean(), color='green', linestyle='--',
           linewidth=1, label='Mean={:.3f}'.format(result_df['improvement'].mean()))
ax.set_xlabel('Improvement ($)')
ax.set_ylabel('Count')
ax.set_title('Pricing Improvement Distribution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) Error by volatility regime
ax = axes[1, 1]
x = np.arange(len(regime_stats.index))
width = 0.35
ax.bar(x - width/2, regime_stats['bsm_mae'], width, label='BSM', color='lightblue')
ax.bar(x + width/2, regime_stats['ml_mae'], width, label='ML-Vol', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(regime_stats.index)
ax.set_ylabel('MAE')
ax.set_title('Error by Volatility Regime')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Week6_Training_Evaluation.png', dpi=150, bbox_inches='tight')
print("  -> 图保存: Week6_Training_Evaluation.png")

# 保存结果
result_df.to_csv('Week6_Evaluation_Results.csv', index=False)
print("  -> 数据保存: Week6_Evaluation_Results.csv")

print(f"\n{'='*55}")
print(f"  Week 6 训练与评估完成!")
print(f"  BSM MAE: {bsm_mae:.4f}  |  ML-Vol MAE: {ml_mae:.4f}  |  改进: {improvement:.2f}%")
print(f"{'='*55}")
