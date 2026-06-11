"""
Week 6 — 性能对比分析 (Performance Comparison)
===============================================
综合对比 BSM Baseline 与 ML-enhanced 模型的定价表现。
对比维度: MAE, RMSE, R2, 改进率
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib
import sys, warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 1. 加载评估结果
# ============================================================

print("=" * 55)
print("  Week 6 — 性能对比分析")
print("=" * 55)

# 读取 Week 6 评估结果
eval_df = pd.read_csv('Week6_Evaluation_Results.csv', parse_dates=['Date'], index_col='Date')

# 读取 Week 4 真实期权数据对比
try:
    opt_df = pd.read_csv('Week4_Final_Error_Matrix.csv')
    has_opt_data = True
except:
    has_opt_data = False

print(f"\n[1/4] 评估数据: {len(eval_df)} 行")

# ============================================================
# 2. 多维度对比矩阵
# ============================================================

print("\n[2/4] 多维度对比矩阵...")

# (A) 全样本对比
bsm_mae = eval_df['bsm_error'].mean()
ml_mae = eval_df['ml_error'].mean()
bsm_rmse = np.sqrt((eval_df['bsm_error'] ** 2).mean())
ml_rmse = np.sqrt((eval_df['ml_error'] ** 2).mean())

# 计算 R2 (以内在价值为基线)
bsm_r2 = r2_score(eval_df['intrinsic'], eval_df['price_bsm'])
ml_r2 = r2_score(eval_df['intrinsic'], eval_df['price_ml'])

improvement_mae = (bsm_mae - ml_mae) / bsm_mae * 100

print(f"\n  {'='*55}")
print(f"  全样本对比 (Test Set: {len(eval_df)} 天)")
print(f"  {'='*55}")
print(f"  {'Metric':<20s} {'BSM Baseline':>12s} {'ML-Vol Model':>12s} {'改进':>10s}")
print(f"  {'-'*54}")
print(f"  {'MAE':<20s} {bsm_mae:>12.4f} {ml_mae:>12.4f} {improvement_mae:>9.2f}%")
print(f"  {'RMSE':<20s} {bsm_rmse:>12.4f} {ml_rmse:>12.4f} {(bsm_rmse-ml_rmse)/bsm_rmse*100:>9.2f}%")
print(f"  {'R2 (vs Intrinsic)':<20s} {bsm_r2:>12.4f} {ml_r2:>12.4f} {'-':>10s}")

# (B) 按年度对比
eval_df['year'] = eval_df.index.year
yearly = eval_df.groupby('year').agg(
    days=('bsm_error', 'count'),
    bsm_mae=('bsm_error', 'mean'),
    ml_mae=('ml_error', 'mean'),
)
yearly['impr%'] = (yearly['bsm_mae'] - yearly['ml_mae']) / yearly['bsm_mae'] * 100

print(f"\n  按年度分层对比:")
print(f"  {'Year':>6s} {'Days':>6s} {'BSM MAE':>10s} {'ML MAE':>10s} {'Impr%':>8s}")
print(f"  {'-'*40}")
for yr, row in yearly.iterrows():
    print(f"  {int(yr):>6d} {int(row['days']):>6d} {row['bsm_mae']:>10.4f} {row['ml_mae']:>10.4f} {row['impr%']:>7.2f}%")

# (C) 波动率分层
vol_bins = [0, 0.12, 0.20, 0.35, 1.0]
vol_labels = ['Low(<12%)', 'Normal(12-20%)', 'High(20-35%)', 'Extreme(>35%)']
eval_df['vol_bin'] = pd.cut(eval_df['sigma_hist'], bins=vol_bins, labels=vol_labels)

print(f"\n  按波动率分层对比:")
vol_stats = eval_df.groupby('vol_bin', observed=True).agg(
    days=('bsm_error', 'count'),
    bsm_mae=('bsm_error', 'mean'),
    ml_mae=('ml_error', 'mean'),
)
vol_stats['impr%'] = (vol_stats['bsm_mae'] - vol_stats['ml_mae']) / vol_stats['bsm_mae'] * 100
print(vol_stats.round(4).to_string())

# ============================================================
# 3. 与 Week 4 期权市场数据对比
# ============================================================

print(f"\n\n[3/4] 与真实期权市场数据对比 (Week 4)...")

if has_opt_data:
    opt_mae = mean_absolute_error(opt_df['Market_Mid'], opt_df['Model_Price'])
    print(f"  BSM vs Market (2026-03-31): MAE={opt_mae:.2f}")
else:
    print("  无期权市场数据, 跳过")

# ============================================================
# 4. 可视化: 对比图
# ============================================================

print(f"\n[4/4] 生成对比可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Week 6: ML vs BSM Performance Comparison', fontsize=14, fontweight='bold')

# (a) MAE 年度趋势
ax = axes[0, 0]
ax.plot(yearly.index, yearly['bsm_mae'], 'bo-', linewidth=1.5, markersize=5, label='BSM')
ax.plot(yearly.index, yearly['ml_mae'], 'rs-', linewidth=1.5, markersize=5, label='ML-Vol')
ax.set_xlabel('Year')
ax.set_ylabel('MAE')
ax.set_title('MAE by Year')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) 改进率年度趋势
ax = axes[0, 1]
ax.bar(yearly.index, yearly['impr%'], color='steelblue', width=0.6)
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Year')
ax.set_ylabel('Improvement (%)')
ax.set_title('ML Improvement over BSM by Year')
ax.grid(True, alpha=0.3, axis='y')

# (c) 波动率分层对比 - 柱状图
ax = axes[1, 0]
x = np.arange(len(vol_stats.index))
width = 0.3
ax.bar(x - width/2, vol_stats['bsm_mae'], width, label='BSM', color='lightblue', edgecolor='gray')
ax.bar(x + width/2, vol_stats['ml_mae'], width, label='ML-Vol', color='coral', edgecolor='gray')
ax.set_xticks(x)
ax.set_xticklabels(vol_stats.index, rotation=30, fontsize=8)
ax.set_ylabel('MAE')
ax.set_title('Pricing Error by Volatility Regime')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# (d) 模型对比总结表 (matplotlib table)
ax = axes[1, 1]
ax.axis('off')
table_data = [
    ['Metric', 'BSM', 'ML-Vol', 'Impr'],
    ['MAE', '{:.4f}'.format(bsm_mae), '{:.4f}'.format(ml_mae), '{:.2f}%'.format(improvement_mae)],
    ['RMSE', '{:.4f}'.format(bsm_rmse), '{:.4f}'.format(ml_rmse), '{:.2f}%'.format((bsm_rmse-ml_rmse)/bsm_rmse*100)],
    ['R2', '{:.4f}'.format(bsm_r2), '{:.4f}'.format(ml_r2), ''],
    ['Best Regime', '-', vol_stats['impr%'].idxmax(), ''],
    ['Worst Regime', '-', vol_stats['impr%'].idxmin(), ''],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.15, 0.2, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif i % 2 == 0:
        cell.set_facecolor('#f0f0f0')
ax.set_title('Performance Summary', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('Week6_Comparison_Analysis.png', dpi=150, bbox_inches='tight')
print("  -> 图保存: Week6_Comparison_Analysis.png")

# 保存对比报告
comparison_summary = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R2'],
    'BSM': [bsm_mae, bsm_rmse, bsm_r2],
    'ML_Vol': [ml_mae, ml_rmse, ml_r2],
    'Improvement_%': [improvement_mae, (bsm_rmse-ml_rmse)/bsm_rmse*100, (ml_r2-bsm_r2)/abs(bsm_r2)*100 if bsm_r2 != 0 else 0],
})
comparison_summary.to_csv('Week6_Comparison_Summary.csv', index=False)
print("  -> 数据保存: Week6_Comparison_Summary.csv")

print(f"\n{'='*55}")
print(f"  对比分析完成!")
print(f"{'='*55}")
