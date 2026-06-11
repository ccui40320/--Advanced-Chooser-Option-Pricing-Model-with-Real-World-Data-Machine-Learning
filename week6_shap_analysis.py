"""
Week 6 — SHAP 可解释性分析 (Interpretability Analysis)
=======================================================
使用 SHAP 解释波动率预测模型的特征重要性。
输出: 摘要图, 特征重要性条形图, 依赖图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib
import sys, warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 1. 加载模型和数据
# ============================================================

print("=" * 55)
print("  Week 6 — SHAP 可解释性分析")
print("=" * 55)

split_data = joblib.load('ML_Dataset_Split.pkl')
train_df = split_data['train']
val_df = split_data['val']
test_df = split_data['test']
features = split_data['features']

target_col = 'Vol_t+1'

train_full = pd.concat([train_df, val_df])
train_ml = train_full.dropna(subset=[target_col]).copy()
test_ml = test_df.dropna(subset=[target_col]).copy()

X_train = train_ml[features].fillna(0)
y_train = train_ml[target_col]
X_test = test_ml[features].fillna(0)
y_test = test_ml[target_col]

# 加载最佳模型
try:
    best_model = joblib.load('Week6_Best_VolModel.pkl')
    model_type = type(best_model).__name__
    print(f"\n[1/4] 模型: {model_type}")
except:
    # 如果没有调优模型, 用 RF 默认
    from sklearn.ensemble import RandomForestRegressor
    best_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    best_model.fit(X_train.values, y_train.values)
    model_type = 'RandomForest (fallback)'
    print(f"\n[1/4] 模型: {model_type} (fallback)")

print(f"  特征数: {len(features)}")
print(f"  测试样本: {len(X_test)}")

# ============================================================
# 2. 计算 SHAP 值
# ============================================================

print("\n[2/4] 计算 SHAP 值 (可能较慢)...")

# XGBoost 可以使用 TreeExplainer (快), 其他用 KernelExplainer
if 'XGB' in model_type or 'Forest' in model_type or 'Tree' in model_type:
    import shap
    explainer = shap.TreeExplainer(best_model)
    # 使用 500 个样本计算 SHAP (平衡速度与精度)
    shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(shap_sample.values)
    print(f"  TreeExplainer: {shap_values.shape}")
else:
    import shap
    X_train_sample = X_train.sample(min(200, len(X_train)), random_state=42)
    explainer = shap.KernelExplainer(best_model.predict, X_train_sample.values)
    shap_sample = X_test.sample(min(200, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(shap_sample.values)
    print(f"  KernelExplainer: {shap_values.shape}")

# ============================================================
# 3. SHAP 可视化
# ============================================================

print("\n[3/4] SHAP 可视化...")

# 特征重要性 DataFrame
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print(f"\n  Top 10 特征 (SHAP importance):")
print(f"  {'Rank':>4s} {'Feature':30s} {'Importance':>12s}")
print(f"  {'-'*46}")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"  {i+1:>4d} {row['feature']:30s} {row['importance']:>12.6f}")

# 图1: SHAP Summary Plot (蜜蜂图)
fig1, ax1 = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values, shap_sample.values,
    feature_names=features,
    show=False, max_display=20
)
plt.tight_layout()
plt.savefig('Week6_SHAP_Summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 图保存: Week6_SHAP_Summary.png (蜜蜂图)")

# 图2: SHAP 特征重要性条形图
fig2, ax2 = plt.subplots(figsize=(10, 7))
top15 = feature_importance.head(15)
ax2.barh(range(len(top15)), top15['importance'].values, color='steelblue')
ax2.set_yticks(range(len(top15)))
ax2.set_yticklabels(top15['feature'].values)
ax2.set_xlabel('mean(|SHAP value|)')
ax2.set_title('SHAP Feature Importance (Top 15)')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('Week6_SHAP_Importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 图保存: Week6_SHAP_Importance.png (条形图)")

# 图3: Top 3 特征的 SHAP 依赖图
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
fig3.suptitle('SHAP Dependence Plots (Top 3 Features)', fontsize=13)

top3_features = feature_importance.head(3)['feature'].values
for i, feat in enumerate(top3_features):
    if feat in shap_sample.columns:
        feat_idx = features.index(feat)
        shap.dependence_plot(
            feat_idx, shap_values, shap_sample.values,
            feature_names=features,
            ax=axes3[i], show=False
        )
        axes3[i].set_title(feat, fontsize=9)
        axes3[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Week6_SHAP_Dependence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 图保存: Week6_SHAP_Dependence.png (依赖图)")

# ============================================================
# 4. 保存 SHAP 数据
# ============================================================

print("\n[4/4] 保存分析结果...")

feature_importance.to_csv('Week6_SHAP_FeatureImportance.csv', index=False)
print("  -> 数据保存: Week6_SHAP_FeatureImportance.csv")

print(f"\nTop 5 关键特征:")
for i in range(min(5, len(feature_importance))):
    feat = feature_importance.iloc[i]
    print(f"  {i+1}. {feat['feature']:30s} (importance={feat['importance']:.6f})")

print(f"\n{'='*55}")
print(f"  SHAP 可解释性分析完成!")
print(f"{'='*55}")
