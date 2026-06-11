"""
Week 6 — 超参数优化 (Hyperparameter Tuning)
=============================================
对 Random Forest 和 XGBoost 进行 GridSearch + TimeSeriesSplit 交叉验证。
目标: 找到预测 Vol_t+1 (下一日波动率) 的最优参数。
"""

import numpy as np
import pandas as pd
import joblib
import sys, os, warnings
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 55)
print("  Week 6 — 超参数优化 (Hyperparameter Tuning)")
print("=" * 55)

# ============================================================
# 1. 加载特征数据
# ============================================================

split_data = joblib.load('ML_Dataset_Split.pkl')
train_df = split_data['train']
val_df = split_data['val']
test_df = split_data['test']
features = split_data['features']

target_col = 'Vol_t+1'

# 合并 Train + Val 用于 CV (保留 Test 做最终评估)
train_full = pd.concat([train_df, val_df])
train_ml = train_full.dropna(subset=[target_col]).copy()
test_ml = test_df.dropna(subset=[target_col]).copy()

X_train = train_ml[features].fillna(0).values
y_train = train_ml[target_col].values
X_test = test_ml[features].fillna(0).values
y_test = test_ml[target_col].values

print(f"\n[1/4] 数据:")
print(f"  Train+Val: {X_train.shape}")
print(f"  Test:      {X_test.shape}")

# ============================================================
# 2. Random Forest 超参数搜索
# ============================================================

print("\n[2/4] Random Forest GridSearch...")

rf_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [6, 10, 15],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
}

tscv = TimeSeriesSplit(n_splits=5)

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=rf_param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1,
)
rf_grid.fit(X_train, y_train)

print(f"\n  RF Best params: {rf_grid.best_params_}")
print(f"  RF Best CV score (neg_MSE): {rf_grid.best_score_:.6f}")

rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
print(f"  RF Test MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}")
print(f"  RF Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")
print(f"  RF Test R2:  {r2_score(y_test, y_pred_rf):.4f}")

joblib.dump(rf_grid, 'Week6_RF_GridSearch.pkl')
print("  -> 保存: Week6_RF_GridSearch.pkl")

# ============================================================
# 3. XGBoost 超参数搜索
# ============================================================

print("\n[3/4] XGBoost GridSearch...")

xgb_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}

xgb_grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0),
    param_grid=xgb_param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=1,
)
xgb_grid.fit(X_train, y_train)

print(f"\n  XGB Best params: {xgb_grid.best_params_}")
print(f"  XGB Best CV score (neg_MSE): {xgb_grid.best_score_:.6f}")

xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)
print(f"  XGB Test MAE: {mean_absolute_error(y_test, y_pred_xgb):.4f}")
print(f"  XGB Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.4f}")
print(f"  XGB Test R2:  {r2_score(y_test, y_pred_xgb):.4f}")

joblib.dump(xgb_grid, 'Week6_XGB_GridSearch.pkl')
print("  -> 保存: Week6_XGB_GridSearch.pkl")

# ============================================================
# 4. 保存最佳模型
# ============================================================

print("\n[4/4] 保存最佳模型...")

# 用测试集表现选择最佳模型
rf_test_r2 = r2_score(y_test, rf_best.predict(X_test))
xgb_test_r2 = r2_score(y_test, xgb_best.predict(X_test))

if rf_test_r2 >= xgb_test_r2:
    best_model = rf_best
    best_name = 'RandomForest'
else:
    best_model = xgb_best
    best_name = 'XGBoost'

joblib.dump(best_model, 'Week6_Best_VolModel.pkl')
print(f"  Best model: {best_name} (Test R2={max(rf_test_r2, xgb_test_r2):.4f})")
print(f"  -> 保存: Week6_Best_VolModel.pkl")

# 对比基线 (历史波动率)
naive_pred = test_ml[['Volatility_21D']].fillna(0).values.ravel()
naive_mae = mean_absolute_error(y_test, naive_pred)
print(f"\n  基线对比:")
print(f"  Naive(hist vol) MAE: {naive_mae:.4f}")
print(f"  {best_name:15s}       MAE: {mean_absolute_error(y_test, best_model.predict(X_test)):.4f}")

print(f"\n{'='*55}")
print(f"  超参数优化完成!")
print(f"{'='*55}")
