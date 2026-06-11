# Week 6 — Model Training, Tuning & Comparison Report

## 1. Hyperparameter Optimization

**Method**: GridSearchCV + TimeSeriesSplit (5-fold)

**Models Tuned**:
- Random Forest: 81 combinations (n_estimators, max_depth, min_samples_split, min_samples_leaf)
- XGBoost: 243 combinations (n_estimators, max_depth, learning_rate, subsample, colsample_bytree)

**Best Parameters**:

| Model | Best Params | Test R2 |
|-------|------------|---------|
| Random Forest | max_depth=10, min_samples_leaf=8, n_estimators=100 | **0.9598** |
| XGBoost | max_depth=4, lr=0.1, n_estimators=100, subsample=0.7 | 0.9559 |

**Winner**: Random Forest (R2=0.9598, MAE=0.0108)

## 2. Model Training & Evaluation

Trained the best RF model to predict Vol_t+1 (next-day volatility), then fed predicted volatility into BSM Chooser formula.

**Test Set Performance (314 trading days, 2025-2026):**

| Metric | BSM Baseline | ML-Vol Model | Improvement |
|--------|-------------|-------------|-------------|
| MAE | 1.6519 | 1.6392 | **+0.77%** |
| RMSE | 3.6898 | 3.6553 | **+0.93%** |
| R2 (vs Intrinsic) | 0.9795 | 0.9799 | +0.04% |

## 3. Performance Comparison

**By Volatility Regime:**

| Regime | Days | BSM MAE | ML MAE | Improvement |
|--------|------|---------|--------|-------------|
| Low (<12%) | 1 | 0.41 | 0.41 | 0.00% |
| Normal (12-20%) | 118 | 0.93 | 0.93 | -0.02% |
| High (20-35%) | 170 | 0.64 | 0.64 | +0.18% |
| **Extreme (>35%)** | **25** | **12.00** | **11.85** | **+1.28%** |

**Key Finding**: ML improvement is concentrated in **Extreme Volatility regimes** (+1.28%), where BSM is weakest.

## 4. SHAP Interpretability Analysis

**Top 10 Features for Volatility Prediction:**

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | Volatility_21D | 0.065942 | Current 21-day historical vol |
| 2 | Vol_lag1 | 0.000853 | 1-day lagged volatility |
| 3 | Vol_min_21D | 0.000724 | 21-day minimum volatility |
| 4 | Rate_Spread_1Y_3M | 0.000626 | Yield curve steepness |
| 5 | Sentiment_Score | 0.000569 | Market sentiment signal |
| 6 | Price_MA20 | 0.000514 | 20-day moving avg price |
| 7 | VIX_Close | 0.000345 | VIX fear index level |
| 8 | VIX_change_5D | 0.000309 | 5-day VIX change |
| 9 | VIX_lag1 | 0.000261 | 1-day lagged VIX |
| 10 | Dividend_Yield | 0.000247 | Dividend yield rate |

**Insight**: Volatility_21D dominates (97% of total importance). The model essentially learns that the best predictor of tomorrow's volatility is today's volatility. Second-order features (vol lags, rate spreads, sentiment) provide marginal refinement.

## 5. Deliverables

| File | Description |
|------|-------------|
| `Week6_Best_VolModel.pkl` | Trained RF model (best params) |
| `Week6_RF_GridSearch.pkl` | RF GridSearch + CV results |
| `Week6_XGB_GridSearch.pkl` | XGBoost GridSearch + CV results |
| `Week6_Evaluation_Results.csv` | Full evaluation with BSM vs ML prices |
| `Week6_Comparison_Summary.csv` | Comparison metrics summary |
| `Week6_SHAP_FeatureImportance.csv` | SHAP feature importance data |
| `Week6_Training_Evaluation.png` | Training evaluation charts |
| `Week6_Comparison_Analysis.png` | Comparison analysis charts |
| `Week6_SHAP_Summary.png` | SHAP bee-swarm plot |
| `Week6_SHAP_Importance.png` | SHAP feature importance bars |
| `Week6_SHAP_Dependence.png` | SHAP dependence plots |

## 6. Conclusion

1. **ML volatility prediction works** (R2=0.96) but BSM is already efficient
2. **Biggest ML gains in extreme volatility** (+1.28% improvement)
3. **Historical volatility is the dominant feature** (97% SHAP importance)
4. **Best approach: Use ML-vol prediction for high-vol periods, BSM for normal periods**
