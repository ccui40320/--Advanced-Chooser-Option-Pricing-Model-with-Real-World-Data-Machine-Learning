# ML Architecture Design — Week 5

## Overview

Two approaches to enhance Chooser Option pricing with machine learning:

```
Approach 1: ML Vol Prediction + BSM      Approach 2: End-to-End ML Pricing
┌────────────┐  ┌────────────┐            ┌─────────────────────────┐
│ Features   │  │ Predicted  │            │ Features → XGBoost/NN  │
│ → ML Model │→ │ σ (vol)    │            │ → Direct Price Output   │
└────────────┘  └─────┬──────┘            └─────────────────────────┘
                      ↓
              ┌────────────┐
              │ BSM Formula│
              │ → Price    │
              └────────────┘
```

## Approach 1: ML Volatility Prediction + BSM

**Idea**: Replace BSM's constant historical volatility with ML-predicted volatility, then feed it into BSM.

**Target**: `Volatility_21D` (or the pricing error residual)

**Models**: XGBoost, Random Forest, Linear Regression

**Why**: BSM's biggest weakness is constant volatility. If ML can predict next-period volatility better, BSM pricing improves.

## Approach 2: End-to-End Supervised Pricing

**Idea**: Skip BSM entirely. Train ML to directly predict Chooser option prices from market features.

**Target**: `Chooser_Price` (or `Pricing_Bias` from Week 4)

**Models**: XGBoost, GBDT, Linear Regression

**Why**: BSM has structural limitations (lognormal assumption, no skew). ML can learn non-linearities from data.

## Data Split (Time-Series)

```
┌─────────────────────────────────────────────────────────┐
│  2018-01  2020-01   2022-01   2024-01   2025-01  2026   │
│  ├──────────┬──────────┬──────────┬──────────┬────────┤
│  │   Train (70%)       │ Val (15%)│  Test (15%)       │
│  │  2018-2022          │2023-2024 │ 2025-2026         │
│  └─────────────────────┴──────────┴────────────────────┘
```

**Critical**: Use `TimeSeriesSplit` or manual cutoff to prevent look-ahead bias. No random shuffle.

## Feature Set (26 → engineered)

| Category | Features |
|----------|----------|
| Price Action | JPM_Close, Daily_Return, Log_Return |
| Volatility | Volatility_21D, VIX_Close, VIX_Return, VIX_JPM_Corr |
| Rates | RiskFreeRate_3M/1Y/10Y_Decimal, Rate_MA5, MA20, Interest_Rate_Momentum |
| Dividend | Dividend_Yield, Dividend_Growth |
| Sentiment | Sentiment_Score |
| Lagged | Vol_21D_lag1/5/21, Return_lag1/5, VIX_lag1, VIX_Close_lag1 |
| Rolling | Vol_max_21D, Vol_min_21D, VIX_MA20 |

## Target Variables

| Approach | Target | Purpose |
|----------|--------|---------|
| 1a | Volatility_21D (t+1) | Predict tomorrow's vol → BSM → price |
| 1b | Pricing_Bias | Predict BSM error → correct price |
| 2 | Chooser_Price | Direct price prediction |

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| MAE | Average dollar error |
| RMSE | Penalize large errors |
| R² | Variance explained |
| MAPE | Percentage error |
| MAE_improvement | % improvement over BSM baseline |

## Deliverables

1. `ml_architecture.md` — This document
2. `ml_feature_pipeline.py` — Feature engineering & dataset splitting
3. `ml_approach1_vol_prediction.py` — ML vol prediction + BSM
4. `ml_approach2_end_to_end.py` — End-to-end ML pricing
