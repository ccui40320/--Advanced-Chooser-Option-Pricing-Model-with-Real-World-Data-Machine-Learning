# Feature Engineering & Data Pipeline Documentation

## 1. Data Scope

| Item | Detail |
|------|--------|
| **Asset** | JPMorgan Chase (JPM) — 2018-01-01 to Today |
| **Macro** | US Treasury 3M/1Y/10Y rates (FRED), CBOE VIX (Yahoo Finance) |
| **Sentiment** | Hybrid: Market proxy (VIX + returns) + NewsAPI NLP (recent 28d) |
| **Row count** | ~2,100 trading days, 26 feature columns |

---

## 2. Feature Catalog

### 2.1 Traditional Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `Daily_Return` | ln(P_t / P_{t-1}) using Adj Close | Stationary return series for BSM |
| `Volatility_21D` | σ(return) × √252, 21d window | Rolling annualized volatility σ |
| `Dividend_Yield` | Σ(div, 252d) / Close | Annualized dividend rate q for BSM |
| `Dividend_Growth` | YoY change in rolling dividend sum | Dividend trend signal |

### 2.2 Advanced Features

| Feature | Construction | Purpose |
|---------|-------------|---------|
| `VIX_Return` | ΔVIX / VIX_{t-1} | Daily change in fear index |
| `VIX_JPM_Corr` | 21d rolling correlation of Daily_Return & VIX_Return | Regime change detection |
| `RiskFreeRate_Xy_Decimal` | FRED rate / 100 | BSM-compatible decimal r |
| `Rate_MA5 / MA20` | 5d / 20d moving avg of risk-free rate | Short/medium rate trend |
| `Interest_Rate_Momentum` | MA5 - MA20 | Monetary policy cycle signal |

### 2.3 Sentiment Features

| Feature | Construction | Purpose |
|---------|-------------|---------|
| `Sentiment_Score` | Hybrid: VIX-inverse (50%) + Return signal (50%), overlaid with NewsAPI NLP (60% weight when available) | Market sentiment [0,1] |
| `Article_Count` | Binary: 1 on days with NewsAPI articles | NLP coverage indicator |

---

## 3. Data Cleaning Pipeline

```
Raw Data → Weekend Filter → Time Interpolation → IQR Capping → Feature Engineering → NaN Drop
```

1. **Weekend Filter**: Keep only weekdays (dayofweek < 5)
2. **Time Interpolation**: Linear time-based for prices/rates/VIX
3. **IQR 3.0× Capping**: Volume and VIX extremes smoothed
4. **NaN Drop**: Remove rows without Daily_Return (first trading day)
5. **No price capping**: Preserve raw price volatility for risk modeling

---

## 4. Sentiment Hybrid Model

```
Historical Data (2018 - Today):
    Sentiment = 0.5 × (1 - VIX_norm) + 0.5 × norm(21d_return)

Recent 28 Days (when NewsAPI available):
    Sentiment = 0.6 × NewsAPI_NLP + 0.4 × Historical_Proxy
```

Rationale: NLP coverage is sparse (only recent 28 days); the historical proxy fills 2018-present seamlessly. Real NewsAPI scores improve recency accuracy.

---

## 5. GitHub Actions Automation

- **Schedule**: Every trading day at UTC 22:00 (Beijing 06:00)
- **Workflow**: auto_daily_pipeline.py → Dynamic_Cleaned_Dataset.csv → commit to repo
- **Secrets needed**: `NEWS_API_KEY`, `FRED_API_KEY`
- **Trigger**: Manual via GitHub UI (workflow_dispatch) or auto-schedule

**Repo**: https://github.com/linkehan/-Advanced-Chooser-Option-Pricing-Model-with-Real-World-Data-Machine-Learning

---

## 6. BSM Parameter Mapping

| BSM Param | Dataset Column | Status |
|-----------|---------------|--------|
| S₀ (spot) | JPM_Close | Ready |
| K (strike) | Config ($150) | Week 3 |
| T (expiry) | Config (1yr) | Week 3 |
| r (risk-free) | RiskFreeRate_3M_Decimal (0.0144) | Ready |
| q (dividend) | Dividend_Yield (0.02–0.04) | Ready |
| σ (volatility) | Volatility_21D (0.08–1.35) | Ready |
| t₁ (choice) | Config | Week 3 |

---

## 7. Change Log

| Date | Change |
|------|--------|
| 2026-06-10 | Added RiskFreeRate_1Y/10Y + Decimal variants |
| 2026-06-10 | Added Volatility_21D (replaced 20D for standard alignment) |
| 2026-06-10 | Added Dividend_Yield (annualized, BSM-ready) |
| 2026-06-10 | Added Article_Count column |
| 2026-06-10 | Improved sentiment: VIX-inverse + return proxy + NLP hybrid |
| 2026-06-10 | Updated GitHub Actions: auto-commit, Python 3.11, gdeltdoc |
