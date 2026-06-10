# Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning

## Project Structure

```
├── week1 — Data Collection & API Setup
│   ├── API_test.py                         # API connectivity tests (Yahoo/FRED/AlphaVantage)
│   ├── project_data_collection.py          # Automated data collection pipeline
│   └── Data specification document.txt     # Data requirement specification
│
├── week2 — Data Preprocessing & Feature Engineering
│   ├── auto_daily_pipeline.py              # Automated daily ETL pipeline
│   ├── Feature engineering documentation.md # Feature catalog & methodology
│   └── .github/workflows/daily_pipeline.yml # GitHub Actions auto-sync
│
├── week3 — BSM Chooser Option Model
│   ├── bsm_model.py                        # BSM pricing model with Greeks
│   ├── config.json                         # Model parameters
│   └── Week3_Validation.ipynb              # Model validation notebook
│
├── Project_Data_JPM_2018_2024.csv          # Static dataset (Week 1)
├── Dynamic_Cleaned_Dataset.csv             # Dynamic dataset (Week 2, auto-updated)
└── .gitignore
```

## BSM Parameters Ready

| Parameter | Source Column | Status |
|-----------|--------------|--------|
| S₀ (spot) | JPM_Close | ✅ |
| r (risk-free) | RiskFreeRate_3M_Decimal | ✅ |
| q (dividend) | Dividend_Yield | ✅ |
| σ (volatility) | Volatility_21D | ✅ |
| VIX (fear index) | VIX_Close | ✅ |

## GitHub Actions

Automated pipeline runs every trading day at UTC 22:00.
