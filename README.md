# 🌾 AgriPrice India — Agricultural Price Forecasting & Decision-Support Dashboard

> A rigorous, data-driven tool for forecasting Indian agricultural commodity prices and generating plain-language market intervention signals.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![Data Source](https://img.shields.io/badge/Data-Agmarknet%20%7C%20data.gov.in-green)](https://data.gov.in/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/jppankajs/-AgriPriceIndia)

**🔗 GitHub:** [github.com/jppankajs/-AgriPriceIndia](https://github.com/jppankajs/-AgriPriceIndia)

---

## 📌 Project Overview

**AgriPrice India** is a five-phase analytics project that applies time-series forecasting to government-sourced Indian agricultural market data (Agmarknet via Kaggle). The result is an interactive Streamlit dashboard that produces **Sell / Hold / Monitor** signals for five key commodities, helping farmers and traders make data-informed decisions.

**Commodities Covered:** Onion · Potato · Wheat · Rice · Tomato

---

## 🏗️ Project Architecture

```
AgriPriceIndia/
├── app/                        # Streamlit dashboard
│   ├── phase5_dashboard.py     # Main dashboard entry point
│   ├── live_data.py            # Data loading & refresh logic
│   └── styles.py               # CSS / glassmorphism styling
│
├── src/                        # Analysis pipeline scripts
│   ├── phase0_inspect.py       # Data inspection & profiling
│   ├── phase0_detail.py        # Detailed data audit
│   ├── phase0_dates.py         # Date range validation
│   ├── pre_phase1_date_check.py # Pre-cleaning date audit
│   ├── phase1_clean.py         # Data cleaning & standardisation
│   ├── phase2_eda.py           # Exploratory data analysis
│   ├── phase3_model.py         # Model training & evaluation
│   ├── phase4_decision.py      # Decision-support signal logic
│   ├── check_onion_ceiling.py  # Onion CI bound validation
│   ├── check_potato_dates.py   # Potato date coverage audit
│   └── check_potato_forecast.py # Potato forecast validation
│
├── data/
│   ├── raw/                    # ⚠️ Raw Kaggle data (not tracked by git)
│   └── processed/
│       ├── agri_prices_clean.csv         # Cleaned master dataset (gitignored — large)
│       └── agri_prices_daily_national.csv # National daily averages
│
├── models/                     # Serialised trained models (.pkl)
│   ├── best_models.json        # Model selection summary
│   ├── ets_rice.pkl
│   ├── ets_tomato.pkl
│   ├── prophet_onion.pkl
│   ├── prophet_potato.pkl
│   ├── prophet_wheat.pkl
│   ├── sarima_onion.pkl
│   ├── sarima_wheat.pkl
│   └── sarima_potato.pkl       # ⚠️ ~30 MB — gitignored
│
├── reports/
│   ├── figures/                # All EDA & forecast charts (PNG)
│   ├── phase1_output.txt       # Phase 1 cleaning audit log
│   ├── phase3_model_results.csv # Model comparison metrics
│   └── phase4_decision_table.csv # Final decision signals
│
├── docs/
│   ├── AgriPriceIndia_Master_Build_Prompt.md
│   └── AgriPriceIndia_Phase0_Review_and_Phase1_Amendment.md
│
├── requirements.txt
└── .gitignore
```

---

## 🚀 Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Environment setup, Kaggle auth, dataset acquisition & profiling | ✅ Complete |
| **Phase 1** | Data cleaning — outlier removal, standardisation, national aggregation | ✅ Complete |
| **Phase 2** | EDA — seasonality, decomposition, correlation, state coverage | ✅ Complete |
| **Phase 3** | Model training — SARIMA / ETS / Prophet per commodity, RMSE comparison | ✅ Complete |
| **Phase 4** | Decision-support layer — Sell / Hold / Monitor signal logic | ✅ Complete |
| **Phase 5** | Streamlit dashboard — interactive UI with price ticker & forecasts | ✅ Complete |

---

## 📊 Model Summary

| Commodity | Best Model | Tier | Notes |
|-----------|-----------|------|-------|
| **Onion** | Prophet | 🟡 Directional | High seasonal volatility; point estimates suppressed |
| **Potato** | Prophet | 🟡 Monitor-only | Structural break detected; forecast extrapolates downtrend |
| **Wheat** | SARIMA | 🟢 Full | Low RMSE; stable seasonal pattern |
| **Rice** | ETS | 🟢 Full | Smooth trend; reliable short-horizon forecast |
| **Tomato** | ETS | 🟡 Directional | Extreme volatility; CI intervals very wide |

---

## 🖥️ Running the Dashboard

### 1. Clone the repo & set up environment
```bash
git clone https://github.com/jppankajs/-AgriPriceIndia.git
cd AgriPriceIndia
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Acquire the data (one-time setup)
The raw dataset is sourced from [Kaggle — Agmarknet](https://www.kaggle.com/). Configure your Kaggle API key at `~/.kaggle/kaggle.json`, then run:
```bash
python src/phase0_inspect.py
python src/phase1_clean.py
```

### 3. Launch the dashboard
```bash
streamlit run app/phase5_dashboard.py
```

---

## ⚠️ Data & Model Transparency

- **Data source**: Agmarknet (Government of India) via Kaggle — static dataset, not a live feed.
- **Onion & Potato** signals are **directional only** (no point estimates) due to structural breaks and high forecast uncertainty.
- All confidence intervals are empirical bootstrap CIs, not parametric.
- The dashboard clearly labels the dataset's temporal coverage and model tier for each commodity.

---

## 📁 Large Files (Not Tracked in Git)

The following files exceed GitHub's recommended size limit and are excluded via `.gitignore`:

| File | Size | Reason |
|------|------|--------|
| `data/raw/` | ~500 MB | Raw Kaggle download |
| `data/processed/agri_prices_clean.csv` | ~62 MB | Full cleaned dataset |
| `models/sarima_potato.pkl` | ~29 MB | SARIMA model for Potato |
| `models/sarima_onion.pkl` | ~3.3 MB | SARIMA model for Onion |
| `models/sarima_wheat.pkl` | ~4.8 MB | SARIMA model for Wheat |

Regenerate these by running the pipeline scripts in order: Phase 0 → 1 → 3.

---

## 🛠️ Tech Stack

- **Python 3.10+** | pandas · numpy · matplotlib · seaborn · plotly
- **Forecasting**: statsmodels (SARIMA/ETS) · Prophet
- **Dashboard**: Streamlit
- **Data**: Agmarknet via Kaggle API

---

## 🚀 Demo

> **Note:** This dashboard runs locally via Streamlit. There is no hosted live demo at this time.
> To see it in action, clone the repo and follow the [Running the Dashboard](#-running-the-dashboard) instructions above.

To deploy on Streamlit Cloud yourself:
1. Fork/clone this repo
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Set main file: `app/phase5_dashboard.py`
4. Ensure `data/processed/agri_prices_daily_national.csv` and model `.pkl` files are present

---

## 👤 Author

**Pankaj** — Data Analytics Internship Project  
GitHub: [@jppankajs](https://github.com/jppankajs)  
Repository: [AgriPriceIndia](https://github.com/jppankajs/-AgriPriceIndia)

---

*This project is part of a rigorous internship analytics portfolio. All forecasts are decision-support tools, not guarantees of future prices.*
