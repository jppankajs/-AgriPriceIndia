# AgriPrice India -- Master Build Prompt
### Codec Technologies Internship -- Project 2 (Sales Forecasting brief, applied to agricultural commodities)

## Read this before pasting anything into Antigravity

This is not a "build everything, walk away, come back to a finished product" prompt. It is a phased specification with mandatory checkpoints. That's deliberate, not something missing -- this exact process (small step, verify, correct, next step) is what caught the disk-full failure, the corrupted `.env` file, and the gitignored model files in CreditBridge. One unattended pass through 7 phases of data we haven't inspected yet would very likely repeat the same categories of mistakes, just later and harder to trace back.

**How to use it:** paste this whole document into Antigravity, but add this line at the top of your message first:

> "Execute only Phase [N]. Stop completely afterward and show me the Report back section before I authorize Phase [N+1]." Repeat that before every phase.

---

## Project goal

Build a transparent, rigorous agricultural price forecasting system for 5 Indian crops (Onion, Potato, Tomato, Wheat, Rice) using real mandi (market) price data. The system should:
- Forecast 8-12 weeks of prices using time-series models (SARIMA, Prophet, Holt-Winters)
- Generate plain-language Sell / Hold / Monitor signals for each crop
- Deploy as a Streamlit dashboard with honest reporting of model performance and limitations

## Architecture constraints

- Flat pandas-based architecture (no PostgreSQL)
- All data in CSV format in the `data/` folder
- Models serialized with joblib in `models/`
- Source code in `src/`, Streamlit app in `app/`
- Reports and figures in `reports/`

## Folder structure

```
AgriPriceIndia/
  data/
    raw/           # Kaggle download (gitignored)
    processed/     # Cleaned CSVs (tracked)
  src/             # Phase scripts
  notebooks/       # EDA notebooks
  models/          # Serialized models (tracked)
  app/             # Streamlit dashboard
  reports/
    figures/       # Saved plots
  docs/            # Project documentation
  venv/            # Virtual environment (gitignored)
```

---

## Phase 0 -- Environment and Data Acquisition

1. Verify disk space (D: drive, not C:)
2. Check for kaggle.json authentication
3. Create folder structure
4. Install dependencies (pin kaggle==1.6.14, attempt Prophet separately)
5. Download dataset from Kaggle: `arjunyadav99/indian-agricultural-mandi-prices-20232025`
6. Extract and inspect the CSV: shape, columns, dtypes, first 5 rows
7. Identify date columns, commodity columns, price columns
8. Check for missing values, duplicates, zero/negative prices
9. Report per-crop date ranges and row counts

**Report back:** file names found, columns + dtypes, row count per crop, date ranges, target crop label matches, any data quality issues.

**Status: COMPLETE** (see Phase 0 report)

---

## Phase 1 -- Data Cleaning and Consolidation

### Original specification:
1. Parse `Price Date` using `pd.to_datetime()` with explicit `format='%m/%d/%Y'`
2. Standardize column names (title case, strip whitespace)
3. Handle data quality issues found in Phase 0:
   - Zero prices (Min_Price=0, Max_Price=0, Modal_Price=0)
   - Modal price outside [Min, Max] bounds
   - Extreme outlier prices
4. Produce a single consolidated clean CSV in `data/processed/`
5. Verify row counts before and after cleaning

### Amended by Phase 0 Review (see AgriPriceIndia_Phase0_Review_and_Phase1_Amendment.md):
- **Amendment 1:** Tiered crop treatment -- all 5 crops stay in scope but Rice/Tomato get trend-only models (no seasonal claims)
- **Amendment 2:** Use FAQ-grade prices as primary series; keep other grades in granular table only
- **Amendment 3:** Two output tables: `agri_prices_clean.csv` (full granular) + `agri_prices_daily_national.csv` (one row per crop per date, FAQ-grade national average)
- **Amendment 4:** Concrete outlier rules -- Max_Price=0 treated as missing; Modal_Price > 5x 95th pctile flagged but not dropped from granular table

**Report back:** row count before/after cleaning per crop, outlier counts, bounds-violation counts, final file sizes of both output tables, sample of the daily national table.

---

## Phase 2 -- Exploratory Data Analysis (EDA)

1. Time-series plots of daily national average Modal_Price per crop
2. Seasonal decomposition (additive) for Onion and Potato
3. Distribution plots (histograms/boxplots) of Modal_Price per crop
4. Correlation analysis between crops
5. State/market coverage heatmap
6. Save all figures to `reports/figures/`

**Report back:** key visual patterns, seasonality observations, any remaining data issues.

---

## Phase 3 -- Model Training

For Onion, Potato (full pipeline):
1. Train/test split (80/20 chronological)
2. SARIMA with auto_arima parameter selection
3. Prophet with default + tuned configurations
4. Holt-Winters exponential smoothing
5. Compare all three using MAE, RMSE, MAPE on test set
6. Select best model per crop

For Wheat (limited pipeline):
1. Attempt SARIMA/Prophet but acknowledge incomplete seasonal cycle
2. Document honestly what can and cannot be validated

For Tomato, Rice (trend-only):
1. Moving average or basic exponential smoothing (no seasonal term)
2. Do NOT generate 8-12 week forecast (exceeds total history)

**Report back:** model comparison table, selected model per crop, actual vs. predicted plots, honest assessment of Wheat/Tomato/Rice limitations.

---

## Phase 4 -- Decision-Support Layer

1. Generate forecasts using best model per crop
2. Calculate confidence intervals
3. Produce Sell / Hold / Monitor signal based on forecast trend and confidence
4. Rice and Tomato automatically get "Monitor -- insufficient history" signal
5. Format output as decision table

**Report back:** decision table for all 5 crops with rationale.

---

## Phase 5 -- Dashboard (Streamlit)

1. Build interactive Streamlit app in `app/`
2. Dashboard pages: Overview, Per-Crop Forecast, Model Performance, About
3. About section must honestly document data limitations (especially Rice/Tomato)
4. Interactive date range selectors, crop filters

**Report back:** screenshots of all dashboard pages, any deployment blockers.

---

## Phase 6 -- Deployment and Documentation

1. Deploy to Streamlit Cloud
2. Update README with live URL, methodology, limitations
3. Final git commit and push

**Report back:** live URL, final project structure, README contents.

---

## Phase 7 -- Final Review

1. End-to-end verification of deployed app
2. Final documentation review
3. Project handoff

**Report back:** verification results, any remaining issues.
