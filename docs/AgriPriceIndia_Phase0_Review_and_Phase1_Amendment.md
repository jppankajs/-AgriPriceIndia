# AgriPrice India -- Phase 0 review and Phase 1 amendment

## Verdict on Phase 0
Approved, with one required check before anything else runs, and four amendments to the original Phase 1 instructions based on what the real data showed.

## Required check before any scope decision is locked in
Verify the Rice/Tomato/Wheat coverage gaps are real, not a parsing artifact:
- Confirm row count after `pd.to_datetime()` matches the raw file's row count per crop -- inconsistent date formatting can silently produce `NaT` and drop rows without erroring
- Check whether `Price Date` uses one consistent format throughout the file, or varies
- Report back: raw row count vs. parsed row count per crop

If this confirms the gaps are genuine, proceed with the amendments below.

## Amendment 1 -- Tiered crop treatment (replaces "same model, lower confidence" for all 5)

- **Onion, Potato** (~2 years each): full SARIMA + Prophet/Holt-Winters comparison, as originally planned.
- **Wheat** (~8 months, one incomplete cycle): attempt SARIMA/Prophet, but don't claim annual seasonality is confirmed -- state plainly that only a trend/short-cycle pattern could be validated with the available history.
- **Tomato** (~5 months, 154 days), **Rice** (~2 months, 62 days): not enough history to fit or validate a seasonal model. Use a trend-only model (moving average or basic exponential smoothing, no seasonal term). Do not generate an 8-12 week forecast -- that extrapolates further than the total history available. Phase 4's decision layer should automatically output "Monitor -- insufficient history for reliable forecast" for these two, not a Sell/Hold signal. State this as a data limitation in the report and the dashboard's About section, not a modeling failure.

All 5 crops stay in scope. Two of them are treated differently, honestly, instead of forced through the same pipeline with a caveat attached.

## Amendment 2 -- Grade handling (not specified in the original Phase 1 spec)

87% of rows are FAQ grade (Fair Average Quality -- the standard benchmark). Use FAQ-grade prices as the primary series for each crop's forecasting model. Keep Local and Non-FAQ grade prices in the granular table for reference and EDA, but don't blend grades into one average -- a modal price mixed across quality tiers isn't a meaningful single number.

## Amendment 3 -- Two output tables, not one

Original Phase 1 target was a single consolidated table. Produce two instead:
1. `agri_prices_clean.csv` -- full granular table (date, state, district, market, commodity, grade, min/max/modal price) for EDA and any dashboard drill-down.
2. `agri_prices_daily_national.csv` -- one row per (crop, date): national daily average `Modal_Price` across FAQ-grade markets reporting that day. This is the actual input to Phase 3 -- SARIMA/Prophet need one series per crop, not hundreds of per-market series.

## Amendment 4 -- Concrete outlier rule (replaces "investigate")

- Any row where `Max_Price = 0` while `Min_Price` or `Modal_Price` is nonzero: treat `Max_Price` as missing, not a real zero. Don't drop the row.
- Any `Modal_Price` more than 5x the crop's own 95th percentile: exclude from the modeling series, but keep the row in the granular table with a `flagged_price_outlier = True` column -- visible, not silently deleted.
- Report the exact count of rows affected by each rule.

---

Run the raw-vs-parsed row count check first. If it confirms the gaps are genuine, these four amendments become the real Phase 1 spec -- paste this alongside the original master prompt when you authorize Phase 1.
