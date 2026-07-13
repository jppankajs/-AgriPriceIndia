"""
Phase 1 -- Data Cleaning and Consolidation
============================================
Implements the amended Phase 1 specification:
  - Amendment 1: Tiered crop treatment (scope tracking only in Phase 1)
  - Amendment 2: FAQ-grade primary series for the daily national table
  - Amendment 3: Two output tables (granular + daily national)
  - Amendment 4: Concrete outlier rules (Max_Price=0 as missing, 5x95th flagging)

Inputs:
  data/raw/Agriculture_price_dataset.csv

Outputs:
  data/processed/agri_prices_clean.csv
  data/processed/agri_prices_daily_national.csv
"""

import pandas as pd
import numpy as np
import os
import sys

# ── Paths ──
BASE_DIR = r"d:\Analytics projects\project5\AgriPriceIndia"
RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "Agriculture_price_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
CLEAN_FILE = os.path.join(OUT_DIR, "agri_prices_clean.csv")
DAILY_FILE = os.path.join(OUT_DIR, "agri_prices_daily_national.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load raw data
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("PHASE 1: DATA CLEANING AND CONSOLIDATION")
print("=" * 90)

df = pd.read_csv(RAW_FILE, low_memory=False)
raw_total = len(df)
print(f"\nRaw data loaded: {raw_total:,} rows x {df.shape[1]} columns")

# ── Record raw row counts per crop ──
raw_counts = df['Commodity'].value_counts()
print("\nRaw row counts per crop:")
for crop in sorted(df['Commodity'].unique()):
    print(f"  {crop}: {raw_counts[crop]:,}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Parse dates
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("STEP 2: DATE PARSING")
print("-" * 90)

df['Price Date'] = pd.to_datetime(df['Price Date'], format='%m/%d/%Y')
nat_count = df['Price Date'].isna().sum()
print(f"NaT after parsing: {nat_count}")
assert nat_count == 0, f"FATAL: {nat_count} dates failed to parse!"

# Rename Price Date to Price_Date for consistency (no spaces in column names)
df = df.rename(columns={'Price Date': 'Price_Date'})

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Standardize column names
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("STEP 3: COLUMN STANDARDIZATION")
print("-" * 90)

# Standardize column names: strip whitespace, replace spaces with underscores
col_rename = {
    'STATE': 'State',
    'District Name': 'District',
    'Market Name': 'Market',
    'Commodity': 'Commodity',
    'Variety': 'Variety',
    'Grade': 'Grade',
    'Min_Price': 'Min_Price',
    'Max_Price': 'Max_Price',
    'Modal_Price': 'Modal_Price',
    'Price_Date': 'Price_Date',
}
df = df.rename(columns=col_rename)
print(f"Columns after standardization: {list(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Amendment 4 -- Outlier Rule 1: Max_Price = 0 treatment
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("STEP 4: MAX_PRICE = 0 TREATMENT (Amendment 4, Rule 1)")
print("-" * 90)

# Any row where Max_Price = 0 while Min_Price or Modal_Price is nonzero:
# treat Max_Price as missing (NaN), not a real zero. Don't drop the row.
max_zero_mask = (df['Max_Price'] == 0) & ((df['Min_Price'] != 0) | (df['Modal_Price'] != 0))
max_zero_count = max_zero_mask.sum()
print(f"Rows with Max_Price=0 but Min_Price or Modal_Price nonzero: {max_zero_count:,}")

# Show per-crop breakdown
if max_zero_count > 0:
    print("\n  Per-crop breakdown:")
    for crop in sorted(df['Commodity'].unique()):
        crop_count = ((df['Commodity'] == crop) & max_zero_mask).sum()
        if crop_count > 0:
            print(f"    {crop}: {crop_count:,}")

# Set Max_Price to NaN for these rows
df.loc[max_zero_mask, 'Max_Price'] = np.nan
print(f"\n  Max_Price set to NaN for {max_zero_count:,} rows.")

# Also handle the truly zero rows (all three prices = 0)
all_zero_mask = (df['Min_Price'] == 0) & (df['Max_Price'] == 0) & (df['Modal_Price'] == 0)
all_zero_count = all_zero_mask.sum()
print(f"\nRows with ALL prices = 0: {all_zero_count}")
# Note: Max_Price was already set to NaN above for some of these -- recheck
# after the NaN assignment
all_zero_remaining = (df['Min_Price'] == 0) & (df['Max_Price'].fillna(0) == 0) & (df['Modal_Price'] == 0)
all_zero_remaining_count = all_zero_remaining.sum()
if all_zero_remaining_count > 0:
    print(f"  Rows with Min=0, Max=0/NaN, Modal=0: {all_zero_remaining_count}")
    print("  These are uninformative -- setting all prices to NaN (rows kept, not dropped).")
    df.loc[all_zero_remaining, ['Min_Price', 'Max_Price', 'Modal_Price']] = np.nan

# Handle Min_Price=0 with nonzero Modal/Max (same logic)
min_zero_mask = (df['Min_Price'] == 0) & ((df['Max_Price'].fillna(0) != 0) | (df['Modal_Price'] != 0))
min_zero_count = min_zero_mask.sum()
print(f"\nRows with Min_Price=0 but Modal/Max nonzero: {min_zero_count:,}")
if min_zero_count > 0:
    print("  Treating Min_Price=0 as missing.")
    df.loc[min_zero_mask, 'Min_Price'] = np.nan

# Handle Modal_Price=0 with nonzero Min/Max
modal_zero_mask = (df['Modal_Price'] == 0) & ((df['Min_Price'].fillna(0) != 0) | (df['Max_Price'].fillna(0) != 0))
modal_zero_count = modal_zero_mask.sum()
print(f"Rows with Modal_Price=0 but Min/Max nonzero: {modal_zero_count:,}")
if modal_zero_count > 0:
    print("  Treating Modal_Price=0 as missing.")
    df.loc[modal_zero_mask, 'Modal_Price'] = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Amendment 4 -- Outlier Rule 2: Modal_Price > 5x 95th percentile
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("STEP 5: PRICE OUTLIER FLAGGING (Amendment 4, Rule 2)")
print("-" * 90)

# Calculate 95th percentile per crop (using non-NaN Modal_Price only)
df['flagged_price_outlier'] = False
total_flagged = 0

print(f"\n{'Crop':<15} {'95th Pctile':>12} {'5x Threshold':>14} {'Flagged':>10}")
print("-" * 60)

for crop in sorted(df['Commodity'].unique()):
    crop_mask = df['Commodity'] == crop
    modal_valid = df.loc[crop_mask, 'Modal_Price'].dropna()
    p95 = modal_valid.quantile(0.95)
    threshold = p95 * 5

    # Flag rows where Modal_Price > 5x the 95th percentile
    outlier_mask = crop_mask & (df['Modal_Price'] > threshold)
    flagged_count = outlier_mask.sum()
    df.loc[outlier_mask, 'flagged_price_outlier'] = True
    total_flagged += flagged_count

    print(f"{crop:<15} {p95:>12,.0f} {threshold:>14,.0f} {flagged_count:>10,}")

print(f"\nTotal rows flagged as price outliers: {total_flagged:,}")

# Show sample of flagged rows
flagged_df = df[df['flagged_price_outlier']]
if len(flagged_df) > 0:
    print(f"\nSample of flagged outlier rows (first 10):")
    print(flagged_df[['Commodity', 'State', 'Market', 'Price_Date',
                       'Modal_Price', 'Min_Price', 'Max_Price']].head(10).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Bounds violation check (post-cleaning)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("STEP 6: BOUNDS VIOLATION CHECK (post-cleaning)")
print("-" * 90)

# After NaN treatment, check Modal outside [Min, Max] where all three are non-NaN
valid_bounds = df[['Min_Price', 'Max_Price', 'Modal_Price']].notna().all(axis=1)
bounds_check = df[valid_bounds]
below_min = bounds_check[bounds_check['Modal_Price'] < bounds_check['Min_Price']]
above_max = bounds_check[bounds_check['Modal_Price'] > bounds_check['Max_Price']]
total_violations = len(below_min) + len(above_max)

print(f"Rows where all prices are non-NaN: {len(bounds_check):,}")
print(f"Modal < Min: {len(below_min):,}")
print(f"Modal > Max: {len(above_max):,}")
print(f"Total bounds violations (post-cleaning): {total_violations:,}")

if total_violations > 0:
    print(f"\n  Per-crop breakdown:")
    for crop in sorted(df['Commodity'].unique()):
        crop_below = ((below_min['Commodity'] == crop).sum() if len(below_min) > 0 else 0)
        crop_above = ((above_max['Commodity'] == crop).sum() if len(above_max) > 0 else 0)
        crop_total = crop_below + crop_above
        if crop_total > 0:
            print(f"    {crop}: {crop_total:,} (Modal<Min: {crop_below:,}, Modal>Max: {crop_above:,})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Sort and finalize the granular table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("STEP 7: FINALIZE GRANULAR TABLE (agri_prices_clean.csv)")
print("-" * 90)

# Sort by Commodity, Price_Date, State, Market
df = df.sort_values(['Commodity', 'Price_Date', 'State', 'Market']).reset_index(drop=True)

# Final column order
col_order = ['Price_Date', 'State', 'District', 'Market', 'Commodity',
             'Variety', 'Grade', 'Min_Price', 'Max_Price', 'Modal_Price',
             'flagged_price_outlier']
df = df[col_order]

print(f"Final granular table shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Row counts per crop after cleaning (no rows dropped -- all kept)
print(f"\nRow counts per crop (post-cleaning, no rows dropped):")
for crop in sorted(df['Commodity'].unique()):
    crop_count = (df['Commodity'] == crop).sum()
    raw_count = raw_counts[crop]
    print(f"  {crop}: {crop_count:,} (raw: {raw_count:,}, diff: {crop_count - raw_count:,})")

# ── Save granular table ──
df.to_csv(CLEAN_FILE, index=False)
clean_size = os.path.getsize(CLEAN_FILE)
print(f"\nSaved: {CLEAN_FILE}")
print(f"File size: {clean_size:,} bytes ({clean_size / 1024 / 1024:.2f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Build daily national table (Amendment 2 + 3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("STEP 8: BUILD DAILY NATIONAL TABLE (Amendment 2 + 3)")
print("-" * 90)

# Filter: FAQ grade only, non-flagged outliers, non-NaN Modal_Price
faq_mask = df['Grade'] == 'FAQ'
not_outlier = ~df['flagged_price_outlier']
modal_valid = df['Modal_Price'].notna()

daily_source = df[faq_mask & not_outlier & modal_valid].copy()
print(f"FAQ-grade, non-outlier, non-NaN Modal rows: {len(daily_source):,}")
print(f"  Excluded by non-FAQ grade: {(~faq_mask).sum():,}")
print(f"  Excluded by outlier flag: {df['flagged_price_outlier'].sum():,}")
print(f"  Excluded by NaN Modal_Price: {(~modal_valid).sum():,}")

# Aggregate: one row per (Commodity, Price_Date)
# National daily average of Modal_Price across all FAQ-grade markets
daily = daily_source.groupby(['Commodity', 'Price_Date']).agg(
    Modal_Price_Avg=('Modal_Price', 'mean'),
    Modal_Price_Median=('Modal_Price', 'median'),
    Min_Price_Avg=('Min_Price', 'mean'),
    Max_Price_Avg=('Max_Price', 'mean'),
    Markets_Reporting=('Market', 'nunique'),
    States_Reporting=('State', 'nunique'),
).reset_index()

# Round prices to 2 decimal places
for col in ['Modal_Price_Avg', 'Modal_Price_Median', 'Min_Price_Avg', 'Max_Price_Avg']:
    daily[col] = daily[col].round(2)

daily = daily.sort_values(['Commodity', 'Price_Date']).reset_index(drop=True)

print(f"\nDaily national table shape: {daily.shape[0]:,} rows x {daily.shape[1]} columns")
print(f"Columns: {list(daily.columns)}")

print(f"\nPer-crop summary in daily table:")
print(f"{'Crop':<15} {'Rows':>8} {'Date Range':>30} {'Avg Markets/Day':>18}")
print("-" * 75)
for crop in sorted(daily['Commodity'].unique()):
    crop_df = daily[daily['Commodity'] == crop]
    date_min = crop_df['Price_Date'].min().date()
    date_max = crop_df['Price_Date'].max().date()
    avg_markets = crop_df['Markets_Reporting'].mean()
    print(f"{crop:<15} {len(crop_df):>8,} {str(date_min) + ' to ' + str(date_max):>30} {avg_markets:>18.1f}")

# ── Save daily national table ──
daily.to_csv(DAILY_FILE, index=False)
daily_size = os.path.getsize(DAILY_FILE)
print(f"\nSaved: {DAILY_FILE}")
print(f"File size: {daily_size:,} bytes ({daily_size / 1024 / 1024:.2f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: Show sample of daily national table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("STEP 9: SAMPLE OF DAILY NATIONAL TABLE")
print("-" * 90)

for crop in sorted(daily['Commodity'].unique()):
    crop_df = daily[daily['Commodity'] == crop]
    print(f"\n{crop} (first 5 rows):")
    print(crop_df.head().to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: Grade distribution in granular table (verification)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("STEP 10: GRADE DISTRIBUTION VERIFICATION")
print("-" * 90)

grade_dist = df['Grade'].value_counts()
print("\nGrade distribution in agri_prices_clean.csv:")
for grade, count in grade_dist.items():
    pct = count / len(df) * 100
    print(f"  {grade}: {count:,} ({pct:.1f}%)")

faq_pct = grade_dist.get('FAQ', 0) / len(df) * 100
print(f"\nFAQ grade represents {faq_pct:.1f}% of the granular table -- "
      f"this is the grade used for the daily national series.")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PHASE 1 COMPLETE -- FINAL SUMMARY")
print("=" * 90)

print(f"""
OUTPUT FILES:
  1. agri_prices_clean.csv
     - {df.shape[0]:,} rows x {df.shape[1]} columns
     - File size: {clean_size:,} bytes ({clean_size/1024/1024:.2f} MB)
     - Full granular table with all grades, flagged_price_outlier column

  2. agri_prices_daily_national.csv
     - {daily.shape[0]:,} rows x {daily.shape[1]} columns
     - File size: {daily_size:,} bytes ({daily_size/1024/1024:.2f} MB)
     - FAQ-grade only, outliers excluded, one row per (crop, date)

CLEANING ACTIONS:
  - Max_Price=0 (nonzero peers) set to NaN: {max_zero_count:,} rows
  - Min_Price=0 (nonzero peers) set to NaN: {min_zero_count:,} rows
  - Modal_Price=0 (nonzero peers) set to NaN: {modal_zero_count:,} rows
  - All-prices-zero set to NaN: {all_zero_remaining_count:,} rows
  - Price outliers flagged (Modal > 5x 95th pctile): {total_flagged:,} rows
  - Bounds violations remaining (post-cleaning): {total_violations:,} rows
  - Rows dropped: 0 (all rows preserved)

NO ROWS WERE DROPPED. All cleaning is via NaN replacement or flagging.
""")
