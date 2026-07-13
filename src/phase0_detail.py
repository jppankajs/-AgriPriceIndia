"""
Phase 0 — Detailed per-crop inspection and data quality flagging.
"""
import pandas as pd

FILE = r"d:\Analytics projects\project5\AgriPriceIndia\data\raw\Agriculture_price_dataset.csv"
df = pd.read_csv(FILE, low_memory=False)
df['Price Date'] = pd.to_datetime(df['Price Date'], format='mixed')

print("=" * 80)
print("PER-CROP DATE RANGES AND ROW COUNTS")
print("=" * 80)

for crop in sorted(df['Commodity'].unique()):
    subset = df[df['Commodity'] == crop]
    print(f"\n{crop}:")
    print(f"  Rows: {len(subset):,}")
    print(f"  Date range: {subset['Price Date'].min().date()} to {subset['Price Date'].max().date()}")
    print(f"  States: {subset['STATE'].nunique()}")
    print(f"  Markets: {subset['Market Name'].nunique()}")
    print(f"  Varieties: {sorted(subset['Variety'].unique())[:10]}")

print("\n" + "=" * 80)
print("DATA QUALITY FLAGS")
print("=" * 80)

# Zero prices
zero_min = df[df['Min_Price'] == 0]
zero_max = df[df['Max_Price'] == 0]
zero_modal = df[df['Modal_Price'] == 0]
print(f"\nZero Min_Price rows: {len(zero_min)}")
print(f"Zero Max_Price rows: {len(zero_max)}")
print(f"Zero Modal_Price rows: {len(zero_modal)}")

if len(zero_modal) > 0:
    print(f"\nZero Modal_Price details:")
    print(zero_modal[['STATE', 'Market Name', 'Commodity', 'Min_Price', 'Max_Price', 'Modal_Price', 'Price Date']].to_string())

# Modal price outside min/max bounds
out_of_bounds = df[(df['Modal_Price'] < df['Min_Price']) | (df['Modal_Price'] > df['Max_Price'])]
print(f"\nModal_Price outside [Min, Max] bounds: {len(out_of_bounds)} rows")
if len(out_of_bounds) > 0:
    print(f"  Sample (first 10):")
    print(out_of_bounds[['Commodity', 'Min_Price', 'Max_Price', 'Modal_Price', 'Price Date']].head(10).to_string())

# Extremely high prices (potential outliers)
for crop in sorted(df['Commodity'].unique()):
    subset = df[df['Commodity'] == crop]
    q99 = subset['Modal_Price'].quantile(0.99)
    q01 = subset['Modal_Price'].quantile(0.01)
    extremes = subset[(subset['Modal_Price'] > q99 * 3) | (subset['Modal_Price'] < q01 * 0.1)]
    if len(extremes) > 0:
        print(f"\n{crop} extreme outliers (>3x the 99th percentile or <0.1x the 1st): {len(extremes)} rows")
        print(f"  99th percentile: {q99:.0f}, 1st percentile: {q01:.0f}")
        print(f"  Extreme range: {extremes['Modal_Price'].min():.0f} to {extremes['Modal_Price'].max():.0f}")

# Check negative prices
neg = df[(df['Min_Price'] < 0) | (df['Max_Price'] < 0) | (df['Modal_Price'] < 0)]
print(f"\nNegative price rows: {len(neg)}")

# Grade distribution
print(f"\nGrade distribution:")
print(df['Grade'].value_counts().to_string())

print("\n" + "=" * 80)
print("DETAILED INSPECTION COMPLETE")
print("=" * 80)
