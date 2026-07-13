"""
Phase 0 — Data inspection script.
Reads the raw data file(s) and prints shape, columns, dtypes, first 5 rows,
date range, unique commodity values, and basic data quality checks.
"""
import pandas as pd
import sys

RAW_DIR = r"d:\Analytics projects\project5\AgriPriceIndia\data\raw"
FILE = f"{RAW_DIR}/Agriculture_price_dataset.csv"

print("=" * 80)
print(f"FILE: Agriculture_price_dataset.csv")
print("=" * 80)

# Read the file
df = pd.read_csv(FILE, low_memory=False)

# Shape
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

# Columns and dtypes
print(f"\nColumns and dtypes:")
for col in df.columns:
    print(f"  {col:30s} -> {df[col].dtype}")

# First 5 rows
print(f"\nFirst 5 rows:")
print(df.head().to_string())

# Check for date columns
date_candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'arrival' in c.lower() and 'quantity' not in c.lower()]
print(f"\nPotential date columns: {date_candidates}")

for col in date_candidates:
    try:
        parsed = pd.to_datetime(df[col], errors='coerce')
        valid_count = parsed.notna().sum()
        if valid_count > len(df) * 0.5:
            print(f"  {col}: parseable as date. Range: {parsed.min()} to {parsed.max()}")
            print(f"    Valid dates: {valid_count}/{len(df)}")
        else:
            print(f"  {col}: only {valid_count}/{len(df)} parseable as date — probably not a date column")
    except Exception as e:
        print(f"  {col}: failed to parse — {e}")

# Commodity column detection
commodity_candidates = [c for c in df.columns if 'commodity' in c.lower() or 'crop' in c.lower() or 'product' in c.lower()]
print(f"\nPotential commodity columns: {commodity_candidates}")

TARGET_CROPS = ['onion', 'tomato', 'potato', 'wheat', 'rice']

for col in commodity_candidates:
    unique_vals = df[col].dropna().unique()
    print(f"\n  {col}: {len(unique_vals)} unique values")
    if len(unique_vals) <= 50:
        print(f"    Values: {sorted(unique_vals)}")
    else:
        print(f"    First 30 values: {sorted(unique_vals)[:30]}")
        print(f"    ...")
    
    # Check which target crops are present (case-insensitive)
    lower_vals = [str(v).lower().strip() for v in unique_vals]
    print(f"\n    Target crop matching:")
    for crop in TARGET_CROPS:
        matches = [v for v in unique_vals if crop in str(v).lower()]
        if matches:
            print(f"      {crop}: FOUND as {matches}")
        else:
            print(f"      {crop}: NOT FOUND")

# Missing values
print(f"\nMissing values per column:")
missing = df.isnull().sum()
for col in df.columns:
    pct = (missing[col] / len(df)) * 100
    print(f"  {col:30s}: {missing[col]:>8d} ({pct:.2f}%)")

# Duplicate check
dup_count = df.duplicated().sum()
print(f"\nExact duplicate rows: {dup_count}")

# Numeric column stats (look for impossible values)
print(f"\nNumeric column summary (looking for zeros, negatives, outliers):")
numeric_cols = df.select_dtypes(include='number').columns.tolist()
for col in numeric_cols:
    desc = df[col].describe()
    print(f"  {col}:")
    print(f"    min={desc['min']}, max={desc['max']}, mean={desc['mean']:.2f}, zeros={int((df[col]==0).sum())}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
