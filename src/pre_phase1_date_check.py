"""
Pre-Phase 1 Check -- Date Parsing Verification
Confirms that pd.to_datetime() on 'Price Date' does not silently drop rows
or produce NaT values due to inconsistent date formatting.

Reports:
  1. Raw row count per crop (before any date parsing)
  2. Parsed row count per crop (after to_datetime with errors='coerce')
  3. Number of NaT values introduced per crop
  4. Date format consistency analysis
"""
import pandas as pd
import re
from collections import Counter

FILE = r"d:\Analytics projects\project5\AgriPriceIndia\data\raw\Agriculture_price_dataset.csv"

print("=" * 90)
print("PRE-PHASE 1 CHECK: DATE PARSING VERIFICATION")
print("=" * 90)

# ── Step 1: Read raw data, keep Price Date as string ──
df = pd.read_csv(FILE, low_memory=False)
total_rows = len(df)
print(f"\nTotal rows in raw file: {total_rows:,}")

# ── Step 2: Analyse raw 'Price Date' string formats ──
print("\n" + "-" * 90)
print("STEP 1: DATE FORMAT CONSISTENCY CHECK (on raw string values)")
print("-" * 90)

raw_dates = df['Price Date'].astype(str)
null_before = df['Price Date'].isna().sum()
print(f"Null/NaN Price Date in raw file: {null_before}")

# Detect formats by regex patterns
def classify_date_format(date_str):
    """Classify a date string into a format pattern."""
    date_str = str(date_str).strip()
    if date_str in ('nan', 'NaT', 'None', ''):
        return 'MISSING'
    # MM/DD/YYYY
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
        return 'M/D/YYYY'
    # DD-MM-YYYY
    if re.match(r'^\d{1,2}-\d{1,2}-\d{4}$', date_str):
        return 'D-M-YYYY'
    # YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return 'YYYY-MM-DD'
    # DD/MM/YYYY
    if re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
        return 'DD/MM/YYYY (ambiguous with MM/DD/YYYY)'
    # YYYY/MM/DD
    if re.match(r'^\d{4}/\d{2}/\d{2}$', date_str):
        return 'YYYY/MM/DD'
    # DD-Mon-YYYY (e.g. 15-Jan-2024)
    if re.match(r'^\d{1,2}-[A-Za-z]{3}-\d{4}$', date_str):
        return 'DD-Mon-YYYY'
    return f'UNKNOWN: {date_str[:30]}'

# Sample analysis: check ALL rows for format patterns
format_counter = Counter()
for val in raw_dates:
    fmt = classify_date_format(val)
    format_counter[fmt] += 1

print(f"\nDate format distribution across {total_rows:,} rows:")
for fmt, count in format_counter.most_common():
    pct = (count / total_rows) * 100
    print(f"  {fmt:45s}: {count:>10,} rows ({pct:.2f}%)")

# Show 5 sample values from each format
print("\nSample values per format pattern:")
format_samples = {}
for val in raw_dates:
    fmt = classify_date_format(val)
    if fmt not in format_samples:
        format_samples[fmt] = []
    if len(format_samples[fmt]) < 5:
        format_samples[fmt].append(val)

for fmt, samples in format_samples.items():
    print(f"  {fmt}: {samples}")

# ── Step 3: Parse dates and compare row counts ──
print("\n" + "-" * 90)
print("STEP 2: RAW vs. PARSED ROW COUNT PER CROP")
print("-" * 90)

# Try three parsing strategies to see which one works best
strategies = {
    "format='%m/%d/%Y' (strict)": lambda s: pd.to_datetime(s, format='%m/%d/%Y', errors='coerce'),
    "format='mixed'": lambda s: pd.to_datetime(s, format='mixed', errors='coerce'),
    "infer (no format)": lambda s: pd.to_datetime(s, errors='coerce'),
}

for strat_name, strat_fn in strategies.items():
    print(f"\n  Strategy: {strat_name}")
    parsed = strat_fn(df['Price Date'])
    nat_count = parsed.isna().sum() - null_before  # subtract already-null
    print(f"    Total NaT introduced by parsing: {nat_count:,} (out of {total_rows:,})")
    if nat_count > 0:
        # Show which raw values became NaT
        nat_mask = parsed.isna() & df['Price Date'].notna()
        nat_samples = df.loc[nat_mask, 'Price Date'].head(10).tolist()
        print(f"    Sample unparseable values: {nat_samples}")

# ── Step 4: Per-crop comparison using the best strategy ──
# Use format='mixed' as the reference (most permissive)
print("\n" + "-" * 90)
print("STEP 3: PER-CROP RAW vs. PARSED ROW COUNT")
print("-" * 90)

df['Price_Date_Parsed'] = pd.to_datetime(df['Price Date'], format='mixed', errors='coerce')

print(f"\n{'Crop':<15} {'Raw Rows':>10} {'Parsed OK':>10} {'NaT':>8} {'Lost %':>8}  {'Date Range (parsed)'}")
print("-" * 90)

for crop in sorted(df['Commodity'].unique()):
    mask = df['Commodity'] == crop
    raw_count = mask.sum()
    parsed_ok = df.loc[mask, 'Price_Date_Parsed'].notna().sum()
    nat_count = raw_count - parsed_ok
    lost_pct = (nat_count / raw_count * 100) if raw_count > 0 else 0

    if parsed_ok > 0:
        date_min = df.loc[mask & df['Price_Date_Parsed'].notna(), 'Price_Date_Parsed'].min().date()
        date_max = df.loc[mask & df['Price_Date_Parsed'].notna(), 'Price_Date_Parsed'].max().date()
        date_range = f"{date_min} to {date_max}"
    else:
        date_range = "N/A"

    flag = " [WARN]" if nat_count > 0 else " [OK]"
    print(f"{crop:<15} {raw_count:>10,} {parsed_ok:>10,} {nat_count:>8,} {lost_pct:>7.2f}%  {date_range}{flag}")

# ── Step 5: Also test with the strict format from phase0_dates.py ──
print("\n" + "-" * 90)
print("STEP 4: PER-CROP COMPARISON WITH STRICT format='%m/%d/%Y'")
print("-" * 90)

df['Price_Date_Strict'] = pd.to_datetime(df['Price Date'], format='%m/%d/%Y', errors='coerce')

print(f"\n{'Crop':<15} {'Raw Rows':>10} {'Strict OK':>10} {'Mixed OK':>10} {'Strict NaT':>10} {'Diff':>8}")
print("-" * 90)

for crop in sorted(df['Commodity'].unique()):
    mask = df['Commodity'] == crop
    raw_count = mask.sum()
    strict_ok = df.loc[mask, 'Price_Date_Strict'].notna().sum()
    mixed_ok = df.loc[mask, 'Price_Date_Parsed'].notna().sum()
    strict_nat = raw_count - strict_ok
    diff = mixed_ok - strict_ok

    flag = " [WARN]" if strict_nat > 0 else " [OK]"
    print(f"{crop:<15} {raw_count:>10,} {strict_ok:>10,} {mixed_ok:>10,} {strict_nat:>10,} {diff:>8,}{flag}")

# ── Step 6: Per-crop format distribution ──
print("\n" + "-" * 90)
print("STEP 5: DATE FORMAT DISTRIBUTION PER CROP")
print("-" * 90)

for crop in sorted(df['Commodity'].unique()):
    mask = df['Commodity'] == crop
    crop_dates = df.loc[mask, 'Price Date'].astype(str)
    crop_formats = Counter(classify_date_format(d) for d in crop_dates)
    print(f"\n  {crop}:")
    for fmt, count in crop_formats.most_common():
        print(f"    {fmt}: {count:,}")

print("\n" + "=" * 90)
print("DATE PARSING VERIFICATION COMPLETE")
print("=" * 90)
