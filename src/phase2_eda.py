"""
Phase 2 -- Exploratory Data Analysis (EDA)
============================================
Adjustments from Phase 1 approval:
  - Use Modal_Price_Median (not Avg) as the primary series
  - For Wheat/Tomato/Rice: mark missing calendar months on seasonal charts
  - Skip arrivals-vs-price scatter (no arrival quantity column)

Outputs all figures to reports/figures/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import os

warnings.filterwarnings('ignore')

# ── Paths ──
BASE_DIR = r"d:\Analytics projects\project5\AgriPriceIndia"
DAILY_FILE = os.path.join(BASE_DIR, "data", "processed", "agri_prices_daily_national.csv")
CLEAN_FILE = os.path.join(BASE_DIR, "data", "processed", "agri_prices_clean.csv")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style setup ──
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
})

CROP_COLORS = {
    'Onion': '#f85149',
    'Potato': '#58a6ff',
    'Tomato': '#f0883e',
    'Wheat': '#d2a8ff',
    'Rice': '#3fb950',
}

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("PHASE 2: EXPLORATORY DATA ANALYSIS")
print("=" * 90)

daily = pd.read_csv(DAILY_FILE, parse_dates=['Price_Date'])
clean = pd.read_csv(CLEAN_FILE, parse_dates=['Price_Date'], low_memory=False)

print(f"Daily national table: {len(daily):,} rows")
print(f"Granular table: {len(clean):,} rows")
print(f"Primary series: Modal_Price_Median (per user adjustment)")

# ── Pivot for correlation analysis ──
price_pivot = daily.pivot_table(
    index='Price_Date', columns='Commodity',
    values='Modal_Price_Median', aggfunc='first'
)

fig_count = 0

def save_fig(fig, name):
    global fig_count
    fig_count += 1
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [{fig_count}] Saved: {name}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Daily price time-series (all crops, one panel each)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("FIGURE 1: Daily Price Time Series")
print("-" * 90)

fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=False)
fig.suptitle('Daily National Median Modal Price by Crop (FAQ Grade)', y=0.98)

for idx, crop in enumerate(['Onion', 'Potato', 'Wheat', 'Tomato', 'Rice']):
    ax = axes[idx]
    crop_df = daily[daily['Commodity'] == crop].sort_values('Price_Date')
    color = CROP_COLORS[crop]

    ax.plot(crop_df['Price_Date'], crop_df['Modal_Price_Median'],
            color=color, linewidth=1.2, alpha=0.9)
    ax.fill_between(crop_df['Price_Date'],
                     crop_df['Min_Price_Avg'], crop_df['Max_Price_Avg'],
                     alpha=0.15, color=color, label='Min-Max range')
    ax.set_ylabel('Rs/Quintal')
    ax.set_title(f'{crop}  ({len(crop_df)} days, '
                 f'{crop_df["Price_Date"].min().strftime("%b %Y")} -- '
                 f'{crop_df["Price_Date"].max().strftime("%b %Y")})',
                 loc='left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(loc='upper right', fontsize=9, framealpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, '01_daily_price_timeseries.png')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: All crops overlaid on same axis (normalized to % change from start)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("FIGURE 2: Normalized Price Comparison")
print("-" * 90)

fig, ax = plt.subplots(figsize=(16, 8))
fig.suptitle('Price Movement (% Change from First Day) -- All Crops')

for crop in ['Onion', 'Potato', 'Wheat', 'Tomato', 'Rice']:
    crop_df = daily[daily['Commodity'] == crop].sort_values('Price_Date')
    start_price = crop_df['Modal_Price_Median'].iloc[0]
    pct_change = ((crop_df['Modal_Price_Median'] - start_price) / start_price) * 100
    ax.plot(crop_df['Price_Date'], pct_change,
            color=CROP_COLORS[crop], linewidth=1.5, label=crop, alpha=0.9)

ax.axhline(0, color='#8b949e', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel('% Change from Start')
ax.set_xlabel('Date')
ax.legend(framealpha=0.3)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
fig.tight_layout()
save_fig(fig, '02_normalized_price_comparison.png')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Seasonal decomposition -- Onion and Potato (full pipeline crops)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("FIGURE 3-4: Seasonal Decomposition (Onion, Potato)")
print("-" * 90)

for crop in ['Onion', 'Potato']:
    crop_df = daily[daily['Commodity'] == crop].sort_values('Price_Date').copy()
    crop_df = crop_df.set_index('Price_Date')['Modal_Price_Median']

    # Resample to regular daily frequency, interpolate gaps
    crop_daily = crop_df.asfreq('D')
    gap_count = crop_daily.isna().sum()
    crop_daily = crop_daily.interpolate(method='linear')

    # Decompose with period=365 (annual cycle)
    decomp = seasonal_decompose(crop_daily, model='additive', period=365)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'{crop} -- Seasonal Decomposition (Additive, Period=365 days)', y=0.98)

    color = CROP_COLORS[crop]
    components = [
        ('Observed', decomp.observed),
        ('Trend', decomp.trend),
        ('Seasonal', decomp.seasonal),
        ('Residual', decomp.resid),
    ]

    for ax, (name, data) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, linewidth=1.0, alpha=0.85)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fname = f'0{3 if crop == "Onion" else 4}_seasonal_decomp_{crop.lower()}.png'
    save_fig(fig, fname)
    print(f"  {crop}: {len(crop_daily)} days (interpolated {gap_count} gaps)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Month-over-month boxplots with missing-month markers
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("FIGURE 5: Monthly Price Distribution (all crops)")
print("-" * 90)

fig, axes = plt.subplots(5, 1, figsize=(16, 22))
fig.suptitle('Monthly Price Distribution (Median Modal Price)', y=0.98)

for idx, crop in enumerate(['Onion', 'Potato', 'Wheat', 'Tomato', 'Rice']):
    ax = axes[idx]
    crop_df = daily[daily['Commodity'] == crop].copy()
    crop_df['Month'] = crop_df['Price_Date'].dt.month
    color = CROP_COLORS[crop]

    # Determine which months have data
    months_with_data = set(crop_df['Month'].unique())
    all_months = set(range(1, 13))
    months_missing = sorted(all_months - months_with_data)

    # Build boxplot data for all 12 months
    box_data = []
    positions = []
    for m in range(1, 13):
        month_vals = crop_df.loc[crop_df['Month'] == m, 'Modal_Price_Median'].values
        if len(month_vals) > 0:
            box_data.append(month_vals)
            positions.append(m)

    if box_data:
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markersize=3,
                                       markerfacecolor=color, alpha=0.4))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color('#c9d1d9')
                line.set_alpha(0.7)
        for line in bp['medians']:
            line.set_color('#f0f6fc')
            line.set_linewidth(1.5)

    # Mark missing months with hatched background
    for m in months_missing:
        ax.axvspan(m - 0.4, m + 0.4, color='#da3633', alpha=0.15,
                   hatch='///', edgecolor='#da3633', linewidth=0)

    ax.set_xlim(0.3, 12.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel('Rs/Quintal')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(True, axis='y', alpha=0.3)

    # Title with explicit missing-month callout
    if months_missing:
        missing_names = ', '.join([MONTH_NAMES[m-1] for m in months_missing])
        ax.set_title(f'{crop}  [NO DATA: {missing_names}]', loc='left',
                     color='#f85149')
    else:
        ax.set_title(f'{crop}  [All 12 months covered]', loc='left')

    # Add legend for missing months if applicable
    if months_missing:
        legend_elements = [Patch(facecolor='#da3633', alpha=0.15,
                                edgecolor='#da3633', hatch='///',
                                label='No data for this month')]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=9, framealpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, '05_monthly_boxplots.png')

# Print missing month summary
for crop in ['Wheat', 'Tomato', 'Rice']:
    crop_df = daily[daily['Commodity'] == crop]
    months_present = sorted(crop_df['Price_Date'].dt.month.unique())
    months_missing = sorted(set(range(1, 13)) - set(months_present))
    present_names = [MONTH_NAMES[m-1] for m in months_present]
    missing_names = [MONTH_NAMES[m-1] for m in months_missing]
    print(f"  {crop}: present={present_names}, MISSING={missing_names}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Distribution plots (histograms + KDE)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("FIGURE 6: Price Distribution (Histogram + KDE)")
print("-" * 90)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle('Distribution of Daily Median Modal Price (FAQ Grade)', y=1.02)

for idx, crop in enumerate(['Onion', 'Potato', 'Wheat', 'Tomato', 'Rice']):
    ax = axes[idx]
    crop_df = daily[daily['Commodity'] == crop]
    values = crop_df['Modal_Price_Median'].dropna()
    color = CROP_COLORS[crop]

    ax.hist(values, bins=40, color=color, alpha=0.5, density=True, edgecolor='none')
    if len(values) > 5:
        values.plot.kde(ax=ax, color=color, linewidth=2)
    ax.set_title(crop)
    ax.set_xlabel('Rs/Quintal')
    if idx == 0:
        ax.set_ylabel('Density')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    ax.grid(True, alpha=0.3)

    # Add stats annotation
    stats_text = (f'n={len(values)}\n'
                  f'med={values.median():,.0f}\n'
                  f'IQR={values.quantile(0.75)-values.quantile(0.25):,.0f}')
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))

fig.tight_layout()
save_fig(fig, '06_price_distributions.png')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Correlation heatmap (Onion, Potato, Wheat only -- shared dates)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("FIGURE 7: Correlation Analysis")
print("-" * 90)

# Full correlation (pairwise, limited by overlapping dates)
corr_data = price_pivot.copy()

# Report overlap
for c1 in corr_data.columns:
    for c2 in corr_data.columns:
        if c1 < c2:
            overlap = corr_data[[c1, c2]].dropna()
            print(f"  {c1} vs {c2}: {len(overlap)} overlapping days")

# Compute pairwise correlation
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle('Price Correlation Between Crops\n(Pairwise on overlapping dates, Median Modal Price)')

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            square=True, linewidths=1, linecolor='#30363d',
            ax=ax, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 13, 'weight': 'bold'})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

fig.tight_layout()
save_fig(fig, '07_correlation_heatmap.png')

print(f"\nCorrelation matrix:")
print(corr_matrix.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Rolling volatility (30-day rolling std)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("FIGURE 8: Rolling 30-Day Price Volatility")
print("-" * 90)

fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=False)
fig.suptitle('30-Day Rolling Standard Deviation (Price Volatility)', y=0.98)

for idx, crop in enumerate(['Onion', 'Potato', 'Wheat', 'Tomato', 'Rice']):
    ax = axes[idx]
    crop_df = daily[daily['Commodity'] == crop].sort_values('Price_Date').copy()
    color = CROP_COLORS[crop]

    if len(crop_df) >= 30:
        rolling_std = crop_df['Modal_Price_Median'].rolling(30, min_periods=15).std()
        ax.fill_between(crop_df['Price_Date'], 0, rolling_std,
                        color=color, alpha=0.4)
        ax.plot(crop_df['Price_Date'], rolling_std,
                color=color, linewidth=1.2)
    else:
        ax.text(0.5, 0.5, f'Insufficient data ({len(crop_df)} days < 30)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='#f85149')

    ax.set_ylabel('Std Dev (Rs)')
    ax.set_title(f'{crop}', loc='left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, '08_rolling_volatility.png')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: State/market coverage heatmap (markets reporting per state per crop)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("FIGURE 9: State Coverage Heatmap")
print("-" * 90)

# Use the granular table for this -- FAQ grade only
faq_clean = clean[clean['Grade'] == 'FAQ']
coverage = faq_clean.groupby(['State', 'Commodity'])['Market'].nunique().unstack(fill_value=0)

# Sort states by total market count
coverage['_total'] = coverage.sum(axis=1)
coverage = coverage.sort_values('_total', ascending=True).drop('_total', axis=1)

# Reorder columns
col_order = [c for c in ['Onion', 'Potato', 'Wheat', 'Tomato', 'Rice'] if c in coverage.columns]
coverage = coverage[col_order]

fig, ax = plt.subplots(figsize=(10, max(12, len(coverage) * 0.45)))
fig.suptitle('Number of FAQ-Grade Markets Reporting per State per Crop', y=0.98)

sns.heatmap(coverage, annot=True, fmt='d', cmap='YlOrRd',
            linewidths=0.5, linecolor='#30363d', ax=ax,
            cbar_kws={'shrink': 0.5, 'label': 'Markets'})
ax.set_xlabel('Crop')
ax.set_ylabel('State')

fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, '09_state_coverage_heatmap.png')

print(f"  States in coverage map: {len(coverage)}")
top5 = coverage.sum(axis=1).nlargest(5)
print(f"  Top 5 states by total FAQ markets: {dict(top5)}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Day-of-week pattern
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("FIGURE 10: Day-of-Week Pattern")
print("-" * 90)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Day-of-Week Patterns')

# Panel 1: Markets reporting by day of week
ax = axes[0]
for crop in ['Onion', 'Potato', 'Wheat', 'Tomato', 'Rice']:
    crop_df = daily[daily['Commodity'] == crop].copy()
    crop_df['DOW'] = crop_df['Price_Date'].dt.dayofweek
    dow_markets = crop_df.groupby('DOW')['Markets_Reporting'].mean()
    ax.plot(dow_markets.index, dow_markets.values,
            color=CROP_COLORS[crop], marker='o', linewidth=1.5, label=crop)

ax.set_xticks(range(7))
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_ylabel('Avg Markets Reporting')
ax.set_title('Markets Reporting by Day of Week', loc='left')
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(True, alpha=0.3)

# Panel 2: Price by day of week (Onion + Potato only, enough data)
ax = axes[1]
for crop in ['Onion', 'Potato']:
    crop_df = daily[daily['Commodity'] == crop].copy()
    crop_df['DOW'] = crop_df['Price_Date'].dt.dayofweek
    dow_price = crop_df.groupby('DOW')['Modal_Price_Median'].mean()
    ax.plot(dow_price.index, dow_price.values,
            color=CROP_COLORS[crop], marker='o', linewidth=1.5, label=crop)

ax.set_xticks(range(7))
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_ylabel('Avg Median Modal Price (Rs)')
ax.set_title('Price by Day of Week (Onion, Potato)', loc='left')
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

fig.tight_layout()
save_fig(fig, '10_day_of_week.png')

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE KEY STATISTICS FOR REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 90)
print("KEY STATISTICS FOR REPORT")
print("-" * 90)

for crop in ['Onion', 'Potato', 'Wheat', 'Tomato', 'Rice']:
    crop_df = daily[daily['Commodity'] == crop].sort_values('Price_Date')
    vals = crop_df['Modal_Price_Median']
    print(f"\n  {crop}:")
    print(f"    Days: {len(crop_df)}")
    print(f"    Date range: {crop_df['Price_Date'].min().date()} to "
          f"{crop_df['Price_Date'].max().date()}")
    print(f"    Median Modal_Price_Median: {vals.median():,.0f}")
    print(f"    Min: {vals.min():,.0f}, Max: {vals.max():,.0f}")
    print(f"    Std: {vals.std():,.0f}")
    print(f"    IQR: {vals.quantile(0.25):,.0f} -- {vals.quantile(0.75):,.0f}")

    # Price at start vs end
    first_price = vals.iloc[0]
    last_price = vals.iloc[-1]
    pct_change = ((last_price - first_price) / first_price) * 100
    print(f"    First day: {first_price:,.0f}, Last day: {last_price:,.0f}")
    print(f"    Overall change: {pct_change:+.1f}%")

    # Months with data
    months_present = sorted(crop_df['Price_Date'].dt.month.unique())
    months_missing = sorted(set(range(1, 13)) - set(months_present))
    print(f"    Months present: {[MONTH_NAMES[m-1] for m in months_present]}")
    if months_missing:
        print(f"    Months MISSING: {[MONTH_NAMES[m-1] for m in months_missing]}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print(f"PHASE 2 COMPLETE -- {fig_count} figures saved to reports/figures/")
print("=" * 90)
