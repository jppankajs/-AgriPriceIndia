"""Quick diagnostic: Potato calendar dates and trend direction for Issue 3."""
import pandas as pd

daily = pd.read_csv(
    r'd:\Analytics projects\project5\AgriPriceIndia\data\processed\agri_prices_daily_national.csv',
    parse_dates=['Price_Date']
)

potato = daily[daily['Commodity']=='Potato'].set_index('Price_Date')['Modal_Price_Median'].sort_index()
potato = potato[~potato.index.duplicated(keep='first')]

print('=== POTATO DATE CHECK (Issue 3) ===')
print(f'Full range: {potato.index[0].date()} to {potato.index[-1].date()}')
print(f'Last price: Rs {potato.iloc[-1]:,.0f} on {potato.index[-1].date()}')
fc_end = potato.index[-1] + pd.Timedelta(weeks=8)
print(f'8-week forecast covers: {potato.index[-1].date()} to {fc_end.date()}')
print(f'  i.e. mid-June to early August 2025')
print()

# Monthly pattern
monthly = potato.groupby(potato.index.month).agg(['mean','min','max'])
print('Monthly historical pattern:')
for m in range(1, 13):
    if m in monthly.index:
        row = monthly.loc[m]
        print(f'  Month {m:2d}: mean=Rs {row["mean"]:,.0f}  range=[{row["min"]:,.0f}, {row["max"]:,.0f}]')
    else:
        print(f'  Month {m:2d}: NO DATA')

# Last 30 days detail
print()
print('Last 20 data points:')
for date, price in potato.iloc[-20:].items():
    print(f'  {date.date()} ({date.strftime("%a")}): Rs {price:,.0f}')

# Trend direction
last_30 = potato.iloc[-30:]
first_half = last_30.iloc[:15].mean()
second_half = last_30.iloc[-15:].mean()
print()
print(f'Recent trend (last 30 obs):')
print(f'  First 15 avg: Rs {first_half:,.0f}')
print(f'  Last 15 avg:  Rs {second_half:,.0f}')
direction = 'RISING' if second_half > first_half else 'FALLING'
pct = (second_half - first_half) / first_half * 100
print(f'  Direction: {direction} ({pct:+.1f}%)')

# Jul-Aug specifically
print()
print('Focus: What does Potato do in Jun/Jul/Aug historically?')
for m in [5, 6, 7, 8, 9, 10, 11]:
    if m in monthly.index:
        row = monthly.loc[m]
        name = ['', 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m]
        print(f'  {name}: mean=Rs {row["mean"]:,.0f}  range=[{row["min"]:,.0f}, {row["max"]:,.0f}]')

# Year-over-year: what did Potato do in Jul-Aug in each year?
print()
print('Year-by-year Jul-Aug prices:')
for yr in potato.index.year.unique():
    jul_aug = potato[(potato.index.year == yr) & (potato.index.month.isin([7, 8]))]
    if len(jul_aug) > 0:
        print(f'  {yr}: {len(jul_aug)} obs, mean=Rs {jul_aug.mean():,.0f}, range=[{jul_aug.min():,.0f}, {jul_aug.max():,.0f}]')
    else:
        print(f'  {yr}: no Jul-Aug data')
