"""Onion forecast vs historical ceiling check for Issue 1 report-back."""
import pandas as pd, numpy as np, joblib, warnings
warnings.filterwarnings('ignore')

daily = pd.read_csv(
    r'd:\Analytics projects\project5\AgriPriceIndia\data\processed\agri_prices_daily_national.csv',
    parse_dates=['Price_Date']
)
onion = daily[daily['Commodity']=='Onion'].set_index('Price_Date')['Modal_Price_Median'].sort_index()
onion = onion[~onion.index.duplicated(keep='first')]

model = joblib.load(r'd:\Analytics projects\project5\AgriPriceIndia\models\sarima_onion.pkl')
pred, ci = model.predict(n_periods=56, return_conf_int=True)

last_date = onion.index[-1]
dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=56, freq='D')

print('=== ONION SARIMA FORECAST vs HISTORICAL CEILING ===')
print(f'Current price: Rs {onion.iloc[-1]:,.0f} (on {last_date.date()})')
print(f'Historical max: Rs {onion.max():,.0f} (on {onion.idxmax().date()})')
print(f'Historical ceiling range (Phase 2): Rs 3,800-4,360')
print()

# Weekly forecast
for wk in range(8):
    start = wk * 7
    end = min(start + 7, 56)
    wk_pred = np.array(pred[start:end], dtype=float)
    wk_lo = np.array(ci[start:end, 0], dtype=float)
    wk_hi = np.array(ci[start:end, 1], dtype=float)
    print(f'  Week {wk+1} ({dates[start].date()} - {dates[min(end-1, 55)].date()}):')
    print(f'    Forecast: Rs {wk_pred.mean():,.0f}  CI: [{wk_lo.mean():,.0f}, {wk_hi.mean():,.0f}]')

fc_end = float(pred.values[-1]) if hasattr(pred, 'values') else float(pred[-1])
print()
print(f'Endpoint forecast: Rs {fc_end:,.0f}')
print(f'  vs historical max: Rs {onion.max():,.0f}')
exceeds = fc_end > onion.max()
print(f'  Exceeds historical ceiling? {"YES" if exceeds else "NO"}')
if exceeds:
    print(f'  By: {(fc_end - onion.max()) / onion.max() * 100:+.1f}%')
else:
    print(f'  Below ceiling by: {(onion.max() - fc_end) / onion.max() * 100:.1f}%')

# Monthly pattern for Jul-Aug context
monthly = onion.groupby(onion.index.month).agg(['mean','min','max'])
print()
print('Onion monthly pattern (for forecast context):')
for m in [6, 7, 8, 9, 10, 11]:
    if m in monthly.index:
        row = monthly.loc[m]
        name = ['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m]
        print(f'  {name}: mean=Rs {row["mean"]:,.0f}  range=[{row["min"]:,.0f}, {row["max"]:,.0f}]')
