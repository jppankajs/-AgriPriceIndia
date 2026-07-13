"""Check what Potato Prophet is actually forecasting and why."""
import pandas as pd, numpy as np, warnings, joblib
warnings.filterwarnings('ignore')

daily = pd.read_csv(
    r'd:\Analytics projects\project5\AgriPriceIndia\data\processed\agri_prices_daily_national.csv',
    parse_dates=['Price_Date']
)
potato = daily[daily['Commodity']=='Potato'].set_index('Price_Date')['Modal_Price_Median'].sort_index()
potato = potato[~potato.index.duplicated(keep='first')]

model = joblib.load(r'd:\Analytics projects\project5\AgriPriceIndia\models\prophet_potato.pkl')

# Forecast 56 days from last date
last_date = potato.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=56, freq='D')
future = pd.DataFrame({'ds': future_dates})
fc = model.predict(future)

print('=== POTATO PROPHET FORECAST DETAIL ===')
print(f'Forecasting from: {last_date.date()} (current: Rs {potato.iloc[-1]:,.0f})')
print(f'Forecasting to:   {future_dates[-1].date()}')
print()

# Weekly summary
fc['week'] = (fc.index // 7) + 1
for wk in range(1, 9):
    wk_data = fc[fc['week'] == wk]
    if len(wk_data) > 0:
        print(f'  Week {wk} ({wk_data["ds"].iloc[0].date()} - {wk_data["ds"].iloc[-1].date()}):')
        print(f'    Forecast: Rs {wk_data["yhat"].mean():,.0f}  CI: [{wk_data["yhat_lower"].mean():,.0f}, {wk_data["yhat_upper"].mean():,.0f}]')

print()
print(f'Endpoint forecast: Rs {fc["yhat"].iloc[-1]:,.0f}')
print(f'CI at endpoint: [{fc["yhat_lower"].iloc[-1]:,.0f}, {fc["yhat_upper"].iloc[-1]:,.0f}]')
print()

# Compare to historical: what was the ACTUAL price in Jul-Aug in prior years?
print('Historical reality check:')
print(f'  Jul-Aug 2023: mean Rs 1,238  (range 1,180-1,286)')
print(f'  Jul-Aug 2024: mean Rs 2,330  (range 2,200-2,400)')
print(f'  Forecast Jul-Aug 2025: mean Rs {fc["yhat"].mean():,.0f}  (range {fc["yhat"].min():,.0f}-{fc["yhat"].max():,.0f})')
print()

# Check: Is Prophet learning a downtrend from the late-2024 decline?
# Look at what happened Oct 2024 -> Jun 2025
oct_on = potato[potato.index >= '2024-10-01']
print(f'Potato Oct 2024 -> Jun 2025 trajectory:')
for date, price in oct_on.iloc[::15].items():  # sample every 15th obs
    print(f'  {date.date()}: Rs {price:,.0f}')
print(f'  (last: {oct_on.index[-1].date()}: Rs {oct_on.iloc[-1]:,.0f})')
print(f'  Oct peak: Rs {oct_on.max():,.0f} -> Jun current: Rs {potato.iloc[-1]:,.0f}')
print(f'  Decline from peak: {(potato.iloc[-1] - oct_on.max()) / oct_on.max() * 100:+.1f}%')
