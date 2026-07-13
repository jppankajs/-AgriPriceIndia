"""
Phase 3 -- Model Training & Evaluation (v2 -- fixed Onion day-count)
=====================================================================
Tier 1 (Full): Onion, Potato -- SARIMA + Prophet + Holt-Winters
Tier 2 (Limited): Wheat -- SARIMA + Prophet (no seasonal confidence claim)
Tier 3 (Trend-only): Tomato, Rice -- Exponential Smoothing + Moving Average
Primary series: Modal_Price_Median | Split: 80/20 on REAL observations
FIX: Onion has 647 real obs across a 737-day calendar span (90 market-closed
     gaps). Previous run inflated all series via asfreq('D').interpolate().
     Now: Prophet uses real dates only; SARIMA uses asfreq grid but metrics
     are evaluated ONLY on real-observation dates.
"""
import os, warnings, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings('ignore')

BASE = r"d:\Analytics projects\project5\AgriPriceIndia"
DAILY = os.path.join(BASE, "data", "processed", "agri_prices_daily_national.csv")
FIG_DIR = os.path.join(BASE, "reports", "figures")
MDL_DIR = os.path.join(BASE, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor':'#0d1117','axes.facecolor':'#161b22',
    'axes.edgecolor':'#30363d','axes.labelcolor':'#c9d1d9',
    'text.color':'#c9d1d9','xtick.color':'#8b949e','ytick.color':'#8b949e',
    'grid.color':'#21262d','grid.alpha':0.6,'font.size':11,
    'axes.titlesize':13,'axes.titleweight':'bold',
})
COLORS = {'Onion':'#f85149','Potato':'#58a6ff','Wheat':'#d2a8ff',
           'Tomato':'#f0883e','Rice':'#3fb950'}

# ── helpers ──────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_real_series(df, crop):
    """Return the raw daily-national series -- real observations only, no fill."""
    s = df[df['Commodity']==crop].set_index('Price_Date')['Modal_Price_Median'].sort_index()
    s = s[~s.index.duplicated(keep='first')]          # safety
    return s

def get_regular_series(real_series):
    """Return asfreq('D') + linear interpolation for SARIMA fitting only."""
    return real_series.asfreq('D').interpolate('linear')

def split_real(series, test_frac=0.20):
    """80/20 split on real observations."""
    n = len(series)
    cut = int(n * (1 - test_frac))
    return series.iloc[:cut], series.iloc[cut:]

def eval_on_real(test_real, pred_series):
    """Align predictions to real-observation dates only, compute metrics."""
    common = test_real.index.intersection(pred_series.index)
    if len(common) == 0:
        return None
    y_true = test_real.loc[common]
    y_pred = pred_series.loc[common]
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'n_eval': len(common),
    }

def plot_comparison(crop, train, test, preds_dict, suffix, title_extra=''):
    """Actual-vs-predicted plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train.index, train.values, color=COLORS[crop], alpha=0.4, lw=0.8, label='Train')
    ax.plot(test.index, test.values, color='#f0f6fc', lw=1.5, label='Actual (Test)')
    dash_colors = {'SARIMA':'#ffa657','Prophet':'#79c0ff','Holt-Winters':'#d2a8ff',
                   'ExpSmoothing':'#ffa657','SMA':'#79c0ff'}
    for name, pred in preds_dict.items():
        if pred is not None:
            c = dash_colors.get(name.split('(')[0].split('-')[0].strip(), '#ffa657')
            ax.plot(pred.index, pred.values, '--', color=c, lw=1.3, label=name)
    ax.axvline(test.index[0], color='#8b949e', ls=':', alpha=0.7, label='Split')
    title_color = '#ffa657' if title_extra else '#c9d1d9'
    ax.set_title(f'{crop} -- Actual vs Predicted {title_extra}', loc='left', color=title_color)
    ax.set_ylabel('Rs/Quintal'); ax.legend(fontsize=9, framealpha=0.3); ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, suffix), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {suffix}")

# ── load data ────────────────────────────────────────────────────────────
print("=" * 90)
print("PHASE 3: MODEL TRAINING & EVALUATION  (v2 -- real-observation splits)")
print("=" * 90)

daily = pd.read_csv(DAILY, parse_dates=['Price_Date'])
results = []   # list of dicts

from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ═════════════════════════════════════════════════════════════════════════
# TIER 1: Onion & Potato -- SARIMA + Prophet + Holt-Winters
# ═════════════════════════════════════════════════════════════════════════
for crop in ['Onion', 'Potato']:
    print(f"\n{'='*90}")
    print(f"TIER 1 -- {crop.upper()}")
    print(f"{'='*90}")

    real = get_real_series(daily, crop)
    train_real, test_real = split_real(real)
    n_real_train, n_real_test = len(train_real), len(test_real)
    cal_span = (real.index.max() - real.index.min()).days + 1
    gap_days = cal_span - len(real)
    print(f"  Real obs: {len(real)} | Calendar span: {cal_span} | Gaps: {gap_days}")
    print(f"  Train: {n_real_train} real obs | Test: {n_real_test} real obs")

    preds = {}

    # ── SARIMA (needs regular freq, evaluate on real dates) ──
    print(f"\n  [1/3] SARIMA (m=7 weekly)...")
    try:
        # Build regular grid from ALL real data's date range, fit on train portion
        reg_full = get_regular_series(real)
        train_cut_date = train_real.index[-1]
        reg_train = reg_full.loc[:train_cut_date]
        n_reg_test = len(reg_full) - len(reg_train)

        sarima_model = auto_arima(
            reg_train, seasonal=True, m=7,
            max_p=3, max_q=3, max_P=2, max_Q=2, max_d=2, max_D=1,
            stepwise=True, suppress_warnings=True, error_action='ignore', n_fits=40
        )
        sarima_pred_all = sarima_model.predict(n_periods=n_reg_test)
        sarima_pred_all = pd.Series(sarima_pred_all,
                                     index=reg_full.index[len(reg_train):])
        # Evaluate ONLY on real-observation dates
        metrics = eval_on_real(test_real, sarima_pred_all)
        preds['SARIMA'] = sarima_pred_all.reindex(test_real.index).dropna()
        r = {'Crop':crop, 'Model':'SARIMA',
             'RMSE':metrics['RMSE'], 'MAE':metrics['MAE'], 'MAPE':metrics['MAPE'],
             'n_train':n_real_train, 'n_test':metrics['n_eval']}
        results.append(r)
        print(f"    Order: {sarima_model.order}, Seasonal: {sarima_model.seasonal_order}")
        print(f"    Eval on {metrics['n_eval']} real dates: RMSE={r['RMSE']:.1f}, MAE={r['MAE']:.1f}, MAPE={r['MAPE']:.2f}%")
        joblib.dump(sarima_model, os.path.join(MDL_DIR, f'sarima_{crop.lower()}.pkl'))
    except Exception as e:
        print(f"    SARIMA FAILED: {e}")
        preds['SARIMA'] = None

    # ── Prophet (handles irregular dates natively) ──
    print(f"\n  [2/3] Prophet (default cp=0.05)...")
    try:
        prophet_df = train_real.reset_index().rename(
            columns={'Price_Date':'ds','Modal_Price_Median':'y'})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, changepoint_prior_scale=0.05)
        m.fit(prophet_df)
        # Predict on test_real dates specifically
        future = pd.DataFrame({'ds': test_real.index})
        forecast = m.predict(future)
        prophet_pred = pd.Series(forecast['yhat'].values, index=test_real.index)
        metrics = eval_on_real(test_real, prophet_pred)
        preds['Prophet'] = prophet_pred
        r = {'Crop':crop, 'Model':'Prophet',
             'RMSE':metrics['RMSE'], 'MAE':metrics['MAE'], 'MAPE':metrics['MAPE'],
             'n_train':n_real_train, 'n_test':metrics['n_eval']}
        results.append(r)
        print(f"    Eval on {metrics['n_eval']} real dates: RMSE={r['RMSE']:.1f}, MAE={r['MAE']:.1f}, MAPE={r['MAPE']:.2f}%")
        joblib.dump(m, os.path.join(MDL_DIR, f'prophet_{crop.lower()}.pkl'))
    except Exception as e:
        print(f"    Prophet FAILED: {e}")
        preds['Prophet'] = None

    # ── Holt-Winters ──
    print(f"\n  [3/3] Holt-Winters (additive, period=365)...")
    try:
        reg_train_hw = get_regular_series(train_real)
        hw_model = ExponentialSmoothing(
            reg_train_hw, trend='add', seasonal='add',
            seasonal_periods=365, initialization_method='estimated'
        ).fit(optimized=True)
        n_forecast = (test_real.index[-1] - train_real.index[-1]).days
        hw_pred_all = hw_model.forecast(n_forecast)
        hw_pred_all.index = pd.date_range(train_real.index[-1] + pd.Timedelta(days=1),
                                           periods=n_forecast, freq='D')
        metrics = eval_on_real(test_real, hw_pred_all)
        if metrics:
            preds['Holt-Winters'] = hw_pred_all.reindex(test_real.index).dropna()
            r = {'Crop':crop, 'Model':'Holt-Winters',
                 'RMSE':metrics['RMSE'], 'MAE':metrics['MAE'], 'MAPE':metrics['MAPE'],
                 'n_train':n_real_train, 'n_test':metrics['n_eval']}
            results.append(r)
            print(f"    Eval on {metrics['n_eval']} real dates: RMSE={r['RMSE']:.1f}, MAE={r['MAE']:.1f}, MAPE={r['MAPE']:.2f}%")
            joblib.dump(hw_model, os.path.join(MDL_DIR, f'holtwinters_{crop.lower()}.pkl'))
        else:
            print(f"    Holt-Winters: no overlapping real dates in forecast range")
            preds['Holt-Winters'] = None
    except Exception as e:
        print(f"    Holt-Winters FAILED: {e}")
        preds['Holt-Winters'] = None

    plot_comparison(crop, train_real, test_real, preds,
                    f'11_model_comparison_{crop.lower()}.png')

# ═════════════════════════════════════════════════════════════════════════
# TIER 2: Wheat -- SARIMA + Prophet (no seasonal confidence)
# ═════════════════════════════════════════════════════════════════════════
crop = 'Wheat'
print(f"\n{'='*90}")
print(f"TIER 2 -- {crop.upper()} (limited; no seasonal confidence claim)")
print(f"{'='*90}")

real = get_real_series(daily, crop)
train_real, test_real = split_real(real)
print(f"  Real obs: {len(real)} | Train: {len(train_real)} | Test: {len(test_real)}")
print(f"  WARNING: Mar-May missing. No annual seasonal claim possible.")

preds = {}

# SARIMA
print(f"\n  [1/2] SARIMA (non-seasonal or weekly)...")
try:
    reg_train = get_regular_series(train_real)
    sarima_model = auto_arima(
        reg_train, seasonal=True, m=7,
        max_p=3, max_q=3, max_P=1, max_Q=1, max_d=2, max_D=1,
        stepwise=True, suppress_warnings=True, error_action='ignore'
    )
    n_forecast = (test_real.index[-1] - train_real.index[-1]).days
    sarima_pred_all = sarima_model.predict(n_periods=n_forecast)
    sarima_pred_all = pd.Series(sarima_pred_all,
        index=pd.date_range(train_real.index[-1]+pd.Timedelta(days=1),
                            periods=n_forecast, freq='D'))
    metrics = eval_on_real(test_real, sarima_pred_all)
    preds['SARIMA'] = sarima_pred_all.reindex(test_real.index).dropna()
    r = {'Crop':crop, 'Model':'SARIMA (no season)',
         'RMSE':metrics['RMSE'], 'MAE':metrics['MAE'], 'MAPE':metrics['MAPE'],
         'n_train':len(train_real), 'n_test':metrics['n_eval']}
    results.append(r)
    print(f"    Order: {sarima_model.order}, Seasonal: {sarima_model.seasonal_order}")
    print(f"    Eval on {metrics['n_eval']} real dates: RMSE={r['RMSE']:.1f}, MAE={r['MAE']:.1f}, MAPE={r['MAPE']:.2f}%")
    joblib.dump(sarima_model, os.path.join(MDL_DIR, f'sarima_{crop.lower()}.pkl'))
except Exception as e:
    print(f"    SARIMA FAILED: {e}")
    preds['SARIMA'] = None

# Prophet
print(f"\n  [2/2] Prophet (yearly_seasonality=False)...")
try:
    prophet_df = train_real.reset_index().rename(
        columns={'Price_Date':'ds','Modal_Price_Median':'y'})
    m = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                daily_seasonality=False, changepoint_prior_scale=0.05)
    m.fit(prophet_df)
    future = pd.DataFrame({'ds': test_real.index})
    forecast = m.predict(future)
    prophet_pred = pd.Series(forecast['yhat'].values, index=test_real.index)
    metrics = eval_on_real(test_real, prophet_pred)
    preds['Prophet'] = prophet_pred
    r = {'Crop':crop, 'Model':'Prophet (no season)',
         'RMSE':metrics['RMSE'], 'MAE':metrics['MAE'], 'MAPE':metrics['MAPE'],
         'n_train':len(train_real), 'n_test':metrics['n_eval']}
    results.append(r)
    print(f"    Eval on {metrics['n_eval']} real dates: RMSE={r['RMSE']:.1f}, MAE={r['MAE']:.1f}, MAPE={r['MAPE']:.2f}%")
    joblib.dump(m, os.path.join(MDL_DIR, f'prophet_{crop.lower()}.pkl'))
except Exception as e:
    print(f"    Prophet FAILED: {e}")
    preds['Prophet'] = None

plot_comparison(crop, train_real, test_real, preds,
                '12_model_comparison_wheat.png',
                title_extra='[NO seasonal confidence]')

# ═════════════════════════════════════════════════════════════════════════
# TIER 3: Tomato & Rice -- trend-only
# ═════════════════════════════════════════════════════════════════════════
for crop in ['Tomato', 'Rice']:
    print(f"\n{'='*90}")
    print(f"TIER 3 -- {crop.upper()} (trend-only; no seasonal term)")
    print(f"{'='*90}")

    real = get_real_series(daily, crop)
    train_real, test_real = split_real(real)
    print(f"  Real obs: {len(real)} | Train: {len(train_real)} | Test: {len(test_real)}")

    preds = {}

    # Exponential Smoothing (trend only)
    print(f"\n  [1/2] Exponential Smoothing (trend only)...")
    try:
        reg_train = get_regular_series(train_real)
        ets_model = ExponentialSmoothing(
            reg_train, trend='add', seasonal=None,
            initialization_method='estimated'
        ).fit(optimized=True)
        n_forecast = (test_real.index[-1] - train_real.index[-1]).days
        ets_pred_all = ets_model.forecast(n_forecast)
        ets_pred_all.index = pd.date_range(train_real.index[-1]+pd.Timedelta(days=1),
                                            periods=n_forecast, freq='D')
        metrics = eval_on_real(test_real, ets_pred_all)
        preds['ExpSmoothing'] = ets_pred_all.reindex(test_real.index).dropna()
        r = {'Crop':crop, 'Model':'ExpSmoothing (trend)',
             'RMSE':metrics['RMSE'], 'MAE':metrics['MAE'], 'MAPE':metrics['MAPE'],
             'n_train':len(train_real), 'n_test':metrics['n_eval']}
        results.append(r)
        print(f"    Eval on {metrics['n_eval']} real dates: RMSE={r['RMSE']:.1f}, MAE={r['MAE']:.1f}, MAPE={r['MAPE']:.2f}%")
        joblib.dump(ets_model, os.path.join(MDL_DIR, f'ets_{crop.lower()}.pkl'))
    except Exception as e:
        print(f"    ETS FAILED: {e}")
        preds['ExpSmoothing'] = None

    # Simple Moving Average
    window = min(14, len(train_real) // 3)
    print(f"\n  [2/2] Simple Moving Average (window={window} days)...")
    try:
        sma_value = train_real.iloc[-window:].mean()
        sma_pred = pd.Series(sma_value, index=test_real.index)
        metrics = eval_on_real(test_real, sma_pred)
        preds[f'SMA-{window}'] = sma_pred
        r = {'Crop':crop, 'Model':f'SMA-{window}',
             'RMSE':metrics['RMSE'], 'MAE':metrics['MAE'], 'MAPE':metrics['MAPE'],
             'n_train':len(train_real), 'n_test':metrics['n_eval']}
        results.append(r)
        print(f"    Constant forecast: {sma_value:.0f}")
        print(f"    Eval on {metrics['n_eval']} real dates: RMSE={r['RMSE']:.1f}, MAE={r['MAE']:.1f}, MAPE={r['MAPE']:.2f}%")
    except Exception as e:
        print(f"    SMA FAILED: {e}")
        preds[f'SMA-{window}'] = None

    plot_comparison(crop, train_real, test_real, preds,
                    f'13_model_trend_{crop.lower()}.png',
                    title_extra='[insufficient history for seasonality]')

# ═════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("MODEL COMPARISON TABLE  (evaluated on real observations only)")
print(f"{'='*90}")

res_df = pd.DataFrame(results)
res_df = res_df.sort_values(['Crop', 'MAPE'])

best = {}
for crop_name in res_df['Crop'].unique():
    crop_rows = res_df[res_df['Crop'] == crop_name]
    best_row = crop_rows.loc[crop_rows['MAPE'].idxmin()]
    best[crop_name] = best_row['Model']

print(f"\n{'Crop':<10} {'Model':<25} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'Train':>7} {'Test':>6}  {'Best?':<6}")
print("-" * 85)
for _, row in res_df.iterrows():
    is_best = '*BEST*' if best[row['Crop']] == row['Model'] else ''
    print(f"{row['Crop']:<10} {row['Model']:<25} {row['RMSE']:>10.1f} {row['MAE']:>10.1f} {row['MAPE']:>9.2f}% {int(row['n_train']):>7} {int(row['n_test']):>6}  {is_best}")

print(f"\n{'='*90}")
print("BEST MODEL PER CROP")
print(f"{'='*90}")
for crop_name, model_name in best.items():
    row = res_df[(res_df['Crop']==crop_name) & (res_df['Model']==model_name)].iloc[0]
    print(f"  {crop_name}: {model_name} (MAPE={row['MAPE']:.2f}%, eval on {int(row['n_test'])} real obs)")

res_df.to_csv(os.path.join(BASE, "reports", "phase3_model_results.csv"), index=False)
with open(os.path.join(MDL_DIR, "best_models.json"), 'w') as f:
    json.dump(best, f, indent=2)
print(f"\n  Saved: reports/phase3_model_results.csv")
print(f"  Saved: models/best_models.json")

print(f"\n{'='*90}")
print("PHASE 3 COMPLETE (v2)")
print(f"{'='*90}")
