"""
Phase 4 -- Decision-Support Layer
===================================
Generates forecasts using best model per crop, calculates confidence intervals,
and produces Sell / Hold / Monitor signals.

Amendments layered on original spec:
  - Amendment 5: Tomato AND Rice get unconditional "Monitor -- insufficient
    stable history" (Rice extended per distinct documented reason).
  - Amendment 6: Threshold scales with each crop's own model error:
      threshold_pct = max(5%, 1.5 x crop's Phase 3 test MAPE)
    Only Onion, Potato, Wheat use the normal signal logic.
  - Carry-forward: Tomato's Monitor is unconditional, not interval-width gated.
  - Amendment 7 (one-off override): Potato's dynamic-threshold Sell signal
    is overridden to Monitor.  Reason: Prophet's trend component extrapolates
    the Oct 2024-Feb 2025 crash past its documented reversal; four months of
    recovery (Rs 1,000 -> Rs 1,228, Feb-Jun 2025) are not reflected in the
    forecast (Rs 770).  This is a one-off override for this specific
    documented case, NOT a change to the general dynamic-threshold rule.
  - Amendment 8 (point-estimate suppression): Onion and Potato forecasts
    must not display a single point estimate prominently in the Phase 5
    dashboard; show direction/range only (CI band + qualitative signal).

Signal logic (for Onion, Potato, Wheat only):
  forecast_move = (forecast_end - current_price) / current_price * 100
  if abs(forecast_move) < threshold_pct:  => Monitor (noise-level move)
  elif forecast_move > threshold_pct:     => Hold   (prices rising meaningfully)
  elif forecast_move < -threshold_pct:    => Sell   (prices falling meaningfully)
  then: apply any per-crop signal overrides from SIGNAL_OVERRIDES dict.
"""
import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import joblib

warnings.filterwarnings('ignore')

BASE = r"d:\Analytics projects\project5\AgriPriceIndia"
DAILY = os.path.join(BASE, "data", "processed", "agri_prices_daily_national.csv")
FIG_DIR = os.path.join(BASE, "reports", "figures")
MDL_DIR = os.path.join(BASE, "models")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor':'#0d1117','axes.facecolor':'#161b22',
    'axes.edgecolor':'#30363d','axes.labelcolor':'#c9d1d9',
    'text.color':'#c9d1d9','xtick.color':'#8b949e','ytick.color':'#8b949e',
    'grid.color':'#21262d','grid.alpha':0.6,'font.size':11,
    'axes.titlesize':13,'axes.titleweight':'bold',
})
COLORS = {'Onion':'#f85149','Potato':'#58a6ff','Wheat':'#d2a8ff',
           'Tomato':'#f0883e','Rice':'#3fb950'}

daily = pd.read_csv(DAILY, parse_dates=['Price_Date'])

# ── Phase 3 results ─────────────────────────────────────────────────────
with open(os.path.join(MDL_DIR, "best_models.json")) as f:
    best_models = json.load(f)

phase3_results = pd.read_csv(os.path.join(BASE, "reports", "phase3_model_results.csv"))

# Build MAPE lookup for best model per crop
mape_lookup = {}
for _, row in phase3_results.iterrows():
    key = (row['Crop'], row['Model'])
    if best_models.get(row['Crop']) == row['Model']:
        mape_lookup[row['Crop']] = row['MAPE']

print("=" * 90)
print("PHASE 4: DECISION-SUPPORT LAYER")
print("=" * 90)

# ── Amendment 6: compute thresholds ────────────────────────────────────
print("\n--- Amendment 6: Crop-specific thresholds ---")
thresholds = {}
for crop in ['Onion', 'Potato', 'Wheat']:
    crop_mape = mape_lookup[crop]
    thr = max(5.0, 1.5 * crop_mape)
    thresholds[crop] = thr
    print(f"  {crop}: test_MAPE={crop_mape:.2f}% => threshold = max(5%, 1.5x{crop_mape:.2f}%) = {thr:.1f}%")

# ── Amendment 5: unconditional Monitor crops ───────────────────────────
UNCONDITIONAL_MONITOR = {
    'Tomato': 'Monitor -- insufficient stable history (154-day window dominated by 2023 monsoon crisis anomaly)',
    'Rice':   'Monitor -- insufficient stable history (62 total days; 4-6 week forecast is largest proportional extrapolation in dataset)',
}

# Amendment 7: one-off signal overrides (crop -> (new_signal, override_rationale))
# This is NOT a general rule; each entry requires a specific documented justification.
SIGNAL_OVERRIDES = {
    'Potato': (
        'Monitor',
        'Override: model likely extrapolating a since-reversed downtrend; '
        'recent recovery pattern (Rs 1000 to Rs 1228 over Feb-Jun 2025) not reflected in the forecast',
    ),
}

# Amendment 8: crops whose point-estimate forecast should be suppressed in Phase 5
# dashboard (show direction/CI band only, not a single number prominently)
POINT_ESTIMATE_SUPPRESS = {'Onion', 'Potato'}

FORECAST_WEEKS = 8  # 8-week horizon = 56 days

# ── Helper functions ───────────────────────────────────────────────────
def get_real_series(df, crop):
    s = df[df['Commodity']==crop].set_index('Price_Date')['Modal_Price_Median'].sort_index()
    return s[~s.index.duplicated(keep='first')]

def forecast_prophet(model, last_date, n_days):
    """Generate Prophet forecast with confidence intervals."""
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                  periods=n_days, freq='D')
    future = pd.DataFrame({'ds': future_dates})
    fc = model.predict(future)
    return pd.DataFrame({
        'date': future_dates,
        'forecast': fc['yhat'].values,
        'lower': fc['yhat_lower'].values,
        'upper': fc['yhat_upper'].values,
    }).set_index('date')

def forecast_sarima(model, n_days):
    """Generate SARIMA forecast with confidence intervals."""
    pred, ci = model.predict(n_periods=n_days, return_conf_int=True)
    dates = pd.date_range(pd.Timestamp.now().normalize(), periods=n_days, freq='D')
    return pd.DataFrame({
        'forecast': pred,
        'lower': ci[:, 0],
        'upper': ci[:, 1],
    }, index=dates)

def forecast_ets(model, n_days, last_date):
    """Generate ETS forecast (no CI from statsmodels, use +/- train RMSE)."""
    pred = model.forecast(n_days)
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days, freq='D')
    return pd.DataFrame({'forecast': pred.values}, index=dates)

def compute_signal(current_price, forecast_end, threshold_pct):
    """Determine Sell / Hold / Monitor based on forecast move vs threshold."""
    move_pct = (forecast_end - current_price) / current_price * 100
    if abs(move_pct) < threshold_pct:
        signal = 'Monitor'
        rationale = f"Predicted move ({move_pct:+.1f}%) is below noise threshold ({threshold_pct:.1f}%)"
    elif move_pct > 0:
        signal = 'Hold'
        rationale = f"Prices predicted to rise {move_pct:+.1f}% (above {threshold_pct:.1f}% threshold)"
    else:
        signal = 'Sell'
        rationale = f"Prices predicted to fall {move_pct:+.1f}% (below -{threshold_pct:.1f}% threshold)"
    return signal, rationale, move_pct

# ── Generate forecasts and signals ─────────────────────────────────────
decision_table = []
n_forecast_days = FORECAST_WEEKS * 7  # 56 days

for crop in ['Onion', 'Potato', 'Wheat', 'Tomato', 'Rice']:
    print(f"\n{'='*90}")
    print(f"{crop.upper()}")
    print(f"{'='*90}")

    series = get_real_series(daily, crop)
    current_price = series.iloc[-1]
    last_date = series.index[-1]
    print(f"  Current price (last obs {last_date.date()}): Rs {current_price:,.0f}/quintal")

    # ── Unconditional Monitor (Amendment 5) ──
    if crop in UNCONDITIONAL_MONITOR:
        print(f"  SIGNAL: {UNCONDITIONAL_MONITOR[crop]}")
        decision_table.append({
            'Crop': crop,
            'Current_Price': current_price,
            'Forecast_End': np.nan,
            'Move_Pct': np.nan,
            'Threshold_Pct': np.nan,
            'Signal': 'Monitor',
            'Rationale': UNCONDITIONAL_MONITOR[crop],
            'Model': best_models[crop],
            'Model_MAPE': mape_lookup[crop],
        })

        # Still generate a lightweight plot for context
        model_name = best_models[crop]
        try:
            if 'ExpSmoothing' in model_name or 'ETS' in model_name:
                model = joblib.load(os.path.join(MDL_DIR, f'ets_{crop.lower()}.pkl'))
                fc_df = forecast_ets(model, n_forecast_days, last_date)
            elif 'SMA' in model_name:
                window = int(model_name.split('-')[1]) if '-' in model_name else 14
                sma_val = series.iloc[-window:].mean()
                dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                       periods=n_forecast_days, freq='D')
                fc_df = pd.DataFrame({'forecast': sma_val}, index=dates)
            else:
                fc_df = None

            if fc_df is not None:
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(series.index, series.values, color=COLORS[crop], lw=1, label='Historical')
                ax.plot(fc_df.index, fc_df['forecast'], '--', color='#8b949e', lw=1.3,
                        label=f'{model_name} (trend only)')
                ax.axvline(last_date, color='#f0883e', ls=':', alpha=0.8)
                ax.set_title(f'{crop} -- Monitor (insufficient history)', loc='left', color='#f0883e')
                ax.set_ylabel('Rs/Quintal'); ax.legend(fontsize=9, framealpha=0.3)
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
                fig.tight_layout()
                fig.savefig(os.path.join(FIG_DIR, f'14_forecast_{crop.lower()}.png'),
                            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
                plt.close(fig)
                print(f"  Saved: 14_forecast_{crop.lower()}.png")
        except Exception as e:
            print(f"  Plot skipped: {e}")
        continue

    # ── Normal signal logic (Onion, Potato, Wheat) ──
    model_name = best_models[crop]
    threshold = thresholds[crop]
    print(f"  Best model: {model_name} | Threshold: {threshold:.1f}%")

    try:
        if 'Prophet' in model_name:
            model = joblib.load(os.path.join(MDL_DIR, f'prophet_{crop.lower()}.pkl'))
            fc_df = forecast_prophet(model, last_date, n_forecast_days)
        elif 'SARIMA' in model_name:
            model = joblib.load(os.path.join(MDL_DIR, f'sarima_{crop.lower()}.pkl'))
            pred, ci = model.predict(n_periods=n_forecast_days, return_conf_int=True)
            # Use .values to detach from SARIMA's internal DatetimeIndex
            dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                   periods=n_forecast_days, freq='D')
            fc_df = pd.DataFrame({
                'forecast': np.array(pred, dtype=float),
                'lower': np.array(ci[:, 0], dtype=float),
                'upper': np.array(ci[:, 1], dtype=float),
            }, index=dates)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        forecast_end = fc_df['forecast'].iloc[-1]
        signal, rationale, move_pct = compute_signal(current_price, forecast_end, threshold)

        # Amendment 7: apply one-off signal overrides
        if crop in SIGNAL_OVERRIDES:
            orig_signal = signal
            signal, rationale = SIGNAL_OVERRIDES[crop]
            print(f"  ** Override applied: {orig_signal} -> {signal} **")

        # Confidence interval width at end
        if 'lower' in fc_df.columns and 'upper' in fc_df.columns:
            ci_width = fc_df['upper'].iloc[-1] - fc_df['lower'].iloc[-1]
            ci_pct = ci_width / current_price * 100
        else:
            ci_width, ci_pct = np.nan, np.nan

        print(f"  Forecast end ({FORECAST_WEEKS}wk): Rs {forecast_end:,.0f}")
        print(f"  Move: {move_pct:+.1f}% vs threshold {threshold:.1f}%")
        if not np.isnan(ci_pct):
            print(f"  CI width at end: Rs {ci_width:,.0f} ({ci_pct:.1f}% of current)")
        print(f"  SIGNAL: {signal}")
        print(f"  Rationale: {rationale}")
        if crop in POINT_ESTIMATE_SUPPRESS:
            print(f"  Note: point-estimate suppressed for Phase 5 display (Amendment 8)")

        decision_table.append({
            'Crop': crop,
            'Current_Price': current_price,
            'Forecast_End': forecast_end,
            'Move_Pct': move_pct,
            'Threshold_Pct': threshold,
            'Signal': signal,
            'Rationale': rationale,
            'Model': model_name,
            'Model_MAPE': mape_lookup[crop],
            'Suppress_Point_Estimate': crop in POINT_ESTIMATE_SUPPRESS,
        })

        # ── Forecast plot ──
        fig, ax = plt.subplots(figsize=(14, 6))
        # Last 90 days of historical
        hist_tail = series.iloc[-90:]
        ax.plot(hist_tail.index, hist_tail.values, color=COLORS[crop], lw=1.2, label='Historical')
        ax.plot(fc_df.index, fc_df['forecast'], '--', color='#f0f6fc', lw=1.5, label='Forecast')
        if 'lower' in fc_df.columns:
            ax.fill_between(fc_df.index, fc_df['lower'], fc_df['upper'],
                            alpha=0.15, color=COLORS[crop], label='80% CI')
        ax.axvline(last_date, color='#8b949e', ls=':', alpha=0.7)
        ax.axhline(current_price, color='#484f58', ls='--', alpha=0.5, lw=0.8)

        # Signal badge
        sig_colors = {'Sell':'#f85149','Hold':'#3fb950','Monitor':'#d29922'}
        badge_color = sig_colors.get(signal, '#8b949e')
        ax.text(0.98, 0.95, signal.upper(), transform=ax.transAxes,
                ha='right', va='top', fontsize=16, fontweight='bold',
                color=badge_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117',
                          edgecolor=badge_color, alpha=0.9))

        ax.set_title(f'{crop} -- {FORECAST_WEEKS}-Week Forecast ({model_name})', loc='left')
        ax.set_ylabel('Rs/Quintal'); ax.legend(fontsize=9, framealpha=0.3); ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f'14_forecast_{crop.lower()}.png'),
                    dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved: 14_forecast_{crop.lower()}.png")

    except Exception as e:
        print(f"  FORECAST FAILED: {e}")
        import traceback
        traceback.print_exc()
        decision_table.append({
            'Crop': crop, 'Current_Price': current_price,
            'Forecast_End': np.nan, 'Move_Pct': np.nan,
            'Threshold_Pct': threshold,
            'Signal': 'Monitor', 'Rationale': f'Forecast generation failed: {e}',
            'Model': model_name, 'Model_MAPE': mape_lookup[crop],
        })

# ═════════════════════════════════════════════════════════════════════════
# DECISION TABLE
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("DECISION TABLE")
print(f"{'='*90}")

dt = pd.DataFrame(decision_table)
print(f"\n{'Crop':<10} {'Signal':<10} {'Current':>10} {'Forecast':>10} {'Move':>8} {'Threshold':>10} {'Model':<25} {'MAPE':>7}")
print("-" * 95)
for _, row in dt.iterrows():
    fc = f"{row['Forecast_End']:,.0f}" if not np.isnan(row['Forecast_End']) else 'N/A'
    mv = f"{row['Move_Pct']:+.1f}%" if not np.isnan(row['Move_Pct']) else 'N/A'
    th = f"{row['Threshold_Pct']:.1f}%" if not np.isnan(row['Threshold_Pct']) else 'N/A'
    print(f"{row['Crop']:<10} {row['Signal']:<10} {row['Current_Price']:>10,.0f} {fc:>10} {mv:>8} {th:>10} {row['Model']:<25} {row['Model_MAPE']:>6.2f}%")

print(f"\nRationale:")
for _, row in dt.iterrows():
    print(f"  {row['Crop']}: {row['Rationale']}")

# Save
dt.to_csv(os.path.join(BASE, "reports", "phase4_decision_table.csv"), index=False)
print(f"\n  Saved: reports/phase4_decision_table.csv")

print(f"\n{'='*90}")
print("PHASE 4 COMPLETE")
print(f"{'='*90}")
