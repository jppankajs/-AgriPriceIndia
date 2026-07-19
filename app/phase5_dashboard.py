"""
Phase 5 — AgriPrice India Decision-Support Dashboard
=====================================================
Agricultural commodity intelligence with interactive Plotly charts,
signal cards, model comparison & limitations.
Static Kaggle dataset (Agmarknet 2023–2025).
"""
import os, sys, json, warnings, traceback
import pandas as pd, numpy as np
import plotly.graph_objects as go
import streamlit as st, joblib

sys.path.insert(0, os.path.dirname(__file__))
from styles import CSS
from live_data import refresh_data, get_latest_prices, COMMODITIES

warnings.filterwarnings('ignore')

BASE     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MDL_DIR  = os.path.join(BASE, "models")
REPORTS  = os.path.join(BASE, "reports")
CROPS    = COMMODITIES
EMOJIS   = {'Onion':'🧅','Potato':'🥔','Wheat':'🌾','Tomato':'🍅','Rice':'🍚'}
COLORS   = {'Onion':'#f85149','Potato':'#58a6ff','Wheat':'#d2a8ff',
            'Tomato':'#f0883e','Rice':'#3fb950'}
GLOWS    = {c: f'rgba({int(v[1:3],16)},{int(v[3:5],16)},{int(v[5:7],16)},0.25)' for c,v in COLORS.items()}
SUPPRESS = {'Onion','Potato'}

CARD_NOTES = {
    'Potato': "This model's forecast turns negative by week 8 — a literal impossibility for a market price — which is why its output isn't shown here. The underlying cause: Prophet is extrapolating a since-reversed downtrend (₹1,000 → ₹1,228 recovery, Feb–Jun 2025) that the model never saw.",
    'Tomato': "Monitor — only 61 data points from 2023, dominated by the monsoon price crisis; no stable baseline to forecast from.",
    'Rice':   "Monitor — only 25 aggregated data points (62 raw records); an 8-week forecast would extrapolate further than the data supports.",
    'Onion':  "MAPE of 117.57% — larger than the forecast itself. Only the directional signal, not the magnitude, should be trusted.",
    'Wheat':  "Forecast excludes annual seasonality — only 8 months of history exist, with Mar–May entirely missing.",
}
NOTE_ICONS = {'Potato':'⚠','Tomato':'⚠','Rice':'⚠','Onion':'⚠','Wheat':'ℹ'}

# ── Modeling tier badges ───────────────────────────────────────────────
TIER_BADGES = {
    'Onion':  'Full seasonal model',
    'Potato': 'Full seasonal model',
    'Wheat':  'Limited — no seasonal claim, 8 months history',
    'Tomato': 'Insufficient history — Monitor only, 61 data points',
    'Rice':   'Insufficient history — Monitor only, 25 aggregated data points',
}
TIER_COLORS = {
    'Onion':  '#3fb950',
    'Potato': '#3fb950',
    'Wheat':  '#d29922',
    'Tomato': '#484f58',
    'Rice':   '#484f58',
}

# ── Safe model loader ─────────────────────────────────────────────────
def _safe_load_model(path: str):
    """Load a joblib model file, returning None on any failure."""
    try:
        if not os.path.exists(path):
            return None
        return joblib.load(path)
    except Exception as e:
        st.warning(f"⚠️ Could not load model `{os.path.basename(path)}`: {type(e).__name__}")
        return None

# ── Load all data ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all():
    daily = refresh_data()
    if daily.empty:
        return daily, {}, pd.DataFrame(), pd.DataFrame(), {}

    best_path = os.path.join(MDL_DIR, "best_models.json")
    if os.path.exists(best_path):
        with open(best_path) as f:
            best = json.load(f)
    else:
        best = {}

    p3_path = os.path.join(REPORTS, "phase3_model_results.csv")
    p3 = pd.read_csv(p3_path) if os.path.exists(p3_path) else pd.DataFrame()

    p4_path = os.path.join(REPORTS, "phase4_decision_table.csv")
    p4 = pd.read_csv(p4_path) if os.path.exists(p4_path) else pd.DataFrame()

    latest = get_latest_prices(daily)
    return daily, best, p3, p4, latest

daily, best_models, phase3_df, decision_df, latest = load_all()

if daily.empty:
    st.error("❌ No data available. Please ensure `data/processed/agri_prices_daily_national.csv` exists.")
    st.stop()

def get_series(crop):
    sub = daily[daily['Commodity'] == crop]
    if sub.empty:
        return pd.Series(dtype=float)
    s = sub.set_index('Price_Date')['Modal_Price_Median'].sort_index()
    return s[~s.index.duplicated(keep='first')]

@st.cache_data
def gen_forecast(crop, _series):
    """Generate forecast with full error handling. Returns DataFrame or None."""
    if _series.empty or len(_series) < 2:
        return None

    last, n = _series.index[-1], 56
    mn = best_models.get(crop, '')

    try:
        if crop in ('Tomato', 'Rice'):
            if 'SMA' in mn:
                w = int(mn.split('-')[1]) if '-' in mn else 14
                d = pd.date_range(last + pd.Timedelta(days=1), periods=n, freq='D')
                return pd.DataFrame({'forecast': _series.iloc[-w:].mean(), 'lower': np.nan, 'upper': np.nan}, index=d)
            elif 'ExpSmoothing' in mn or 'ETS' in mn or 'Smoothing' in mn:
                m = _safe_load_model(os.path.join(MDL_DIR, f'ets_{crop.lower()}.pkl'))
                d = pd.date_range(last + pd.Timedelta(days=1), periods=n, freq='D')
                if m is not None:
                    try:
                        fc_vals = m.forecast(n)
                    except Exception:
                        fc_vals = pd.Series([float(_series.iloc[-1])] * n)
                else:
                    fc_vals = pd.Series([float(_series.iloc[-1])] * n)
                return pd.DataFrame({'forecast': np.array(fc_vals, dtype=float), 'lower': np.nan, 'upper': np.nan}, index=d)
            # fallback: flat forecast at last known price
            d = pd.date_range(last + pd.Timedelta(days=1), periods=n, freq='D')
            return pd.DataFrame({'forecast': float(_series.iloc[-1]), 'lower': np.nan, 'upper': np.nan}, index=d)

        if 'Prophet' in mn:
            m = _safe_load_model(os.path.join(MDL_DIR, f'prophet_{crop.lower()}.pkl'))
            if m is None:
                return None
            fut = pd.DataFrame({'ds': pd.date_range(last + pd.Timedelta(days=1), periods=n, freq='D')})
            fc = m.predict(fut)
            return pd.DataFrame({
                'forecast': fc['yhat'].values,
                'lower': fc['yhat_lower'].values,
                'upper': fc['yhat_upper'].values
            }, index=fut['ds'].values)

        elif 'SARIMA' in mn:
            m = _safe_load_model(os.path.join(MDL_DIR, f'sarima_{crop.lower()}.pkl'))
            if m is None:
                return None
            p, ci = m.predict(n_periods=n, return_conf_int=True)
            d = pd.date_range(last + pd.Timedelta(days=1), periods=n, freq='D')
            return pd.DataFrame({
                'forecast': np.array(p, dtype=float),
                'lower': np.array(ci[:, 0], dtype=float),
                'upper': np.array(ci[:, 1], dtype=float)
            }, index=d)

    except Exception as e:
        st.warning(f"⚠️ Forecast generation failed for {crop}: {type(e).__name__} — {e}")
        return None

    return None

def sparkline_svg(vals, color, w=120, h=28):
    if not vals or len(vals) < 2:
        return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1
    pts = " ".join(f"{i*w/(len(vals)-1):.1f},{h - (v-mn)/rng*(h-4) - 2:.1f}" for i, v in enumerate(vals))
    return f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/></svg>'

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(page_title="AgriPrice India", page_icon="🌾", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero"><h1>🌾 AgriPrice India</h1><p>Agricultural Commodity Decision-Support Dashboard · Static Kaggle Dataset (Agmarknet 2023–2025) · Coverage varies by crop — see individual cards</p></div>', unsafe_allow_html=True)

# ── State ──────────────────────────────────────────────────────────────
if 'crop' not in st.session_state:
    st.session_state.crop = 'Wheat'
sel = st.session_state.crop

# ── Overview cards ─────────────────────────────────────────────────────
cards_html = '<div class="ov-grid">'
for c in CROPS:
    p = latest.get(c, {})
    if not p or p.get('price') is None:
        continue
    active = "active" if c == sel else ""
    ch7 = p.get('change_7d')
    ch_cls = "ticker-up" if ch7 and ch7 > 0 else "ticker-down" if ch7 and ch7 < 0 else "ticker-flat"
    ch_str = f"{'▲' if ch7 and ch7 > 0 else '▼' if ch7 and ch7 < 0 else '–'} {ch7:+.1f}% 7-sess" if ch7 else "–"
    spark = sparkline_svg(p.get('sparkline', []), COLORS[c])
    cards_html += f'''<div class="ov-card {active}" style="--crop-color:{COLORS[c]};--glow:{GLOWS[c]}">
      <span class="ov-emoji">{EMOJIS[c]}</span>
      <div class="ov-name">{c}</div>
      <div class="ov-price">₹{p["price"]:,.0f}</div>
      <div class="ov-change {ch_cls}">{ch_str}</div>
      <div class="ov-spark">{spark}</div></div>'''
cards_html += '</div>'
st.markdown(cards_html, unsafe_allow_html=True)

# Crop selector buttons (functional)
bcols = st.columns(5)
for i, c in enumerate(CROPS):
    with bcols[i]:
        if st.button(f"{EMOJIS[c]} {c}", key=f"b_{c}", width='stretch',
                     type="primary" if c == sel else "secondary"):
            st.session_state.crop = c
            st.rerun()
sel = st.session_state.crop

# ── Market stats row ──────────────────────────────────────────────────
p = latest.get(sel, {})
series = get_series(sel)

if series.empty:
    st.warning(f"No data available for {sel}.")
    st.stop()

cur_price = float(series.iloc[-1])
last_date = series.index[-1]

# Safe decision table lookup
dec_rows = decision_df[decision_df['Crop'] == sel] if not decision_df.empty else pd.DataFrame()
if not dec_rows.empty:
    row = dec_rows.iloc[0]
    signal = str(row.get('Signal', 'Monitor'))
    model_mape = float(row.get('Model_MAPE', 0)) if pd.notna(row.get('Model_MAPE')) else 0.0
else:
    signal = 'Monitor'
    model_mape = 0.0

vol = p.get('volatility_30d')
hi52 = p.get('high_52w', cur_price)
lo52 = p.get('low_52w', cur_price)
mkts = p.get('markets', 0)

if vol is not None and vol > 0:
    vol_color = '#3fb950' if vol < 10 else '#d29922' if vol < 25 else '#f85149'
    vol_str = f'{vol:.1f}%'
else:
    vol_color = '#8b949e'
    vol_str = '–'

st.markdown(f'''<div class="stats-row">
  <div class="stat-box"><div class="stat-num" style="color:{COLORS[sel]}">₹{cur_price:,.0f}</div><div class="stat-lbl">Current Price</div></div>
  <div class="stat-box"><div class="stat-num" style="color:{vol_color}">{vol_str}</div><div class="stat-lbl">30-Session Volatility</div></div>
  <div class="stat-box"><div class="stat-num">₹{lo52:,.0f} – ₹{hi52:,.0f}</div><div class="stat-lbl">52-Week Range</div></div>
  <div class="stat-box"><div class="stat-num">{mkts:,}</div><div class="stat-lbl">Markets Reporting</div></div>
</div>''', unsafe_allow_html=True)

# ── Signal card ───────────────────────────────────────────────────────
st.markdown(f'<div class="sec-hdr">📊 Signal — {sel}</div>', unsafe_allow_html=True)
sig_lo = signal.lower()
mets = f'<div class="met-item"><div class="met-label">Current Price</div><div class="met-val">₹{cur_price:,.0f}</div></div>'
mets += f'<div class="met-item"><div class="met-label">Model</div><div class="met-val">{best_models.get(sel, "N/A")}</div></div>'

fc_df = gen_forecast(sel, series)

if sel in SUPPRESS:
    if fc_df is not None:
        fc_end = fc_df['forecast'].iloc[-1]
        fc_min = fc_df['forecast'].min()
        if fc_min <= 0:
            d_html = '<span style="color:#f85149">Unreliable — model output diverges to implausible values</span>'
        else:
            d = "Upward ↑" if fc_end > cur_price else "Downward ↓" if fc_end < cur_price else "Flat →"
            d_html = d
        mets += f'<div class="met-item"><div class="met-label">Direction (8-wk)</div><div class="met-val">{d_html}</div></div>'
        lo_val, hi_val = fc_df['lower'].iloc[-1], fc_df['upper'].iloc[-1]
        if not np.isnan(lo_val) and lo_val > 0 and hi_val > 0:
            mets += f'<div class="met-item"><div class="met-label">CI Range</div><div class="met-val">₹{lo_val:,.0f} – ₹{hi_val:,.0f}</div></div>'
        elif not np.isnan(lo_val):
            mets += '<div class="met-item"><div class="met-label">CI Range</div><div class="met-val" style="color:#8b949e">N/A (model breakdown)</div></div>'
    mets += f'<div class="met-item"><div class="met-label">Model MAPE</div><div class="met-val" style="color:#d29922">{model_mape:.2f}%</div></div>'
elif sel in ('Tomato', 'Rice'):
    mets += '<div class="met-item"><div class="met-label">8-wk Forecast</div><div class="met-val" style="color:#8b949e">N/A</div></div>'
    mets += f'<div class="met-item"><div class="met-label">Model MAPE</div><div class="met-val">{model_mape:.2f}%</div></div>'
else:
    if not dec_rows.empty:
        fe = row.get('Forecast_End')
        if fe is not None and pd.notna(fe):
            fe = float(fe)
            mets += f'<div class="met-item"><div class="met-label">8-wk Forecast</div><div class="met-val">₹{fe:,.0f}</div></div>'
            move_pct = row.get('Move_Pct')
            thresh_pct = row.get('Threshold_Pct')
            if pd.notna(move_pct):
                mets += f'<div class="met-item"><div class="met-label">Move</div><div class="met-val">{float(move_pct):+.1f}%</div></div>'
            if pd.notna(thresh_pct):
                mets += f'<div class="met-item"><div class="met-label">Threshold</div><div class="met-val">{float(thresh_pct):.1f}%</div></div>'
    mets += f'<div class="met-item"><div class="met-label">Model MAPE</div><div class="met-val">{model_mape:.2f}%</div></div>'

note = f'<div class="card-note">{NOTE_ICONS.get(sel, "ℹ")} {CARD_NOTES[sel]}</div>' if sel in CARD_NOTES else ""
tier_label = TIER_BADGES.get(sel, 'Unknown')
tier_color = TIER_COLORS.get(sel, '#484f58')
tier_html = f'<span class="tier-badge" style="color:{tier_color};border-color:{tier_color}">{tier_label}</span>'
st.markdown(f'<div class="sig-card {sig_lo}"><span class="sig-badge badge-{sig_lo}">{signal.upper()}</span><span class="mape-tag">MAPE: {model_mape:.2f}%</span>{tier_html}<div class="met-row">{mets}</div>{note}</div>', unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-hdr">📈 Price History & Forecast — {sel}</div>', unsafe_allow_html=True)
fig = go.Figure()
# Convert datetime index to Python datetime to avoid Plotly datetime64[us] serialization bugs
_hist_x = series.index.to_pydatetime()
fig.add_trace(go.Scatter(x=_hist_x, y=series.values, name='Historical',
    line=dict(color=COLORS[sel], width=1.8),
    hovertemplate='%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>'))

# Only plot forecast if values are plausible (all positive)
fc_plausible = (fc_df is not None and len(fc_df) > 0 and fc_df['forecast'].min() > 0)
if fc_plausible:
    _fc_x = fc_df.index.to_pydatetime()
    fig.add_trace(go.Scatter(x=_fc_x, y=fc_df['forecast'].values, name='Forecast',
        line=dict(color='#f0f6fc', width=2, dash='dash'),
        hovertemplate='%{x|%d %b %Y}<br>₹%{y:,.0f}<extra>Forecast</extra>'))
    if not fc_df['lower'].isna().all() and fc_df['lower'].min() > 0:
        r, g, b = int(COLORS[sel][1:3], 16), int(COLORS[sel][3:5], 16), int(COLORS[sel][5:7], 16)
        _fc_x_rev = _fc_x[::-1].tolist()
        fig.add_trace(go.Scatter(x=list(_fc_x) + _fc_x_rev,
            y=list(fc_df['upper'].values) + list(fc_df['lower'].values[::-1]),
            fill='toself', fillcolor=f'rgba({r},{g},{b},0.1)', line=dict(width=0),
            name='80% CI', hoverinfo='skip'))

# Divider line
if fc_plausible:
    ylo = float(np.nanmin([series.min(), np.nanmin(fc_df['forecast'].values)]))
    yhi = float(np.nanmax([series.max(), np.nanmax(fc_df['forecast'].values)]))
else:
    ylo = float(series.min())
    yhi = float(series.max())
pad = (yhi - ylo) * 0.05 if yhi != ylo else max(ylo * 0.05, 50)
_last_py = last_date.to_pydatetime()
fig.add_trace(go.Scatter(x=[_last_py, _last_py], y=[ylo - pad, yhi + pad], mode='lines',
    line=dict(color='#484f58', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))

fig.update_layout(template=None, plot_bgcolor='#0a0e17', paper_bgcolor='#0a0e17',
    font=dict(color='#e6edf3', family='Inter'), height=460,
    margin=dict(l=55, r=25, t=30, b=45),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
    xaxis=dict(showgrid=True, gridcolor='#161b22', tickfont=dict(color='#8b949e'), linecolor='#21262d'),
    yaxis=dict(title='₹/Quintal', showgrid=True, gridcolor='#161b22',
               tickfont=dict(color='#8b949e'), linecolor='#21262d', tickformat=',',
               range=[ylo - pad, yhi + pad]),
    hovermode='x unified')
st.plotly_chart(fig, width='stretch', config={'displayModeBar': True, 'scrollZoom': True})

# ── Model Comparison ──────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔬 Model Comparison — All Crops (Phase 3)</div>', unsafe_allow_html=True)
if not phase3_df.empty:
    rows_html = ""
    for _, r in phase3_df.iterrows():
        w = best_models.get(r['Crop']) == r['Model']
        rc = 'win-row' if w else ''
        b = '<span class="win-b">✓ BEST</span>' if w else '<span class="los-b">—</span>'
        rows_html += f'<tr class="{rc}"><td>{r["Crop"]}</td><td>{r["Model"]}</td><td>{r["RMSE"]:,.1f}</td><td>{r["MAE"]:,.1f}</td><td>{r["MAPE"]:.2f}%</td><td>{int(r["n_train"])}</td><td>{int(r["n_test"])}</td><td>{b}</td></tr>'
    st.markdown(f'<table class="mdl-tbl"><thead><tr><th>Crop</th><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>Train</th><th>Test</th><th>Status</th></tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)
else:
    st.info("Model comparison data not available.")

# ── About & Limitations ──────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📋 About & Limitations</div>', unsafe_allow_html=True)
st.markdown('''<div class="abt-box">
<h3>Data Source</h3>
<ul>
<li>Daily modal prices from <strong>data.gov.in / Agmarknet</strong> (Ministry of Consumer Affairs, Government of India), aggregated as national median across all reporting markets.</li>
<li>Five commodities tracked: Onion, Potato, Wheat, Tomato, Rice.</li>
<li>Data comes from a static Kaggle dataset snapshot — there is no live feed.</li>
</ul>
<h3>What This Dashboard Does</h3>
<ul>
<li>Compares candidate time-series models (SARIMA, Prophet, ETS, SMA) per crop on an 80/20 train-test split.</li>
<li>Generates an 8-week (56-day) forward forecast from the best-performing model per crop.</li>
<li>Produces a Sell / Hold / Monitor signal using a dynamic noise threshold (max of 5% or 1.5× the model's own test MAPE).</li>
</ul>

<h3>Crop Mix — Design Choice</h3>
<p style="color:#8b949e;line-height:1.7">This dashboard tracks three perishable vegetables (Onion, Potato, Tomato) alongside two MSP-stabilized grains (Wheat, Rice), chosen because they behave in genuinely different ways. Wheat's 30-day price volatility is roughly a tenth of Onion's, specifically because MSP floor pricing insulates it from the seasonal swings that dominate the vegetables. However, the two grains should not be treated as equally well-understood: Wheat's low volatility is supported by 8 months of consistent data (MAPE 0.82%), while Rice's apparent stability is based on only 62 days — it may simply be a calm two-month window, not a confirmed structural pattern.</p>

<h3 style="color:#d29922">Limitations — Specific Findings From This Build</h3>

<p style="color:#d29922;line-height:1.7;margin-bottom:0.8rem"><strong>Onion:</strong> Onion's first-pass forecast used a non-default Prophet parameter that produced a price exceeding anything observed in two years of data. Reverting to Prophet's default settings produced a more plausible, though still highly uncertain, forecast — reflected in its Very Low confidence rating. The final best model (SARIMA) still carries a <strong>117.57% MAPE</strong> — larger than the forecast itself — so only the directional signal, not the magnitude, should be trusted.</p>

<p style="color:#d29922;line-height:1.7;margin-bottom:0.8rem"><strong>Potato:</strong> Potato's Sell signal was initially generated by Prophet extrapolating a price crash from Oct 2024–Feb 2025. That crash had already reversed by the time of this forecast — prices recovered from around ₹1,000 (trough in early 2025) to ₹1,228 (current price at decision time) — so the signal was corrected to Monitor. More concretely, the same Prophet model produces a <strong>negative price forecast (~₹ −316) by day 56</strong> — a physical impossibility for any market price — confirming that the trend it learned is not a valid basis for projection.</p>

<p style="color:#d29922;line-height:1.7;margin-bottom:0.8rem"><strong>Key takeaway on backtesting:</strong> Potato's Prophet model has the <em>best</em> historical test-period accuracy of any candidate for that crop (MAPE 12.74%), and is simultaneously the one producing an impossible forecast at the 8-week horizon. Strong backtested accuracy does not guarantee reliable extrapolation — a model can fit historical patterns well and still diverge when asked to project beyond the conditions it was trained on.</p>

<ul>
<li class="lim"><strong>Tomato:</strong> Only 154 days of history, dominated by the 2023 monsoon price crisis. No stable baseline — intentionally Monitor.</li>
<li class="lim"><strong>Rice:</strong> Only 62 days of history. An 8-week forecast would extrapolate further than the available data supports — intentionally Monitor.</li>
<li class="lim"><strong>Wheat:</strong> Forecast <strong>excludes annual seasonality</strong> — only 8 months of history exist, with Mar–May entirely missing.</li>
</ul>
</div>''', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown('<div class="foot">AgriPrice India · Decision-Support Dashboard · Built with Streamlit & Plotly · Data: data.gov.in / Agmarknet (static Kaggle snapshot)</div>', unsafe_allow_html=True)
