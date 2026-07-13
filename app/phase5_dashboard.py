"""
Phase 5 — AgriPrice India Decision-Support Dashboard
=====================================================
Agricultural commodity intelligence with interactive Plotly charts,
signal cards, model comparison & limitations.
Static Kaggle dataset (Agmarknet 2023–2025).
"""
import os, sys, json, warnings
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
    'Potato': "Model likely extrapolating a since-reversed downtrend; recent recovery pattern (₹1,000 → ₹1,228, Feb–Jun 2025) not reflected in the forecast.",
    'Tomato': "Monitor — 154-day window dominated by the 2023 monsoon price crisis; no stable baseline to forecast from.",
    'Rice':   "Monitor — only 62 days of history; an 8-week forecast would extrapolate further than the available data supports.",
    'Onion':  "MAPE of 117.57% — larger than the forecast itself. Only the directional signal, not the magnitude, should be trusted.",
    'Wheat':  "Forecast excludes annual seasonality — only 8 months of history exist, with Mar–May entirely missing.",
}
NOTE_ICONS = {'Potato':'⚠','Tomato':'⚠','Rice':'⚠','Onion':'⚠','Wheat':'ℹ'}

# ── Modeling tier badges (Part 1) ──────────────────────────────────────
TIER_BADGES = {
    'Onion':  'Full seasonal model',
    'Potato': 'Full seasonal model',
    'Wheat':  'Limited — no seasonal claim, 8 months history',
    'Tomato': 'Insufficient history — Monitor only, 154 days',
    'Rice':   'Insufficient history — Monitor only, 62 days',
}
TIER_COLORS = {
    'Onion':  '#3fb950',  # green — full model
    'Potato': '#3fb950',  # green — full model
    'Wheat':  '#d29922',  # yellow — limited
    'Tomato': '#484f58',  # gray — insufficient
    'Rice':   '#484f58',  # gray — insufficient
}

# ── Load all data ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all():
    daily = refresh_data()
    with open(os.path.join(MDL_DIR, "best_models.json")) as f:
        best = json.load(f)
    p3 = pd.read_csv(os.path.join(REPORTS, "phase3_model_results.csv"))
    p4 = pd.read_csv(os.path.join(REPORTS, "phase4_decision_table.csv"))
    latest = get_latest_prices(daily)
    return daily, best, p3, p4, latest

daily, best_models, phase3_df, decision_df, latest = load_all()

def get_series(crop):
    s = daily[daily['Commodity']==crop].set_index('Price_Date')['Modal_Price_Median'].sort_index()
    return s[~s.index.duplicated(keep='first')]

@st.cache_data
def gen_forecast(crop, _series):
    last, n = _series.index[-1], 56
    mn = best_models[crop]
    if crop in ('Tomato','Rice'):
        if 'SMA' in mn:
            w = int(mn.split('-')[1]) if '-' in mn else 14
            d = pd.date_range(last+pd.Timedelta(days=1), periods=n, freq='D')
            return pd.DataFrame({'forecast':_series.iloc[-w:].mean(),'lower':np.nan,'upper':np.nan}, index=d)
        elif 'ExpSmoothing' in mn or 'ETS' in mn:
            m = joblib.load(os.path.join(MDL_DIR, f'ets_{crop.lower()}.pkl'))
            d = pd.date_range(last+pd.Timedelta(days=1), periods=n, freq='D')
            return pd.DataFrame({'forecast':m.forecast(n).values,'lower':np.nan,'upper':np.nan}, index=d)
        return None
    if 'Prophet' in mn:
        m = joblib.load(os.path.join(MDL_DIR, f'prophet_{crop.lower()}.pkl'))
        fut = pd.DataFrame({'ds':pd.date_range(last+pd.Timedelta(days=1), periods=n, freq='D')})
        fc = m.predict(fut)
        return pd.DataFrame({'forecast':fc['yhat'].values,'lower':fc['yhat_lower'].values,'upper':fc['yhat_upper'].values}, index=fut['ds'].values)
    elif 'SARIMA' in mn:
        m = joblib.load(os.path.join(MDL_DIR, f'sarima_{crop.lower()}.pkl'))
        p,ci = m.predict(n_periods=n, return_conf_int=True)
        d = pd.date_range(last+pd.Timedelta(days=1), periods=n, freq='D')
        return pd.DataFrame({'forecast':np.array(p,dtype=float),'lower':np.array(ci[:,0],dtype=float),'upper':np.array(ci[:,1],dtype=float)}, index=d)
    return None

def sparkline_svg(vals, color, w=120, h=28):
    if not vals or len(vals) < 2: return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1
    pts = " ".join(f"{i*w/(len(vals)-1):.1f},{h - (v-mn)/rng*(h-4) - 2:.1f}" for i,v in enumerate(vals))
    return f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/></svg>'

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(page_title="AgriPrice India", page_icon="🌾", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero"><h1>🌾 AgriPrice India</h1><p>Agricultural Commodity Decision-Support Dashboard · Static Kaggle Dataset (Agmarknet 2023–2025)</p></div>', unsafe_allow_html=True)

# ── State ──────────────────────────────────────────────────────────────
if 'crop' not in st.session_state:
    st.session_state.crop = 'Wheat'
sel = st.session_state.crop

# ── Overview cards ─────────────────────────────────────────────────────
cards_html = '<div class="ov-grid">'
for c in CROPS:
    p = latest[c]
    if p['price'] is None: continue
    active = "active" if c == sel else ""
    ch7 = p.get('change_7d')
    ch_cls = "ticker-up" if ch7 and ch7>0 else "ticker-down" if ch7 and ch7<0 else "ticker-flat"
    ch_str = f"{'▲' if ch7 and ch7>0 else '▼' if ch7 and ch7<0 else '–'} {ch7:+.1f}% 7d" if ch7 else "–"
    spark = sparkline_svg(p.get('sparkline',[]), COLORS[c])
    cards_html += f'''<div class="ov-card {active}" style="--crop-color:{COLORS[c]};--glow:{GLOWS[c]}"
      onclick="window.parent.postMessage({{type:'streamlit:setComponentValue',value:'{c}'}},'*')">
      <span class="ov-emoji">{EMOJIS[c]}</span>
      <div class="ov-name">{c}</div>
      <div class="ov-price">₹{p["price"]:,.0f}</div>
      <div class="ov-change {ch_cls}">{ch_str}</div>
      <div class="ov-spark">{spark}</div></div>'''
cards_html += '</div>'
st.markdown(cards_html, unsafe_allow_html=True)

# Crop selector buttons (functional)
bcols = st.columns(5)
for i,c in enumerate(CROPS):
    with bcols[i]:
        if st.button(f"{EMOJIS[c]} {c}", key=f"b_{c}", use_container_width=True,
                     type="primary" if c==sel else "secondary"):
            st.session_state.crop = c; st.rerun()
sel = st.session_state.crop

# ── Market stats row ───────────────────────────────────────────────────
p = latest[sel]
series = get_series(sel)
cur_price = float(series.iloc[-1])
last_date = series.index[-1]
row = decision_df[decision_df['Crop']==sel].iloc[0]
signal = row['Signal']; model_mape = row['Model_MAPE']

vol = p.get('volatility_30d')
hi52 = p.get('high_52w', cur_price)
lo52 = p.get('low_52w', cur_price)
mkts = p.get('markets', 0)

vol_color = '#3fb950' if vol and vol<10 else '#d29922' if vol and vol<25 else '#f85149'
st.markdown(f'''<div class="stats-row">
  <div class="stat-box"><div class="stat-num" style="color:{COLORS[sel]}">₹{cur_price:,.0f}</div><div class="stat-lbl">Current Price</div></div>
  <div class="stat-box"><div class="stat-num" style="color:{vol_color}">{vol:.1f}%</div><div class="stat-lbl">30-Day Volatility</div></div>
  <div class="stat-box"><div class="stat-num">₹{lo52:,.0f} – ₹{hi52:,.0f}</div><div class="stat-lbl">52-Week Range</div></div>
  <div class="stat-box"><div class="stat-num">{mkts:,}</div><div class="stat-lbl">Markets Reporting</div></div>
</div>''' if vol else f'''<div class="stats-row">
  <div class="stat-box"><div class="stat-num" style="color:{COLORS[sel]}">₹{cur_price:,.0f}</div><div class="stat-lbl">Current Price</div></div>
  <div class="stat-box"><div class="stat-num">–</div><div class="stat-lbl">30-Day Volatility</div></div>
  <div class="stat-box"><div class="stat-num">₹{lo52:,.0f} – ₹{hi52:,.0f}</div><div class="stat-lbl">52-Week Range</div></div>
  <div class="stat-box"><div class="stat-num">{mkts:,}</div><div class="stat-lbl">Markets Reporting</div></div>
</div>''', unsafe_allow_html=True)

# ── Signal card ────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-hdr">📊 Signal — {sel}</div>', unsafe_allow_html=True)
sig_lo = signal.lower()
mets = f'<div class="met-item"><div class="met-label">Current Price</div><div class="met-val">₹{cur_price:,.0f}</div></div>'
mets += f'<div class="met-item"><div class="met-label">Model</div><div class="met-val">{best_models[sel]}</div></div>'

fc_df = gen_forecast(sel, series)
if sel in SUPPRESS:
    if fc_df is not None:
        fc_end = fc_df['forecast'].iloc[-1]
        d = "Upward ↑" if fc_end>cur_price else "Downward ↓" if fc_end<cur_price else "Flat →"
        mets += f'<div class="met-item"><div class="met-label">Direction (8-wk)</div><div class="met-val">{d}</div></div>'
        lo_val, hi_val = fc_df['lower'].iloc[-1], fc_df['upper'].iloc[-1]
        if not np.isnan(lo_val) and lo_val > 0 and hi_val > 0:
            mets += f'<div class="met-item"><div class="met-label">CI Range</div><div class="met-val">₹{lo_val:,.0f} – ₹{hi_val:,.0f}</div></div>'
        elif not np.isnan(lo_val):
            mets += '<div class="met-item"><div class="met-label">CI Range</div><div class="met-val" style="color:#8b949e">N/A (model breakdown)</div></div>'
    mets += f'<div class="met-item"><div class="met-label">Model MAPE</div><div class="met-val" style="color:#d29922">{model_mape:.2f}%</div></div>'
elif sel in ('Tomato','Rice'):
    mets += '<div class="met-item"><div class="met-label">8-wk Forecast</div><div class="met-val" style="color:#8b949e">N/A</div></div>'
    mets += f'<div class="met-item"><div class="met-label">Model MAPE</div><div class="met-val">{model_mape:.2f}%</div></div>'
else:
    fe = row['Forecast_End']
    if not np.isnan(fe):
        mets += f'<div class="met-item"><div class="met-label">8-wk Forecast</div><div class="met-val">₹{fe:,.0f}</div></div>'
        mets += f'<div class="met-item"><div class="met-label">Move</div><div class="met-val">{row["Move_Pct"]:+.1f}%</div></div>'
        mets += f'<div class="met-item"><div class="met-label">Threshold</div><div class="met-val">{row["Threshold_Pct"]:.1f}%</div></div>'
    mets += f'<div class="met-item"><div class="met-label">Model MAPE</div><div class="met-val">{model_mape:.2f}%</div></div>'

note = f'<div class="card-note">{NOTE_ICONS.get(sel,"ℹ")} {CARD_NOTES[sel]}</div>' if sel in CARD_NOTES else ""
tier_label = TIER_BADGES[sel]
tier_color = TIER_COLORS[sel]
tier_html = f'<span class="tier-badge" style="color:{tier_color};border-color:{tier_color}">{tier_label}</span>'
st.markdown(f'<div class="sig-card {sig_lo}"><span class="sig-badge badge-{sig_lo}">{signal.upper()}</span><span class="mape-tag">MAPE: {model_mape:.2f}%</span>{tier_html}<div class="met-row">{mets}</div>{note}</div>', unsafe_allow_html=True)

# ── Chart ──────────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-hdr">📈 Price History & Forecast — {sel}</div>', unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series.values, name='Historical',
    line=dict(color=COLORS[sel], width=1.8),
    hovertemplate='%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>'))
# Only plot forecast if values are plausible (all positive).
# Potato's Prophet model produces negative prices — plotting that is misleading.
fc_plausible = (fc_df is not None and fc_df['forecast'].min() > 0)
if fc_plausible:
    fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df['forecast'], name='Forecast',
        line=dict(color='#f0f6fc', width=2, dash='dash'),
        hovertemplate='%{x|%d %b %Y}<br>₹%{y:,.0f}<extra>Forecast</extra>'))
    if not fc_df['lower'].isna().all() and fc_df['lower'].min() > 0:
        r,g,b = int(COLORS[sel][1:3],16), int(COLORS[sel][3:5],16), int(COLORS[sel][5:7],16)
        fig.add_trace(go.Scatter(x=list(fc_df.index)+list(fc_df.index[::-1]),
            y=list(fc_df['upper'])+list(fc_df['lower'][::-1]),
            fill='toself', fillcolor=f'rgba({r},{g},{b},0.1)', line=dict(width=0),
            name='80% CI', hoverinfo='skip'))

# Divider line
ylo = float(np.nanmin([series.min(), np.nanmin(fc_df['forecast'].values)] if fc_plausible else [series.min()]))
yhi = float(np.nanmax([series.max(), np.nanmax(fc_df['forecast'].values)] if fc_plausible else [series.max()]))
pad = (yhi-ylo)*0.05
fig.add_trace(go.Scatter(x=[last_date,last_date], y=[ylo-pad, yhi+pad], mode='lines',
    line=dict(color='#484f58', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))

fig.update_layout(template=None, plot_bgcolor='#0a0e17', paper_bgcolor='#0a0e17',
    font=dict(color='#e6edf3', family='Inter'), height=460,
    margin=dict(l=55, r=25, t=30, b=45),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
    xaxis=dict(showgrid=True, gridcolor='#161b22', tickfont=dict(color='#8b949e'), linecolor='#21262d'),
    yaxis=dict(title='₹/Quintal', showgrid=True, gridcolor='#161b22',
               tickfont=dict(color='#8b949e'), linecolor='#21262d', tickformat=','),
    hovermode='x unified')
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':True,'scrollZoom':True})

# ── Model Comparison ───────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔬 Model Comparison — All Crops (Phase 3)</div>', unsafe_allow_html=True)
rows = ""
for _, r in phase3_df.iterrows():
    w = best_models.get(r['Crop']) == r['Model']
    rc = 'win-row' if w else ''
    b = '<span class="win-b">✓ BEST</span>' if w else '<span class="los-b">—</span>'
    rows += f'<tr class="{rc}"><td>{r["Crop"]}</td><td>{r["Model"]}</td><td>{r["RMSE"]:,.1f}</td><td>{r["MAE"]:,.1f}</td><td>{r["MAPE"]:.2f}%</td><td>{int(r["n_train"])}</td><td>{int(r["n_test"])}</td><td>{b}</td></tr>'
st.markdown(f'<table class="mdl-tbl"><thead><tr><th>Crop</th><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>Train</th><th>Test</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

# ── About & Limitations ───────────────────────────────────────────────
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

<p style="color:#d29922;line-height:1.7;margin-bottom:0.8rem"><strong>Potato:</strong> Potato's Sell signal was initially generated by Prophet extrapolating a price crash from Oct 2024–Feb 2025. That crash had already reversed by the time of this forecast — prices recovered from around Rs 800 (trough in early 2025) to Rs 1,228 (current price at decision time) — so the signal was corrected to Monitor, with the reasoning shown on its card.</p>

<ul>
<li class="lim"><strong>Tomato:</strong> Only 154 days of history, dominated by the 2023 monsoon price crisis. No stable baseline — intentionally Monitor.</li>
<li class="lim"><strong>Rice:</strong> Only 62 days of history. An 8-week forecast would extrapolate further than the available data supports — intentionally Monitor.</li>
<li class="lim"><strong>Wheat:</strong> Forecast <strong>excludes annual seasonality</strong> — only 8 months of history exist, with Mar–May entirely missing.</li>
</ul>
</div>''', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown('<div class="foot">AgriPrice India · Decision-Support Dashboard · Built with Streamlit & Plotly · Data: data.gov.in / Agmarknet (static Kaggle snapshot)</div>', unsafe_allow_html=True)
