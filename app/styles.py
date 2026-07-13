"""CSS styles for AgriPrice India dashboard."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main,.stApp { background: #0a0e17; color: #e6edf3; }
[data-testid="stHeader"] { background: transparent; }

/* ── Change indicators (overview cards) ── */
.ticker-up { color:#3fb950; font-size:0.78rem; }
.ticker-down { color:#f85149; font-size:0.78rem; }
.ticker-flat { color:#8b949e; font-size:0.78rem; }

/* ── Hero ── */
.hero { text-align:center; padding:1.2rem 0 0.3rem; }
.hero h1 { font-size:2.6rem; font-weight:800; margin:0;
  background:linear-gradient(135deg,#58a6ff 0%,#d2a8ff 40%,#f0883e 70%,#3fb950 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#8b949e; font-size:0.92rem; margin:0.2rem 0 0; }


/* ── Overview cards ── */
.ov-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:0.8rem; margin:0.8rem 0; }
@media(max-width:900px){ .ov-grid{grid-template-columns:repeat(2,1fr)} }
.ov-card { background:linear-gradient(135deg,#161b22 0%,#1c2333 100%);
  border:1px solid #21262d; border-radius:14px; padding:1rem; cursor:pointer;
  transition:all 0.3s ease; position:relative; overflow:hidden; }
.ov-card:hover { transform:translateY(-3px); border-color:#30363d;
  box-shadow:0 8px 25px rgba(0,0,0,0.4); }
.ov-card.active { border-color:var(--crop-color); box-shadow:0 0 20px var(--glow); }
.ov-card::after { content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:var(--crop-color); opacity:0.6; }
.ov-emoji { font-size:1.4rem; }
.ov-name { color:#8b949e; font-size:0.72rem; text-transform:uppercase; letter-spacing:1px; margin-top:0.3rem; }
.ov-price { color:#e6edf3; font-size:1.3rem; font-weight:700; margin:0.2rem 0; }
.ov-change { font-size:0.78rem; font-weight:600; }
.ov-spark { margin-top:0.4rem; }
.ov-spark svg { width:100%; height:32px; }

/* ── Signal card ── */
.sig-card { background:linear-gradient(135deg,#161b22,#1a2332); border:1px solid #30363d;
  border-radius:16px; padding:1.5rem; margin:0.8rem 0; position:relative; overflow:hidden; }
.sig-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.sig-card.hold::before { background:linear-gradient(90deg,#3fb950,#56d364); }
.sig-card.sell::before { background:linear-gradient(90deg,#f85149,#ff7b72); }
.sig-card.monitor::before { background:linear-gradient(90deg,#d29922,#e3b341); }
.sig-badge { display:inline-block; padding:0.35rem 1.2rem; border-radius:20px;
  font-weight:700; font-size:1.15rem; letter-spacing:1.5px; }
.badge-hold { background:rgba(63,185,80,0.12); color:#3fb950; border:1px solid rgba(63,185,80,0.25); }
.badge-sell { background:rgba(248,81,73,0.12); color:#f85149; border:1px solid rgba(248,81,73,0.25); }
.badge-monitor { background:rgba(210,153,34,0.12); color:#d29922; border:1px solid rgba(210,153,34,0.25); }
.mape-tag { color:#8b949e; font-size:0.78rem; margin-left:0.8rem; }
.tier-badge { display:inline-block; font-size:0.72rem; font-weight:600; padding:0.2rem 0.7rem;
  margin-left:0.6rem; border:1px solid; border-radius:4px; letter-spacing:0.3px;
  background:transparent; vertical-align:middle; }
.met-row { display:flex; gap:1rem; margin-top:1rem; flex-wrap:wrap; }
.met-item { background:rgba(13,17,23,0.7); border:1px solid #21262d; border-radius:10px;
  padding:0.6rem 1rem; min-width:110px; backdrop-filter:blur(8px); }
.met-label { color:#8b949e; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.5px; }
.met-val { color:#e6edf3; font-size:1rem; font-weight:600; margin-top:2px; }
.card-note { color:#d29922; font-size:0.8rem; margin-top:0.8rem; padding:0.6rem 0.8rem;
  background:rgba(210,153,34,0.06); border-left:3px solid #d29922; border-radius:0 8px 8px 0; }

/* ── Stats row ── */
.stats-row { display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; margin:0.8rem 0; }
@media(max-width:768px){ .stats-row{grid-template-columns:repeat(2,1fr)} }
.stat-box { background:linear-gradient(135deg,#161b22,#1a2332); border:1px solid #21262d;
  border-radius:12px; padding:1rem; text-align:center; }
.stat-num { font-size:1.5rem; font-weight:800; }
.stat-lbl { color:#8b949e; font-size:0.72rem; text-transform:uppercase; margin-top:0.2rem; }

/* ── Section headers ── */
.sec-hdr { color:#e6edf3; font-size:1.25rem; font-weight:700; margin:1.8rem 0 0.5rem;
  padding-bottom:0.4rem; border-bottom:1px solid #21262d;
  display:flex; align-items:center; gap:0.5rem; }

/* ── Model table ── */
.mdl-tbl { width:100%; border-collapse:separate; border-spacing:0;
  background:#161b22; border-radius:14px; overflow:hidden; margin:0.8rem 0; }
.mdl-tbl th { background:#1c2128; color:#8b949e; font-size:0.72rem;
  text-transform:uppercase; letter-spacing:0.5px; padding:0.75rem 1rem; text-align:left; }
.mdl-tbl td { padding:0.65rem 1rem; border-bottom:1px solid #21262d; color:#e6edf3; font-size:0.85rem; }
.mdl-tbl tr:last-child td { border-bottom:none; }
.win-row { background:rgba(63,185,80,0.04); }
.win-b { background:rgba(63,185,80,0.12); color:#3fb950; padding:2px 8px; border-radius:10px; font-size:0.7rem; font-weight:600; }
.los-b { color:#484f58; padding:2px 8px; font-size:0.7rem; }

/* ── About ── */
.abt-box { background:linear-gradient(135deg,#161b22,#1a2332); border:1px solid #30363d;
  border-radius:14px; padding:1.5rem; margin:0.8rem 0; }
.abt-box h3 { color:#e6edf3; margin-top:0.5rem; font-size:1.05rem; }
.abt-box h3:first-child { margin-top:0; }
.abt-box ul { color:#8b949e; line-height:1.7; padding-left:1.2rem; }
.abt-box li { margin-bottom:0.2rem; }
.lim { color:#d29922; }

/* ── Footer ── */
.foot { text-align:center; color:#484f58; font-size:0.72rem; padding:1.5rem 0 0.8rem;
  border-top:1px solid #21262d; margin-top:2rem; }
</style>
"""
