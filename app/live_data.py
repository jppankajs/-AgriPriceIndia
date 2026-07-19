"""
Live Data Module — AgriPrice India
====================================
Fetches the latest commodity prices from the Kaggle dataset and merges
with local historical data.  Falls back gracefully to local data when
the network is unavailable.
"""
import os, subprocess, shutil, datetime as dt
import pandas as pd
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR  = os.path.join(BASE, "data", "raw")
PROC_DIR = os.path.join(BASE, "data", "processed")
DAILY_CSV = os.path.join(PROC_DIR, "agri_prices_daily_national.csv")

KAGGLE_SLUG = "saurabhbadole/indian-agricultural-mandi-prices-20232025"
COMMODITIES = ["Onion", "Potato", "Wheat", "Tomato", "Rice"]

# Detect Streamlit Cloud (no Kaggle CLI / API key available)
_ON_CLOUD = bool(os.environ.get("STREAMLIT_SHARING") or os.path.exists("/mount/src"))

# ── refresh via Kaggle CLI (idempotent) ────────────────────────────────
def _download_kaggle(force: bool = False) -> str | None:
    """Download the Kaggle dataset zip; return path to CSV or None."""
    zip_name = "indian-agricultural-mandi-prices-20232025.zip"
    zip_path = os.path.join(RAW_DIR, zip_name)
    csv_path = os.path.join(RAW_DIR, "Agriculture_price_dataset.csv")

    if not force and os.path.exists(csv_path):
        age_h = (dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(csv_path))).total_seconds() / 3600
        if age_h < 24:
            return csv_path  # fresh enough

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", RAW_DIR, "--force"],
            capture_output=True, text=True, timeout=120,
        )
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(RAW_DIR)
        return csv_path if os.path.exists(csv_path) else None
    except Exception:
        return csv_path if os.path.exists(csv_path) else None


def _clean_raw(csv_path: str) -> pd.DataFrame:
    """Reproduce Phase 1 cleaning on the raw Kaggle CSV."""
    raw = pd.read_csv(csv_path, low_memory=False)

    # Normalise column names
    col_map = {}
    for c in raw.columns:
        cl = c.strip().lower().replace(" ", "_")
        if "commodity" in cl:   col_map[c] = "Commodity"
        elif "modal" in cl and "price" in cl: col_map[c] = "Modal_Price"
        elif "min" in cl and "price" in cl:   col_map[c] = "Min_Price"
        elif "max" in cl and "price" in cl:   col_map[c] = "Max_Price"
        elif "date" in cl or "arrival" in cl: col_map[c] = "Price_Date"
        elif "state" in cl:    col_map[c] = "State"
        elif "market" in cl or "district" in cl: col_map[c] = "Market"
    raw.rename(columns=col_map, inplace=True)

    needed = ["Commodity", "Price_Date", "Modal_Price"]
    if not all(c in raw.columns for c in needed):
        return pd.DataFrame()

    raw["Price_Date"] = pd.to_datetime(raw["Price_Date"], errors="coerce", dayfirst=True)
    raw.dropna(subset=["Price_Date", "Modal_Price"], inplace=True)

    # Filter commodities
    raw["Commodity"] = raw["Commodity"].str.strip().str.title()
    # Map common name variants
    name_map = {"Onion": "Onion", "Potato": "Potato", "Wheat(Husked)": "Wheat",
                "Wheat": "Wheat", "Tomato": "Tomato", "Rice": "Rice",
                "Rice(Paddy-Husked)": "Rice", "Paddy(Dhan)(Common)": "Rice"}
    raw["Commodity"] = raw["Commodity"].map(lambda x: name_map.get(x, x))
    raw = raw[raw["Commodity"].isin(COMMODITIES)].copy()

    for pc in ["Modal_Price", "Min_Price", "Max_Price"]:
        if pc in raw.columns:
            raw[pc] = pd.to_numeric(raw[pc], errors="coerce")

    raw.dropna(subset=["Modal_Price"], inplace=True)
    raw = raw[raw["Modal_Price"] > 0]

    # Aggregate to national daily
    agg = raw.groupby(["Commodity", "Price_Date"]).agg(
        Modal_Price_Avg=("Modal_Price", "mean"),
        Modal_Price_Median=("Modal_Price", "median"),
        Min_Price_Avg=("Min_Price", "mean") if "Min_Price" in raw.columns else ("Modal_Price", "min"),
        Max_Price_Avg=("Max_Price", "mean") if "Max_Price" in raw.columns else ("Modal_Price", "max"),
        Markets_Reporting=("Modal_Price", "count"),
    ).reset_index()
    if "State" in raw.columns:
        states = raw.groupby(["Commodity", "Price_Date"])["State"].nunique().reset_index()
        states.columns = ["Commodity", "Price_Date", "States_Reporting"]
        agg = agg.merge(states, on=["Commodity", "Price_Date"], how="left")
    else:
        agg["States_Reporting"] = np.nan

    agg.sort_values(["Commodity", "Price_Date"], inplace=True)
    return agg


def refresh_data(force: bool = False) -> pd.DataFrame:
    """Main entry: download + clean → return national daily DataFrame."""
    # On Streamlit Cloud, skip Kaggle CLI (no API key / CLI available)
    if not _ON_CLOUD:
        csv = _download_kaggle(force=force)
        if csv and os.path.exists(csv):
            df = _clean_raw(csv)
            if not df.empty:
                df.to_csv(DAILY_CSV, index=False)
                return df

    # fallback: read existing processed file
    if os.path.exists(DAILY_CSV):
        return pd.read_csv(DAILY_CSV, parse_dates=["Price_Date"])
    return pd.DataFrame()


def get_latest_prices(df: pd.DataFrame) -> dict:
    """Return {crop: {price, date, change_1d, change_7d, change_30d, ...}}."""
    out = {}
    for crop in COMMODITIES:
        s = df[df["Commodity"] == crop].sort_values("Price_Date")
        if s.empty:
            out[crop] = {"price": None}
            continue
        last = s.iloc[-1]
        price = float(last["Modal_Price_Median"])
        date = pd.Timestamp(last["Price_Date"])
        markets = int(last.get("Markets_Reporting", 0))

        # changes
        def _change(n):
            if len(s) < n + 1:
                return None
            old = float(s.iloc[-(n+1)]["Modal_Price_Median"])
            return ((price - old) / old) * 100 if old else None

        # 30-day rolling volatility
        vals = s["Modal_Price_Median"].astype(float).values
        vol = float(np.std(vals[-30:]) / np.mean(vals[-30:]) * 100) if len(vals) >= 30 else None

        out[crop] = {
            "price": price, "date": date, "markets": markets,
            "change_1d": _change(1), "change_7d": _change(7), "change_30d": _change(30),
            "volatility_30d": vol,
            "high_52w": float(vals[-min(365, len(vals)):].max()),
            "low_52w": float(vals[-min(365, len(vals)):].min()),
            "sparkline": vals[-60:].tolist() if len(vals) >= 10 else vals.tolist(),
        }
    return out
