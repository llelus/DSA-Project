"""Project-wide configuration: paths, study window and analysis constants.

Every notebook imports from here, so changing the study window or a threshold
in one place changes it everywhere.
"""
import os
from pathlib import Path

# Repository root. DSA_ROOT lets tests point the pipeline at a scratch copy.
ROOT = Path(os.environ.get("DSA_ROOT", Path(__file__).resolve().parents[1]))
REPO_SLUG = "llelus/DSA-Project"

# ── Paths ────────────────────────────────────────────────────────────────────
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_MARKETS = RAW / "polymarket_markets.csv"   # one row per market (metadata + outcome)
RAW_PRICES_DIR = RAW / "polymarket_prices"     # one parquet per market-expiry UTC day
RAW_BTC_DIR = RAW / "btc_1m"                   # one parquet per UTC day (Binance klines)
PROCESSED = DATA / "processed"
PANEL = PROCESSED / "panel_1m.parquet"         # derived, rebuilt from raw (git-ignored)
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics.json"

# ── Study window (UTC). Markets are selected by EXPIRY time in [START, END). ──
START_DATE = "2026-09-01"
END_DATE = "2026-11-01"

# ── Market selection ─────────────────────────────────────────────────────────
# Matches e.g. "Bitcoin above 71,600 on April 9, 4PM ET?"
MARKET_PATTERN = r"^Bitcoin above \$?([\d,]+(?:\.\d+)?) on (.+?)\?$"
MAX_MARKET_HOURS = 3          # keep short-lived (hourly) markets only
PRICE_FIDELITY_MIN = 1        # requested Polymarket price-history resolution

# ── BTC spot data ────────────────────────────────────────────────────────────
BTC_SYMBOL = "BTCUSDT"        # Binance spot, 1-minute klines (data.binance.vision)
# Settlement candle = the 1-minute candle that OPENS at market_end + this offset.
# Notebook 01 checks which offset reproduces the recorded outcomes best.
SETTLE_CANDLE_OFFSET_MIN = 0

# ── Analysis constants ───────────────────────────────────────────────────────
VOL_WINDOW_MIN = 60           # trailing window for realised BTC volatility
MIN_TAU_MIN = 2               # drop the last minutes before settlement
MID_LOW, MID_HIGH = 0.05, 0.95  # "uncertain" price band used by tests and ML
LP_HORIZONS = list(range(0, 31))  # local-projection horizons (minutes)
ML_HORIZON_MIN = 5            # ML target: price change over the next 5 minutes
N_SPLITS = 5
N_BOOT = 1000
RANDOM_STATE = 42
