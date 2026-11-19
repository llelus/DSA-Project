"""Build the 1-minute market panel used by every analysis notebook.

One row = one market at one minute t strictly before expiry, with
  price      last Polymarket YES price observed at or before t
  stale_min  minutes since that price point was published
  btc        last *closed* BTC 1-minute candle close known at t (no look-ahead)
  sigma_1m   trailing realised volatility of 1-minute BTC log returns
  tau_min    minutes from t until the settlement price is known
  fair       BTC-implied probability that the market resolves YES
  gap        fair - price  (positive = Polymarket too low relative to BTC)
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config
from .collect import load_btc, load_markets_csv, load_prices


def btc_minute_frame(btc):
    """BTC close indexed by the minute it becomes KNOWN (candle close time).

    A candle that opens at 12:00 closes at 12:01, so its close is only usable
    from 12:01 on. Indexing by open time would leak one minute of future data.
    """
    s = btc.set_index("open_time")["close"].sort_index()
    s.index = s.index + pd.Timedelta(minutes=1)
    full = pd.date_range(s.index.min(), s.index.max(), freq="1min")
    s = s.reindex(full).ffill()
    logret = np.log(s).diff()
    sigma = logret.rolling(config.VOL_WINDOW_MIN,
                           min_periods=config.VOL_WINDOW_MIN // 2).std()
    out = pd.DataFrame({"btc": s, "btc_logret": logret, "sigma_1m": sigma})
    out.index.name = "ts"
    return out


def settlement_close(markets, btc, offset_min=config.SETTLE_CANDLE_OFFSET_MIN):
    """Close of the candle opening at market_end + offset (NaN if missing)."""
    closes = btc.set_index("open_time")["close"]
    keys = markets["market_end"] + pd.Timedelta(minutes=offset_min)
    return pd.Series(closes.reindex(keys).values, index=markets.index)


def resolution_check(markets, btc, offsets=(-1, 0, 1)):
    """How often 'settlement close > strike' reproduces the recorded outcome.

    Accuracy near 100% for one offset confirms both the settlement rule and
    that our BTC series is the resolution source; much lower values mean the
    fair-value model is anchored to the wrong price.
    """
    m = markets.dropna(subset=["result_yes_won"])
    rows = []
    for off in offsets:
        close = settlement_close(m, btc, off)
        ok = close.notna()
        pred = (close[ok] > m.loc[ok, "strike"]).astype(int)
        acc = (pred == m.loc[ok, "result_yes_won"].astype(int)).mean()
        rows.append({"offset_min": off, "n": int(ok.sum()), "accuracy": float(acc)})
    return pd.DataFrame(rows)


def fair_probability(S, K, sigma_1m, tau_min):
    """P(S_T > K) under driftless log-normal BTC with per-minute vol sigma_1m."""
    S, K = np.asarray(S, float), np.asarray(K, float)
    sd = np.asarray(sigma_1m, float) * np.sqrt(np.asarray(tau_min, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (np.log(S / K) - 0.5 * sd ** 2) / sd
    p = norm.cdf(d)
    p[~np.isfinite(d) | ~(sd > 0)] = np.nan
    return p


def build_panel(markets=None, prices=None, btc=None,
                offset_min=config.SETTLE_CANDLE_OFFSET_MIN):
    markets = load_markets_csv() if markets is None else markets
    prices = load_prices() if prices is None else prices
    btc = load_btc() if btc is None else btc

    m = markets.dropna(subset=["result_yes_won"]).copy()
    m = m[(m["market_end"] - m["market_start"]) <= pd.Timedelta(hours=config.MAX_MARKET_HOURS)]
    by_market = {k: g for k, g in prices.groupby("condition_id")}

    frames = []
    for r in m.itertuples(index=False):
        p = by_market.get(r.condition_id)
        if p is None or p.empty:
            continue
        p = p.sort_values("ts")
        p_ts = pd.DatetimeIndex(p["ts"])
        p_val = p["price"].to_numpy()
        start = max(p_ts.min(), r.market_start).ceil("1min")
        grid = pd.date_range(start, r.market_end.floor("1min"), freq="1min", inclusive="left")
        if len(grid) == 0:
            continue
        idx = p_ts.searchsorted(grid, side="right") - 1      # last point at or before t
        frames.append(pd.DataFrame({
            "condition_id": r.condition_id,
            "ts": grid,
            "market_end": r.market_end,
            "strike": r.strike,
            "result_yes_won": int(r.result_yes_won),
            "price": p_val[idx],
            "stale_min": (grid - p_ts[idx]).total_seconds() / 60,
        }))
    if not frames:
        raise ValueError("No market has price history inside its lifetime.")
    panel = pd.concat(frames, ignore_index=True)

    panel = panel.join(btc_minute_frame(btc), on="ts")
    settle_known = panel["market_end"] + pd.Timedelta(minutes=offset_min + 1)
    panel["tau_min"] = (settle_known - panel["ts"]).dt.total_seconds() / 60
    panel["fair"] = fair_probability(panel["btc"], panel["strike"],
                                     panel["sigma_1m"], panel["tau_min"])
    panel["gap"] = panel["fair"] - panel["price"]
    panel["log_moneyness"] = np.log(panel["btc"] / panel["strike"])
    panel = panel.sort_values(["condition_id", "ts"]).reset_index(drop=True)
    return panel


def load_panel(rebuild=False, **kwargs):
    """Read the cached panel, or build it from data/raw and cache it."""
    if config.PANEL.exists() and not rebuild:
        return pd.read_parquet(config.PANEL)
    panel = build_panel(**kwargs)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.PANEL, index=False)
    return panel


def add_market_diffs(panel):
    """Within-market 1-minute changes: dp (Polymarket) and d_fair (BTC-implied)."""
    g = panel.groupby("condition_id", sort=False)
    panel["dp"] = g["price"].diff()
    panel["d_fair"] = g["fair"].diff()
    return panel


def analysis_sample(panel, mid_only=False):
    """Rows usable for inference: fair value defined, not in the final minutes."""
    ok = panel["fair"].notna() & (panel["tau_min"] >= config.MIN_TAU_MIN)
    if mid_only:
        ok &= panel["price"].between(config.MID_LOW, config.MID_HIGH)
    return panel[ok]
