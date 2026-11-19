"""Data collection.

* Polymarket: closed "Bitcoin above K on <date>, <hour> ET?" markets (Gamma API)
  and their 1-minute YES-price history (CLOB API).
* BTC spot: Binance BTC/USDT 1-minute klines from the public archive
  data.binance.vision, which has full history (the Kraken OHLC endpoint only
  returns the most recent 720 candles, so it cannot back-fill a study window).

Everything is written under data/raw/ and is resumable: files that already
exist are skipped, so a disconnected Colab session can simply re-run the cell.
"""
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
BINANCE_DAILY = ("https://data.binance.vision/data/spot/daily/klines/"
                 "{sym}/1m/{sym}-1m-{day}.zip")
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore"]


def make_session():
    """requests.Session with retry/back-off on rate limits and server errors."""
    s = requests.Session()
    retry = Retry(total=6, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    return s


def _as_list(x):
    """Gamma returns some list fields as JSON strings."""
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip():
        return json.loads(x)
    return []


def _iso(ts):
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Polymarket: market list ──────────────────────────────────────────────────
def parse_market(m):
    """Flatten one Gamma market dict; None if it is not a BTC threshold market."""
    q = (m.get("question") or "").strip()
    match = re.match(config.MARKET_PATTERN, q)
    if not match:
        return None
    outcomes = [str(o).strip().lower() for o in _as_list(m.get("outcomes"))]
    tokens = [str(t) for t in _as_list(m.get("clobTokenIds"))]
    prices = [float(p) for p in _as_list(m.get("outcomePrices"))]
    if "yes" not in outcomes or len(tokens) != len(outcomes):
        return None
    i = outcomes.index("yes")              # never assume YES is element 0
    start = pd.to_datetime(m.get("startDate"), utc=True, errors="coerce")
    end = pd.to_datetime(m.get("endDate"), utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None
    final = prices[i] if len(prices) == len(outcomes) else np.nan
    # Resolved markets settle at exactly 1/0; anything else is unresolved/void.
    won = 1.0 if final >= 0.99 else (0.0 if final <= 0.01 else np.nan)
    return {
        "condition_id": m.get("conditionId"),
        "question": q,
        "strike": float(match.group(1).replace(",", "")),
        "market_start": start,
        "market_end": end,
        "yes_token": tokens[i],
        "yes_final_price": final,
        "result_yes_won": won,
        "volume": float(m.get("volumeNum") or m.get("volume") or 0.0),
    }


def fetch_markets_page(session, day_start, day_end, offset, limit=500):
    params = {"closed": "true", "limit": limit, "offset": offset,
              "end_date_min": _iso(day_start), "end_date_max": _iso(day_end)}
    r = session.get(GAMMA_MARKETS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_markets(start=config.START_DATE, end=config.END_DATE, session=None,
                  limit=500, max_pages=200, verbose=True):
    """All closed BTC threshold markets whose expiry lies in [start, end).

    Queried one UTC day at a time with the Gamma end-date filter, so the
    result does not depend on how many unrelated markets exist (the v1 code
    only scanned the newest 5,000 closed markets and silently missed the rest).
    """
    s = session or make_session()
    days = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"),
                         freq="D", inclusive="left")
    records, seen = [], set()
    for d0 in days:
        d1 = d0 + pd.Timedelta(days=1)
        n_day = 0
        for page in range(max_pages):
            batch = fetch_markets_page(s, d0, d1, page * limit, limit)
            if not batch:
                break
            if page == 0:
                ends = pd.to_datetime([b.get("endDate") for b in batch], utc=True, errors="coerce")
                in_window = ((ends >= d0 - pd.Timedelta(days=1)) & (ends <= d1 + pd.Timedelta(days=1))).sum()
                if in_window == 0:
                    raise RuntimeError("Gamma API ignored end_date_min/max; "
                                       "check the parameter names before collecting.")
            for m in batch:
                rec = parse_market(m)
                if rec and rec["condition_id"] not in seen and d0 <= rec["market_end"] < d1:
                    seen.add(rec["condition_id"])
                    records.append(rec)
                    n_day += 1
            if len(batch) < limit:
                break
        else:
            print(f"WARNING {d0.date()}: hit max_pages, day may be incomplete")
        if verbose:
            print(f"{d0.date()}: {n_day} markets")
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["market_end", "strike"]).reset_index(drop=True)
    return df


def save_markets(df, path=config.RAW_MARKETS):
    """Merge with any existing market file (keyed by condition_id) and save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = load_markets_csv(path)
        df = pd.concat([old, df]).drop_duplicates("condition_id", keep="last")
    df = df.sort_values(["market_end", "strike"]).reset_index(drop=True)
    out = df.copy()
    for c in ("market_start", "market_end"):
        out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(path, index=False)
    return df


def load_markets_csv(path=config.RAW_MARKETS):
    df = pd.read_csv(path, dtype={"yes_token": str, "condition_id": str})
    for c in ("market_start", "market_end"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


# ── Polymarket: price history ────────────────────────────────────────────────
def fetch_price_history(session, token, start, end, fidelity=config.PRICE_FIDELITY_MIN):
    """YES-token price history between two timestamps → list of (unix_s, price)."""
    params = {"market": token, "startTs": int(pd.Timestamp(start).timestamp()),
              "endTs": int(pd.Timestamp(end).timestamp()), "fidelity": fidelity}
    r = session.get(CLOB_HISTORY, params=params, timeout=30)
    r.raise_for_status()
    return [(int(h["t"]), float(h["p"])) for h in r.json().get("history", [])]


def _history_frame(row, session):
    pts = fetch_price_history(session, row.yes_token,
                              row.market_start - pd.Timedelta(minutes=5),
                              row.market_end + pd.Timedelta(minutes=5))
    if not pts:
        return None
    df = pd.DataFrame(pts, columns=["t", "price"])
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["condition_id"] = row.condition_id
    return df[["condition_id", "ts", "price"]]


def _safe_history_frame(row, session):
    try:
        return _history_frame(row, session), None
    except requests.RequestException as e:
        return None, f"{row.condition_id}: {e}"


def collect_prices(markets, out_dir=config.RAW_PRICES_DIR, workers=4, session=None):
    """Download price histories, one parquet per expiry day (resumable).

    A failing market is logged and skipped; if every market of a day fails the
    API itself is the problem, so collection stops with the first error.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    s = session or make_session()
    summary = []
    for day, grp in markets.groupby(markets["market_end"].dt.strftime("%Y-%m-%d")):
        path = out_dir / f"{day}.parquet"
        if path.exists():
            continue
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda r: _safe_history_frame(r, s), grp.itertuples(index=False)))
        ok = [f for f, _ in results if f is not None]
        errors = [e for _, e in results if e]
        if errors and len(errors) == len(grp):
            raise RuntimeError(f"All {len(grp)} requests failed on {day}; first error: {errors[0]}")
        if ok:
            pd.concat(ok, ignore_index=True).to_parquet(path, index=False)
        summary.append({"day": day, "markets": len(grp), "with_history": len(ok), "errors": len(errors)})
        print(f"{day}: {len(ok)}/{len(grp)} markets with price history"
              + (f", {len(errors)} errors (e.g. {errors[0]})" if errors else ""))
    return pd.DataFrame(summary)


def load_prices(in_dir=config.RAW_PRICES_DIR):
    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No price files in {in_dir}; run notebook 01 first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.drop_duplicates(["condition_id", "ts"]).sort_values(["condition_id", "ts"])


# ── BTC spot: Binance 1-minute klines ────────────────────────────────────────
def parse_binance_klines(raw_csv_bytes):
    df = pd.read_csv(io.BytesIO(raw_csv_bytes), header=None)
    if not str(df.iloc[0, 0]).isdigit():          # some archive files carry a header row
        df = df.iloc[1:]
    df = df.iloc[:, :len(KLINE_COLS)]
    df.columns = KLINE_COLS[:df.shape[1]]
    t = df["open_time"].astype("int64")
    # The archive switched from milliseconds to microseconds in 2025.
    unit = "us" if t.max() > 10**14 else "ms"
    out = pd.DataFrame({"open_time": pd.to_datetime(t, unit=unit, utc=True)})
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = df[c].astype(float).values
    return out


def download_btc_day(day, session, symbol=config.BTC_SYMBOL):
    url = BINANCE_DAILY.format(sym=symbol, day=day)
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return None                                  # not published yet
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        return parse_binance_klines(z.read(z.namelist()[0]))


def collect_btc(start=config.START_DATE, end=config.END_DATE, out_dir=config.RAW_BTC_DIR,
                session=None, warmup_days=1):
    """Download daily 1-minute klines for [start - warmup, end) (resumable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s = session or make_session()
    days = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=warmup_days),
                         pd.Timestamp(end), freq="D", inclusive="left")
    missing = []
    for d in days:
        day = d.strftime("%Y-%m-%d")
        path = out_dir / f"{day}.parquet"
        if path.exists():
            continue
        df = download_btc_day(day, s)
        if df is None:
            missing.append(day)
            continue
        df.to_parquet(path, index=False)
    print(f"BTC days present: {len(list(out_dir.glob('*.parquet')))}, missing: {missing or 'none'}")
    return missing


def load_btc(in_dir=config.RAW_BTC_DIR):
    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No BTC files in {in_dir}; run notebook 01 first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
