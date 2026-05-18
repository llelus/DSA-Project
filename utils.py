import pandas as pd
import numpy as np


def colab_setup():
    """Install required packages in a Colab session."""
    import subprocess, sys
    pkgs = ["ccxt", "lightgbm", "shap", "statsmodels"]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + pkgs,
        check=True
    )
    print("Packages ready:", ", ".join(pkgs))


def load_and_prepare(csv_path="data/processed/merged_dataset.csv"):
    """Load merged dataset, filter threshold markets, compute standard features.

    Returns a DataFrame with columns:
        threshold, btc_to_threshold_pct, btc_return_1m,
        poly_change_1m, rolling_volatility_15m
    """
    df = pd.read_csv(csv_path)
    df["timestamp"]    = pd.to_datetime(df["timestamp"],    utc=True)
    df["market_start"] = pd.to_datetime(df["market_start"], utc=True)
    df["market_end"]   = pd.to_datetime(df["market_end"],   utc=True)

    # Exclude "Up or Down" markets -- no fixed threshold, structurally different
    before = len(df)
    df = df[df["question"].str.contains("above|below", case=False, na=False)].copy()
    print(f"Loaded {len(df):,} rows ({before - len(df):,} Up-or-Down rows excluded)")
    print(f"Unique markets : {df['conditionId'].nunique()}")
    print(f"Date range     : {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # Threshold price from question text
    df["threshold"] = (
        df["question"]
        .str.extract(r"above ([\d,]+)")
        .replace(",", "", regex=True)
        .astype(float)
    )
    df["btc_to_threshold_pct"] = (
        (df["btc_price"] - df["threshold"]) / df["threshold"] * 100
    )

    # Per-market time-series features
    df = df.sort_values("timestamp")
    df["btc_return_1m"] = (
        df.groupby("conditionId")["btc_price"]
        .pct_change(fill_method=None) * 100
    )
    df["poly_change_1m"] = df.groupby("conditionId")["yes_price"].diff()
    df["rolling_volatility_15m"] = (
        df.groupby("conditionId")["btc_price"]
        .transform(
            lambda x: x.pct_change(fill_method=None)
            .rolling(window=15, min_periods=3).std() * 100
        )
    )

    return df
