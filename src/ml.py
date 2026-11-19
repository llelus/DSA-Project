"""Out-of-sample predictability of Polymarket price moves (notebook 04).

Question: does BTC information that is public at minute t help predict the
Polymarket price change over the next H minutes, beyond Polymarket's own
history? Under semi-strong efficiency it should not.

Leakage rules:
  * every feature is known at t (BTC from closed candles, lags only);
  * folds are blocks of whole expiry hours in time order, so markets that
    expire together never sit on both sides of a split;
  * early stopping uses the last 20% of the TRAINING hours, never the test fold.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .analysis import within_shift

# eval_set still works in every lightgbm 4.x; newer releases only warn about it.
warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated.*")

POLY_FEATURES = ["price", "dp_l0", "dp_l1", "dp_l2", "dp_l3", "dp_l4",
                 "stale_min", "tau_min"]
BTC_FEATURES = ["gap", "fair", "log_moneyness", "sigma_1m",
                "d_fair_l0", "d_fair_l1", "d_fair_l2", "d_fair_l3", "d_fair_l4"]
FULL_FEATURES = POLY_FEATURES + BTC_FEATURES


def make_dataset(panel, horizon=config.ML_HORIZON_MIN, mask=None):
    """Features at t and target y = price_{t+h} - price_t (within market)."""
    df = panel
    feats = pd.DataFrame(index=df.index)
    for j in range(5):
        feats[f"dp_l{j}"] = within_shift(df, "dp", j)
        feats[f"d_fair_l{j}"] = within_shift(df, "d_fair", j)
    for c in ("price", "stale_min", "tau_min", "gap", "fair", "log_moneyness", "sigma_1m"):
        feats[c] = df[c]
    y = within_shift(df, "price", -horizon) - df["price"]
    ok = y.notna() & feats[FULL_FEATURES].notna().all(axis=1)
    if mask is not None:
        ok &= mask.reindex(df.index, fill_value=False)
    out = feats.loc[ok, FULL_FEATURES].copy()
    out["y"] = y[ok]
    out["market_end"] = df.loc[ok, "market_end"]
    return out.sort_values("market_end", kind="stable").reset_index(drop=True)


def hour_block_folds(expiry, n_splits=config.N_SPLITS):
    """Expanding-window folds over sorted unique expiry hours."""
    hours = np.sort(pd.unique(expiry))
    blocks = np.array_split(hours, n_splits + 1)
    for i in range(1, n_splits + 1):
        train_h = np.concatenate(blocks[:i])
        test_h = blocks[i]
        yield np.flatnonzero(np.isin(expiry, train_h)), np.flatnonzero(np.isin(expiry, test_h)), train_h


def _model(n_estimators=2000, seed=config.RANDOM_STATE):
    return lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
                             min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed, verbose=-1)


def cv_predict(data, features, n_splits=config.N_SPLITS):
    """Out-of-fold predictions (NaN for the first, training-only block)."""
    expiry = data["market_end"].to_numpy()
    X, y = data[features], data["y"].to_numpy()
    pred = np.full(len(data), np.nan)
    best_iters = []
    for tr, te, train_h in hour_block_folds(expiry, n_splits):
        cut = train_h[int(len(train_h) * 0.8)]
        fit_idx, val_idx = tr[expiry[tr] < cut], tr[expiry[tr] >= cut]
        m = _model()
        m.fit(X.iloc[fit_idx], y[fit_idx], eval_set=[(X.iloc[val_idx], y[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        best_iters.append(m.best_iteration_ or m.n_estimators)
        pred[te] = m.predict(X.iloc[te], num_iteration=m.best_iteration_)
    return pred, best_iters


def fit_final(data, features, n_estimators):
    m = _model(n_estimators=max(int(n_estimators), 10))
    m.fit(data[features], data["y"])
    return m


def r2_vs_zero(y, pred):
    """R² against the random-walk forecast 'no change' (the efficient benchmark)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(1 - np.sum((y - pred) ** 2) / np.sum(y ** 2))
