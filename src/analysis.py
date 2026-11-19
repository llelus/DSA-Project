"""Statistical tools for the price-discovery analysis.

Conventions (stated explicitly because v1 got the CCF direction backwards):
  * lag k > 0 always means "BTC information at t, Polymarket at t + k".
  * Standard errors are clustered by expiry hour (market_end): all markets that
    expire together see the same BTC path, so they are not independent.
  * Functions take the FULL panel plus an optional boolean `mask` selecting the
    rows (time t) to use. Shifts are computed on the full panel first; filtering
    before shifting would pair minutes that are not adjacent.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

GROUP = "condition_id"
CLUSTER = "market_end"


def within_shift(df, col, k, group=GROUP):
    """col shifted by k rows inside each market (k < 0 = future values).

    Assumes each market's rows are consecutive 1-minute rows sorted by ts,
    which dataset.build_panel guarantees.
    """
    return df.groupby(group, sort=False)[col].shift(k)


def _mask(df, mask):
    return pd.Series(True, index=df.index) if mask is None else mask.reindex(df.index, fill_value=False)


def lead_lag(df, x, y, max_lag=15, mask=None, group=GROUP):
    """corr(x_t, y_{t+k}) pooled across markets, for k = -max_lag..max_lag.

    k > 0: x leads y (BTC → Polymarket if x is the BTC signal).
    k < 0: y leads x.
    """
    m = _mask(df, mask)
    rows = []
    for k in range(-max_lag, max_lag + 1):
        yk = within_shift(df, y, -k, group)
        ok = m & df[x].notna() & yk.notna()
        n = int(ok.sum())
        r = np.corrcoef(df.loc[ok, x], yk[ok])[0, 1] if n > 2 else np.nan
        rows.append({"lag": k, "corr": r, "n": n, "ci95": 1.96 / np.sqrt(n) if n else np.nan})
    return pd.DataFrame(rows)


def _ols_cluster(y, X, clusters):
    X = sm.add_constant(X, has_constant="add")
    codes = pd.factorize(clusters)[0]
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": codes})


def local_projections(df, horizons, shock="d_fair", n_ctrl=3, mask=None,
                      group=GROUP, cluster=CLUSTER):
    """Impulse response of the Polymarket price to a BTC-implied fair-value move.

    For each horizon h:  p_{t+h} - p_{t-1} = a + b_h * d_fair_t
                                           + controls(d_fair_{t-1..t-n}, dp_{t-1..t-n}) + e
    b_h is the share of a fair-value move that is reflected in the price h minutes
    later. Instant pricing → b_0 ≈ b_long; slow pricing → b_h rises gradually.
    Rows whose horizon runs past expiry drop out automatically (NaN lead).
    """
    m = _mask(df, mask)
    base = within_shift(df, "price", 1, group)
    ctrl = {}
    for j in range(1, n_ctrl + 1):
        ctrl[f"{shock}_l{j}"] = within_shift(df, shock, j, group)
        ctrl[f"dp_l{j}"] = within_shift(df, "dp", j, group)
    ctrl = pd.DataFrame(ctrl, index=df.index)
    rows = []
    for h in horizons:
        yh = within_shift(df, "price", -h, group) - base
        X = pd.concat([df[[shock]], ctrl], axis=1)
        ok = m & yh.notna() & X.notna().all(axis=1)
        if ok.sum() < 50:
            continue
        res = _ols_cluster(yh[ok], X[ok], df.loc[ok, cluster])
        b, se = res.params[shock], res.bse[shock]
        rows.append({"h": h, "b": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se,
                     "n": int(ok.sum())})
    return pd.DataFrame(rows)


def response_lags(lp, shares=(0.5, 0.9), long_run_h=None):
    """First horizon (minutes) at which b_h reaches each share of the long-run response."""
    long_run_h = lp["h"].max() if long_run_h is None else long_run_h
    b_long = lp.loc[lp["h"] == long_run_h, "b"].iloc[0]
    out = {"b_long": float(b_long), "b0_share": float(lp["b"].iloc[0] / b_long)}
    for s in shares:
        hit = lp.loc[lp["b"] >= s * b_long, "h"]
        out[f"lag_{int(s * 100)}_min"] = int(hit.iloc[0]) if len(hit) else None
    return out


def error_correction(df, mask=None, demean_gap=True, group=GROUP, cluster=CLUSTER):
    """dp_{t+1} = a + beta * gap_t + e.  Half-life of a mispricing = ln 0.5 / ln(1 - beta).

    demean_gap removes each market's average gap, so a constant model bias
    (e.g. fat tails the log-normal model misses) is not mistaken for slow pricing.
    """
    m = _mask(df, mask)
    y = within_shift(df, "price", -1, group) - df["price"]
    gap = df["gap"]
    if demean_gap:
        gap = gap - gap.where(m).groupby(df[group], sort=False).transform("mean")
    ok = m & y.notna() & gap.notna()
    res = _ols_cluster(y[ok], gap[ok].rename("gap"), df.loc[ok, cluster])
    beta, se = res.params["gap"], res.bse["gap"]
    hl = np.log(0.5) / np.log(1 - beta) if 0 < beta < 1 else np.nan
    return {"beta": float(beta), "se": float(se), "t": float(beta / se),
            "half_life_min": float(hl), "n": int(ok.sum())}


def efficiency_test(df, n_lags=5, mask=None, group=GROUP, cluster=CLUSTER):
    """Semi-strong efficiency test.

    dp_{t+1} = a + sum_{j=0..L} c_j d_fair_{t-j} + sum_{j=0..L} d_j dp_{t-j} + e
    H0 (efficient): every c_j = 0 — BTC information already public at t does not
    predict the next Polymarket move. Joint cluster-robust F-test.
    """
    y = within_shift(df, "price", -1, group) - df["price"]
    X = {}
    for j in range(n_lags + 1):
        X[f"d_fair_l{j}"] = within_shift(df, "d_fair", j, group)
        X[f"dp_l{j}"] = within_shift(df, "dp", j, group)
    X = pd.DataFrame(X, index=df.index)
    ok = _mask(df, mask) & y.notna() & X.notna().all(axis=1)
    res = _ols_cluster(y[ok], X[ok], df.loc[ok, cluster])
    names = list(res.params.index)
    fair_cols = [c for c in names if c.startswith("d_fair_l")]
    R = np.zeros((len(fair_cols), len(names)))
    for i, c in enumerate(fair_cols):
        R[i, names.index(c)] = 1.0
    ft = res.f_test(R)
    coefs = pd.DataFrame({"coef": res.params[fair_cols], "se": res.bse[fair_cols]})
    return {"F": float(np.squeeze(ft.fvalue)), "p_value": float(ft.pvalue),
            "n": int(ok.sum()), "sum_coef": float(res.params[fair_cols].sum())}, coefs


def two_group_test(a, b):
    """Welch t-test + Mann-Whitney U (robust to the heavy zero-inflation of |dp|)."""
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    w = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "welch_t": float(w.statistic), "welch_p": float(w.pvalue),
            "mwu_p": float(u.pvalue)}


def mean_diff_cluster(y, flag, clusters):
    """Difference in means of y between flag==1 and flag==0, cluster-robust SE.

    Minute-level rows of the same expiry hour are strongly dependent, so a
    plain t-test overstates significance; this is the regression y = a + d*flag.
    """
    y = pd.Series(np.asarray(y, float))
    flag = pd.Series(np.asarray(flag, float), name="flag")
    ok = y.notna() & flag.notna()
    res = _ols_cluster(y[ok], flag[ok], np.asarray(clusters)[ok.to_numpy()])
    d, se = res.params["flag"], res.bse["flag"]
    return {"diff": float(d), "se": float(se), "t": float(d / se), "p_value": float(res.pvalues["flag"]),
            "n": int(ok.sum())}


def brier(y, p):
    """Brier score; argument order (y_true, y_pred) like scikit-learn."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[ok] - y[ok]) ** 2))


def cluster_bootstrap_diff(metric, a, b, y, clusters, n_boot=1000, seed=42):
    """95% CI for metric(y, a) - metric(y, b), resampling whole clusters.

    metric takes (y_true, y_pred) like scikit-learn metrics.
    """
    rng = np.random.default_rng(seed)
    a, b, y = np.asarray(a), np.asarray(b), np.asarray(y)
    codes, uniq = pd.factorize(np.asarray(clusters))
    members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        try:
            diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
        except ValueError:          # e.g. a resample with a single class
            continue
    diffs = np.array(diffs)
    return {"diff": float(metric(y, a) - metric(y, b)),
            "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))}
