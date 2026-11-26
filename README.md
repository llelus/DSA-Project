# DSA-210 Term Project — How Fast Does Polymarket Price Bitcoin Information?

**Sabanci University | DSA-210**

## Research question

Polymarket lists hourly binary markets such as *"Bitcoin above 71,600 on April 9, 4PM ET?"*.
Their outcome is fully determined by the BTC spot price at expiry, so at every minute the
spot price, the strike, the time left and BTC volatility imply a **fair probability** for YES.

> Does the Polymarket price incorporate public BTC price information immediately, or with a
> measurable delay — and if so, how many minutes?

This is a test of **semi-strong form efficiency**: information that is public at time *t*
(the BTC spot price) should not predict Polymarket's subsequent moves.

## Method in one picture

```
Binance BTC/USDT 1-min ──┐
                          ├─► fair_t = P(BTC_T > K | BTC_t, σ_t, τ_t)  ─┐
strike K, expiry T ───────┘                                              ├─► gap_t = fair_t − price_t
Polymarket YES price 1-min ─────────────────────────────────────────────┘
```

| Measurement | Question it answers | Notebook |
|---|---|---|
| **H1 validity / calibration** | Are Polymarket prices informative? Brier score vs the BTC model | 02 |
| **H2 lead-lag** | Do BTC-implied moves come *before* Polymarket moves? (k > 0 = BTC leads) | 02 |
| **H3 volatility** | Does Polymarket move more when BTC is volatile? (cluster-robust) | 02 |
| **Efficiency test** | Do lagged BTC-implied changes predict the next Polymarket move? (joint F-test) | 03 |
| **Impulse response** | Minutes until 50% / 90% of a BTC-implied move is priced — **the lag** | 03 |
| **Mispricing half-life** | How fast the price-vs-fair gap closes | 03 |
| **ML (LightGBM + SHAP)** | Out-of-sample: does BTC information improve forecasts of the next 5-min move? | 04 |

Design choices that matter for correctness:

* **No look-ahead.** A BTC candle's close is used only from its close time; volatility uses a
  trailing window; ML features are lags only.
* **Only pre-expiry minutes.** After expiry the price is mechanically 0/1.
* **Inference on the 0.05–0.95 band**, where the price can still move.
* **Dependence.** Markets that expire in the same hour share one BTC path, so standard errors and
  bootstrap CIs are clustered by expiry hour, and ML folds are blocks of whole expiry hours in time order.
* **Settlement check.** Each market resolves YES if the Binance BTC/USDT **1-hour candle that ends
  at expiry** closes above the strike. That close equals the close of the 1-minute candle opening at
  expiry − 1 min. Notebook 01 recomputes every outcome this way: it reproduced 100% of the 29,228
  markets of Sep–Oct 2026. This confirms that the BTC data is the resolution source and that the
  fair value is anchored to the right price.

## Results

Every number is written by the notebooks to [`results/metrics.json`](results/metrics.json), and
the figures go to [`figures/`](figures/). This section is filled from those files after the
notebooks have run on the collected data. Do not type numbers by hand.

| Quantity | metrics.json key |
|---|---|
| Markets / market-minutes | `data.n_markets`, `dataset.n_market_minutes` |
| Settlement-rule accuracy | `data.settle_accuracy_config` |
| Brier: Polymarket vs BTC model | `h1_validity.brier_polymarket`, `h1_validity.brier_btc_model` |
| Lead-lag peak | `h2_lead_lag.peak_lag_min`, `h2_lead_lag.share_delayed` |
| Efficiency test | `efficiency_test.F`, `efficiency_test.p_value` |
| **Lag (50% / 90% priced)** | `impulse_response.lag_50_min`, `impulse_response.lag_90_min` |
| Mispricing half-life | `error_correction.within.half_life_min` |
| ML: Full − Poly-only | `ml.full_minus_poly_r2`, `ml.full_minus_poly_auc` |

## Repository structure

```
DSA-Project/
├── notebooks/
│   ├── 01_data_collection.ipynb   # download markets, prices, BTC; quality checks; build panel
│   ├── 02_eda_hypothesis.ipynb    # EDA, H1 calibration, H2 lead-lag, H3 volatility
│   ├── 03_fair_value_lag.ipynb    # main analysis: efficiency test, impulse response, half-life
│   └── 04_ml_shap.ipynb           # LightGBM out-of-sample test + SHAP
├── src/                           # all logic lives here; notebooks only call it
│   ├── config.py                  # paths, study window, constants (single source of truth)
│   ├── collect.py                 # Polymarket Gamma/CLOB + Binance archive downloaders
│   ├── dataset.py                 # 1-minute panel, fair value, settlement check
│   ├── analysis.py                # lead-lag, local projections, efficiency test, bootstrap
│   ├── ml.py                      # leakage-safe features, time-block CV
│   ├── results.py                 # save_metrics() → results/metrics.json, savefig() → figures/
│   └── colab.py                   # commit/push from Colab using secrets
├── data/
│   ├── raw/                       # committed: polymarket_markets.csv, polymarket_prices/, btc_1m/
│   └── processed/                 # panel_1m.parquet (derived, git-ignored)
├── figures/                       # every plot, written by the notebooks
├── results/metrics.json           # every reported number, written by the notebooks
├── archive/v1/                    # first version (notebooks, data, README) — see below
└── requirements.txt
```

Raw data is split into **one file per day**. Collection can therefore resume after a Colab
disconnect, git diffs stay small, and the processed panel can always be rebuilt from `data/raw`.

## How to run (Google Colab)

1. Open a notebook from GitHub in Colab (badge at the top of each notebook). The first cell
   clones the repo and installs `requirements.txt`.
2. Optional, to push results back: add Colab secrets `GITHUB_TOKEN`, `GIT_NAME` and `GIT_EMAIL`,
   then set `PUSH = True` in the last cell.
3. Set the study window in `src/config.py` (`START_DATE`, `END_DATE`) and run, in order:
   **01 → 02 → 03 → 04**. Notebook 01 starts with a one-day probe. It shows the market wording and
   resolution rules, so check them before the full download.

Locally: `pip install -r requirements.txt`, then run the notebooks from the repo root or from `notebooks/`.

## What changed from v1 (and why)

The first version is kept in [`archive/v1/`](archive/v1/) for reference. A review found problems
that invalidate its headline result ("Polymarket lags BTC by ~3.5 hours"):

| v1 issue | Consequence | v2 fix |
|---|---|---|
| `statsmodels.ccf(btc, poly)` reads lag k as corr(btc₍t+k₎, poly₍t₎) | The "4h / 3.5h lag" was Polymarket *leading* BTC; the correct direction peaked at 1h | Explicit lead-lag with documented sign; impulse response in minutes |
| 64% of observations were **after market expiry** | Prices were mechanically 0/1, which inflated correlations and ML AUC | Panel uses pre-expiry minutes only |
| Polymarket ticks were 10 minutes apart; markets lived ~1.3 h, not 2–4 days | "1-minute" features were 10-minute features; per-market "lag in hours" was in ticks | 1-minute price history; correct time units |
| Data existed in only 109 of 617 hours (Apr 17–May 2 missing); gaps were forward-filled | Hourly CCF ran mostly on artificial zero changes | Full-history collection by date; coverage reported |
| H3: rows with missing volatility were assigned to the "low volatility" group | The significant result disappeared once these rows were excluded (Welch t = −1.61) | Missing values excluded; cluster-robust test |
| Kraken OHLC API returns only the latest 720 candles | Back-filling silently failed; rows dropped at merge | Binance public archive (full history; checked against outcomes) |
| ML target "next tick up", test fold used for early stopping and threshold choice | Mostly measured "has the market resolved?"; optimistic scores | 5-min price-change target, time-block CV, early stopping inside training |
| SHAP "final model" cell was a Markdown cell | SHAP came from the last fold's model | Final model trained on all data |
| README numbers were typed by hand | Several numbers disagreed with the notebook outputs | All numbers come from `results/metrics.json` |

## Limitations

* The fair value is a simple driftless log-normal model. A persistent gap can mean that Polymarket
  is slow, *or* that it prices something the model ignores (fat tails, jumps). The impulse response
  measures the timing of the reaction rather than its level, so it is less sensitive to this.
* The Polymarket price history is the platform's sampled price, not order-book data. In thin
  markets, a stale price can reflect the absence of trades rather than slow beliefs (`stale_min`
  tracks this).
* Results hold for the chosen study window and for hourly Bitcoin markets. Other assets and other
  market types are untested.

## AI usage disclosure

This project was developed with assistance from Claude (Anthropic). AI was used for API
integration, code review and debugging, statistical method selection and refactoring. All
analytical decisions and the interpretation of results were reviewed by the student.
