# DSA-210 Term Project — Spot Market vs Prediction Market Efficiency

**Sabanci University | DSA-210 | Spring 2026**

## Research Question

Does Polymarket efficiently price Bitcoin-related outcomes in real time, or is there a measurable lag compared to the spot market (Kraken)? This project compares two fundamentally different market mechanisms — a continuous spot exchange and a binary prediction market — and examines how quickly new price information propagates between them.

---

## Key Findings

### Hypothesis 1 — BTC Position vs Polymarket Probability
- Pearson r = **0.852**, p < 0.001
- Strong positive correlation between BTC's distance from the prediction threshold and Polymarket yes_price.
- **Result:** Polymarket correctly reflects BTC's general position.

### Hypothesis 2 — Lag Detection
- Cross-lag correlation peaks at **lag = 4 hours** (CCF = 0.0935, Granger p < 0.0001 for all lags 1–6h)
- Confirmed independently by both CCF on hourly aggregated signal (02_eda_hypothesis) and per-market Pearson cross-lag (03_ml_shap).
- **Result:** Real-time pricing is not occurring; Polymarket systematically reprices ~4 hours after BTC moves.

### Hypothesis 3 — Volatility Effect
- t = 5.68, p < 0.0001
- Larger Polymarket updates occur during **low** volatility periods.
- **Result:** High volatility drives prices to 0/1 extremes (ceiling/floor effect), leaving no room for adjustment. Low volatility keeps prices in the uncertain range where updates are more frequent.

### ML Efficiency Measurement

| Model | AUC-ROC | 95% CI | PR-AUC |
|---|---|---|---|
| Full model (BTC + Polymarket history) | 0.9775 | [0.9724, 0.9825] | 0.8343 |
| Poly-only (autocorrelation baseline) | 0.9705 | [0.9631, 0.9772] | — |
| BTC-only model | 0.6800 | [0.6576, 0.7015] | 0.1569 |
| **Full vs BTC-only gap** | **0.2975** | | |

Confidence intervals computed via bootstrap resampling (n=1000).

The full model's predictive power comes primarily from Polymarket's own price momentum (`yes_price`, `poly_lag_3`), confirmed by the Poly-only ablation (AUC 0.9705 ≈ full model). BTC-only model drops to AUC 0.68 — statistically significantly better than random (lower CI 0.66 > 0.50) but far below the full model (CIs do not overlap). This confirms Polymarket moves largely on internal momentum rather than real-time BTC information.

**Overall conclusion:** Polymarket is **partially efficient**. It correctly reflects BTC's direction (~97.3% agreement) but reacts with a ~4-hour delay and relies primarily on its own momentum rather than spot market signals.

---

## Dataset

| | Value |
|---|---|
| Total rows | 96,944 |
| Unique markets | 1,258 |
| Date range | April 9 – May 5, 2026 |
| BTC data | 1-minute Kraken OHLCV |
| Missing btc_price | 0 |

Key features: `yes_price`, `btc_price`, `threshold`, `btc_to_threshold_pct`, `btc_return_1m`, `rolling_volatility_15m`, `poly_lag_1`, `poly_lag_3`

---

## Repository Structure

```
DSA-Project/
├── 01_data_collection.ipynb   # Data pipeline: Polymarket + Kraken fetch & merge
├── 02_eda_hypothesis.ipynb    # EDA, visualizations, hypothesis tests (H1, H2, H3)
├── 03_ml_shap.ipynb           # LightGBM, AUC-ROC, threshold opt, BTC-only, lag measurement
├── data/
│   ├── raw/                   # kraken_raw.csv, polymarket_raw.csv
│   └── processed/             # merged_dataset.csv, all plots
├── requirements.txt
└── README.md
```

---

## How to Reproduce

```bash
git clone https://github.com/llelus/DSA-Project.git
cd DSA-Project
pip install -r requirements.txt
```

Open notebooks in order on Google Colab:
1. `01_data_collection.ipynb` — fetch and merge data
2. `02_eda_hypothesis.ipynb` — EDA and hypothesis tests
3. `03_ml_shap.ipynb` — ML pipeline and efficiency measurement

---

## SHAP Feature Importance

| Rank | Feature | Mean SHAP |
|---|---|---|
| 1 | yes_price | +2.95 |
| 2 | poly_lag_3 | +1.05 |
| 3 | btc_to_threshold_pct | +0.56 |
| 4 | poly_lag_1 | +0.49 |
| 5 | btc_lag_1 | +0.15 |
| 6 | btc_return_1m | +0.11 |
| 7 | rolling_volatility_15m | +0.06 |

---

## Discussion — Practical Implications of the 4-Hour Lag

### What does the lag mean?
When BTC crosses a prediction market threshold, Polymarket participants take approximately
4 hours to fully reprice the outcome. During this window, the market trades at a
probability that no longer reflects available spot price information — a clear departure
from strong-form efficiency.

### Is arbitrage theoretically possible?
Yes, under idealized conditions. A trader who monitors BTC spot price in real time could:
1. Observe BTC move above/below a threshold at time T
2. Buy the underpriced outcome on Polymarket while it still reflects T−4h information
3. Capture the repricing as other participants update over the next ~4 hours

The 2.7% disagreement rate (226/8,269 observations where BTC and Polymarket disagree
on direction) represents the observable arbitrage surface. The highest disagreement
occurs when Polymarket certainty is 0.1–0.2 (16.7% disagreement) — these are the
most exploitable moments.

### Why is the lag not fully arbitraged away?
Several structural frictions prevent efficient exploitation:

- **Liquidity**: Bitcoin prediction markets on Polymarket are thinly traded. Large
  positions move the price and erode the edge before it is captured.
- **Transaction costs**: Polymarket operates on-chain (Polygon). Gas fees and
  USDC bridging costs reduce net profit on small mispricings.
- **Market lifetime**: These markets expire in 2–4 days. A 4-hour lag consumes
  a significant fraction of remaining market life, compressing the holding window.
- **Participant composition**: Polymarket's Bitcoin market participants are largely
  retail traders without automated BTC monitoring — the lag exists precisely because
  those who could close it choose not to (or cannot afford to).

### Who could exploit it?
The lag is most actionable for algorithmic traders with real-time BTC price feeds,
low transaction costs, and the ability to hold positions across multiple concurrent
markets. For a retail participant operating manually, the combination of low
liquidity, on-chain friction, and the need for continuous monitoring makes systematic
exploitation impractical.

This finding is consistent with the **adaptive market hypothesis** (Lo, 2004):
markets are not statically efficient or inefficient — they reflect the capabilities
and incentives of their current participant base. Polymarket's 4-hour lag will likely
compress as the platform matures and attracts more sophisticated liquidity providers.

---

## AI Usage Disclosure

This project was developed with assistance from Claude (Anthropic). AI was used for: API integration guidance, code debugging, statistical method selection, and interpretation of results. All analytical decisions, data collection strategy, and findings interpretation were reviewed and validated by the student.

## Dependencies

See `requirements.txt`
