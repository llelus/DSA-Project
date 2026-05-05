# DSA-210 Term Project — Spot Market vs Prediction Market Efficiency

**Sabanci University | DSA-210 | Spring 2026**

## Research Question

Does Polymarket efficiently price Bitcoin-related outcomes in real time, or is there a measurable lag compared to the spot market (Kraken)? This project compares two fundamentally different market mechanisms — a continuous spot exchange and a binary prediction market — and examines how quickly new price information propagates between them.

---

## Key Findings

### Hypothesis 1 — BTC Position vs Polymarket Probability
- Pearson r = **0.862**, p < 0.001
- Strong positive correlation between BTC's distance from the prediction threshold and Polymarket yes_price.
- **Result:** Polymarket correctly reflects BTC's general position.

### Hypothesis 2 — Lag Detection
- Cross-lag correlation peaks at **lag = 4 hours**
- Polymarket lags the BTC spot market by approximately 4 hours on average.
- **Result:** Real-time pricing is not occurring; measurable delay exists.

### Hypothesis 3 — Volatility Effect
- t = 3.14, p = 0.0017
- Larger Polymarket updates occur during **low** volatility periods.
- **Result:** High volatility drives prices to 0/1 extremes (ceiling/floor effect), leaving no room for adjustment. Low volatility keeps prices in the uncertain range where updates are more frequent.

### ML Efficiency Measurement

| Model | AUC-ROC | PR-AUC |
|---|---|---|
| Full model (BTC + Polymarket history) | 0.9705 | 0.8167 |
| BTC-only model | 0.6800 | 0.1569 |
| **AUC Gap** | **0.2905** | |

The full model's predictive power comes primarily from Polymarket's own price momentum (`yes_price`, `poly_lag_3`), not from BTC signals. BTC-only model drops to AUC 0.68, confirming that Polymarket moves largely on internal momentum rather than real-time BTC information.

**Overall conclusion:** Polymarket is **partially efficient**. It correctly reflects BTC's direction (~97.3% agreement) but reacts with a ~4-hour delay and relies primarily on its own momentum rather than spot market signals.

---

## Dataset

| | Value |
|---|---|
| Total rows | 96,944 |
| Unique markets | 1,258 |
| Date range | April 9 – May 5, 2026 |
| BTC data | Hourly Kraken OHLCV |
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
| 1 | yes_price | +3.00 |
| 2 | poly_lag_3 | +1.16 |
| 3 | btc_to_threshold_pct | +0.63 |
| 4 | poly_lag_1 | +0.52 |
| 5 | btc_lag_1 | +0.19 |
| 6 | btc_return_1m | +0.14 |
| 7 | rolling_volatility_15m | +0.07 |

---

## AI Usage Disclosure

This project was developed with assistance from Claude (Anthropic). AI was used for: API integration guidance, code debugging, statistical method selection, and interpretation of results. All analytical decisions, data collection strategy, and findings interpretation were reviewed and validated by the student.

## Dependencies

See `requirements.txt`
