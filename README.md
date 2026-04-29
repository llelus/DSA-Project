# DSA-Project
Sabancı University dsa-210 term project
### Proposal
## Spot Market vs. Prediction Market Lag: Binance vs. Polymarket

# Data Source and Collection

The primary dataset will be sourced from the Polymarket CLOB (Central Limit Order Book) API, which provides public, unauthenticated access to historical per-minute price series for prediction market contracts. A BTC price-related market will be selected based on three criteria: total trading volume exceeding $50,000, a current contract price between 0.15 and 0.85 (ensuring the outcome remains genuinely uncertain), and a resolution date at least four weeks in the future. This selection process will be performed programmatically before data collection begins.
As the enrichment dataset, one-minute OHLCV (Open, High, Low, Close, Volume) data for the BTC/USDT pair will be collected via CCXT, a unified cryptocurrency exchange library, connecting to Binance as the primary source. CCXT allows seamless switching to alternative exchanges (Kraken, Coinbase) if access issues arise, without modifying any downstream code. Both datasets will cover a three-month window (January 2025 — March 2025) and will be aligned to a common one-minute timestamp index using pandas merge_asof, which matches each Polymarket observation to the nearest available Binance bar within a five-minute tolerance.

# Dataset Characteristics

The merged dataset is expected to contain approximately 130,000 rows, each representing one minute of observations. Key columns will include the Binance close price, one-minute percentage return, and trading volume, alongside the Polymarket yes-token price (a value in [0, 1] representing the implied probability of the event), and its minute-over-minute change rate. Derived features will include 15-minute rolling returns and volatility for both series, lag-shifted Polymarket prices, and a binary drop-event indicator triggered when the BTC return falls below a predefined threshold (e.g., −2% within 15 minutes).



## Project Overview
This project investigates whether prediction markets (Polymarket) efficiently 
price Bitcoin-related outcomes in real time, or whether there is a measurable 
lag compared to the spot market (Kraken/Binance). Unlike standard crypto price 
analyses, this project compares two fundamentally different market mechanisms: 
a continuous spot exchange and a binary prediction market — examining how 
quickly new price information propagates between them.

## Research Questions
1. Does BTC's position relative to a prediction market threshold explain 
   Polymarket probability levels? (Hypothesis 1)
2. Do Polymarket probabilities update with a measurable delay after BTC 
   spot price movements? (Hypothesis 2)
3. Does market volatility affect the speed of Polymarket's price updates? 
   (Hypothesis 3)

## Data Sources
- **Primary:** Polymarket CLOB API — per-minute price history of 
  "Bitcoin above $X" prediction markets
- **Enrichment:** Kraken API via CCXT — 1-minute BTC/USDT OHLCV spot data

Both datasets are merged on a shared minute-level timestamp index using 
`pandas.merge_asof`.

## Dataset
- ~1,000+ rows (growing daily), each representing one minute of observation
- 60+ unique prediction market contracts
- Key variables: `yes_price`, `btc_price`, `threshold`, 
  `btc_to_threshold_pct`, `btc_return_1m`, `poly_change_1m`, 
  `rolling_volatility_15m`

## Repository Structure

DSA-Project/
├── data/
│   ├── raw/                  # Raw API data
│   └── processed/            # Merged dataset and plot outputs
├── notebooks/
│   ├── 01_data_collection.ipynb   # Data fetching and merging pipeline
│   └── 02_eda_hypothesis.ipynb    # EDA, visualizations, hypothesis tests
├── README.md
└── requirements.txt


## How to Reproduce
1. Clone the repo
```bash
git clone https://github.com/llelus/DSA-Project.git
cd DSA-Project
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run data collection

Open notebooks/01_data_collection.ipynb and run all cells

4. Run analysis
Open notebooks/02_eda_hypothesis.ipynb and run all cells

## Key Findings (so far)
- **Hypothesis 1:** Strong positive correlation (r=0.862, p<0.001) between 
  BTC's distance from the prediction threshold and Polymarket probability — 
  the market accurately prices BTC position.
- **Hypothesis 2:** No statistically significant lag detected with current 
  data volume; more data required for reliable CCF/Granger results.
- **Hypothesis 3:** Counterintuitively, Polymarket shows larger price updates 
  during low volatility periods (p=0.0017), likely because high volatility 
  drives prices to 0/1 extremes leaving no room for adjustment.

## AI Usage Disclosure
This project was developed with assistance from Claude (Anthropic). 
AI was used for: API integration guidance, code debugging, statistical 
method selection, and interpretation of results. All analytical decisions, 
data collection strategy, and findings interpretation were reviewed and 
validated by the student.

## Dependencies
See `requirements.txt`




