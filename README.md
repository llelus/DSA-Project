# DSA-Project
Sabancı University dsa-210 term project
### Proposal
## Spot Market vs. Prediction Market Lag: Binance vs. Polymarket

# Data Source and Collection

The primary dataset will be sourced from the Polymarket CLOB (Central Limit Order Book) API, which provides public, unauthenticated access to historical per-minute price series for prediction market contracts. A BTC price-related market will be selected based on three criteria: total trading volume exceeding $50,000, a current contract price between 0.15 and 0.85 (ensuring the outcome remains genuinely uncertain), and a resolution date at least four weeks in the future. This selection process will be performed programmatically before data collection begins.
As the enrichment dataset, one-minute OHLCV (Open, High, Low, Close, Volume) data for the BTC/USDT pair will be collected via CCXT, a unified cryptocurrency exchange library, connecting to Binance as the primary source. CCXT allows seamless switching to alternative exchanges (Kraken, Coinbase) if access issues arise, without modifying any downstream code. Both datasets will cover a three-month window (January 2025 — March 2025) and will be aligned to a common one-minute timestamp index using pandas merge_asof, which matches each Polymarket observation to the nearest available Binance bar within a five-minute tolerance.

# Dataset Characteristics

The merged dataset is expected to contain approximately 130,000 rows, each representing one minute of observations. Key columns will include the Binance close price, one-minute percentage return, and trading volume, alongside the Polymarket yes-token price (a value in [0, 1] representing the implied probability of the event), and its minute-over-minute change rate. Derived features will include 15-minute rolling returns and volatility for both series, lag-shifted Polymarket prices, and a binary drop-event indicator triggered when the BTC return falls below a predefined threshold (e.g., −2% within 15 minutes).


