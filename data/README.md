# Data Directory Documentation

This directory contains the datasets used for the cryptocurrency event study analysis.

## Data Sources

### Price Data Files
- **btc.csv, eth.csv, xrp.csv, bnb.csv, ltc.csv, ada.csv**
  - Source: CoinGecko API
  - Coverage: January 2019 - August 2025
  - Frequency: Daily
  - Contains: OHLC prices, volume, market cap
  - Format: `snapped_at` (timestamp), `price`, `market_cap`, `total_volume`

### Event Data
- **events.csv**
  - Source: Manual classification from news sources
  - Coverage: 50 events from 2019-2025
  - Types: 27 Infrastructure events, 23 Regulatory events
  - Format: `event_id`, `date`, `name`, `type`, `description`

### Sentiment Data
- **gdelt.csv**
  - Source: GDELT Project (Global Database of Events, Language, and Tone)
  - Coverage: Weekly sentiment indicators
  - Processing: Z-score normalized with 52-week rolling window
  - Decomposition: Infrastructure and regulatory proportions
  - Format: `week_start`, `S_gdelt_raw`, `reg_proportion`, `infra_proportion`

## Data Processing Notes

1. **Timezone Handling**: All timestamps are converted to UTC for consistency
2. **Returns Calculation**: Log returns calculated as ln(P_t/P_{t-1}) × 100
3. **Winsorization**: Returns winsorized at 5 standard deviations
4. **Sentiment Forward-Fill**: Weekly sentiment data forward-filled to daily frequency

## Event Window Specifications

### Standard Events
- Window: [-3, +3] days around event date
- Total: 7 days per event

### Special Event Handling
1. **SEC Twin Suits (Events 31-32, June 2023)**: 
   - Combined into single composite dummy
   - Window: June 2-9, 2023

2. **EIP-1559 & Polygon Hack Overlap (Events 17-18, August 2021)**:
   - Overlapping days (Aug 7-8) given 0.5 weight each
   - Prevents double-counting volatility

3. **Bybit & SEC Dismissal (Events 43-44, February 2025)**:
   - Truncated windows to prevent overlap
   - Bybit: ends Feb 23
   - SEC: starts Feb 27

## Data Quality Notes

- Missing values before June 2019 in sentiment data (initialization period)
- All cryptocurrency data complete within analysis window
- Event classifications validated against multiple news sources

## Usage

To use this data with the analysis scripts:

```python
from code.data_preparation import DataPreparation

# Load and prepare all data
prep = DataPreparation()
crypto_data = prep.prepare_all_cryptos(
    include_events=True,
    include_sentiment=True
)
```

## License

Price data subject to CoinGecko API terms of service.
GDELT data is publicly available under Creative Commons Attribution 4.0.
Event classifications are original research contributions.
