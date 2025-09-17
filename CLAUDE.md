# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Cryptocurrency Event Study Analysis project that examines differential volatility impacts of Infrastructure vs Regulatory events using GARCH models with sentiment controls. The project analyzes 50 classified events (27 Infrastructure, 23 Regulatory) from 2019-2025 across major cryptocurrencies (BTC, ETH, XRP, BNB, LTC, ADA).

## Key Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Data Fetching
```python
from code.coingecko_fetcher import CoinGeckoFetcher

fetcher = CoinGeckoFetcher()
# Fetch event window data
data = fetcher.fetch_event_window(['BTC', 'ETH'], '2020-03-12', days_before=30, days_after=30)
```

### Testing (when implemented)
```bash
pytest tests/
```

## Architecture

### Core Components

1. **CoinGeckoFetcher** (`code/coingecko_fetcher.py`): Main data fetching class
   - Handles API rate limiting (1.2 second intervals)
   - Fetches OHLC, volume, and market cap data
   - Supports batch processing with event windows
   - Auto-retries on rate limit errors (429)

2. **Data Organization**:
   - `data/`: Contains crypto price CSVs, events.csv (50 classified events), gdelt.csv (sentiment data)
   - `outputs/`: For analysis results (figures/ and tables/ subdirectories)
   - `docs/`: Comprehensive methodology documentation

3. **Analysis Framework** (planned):
   - GARCH(1,1): Baseline volatility model
   - TARCH(1,1): Asymmetric volatility responses
   - TARCH-X: Extended model with event dummies and sentiment

### Event Window Specifications

- Standard window: [-3, +3] days around events
- Special overlapping events require custom handling:
  - SEC Twin Suits (June 2023): Composite dummy
  - EIP-1559 & Poly Hack (August 2021): Overlap adjustment
  - Bybit Hack & SEC Dismissal (February 2025): Truncated windows

### Data Processing Standards

- Returns calculation: Log returns r_t = ln(P_t/P_{t-1}) × 100
- Outlier treatment: Winsorization at 5 standard deviations
- Sentiment data: Forward-fill weekly to daily frequency

## Environment Configuration

- Requires `COINGECKO_API_KEY` in `.env` file
- Python 3.x with dependencies specified in requirements.txt
- Key packages: pandas, numpy, arch (for GARCH), statsmodels, matplotlib, plotly

## Important Notes

- API calls include automatic rate limiting (1.2s between requests)
- Event data uses UTC timestamps (00:00:00)
- Three model progression: GARCH → TARCH → TARCH-X with increasing complexity
- Project follows academic research standards with FDR correction for multiple testing