#  Differential Volatility Responses to Infrastructure and Regulatory Events in Cryptocurrency Markets: A TARCH-X Analysis with Sentiment Decomposition

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive analysis of differential volatility impacts between Infrastructure and Regulatory events on major cryptocurrencies using GARCH models with sentiment controls.


## Project Overview

This research examines how different types of events affect cryptocurrency market volatility. Using a dataset of 50 classified events (27 Infrastructure, 23 Regulatory) from 2019-2025, where I apply advanced econometric models to quantify and compare their impacts on six major cryptocurrencies: Bitcoin (BTC), Ethereum (ETH), Ripple (XRP), Binance Coin (BNB), Litecoin (LTC), and Cardano (ADA).


## Features

- **Volatility Modeling**: Implementation of GARCH(1,1), TARCH(1,1), and custom TARCH-X models
- **Event Window Analysis**: Automated handling of overlapping and special events
- **Sentiment Integration**: GDELT data processing and decomposition (though SQL query handling needs work)
- **Robustness Framework**: Bootstrap inference, placebo tests, and sensitivity analysis

## Prerequisites

- Python 3.8 or higher
- CoinGecko API key (free tier available at [CoinGecko](https://www.coingecko.com/en/api))
- (NOTE: Due to usage limits I was personally unable to implement this, however, you may have more luck)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/studiofarzulla/event_study.git
cd event_study
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your CoinGecko API key
```

## Usage

### Quick Start
Run the complete analysis pipeline:

```python
from code.run_event_study_analysis import main

# Run analysis with default settings
results = main(
    run_robustness=True,      # Include robustness checks
    run_bootstrap=False,       # Bootstrap inference (computationally intensive)
    generate_publication=True  # Create publication outputs
)
```

### Data Fetching
Fetch cryptocurrency data for specific events:

```python
from code.coingecko_fetcher import CoinGeckoFetcher

fetcher = CoinGeckoFetcher()
data = fetcher.fetch_event_window(
    coins=['BTC', 'ETH'],
    event_date='2020-03-12',
    days_before=30,
    days_after=30
)
```
**NOTE** I personally wanted to implement the OHLC method also, but hit API limits thus I manually installed the .csv files from CoinGecko (in the data folder), but if you have API access you can probably have richer analysis and/or not need to manually install data (though I kept the data files in the repository for convenience)

### Model Estimation
Run GARCH models on prepared data:

```python
from code.data_preparation import DataPreparation
from code.garch_models import estimate_models_for_all_cryptos

# Prepare data
data_prep = DataPreparation(data_path="data")
crypto_data = data_prep.prepare_all_cryptos(
    include_events=True,
    include_sentiment=True
)

# Estimate models
model_results = estimate_models_for_all_cryptos(crypto_data)
```


## Data Sources

- **Price Data**: CoinGecko API (requires API key)
- **Event Data**: Manually classified cryptocurrency events (2019-2025), with prior 208 events found also available in the /data directory
- **Sentiment Data**: GDELT Project news sentiment indicators, the SQL code used in BigQuery is also provided as well as previous iterations (pretty graphs for showing difference between the qualities of different searches also available)

## Methodology (full literature review + methodology is available in the /docs directory which was what was included in my thesis submission + references added)

### Event Classification
Events are classified into two categories:
- **Infrastructure** (27 events): Technical upgrades, partnerships, product launches
- **Regulatory** (23 events): Government actions, legal proceedings, policy changes

### Model Specifications
1. **GARCH(1,1)**: Baseline volatility model
2. **TARCH(1,1)**: Captures asymmetric responses to positive/negative shocks
3. **TARCH-X**: Extended model with event dummies and sentiment controls

### Event Windows
- Standard: [-3, +3] days around event
- Special handling for overlapping events (e.g., SEC Twin Suits, EIP-1559 & Poly Hack)

## Results

Key findings from the analysis:
- Infrastructure events increase volatility by approximately 2.5% on average
- Regulatory events show smaller but significant impacts (~1.2%)
- Effects are consistent across different cryptocurrencies and model specifications
- Overall nothing spectacular but the use of TARCH-X & GDELT in this case were unique

## Contributing

- Though this was a thesis project, if you find value in the work I've done then feel free to contribute! I'm planning on maintaining this code and further refining it for potential research submission, so even advice would be amazing!
- Please reach out to murad@farzulla.org and see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. 

## Citation

If you would want the full paper (doesn't include codebase) please contact me at murad@farzulla.org, otherwise if you use this code in your research, please cite (though you don't have to):

```bibtex
@software{crypto_event_study_2025,
  title = {Differential Volatility Responses to Infrastructure and Regulatory Events in Cryptocurrency Markets: A TARCH-X Analysis with Sentiment Decomposition},
  author = {Murad Farzulla},
  year = {2025},
  url = {https://github.com/studiofarzulla/event_study}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for research purposes only. Cryptocurrency markets are highly volatile and risky. This analysis should not be considered as financial advice.

## Acknowledgments

- CoinGecko for providing cryptocurrency data API
- Google Cloud & BigQuery for GDELT Project sentiment data
- The cryptocurrency research community for valuable insights 

## Contact

For questions or collaboration opportunities, please open an issue or contact murad@farzulla.org

---

**Note**: Before running the analysis, ensure you have obtained proper API credentials and understand the rate limits for data fetching (or just manually grab the historic data if you want to use other coins/periods)

## Related Work & Profiles

- [ORCID](https://orcid.org/0009-0002-7164-8704)
-  [Academia.edu](https://kcl.academia.edu/MuradFarzulla)
-   [LinkedIn](www.linkedin.com/in/farzulla)

**Other Projects:**
-  [Adversarial Security Agents](coming soon)
-  [DeFi Rug Pull Detection](coming soon)

**Blog/Writing:**
-  [Personal Site](https://farzulla.com) 
-  Research journey blog posts & YouTube (coming soon)
