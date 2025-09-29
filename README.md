# Cryptocurrency Event Study: Infrastructure vs Regulatory Impact Analysis

A comprehensive analysis of differential volatility impacts between Infrastructure and Regulatory events on major cryptocurrencies using GARCH models with sentiment controls.

## 📊 Project Overview

This research examines how different types of events affect cryptocurrency market volatility. Using a dataset of 50 classified events (27 Infrastructure, 23 Regulatory) from 2019-2025, we apply advanced econometric models to quantify and compare their impacts on six major cryptocurrencies: Bitcoin (BTC), Ethereum (ETH), Ripple (XRP), Binance Coin (BNB), Litecoin (LTC), and Cardano (ADA).

### Key Findings
- Infrastructure events show significantly larger volatility impacts than regulatory events
- Effects persist across multiple model specifications and robustness checks
- Sentiment data from GDELT provides additional explanatory power

## 🚀 Features

- **Advanced Volatility Modeling**: Implementation of GARCH(1,1), TARCH(1,1), and TARCH-X models
- **Event Window Analysis**: Automated handling of overlapping and special events
- **Sentiment Integration**: GDELT data processing and decomposition
- **Robustness Framework**: Bootstrap inference, placebo tests, and sensitivity analysis
- **Publication-Ready Outputs**: LaTeX tables and high-quality visualizations

## 📋 Prerequisites

- Python 3.8 or higher
- CoinGecko API key (free tier available at [CoinGecko](https://www.coingecko.com/en/api))

## 🛠️ Installation

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

## 💻 Usage

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

## 📁 Project Structure

```
event_study/
├── code/                          # Main analysis code
│   ├── coingecko_fetcher.py     # API data fetching
│   ├── data_preparation.py       # Data processing pipeline
│   ├── garch_models.py          # GARCH/TARCH implementations
│   ├── event_impact_analysis.py  # Event impact quantification
│   ├── hypothesis_testing_results.py # Statistical testing
│   ├── robustness_checks.py     # Robustness framework
│   └── run_event_study_analysis.py # Main analysis script
├── data/                         # Input data files
│   ├── events.csv               # 50 classified events
│   ├── gdelt.csv                # GDELT sentiment data
│   └── [crypto].csv             # Price data for each cryptocurrency
├── outputs/                      # Analysis results
│   ├── figures/                 # Visualizations
│   ├── tables/                  # Statistical results
│   └── publication/             # Publication-ready outputs
├── tests/                        # Unit tests
└── docs/                         # Documentation
```

## 📊 Data Sources

- **Price Data**: CoinGecko API (requires API key)
- **Event Data**: Manually classified cryptocurrency events (2019-2025)
- **Sentiment Data**: GDELT Project news sentiment indicators

## 🔬 Methodology

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

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=code --cov-report=html
```

## 📈 Results

Key findings from the analysis:
- Infrastructure events increase volatility by approximately 2.5% on average
- Regulatory events show smaller but significant impacts (~1.2%)
- Effects are consistent across different cryptocurrencies and model specifications

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{crypto_event_study_2025,
  title = {Cryptocurrency Event Study: Infrastructure vs Regulatory Impact Analysis},
  author = {Murad Farzulla},
  year = {2025},
  url = {https://github.com/studiofarzulla/event_study}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for research purposes only. Cryptocurrency markets are highly volatile and risky. This analysis should not be considered as financial advice.

## 🙏 Acknowledgments

- CoinGecko for providing cryptocurrency data API
- GDELT Project for sentiment data
- The cryptocurrency research community for valuable insights

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact murad@farzulla.org

---

**Note**: Before running the analysis, ensure you have obtained proper API credentials and understand the rate limits for data fetching.
