# Cryptocurrency Event Study: Master Reference Document

## Project Overview

**Objective**: Analyze differential volatility impacts of Infrastructure vs Regulatory events in cryptocurrency markets using GARCH models with sentiment controls.

**Core Hypothesis**: Infrastructure events (exchange failures, hacks, protocol changes) generate larger and more persistent volatility increases than regulatory events (announcements, enforcement actions, legislation).

## Research Hypotheses Testing Requirements

### H1: Asymmetric Volatility Response

- Compare mean volatility impacts: Infrastructure vs Regulatory
- Expected: Infrastructure coefficient > Regulatory coefficient
- Statistical tests: T-test, Mann-Whitney U, Cohen's d
- Persistence analysis: Half-life of volatility shocks

### H2: Sentiment Leading Indicator

- Status: Implemented (cross-correlation and Granger tests)
- Cross-correlation analysis between sentiment and volatility at lags -4 to +4 weeks
- Expected: Infrastructure shows lag=0 correlation, Regulatory shows negative lag
- Granger causality tests for sentiment→volatility relationship using weekly `S_reg_decomposed` and `S_infra_decomposed` (forward-filled to daily)

### H3: TARCH-X Model Superiority

- Model comparison via AIC/BIC
- Diagnostic tests: Ljung-Box, ARCH-LM
- Expected: TARCH-X < TARCH < GARCH for information criteria

**Timeline**: Analysis period January 2019 - August 2025

## Data Components

### 1. Cryptocurrency Price Data

- **Source**: CoinGecko API
- **Coins**: BTC, ETH, XRP, BNB, LTC, ADA
- **Frequency**: Daily prices at 00:00:00 UTC
- **Required transformations**:
  - Log returns: r*t = ln(P_t/P*{t-1}) \* 100
  - Winsorize outliers at 5 standard deviations (rolling 30-day window)

### 2. Event Data (`Crypto_Events__Shorthand__2_classes__2019-2025_noAug29_plus1.csv`)

- **Total events**: 50 (27 Infrastructure, 23 Regulatory)
- **Structure**: event_id, date, label, title, type
- **Event window**: [-3, +3] days around event date

### 3. GDELT Sentiment (`bquxjob_2bfc9768_19940c5e67e.csv`)

- **Frequency**: Weekly (aggregated to Monday)
- **Key columns**:
  - `S_gdelt_normalized`: Z-score normalized sentiment
  - `S_reg_decomposed`: Regulatory sentiment component
  - `S_infra_decomposed`: Infrastructure sentiment component
- **Merge strategy**: Forward-fill weekly sentiment to daily data

## Special Event Handling

### Overlapping Events (3 cases requiring special treatment):

1. **SEC Twin Suits (June 5-6, 2023)**:

   - Events: SEC v Binance (ID 31), SEC v Coinbase (ID 32)
   - Treatment: Single composite dummy `D_SEC_enforcement_2023` for [June 2-9]

2. **EIP-1559 & Poly Hack (Aug 5 & 10, 2021)**:

   - Events: Ethereum EIP-1559 (ID 17), Poly Network hack (ID 18)
   - Treatment: Individual dummies with -0.5 adjustment on overlap days [Aug 7-8]

3. **Bybit Hack & SEC Dismissal (Feb 21 & 27, 2025)**:
   - Events: Bybit hack (ID 43), SEC-Coinbase dismissal (ID 44)
   - Treatment: Truncated windows - Bybit ends Feb 23, SEC starts Feb 27

## Model Specifications

### Model Progression:

1. **GARCH(1,1)**: σ²*t = ω + α₁ε²*{t-1} + β₁σ²\_{t-1}
2. **TARCH(1,1)**: σ²*t = ω + α₁ε²*{t-1} + γ₁ε²*{t-1}I(ε*{t-1}<0) + β₁σ²\_{t-1}
3. **TARCH-X**: Adds event dummies and sentiment to variance equation

### Estimation Details:

- Distribution: Student-t
- Method: Quasi-Maximum Likelihood (QMLE)
- Standard errors: Robust (sandwich)
- Bootstrap: 500-1000 replications for key parameters

## Analysis Pipeline

### Step 1: Data Preparation

```python
# Required operations:
1. Load and filter crypto prices (2019-01-01 to 2025-08-31)
2. Calculate log returns
3. Winsorize outliers
4. Merge GDELT weekly sentiment (forward-fill)
5. Create event dummies with overlap handling
```

### Step 2: Model Estimation

```python
# For each cryptocurrency:
1. Estimate GARCH(1,1) baseline
2. Estimate TARCH(1,1) for asymmetry
3. Estimate TARCH-X with events + sentiment
4. Compare models via AIC/BIC
```

### Step 3: Event Impact Analysis

```python
# Extract and compare:
1. Event coefficients from TARCH-X variance equation
2. Group by Infrastructure vs Regulatory
3. Calculate mean/median effects
4. Test difference (t-test and Mann-Whitney U)
```

### Step 4: Statistical Inference

```python
# Multiple testing corrections:
1. Apply FDR correction at α=0.10
2. Bootstrap confidence intervals
3. Report both raw and adjusted p-values
```

### Step 5: Robustness Checks

```python
# Diagnostic tests:
1. Ljung-Box test on residuals
2. ARCH-LM test for remaining heteroskedasticity
3. Sign bias test for asymmetry

### Additional Robustness Tests:
1. **OHLC-based volatility**: Garman-Klass estimator for intraday volatility
2. **Placebo events**: 1,000 random dates tested for spurious effects
3. **Winsorization sensitivity**: Compare raw vs winsorized with T-distribution
4. **Expected results**:
   - Real events > 95th percentile of placebo distribution
   - T-distribution df < 10 suggests heavy tails captured
   - OHLC volatility shows larger event impacts than close-only
```

## Expected Outputs

### Primary Results:

1. **Table 1**: Model comparison (AIC, BIC, leverage parameters)
2. **Table 2**: Event type comparison (mean effects, significance counts)
3. **Table 3**: Individual event coefficients with FDR correction

### Key Metrics:

- Average volatility increase: Infrastructure vs Regulatory
- Persistence measures (half-life of shocks)
- Leverage effect (γ parameter) by cryptocurrency
- Sentiment-volatility correlation by event type

### Visualizations:

1. Volatility around major events (FTX, Terra, BTC ETF approval)
2. Event coefficient distribution by type
3. GARCH model diagnostics (ACF, Q-Q plots)

## Critical Implementation Notes

1. **Returns scaling**: Multiply returns by 100 for percentage interpretation
2. **Missing sentiment**: Use 0 for pre-June 2019 (before 26-week initialization)
3. **Event dummy alignment**: Ensure dates match exactly (UTC midnight)
4. **Memory management**: Process coins sequentially, not all at once
5. **Convergence issues**: If TARCH-X fails, try reducing event dummies to aggregated type indicators
6. **OHLC derivation**: Daily OHLC is derived by resampling CoinGecko market-chart price time series (hourly→daily open/high/low/close). No fabricated OHLC values are used.

## File Structure

```
project/
├── data/
│   ├── [cryptocurrency tag].csv  # Historical data from CoinGecko
│   ├── events.csv                # 50 events to be analysed
│   └── gdelt_sentiment.csv       # Weekly sentiment proxy
├── code/
│   ├── data_prep.py             # Data cleaning and merging
│   ├── garch_models.py          # Model estimation
│   └── analysis.py              # Results and tests
└── output/
    ├── model_results.csv         # Estimation output
    ├── event_impacts.csv         # Event coefficients
    └── figures/                  # Visualizations
```

## Python Package Requirements

```python
# Core data manipulation
pandas>=2.0.0
numpy>=1.24.0

# API and web requests
requests>=2.31.0
python-dotenv>=1.0.0

# Data analysis and statistics
scipy>=1.10.0
statsmodels>=0.14.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Time series analysis
arch>=6.0.0  # For GARCH models

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Installation:
pip install arch pandas numpy scipy statsmodels matplotlib seaborn
```

## Success Criteria

1. Models converge for all 6 cryptocurrencies
2. Leverage effect (γ) significant for majority of coins
3. Clear difference between Infrastructure and Regulatory impacts
4. Results robust to FDR correction
5. Diagnostic tests pass (no remaining ARCH effects)
