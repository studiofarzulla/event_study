# Results and Discussion: Cryptocurrency Event Study Analysis

## Executive Summary

This study examined the differential volatility impacts of Infrastructure vs Regulatory events on six major cryptocurrencies (BTC, ETH, XRP, BNB, LTC, ADA) using GARCH family models with sentiment controls. Analysis of 50 classified events (27 Infrastructure, 23 Regulatory) from 2019-2025 reveals no statistically significant differential impact between event types after correcting for multiple testing.

## 1. Model Performance and Selection

### 1.1 Model Comparison Results

The progression from GARCH(1,1) to TARCH(1,1) to TARCH-X models shows mixed results across cryptocurrencies:

```
Model AIC/BIC Comparison (Best Model by Cryptocurrency):
┌──────────┬────────────────┬──────────────┬──────────────┐
│ Crypto   │ Best Model     │ AIC          │ BIC          │
├──────────┼────────────────┼──────────────┼──────────────┤
│ BTC      │ GARCH(1,1)     │ 11904.02     │ 11933.01     │
│ ETH      │ GARCH(1,1)     │ 13344.71     │ 13373.69     │
│ XRP      │ GARCH(1,1)     │ 13324.30     │ 13353.28     │
│ BNB      │ GARCH(1,1)     │ 11400.37     │ 11428.83     │
│ LTC      │ TARCH(1,1)     │ 13773.56     │ 13808.34     │
│ ADA      │ GARCH(1,1)     │ 14091.20     │ 14120.18     │
└──────────┴────────────────┴──────────────┴──────────────┘
```

**Key Finding**: The simpler GARCH(1,1) model outperforms more complex specifications for most cryptocurrencies, suggesting that additional complexity from asymmetric effects and exogenous variables does not improve model fit sufficiently to justify increased parameters.

### 1.2 Leverage Effects

TARCH models reveal minimal leverage effects across all cryptocurrencies:

```
Leverage Effects (γ parameter from TARCH models):
BTC: -0.0121 (not significant)
ETH: -0.0098 (not significant)
XRP:  0.0073 (not significant)
BNB: -0.0153 (not significant)
LTC: -0.0323 (not significant)
ADA:  0.0093 (not significant)
```

**Interpretation**: The lack of significant leverage effects suggests symmetric volatility responses to positive and negative shocks during event windows, contrary to typical financial market behavior.

## 2. Event Impact Analysis

### 2.1 Individual Cryptocurrency Results

From the TARCH-X model event dummy coefficients:

```
Event Coefficients (% increase in conditional volatility):
┌──────┬─────────────────────┬────────────────────┬─────────────┐
│Crypto│ Infrastructure (%)  │ Regulatory (%)     │ Difference  │
├──────┼─────────────────────┼────────────────────┼─────────────┤
│ BTC  │ 0.463 (p=0.628)    │ 0.488 (p=0.466)   │ -0.025      │
│ ETH  │ 0.090 (p=0.909)    │ 0.094 (p=0.809)   │ -0.003      │
│ XRP  │ 0.717 (p=0.512)    │ 0.863 (p=0.116)   │ -0.146      │
│ BNB  │ 1.131 (p=0.022)*   │ 0.763 (p=0.083)   │  0.368      │
│ LTC  │ 0.009 (p=0.980)    │ -0.064 (p=0.867)  │  0.074      │
│ ADA  │ 0.091 (p=0.843)    │ 0.350 (p=0.373)   │ -0.259      │
└──────┴─────────────────────┴────────────────────┴─────────────┘
* Significant at 5% before FDR correction
```

### 2.2 FDR-Corrected Results

After applying False Discovery Rate correction for multiple testing:

```
FDR-Corrected P-values:
Infrastructure Events:
- BNB: p_raw=0.022 → p_FDR=0.259 (not significant)
- All others: p_FDR > 0.46

Regulatory Events:
- All: p_FDR > 0.46

Significant findings after FDR: NONE
```

**Critical Result**: No event coefficient remains statistically significant after correcting for multiple testing, failing to support differential event impacts.

### 2.3 Meta-Analysis Results

Inverse variance weighted meta-analysis provides pooled estimates:

```
Pooled Event Effects (Inverse Variance Weighted):
┌──────────────────┬─────────────┬──────────────┬─────────────────┐
│ Event Type       │ Coefficient │ Std Error    │ 95% CI          │
├──────────────────┼─────────────┼──────────────┼─────────────────┤
│ Infrastructure   │ 0.338       │ 0.228        │ [-0.108, 0.785] │
│ Regulatory       │ 0.340       │ 0.180        │ [-0.014, 0.693] │
│ Difference       │ -0.001      │ 0.290        │ [-0.570, 0.567] │
└──────────────────┴─────────────┴──────────────┴─────────────────┘

Test for Differential Impact:
z-statistic: -0.004
p-value: 0.997
```

**Key Finding**: The difference between infrastructure and regulatory impacts is essentially zero (-0.001%), with extremely high p-value (0.997), providing strong evidence for the null hypothesis.

## 3. Hypothesis Testing Results

### 3.1 Primary Hypothesis Test

**H₀**: No differential impact between Infrastructure and Regulatory events
**H₁**: Infrastructure events have different volatility impact than Regulatory events

```
Hypothesis Test Summary:
┌─────────────────────┬────────────────────────┐
│ Test Statistic      │ Value                  │
├─────────────────────┼────────────────────────┤
│ Mean Difference     │ 0.00144%               │
│ t-statistic         │ 0.005                  │
│ p-value (two-sided) │ 0.996                  │
│ Decision            │ Fail to Reject H₀      │
└─────────────────────┴────────────────────────┘
```

### 3.2 Distribution of Effects

Comparing the distribution of event coefficients:

```
Distribution Statistics:
Infrastructure Events (n=6):
- Mean: 0.417%
- Median: 0.277%
- Std Dev: 0.404%
- Range: [0.009%, 1.131%]

Regulatory Events (n=6):
- Mean: 0.415%
- Median: 0.419%
- Std Dev: 0.333%
- Range: [-0.064%, 0.863%]
```

The similar means (0.417% vs 0.415%) and overlapping distributions further support the null hypothesis.

## 4. Sentiment Analysis Results

### 4.1 GDELT Sentiment Impact

From TARCH-X models, normalized GDELT sentiment shows mixed effects:

```
GDELT Sentiment Coefficients (S_gdelt_normalized):
BTC:  0.366 (positive, not significant)
ETH: -0.089 (negative, not significant)
XRP:  0.234 (positive, not significant)
BNB: -0.197 (negative, not significant)
LTC:  0.149 (positive, not significant)
ADA: -0.042 (negative, not significant)
```

### 4.2 Decomposed Sentiment Effects

Event-specific sentiment decomposition reveals no consistent patterns:

```
Sentiment Decomposition (Average across cryptocurrencies):
S_reg_decomposed mean: -0.112
S_infra_decomposed mean: -0.267
```

Both decomposed sentiment variables show negative average coefficients, suggesting sentiment may dampen rather than amplify volatility, contrary to expectations.

## 5. Discussion

### 5.1 Interpretation of Null Results

The failure to find differential impacts between Infrastructure and Regulatory events has several potential explanations:

1. **Market Efficiency**: Cryptocurrency markets may efficiently price both event types similarly, reflecting their comparable information content for market participants.

2. **Event Heterogeneity**: Within-category variation may exceed between-category differences. Infrastructure events range from major protocol upgrades to exchange launches, while regulatory events span from outright bans to favorable legislation.

3. **Temporal Evolution**: The 2019-2025 period witnessed market maturation, potentially homogenizing responses to different event types over time.

4. **Sample Size Limitations**: With only 50 events split across six cryptocurrencies, statistical power may be insufficient to detect subtle differences.

### 5.2 Comparison with Research Hypotheses

**Original Hypothesis**: Infrastructure events would show lower volatility impacts due to their planned nature and technical focus, while regulatory events would trigger higher volatility due to uncertainty and broader implications.

**Actual Findings**:
- No systematic difference in volatility impacts (difference = -0.001%, p = 0.997)
- BNB showed the largest infrastructure effect (1.13%), contradicting the hypothesis
- XRP showed the largest regulatory effect (0.86%), though not statistically significant
- Overall volatility increases were modest (0.3-0.4% on average) for both event types

### 5.3 Methodological Insights

1. **Model Selection**: The preference for simpler GARCH(1,1) models suggests that event-window volatility dynamics don't exhibit strong asymmetries or benefit from exogenous variables.

2. **Multiple Testing**: The importance of FDR correction is evident—BNB's infrastructure effect appeared significant (p=0.022) before correction but not after (p_FDR=0.259).

3. **Meta-Analysis Value**: Inverse variance weighting provided robust pooled estimates, confirming individual cryptocurrency findings.

### 5.4 Practical Implications

1. **Risk Management**: Traders and risk managers need not differentiate between infrastructure and regulatory events when adjusting volatility forecasts or position sizes.

2. **Event Trading**: The lack of differential impact suggests event-based trading strategies should not prioritize event type as a primary signal.

3. **Market Maturity**: The homogeneous response to different event types may indicate increasing cryptocurrency market maturity and integration.

## 6. Limitations and Future Research

### 6.1 Study Limitations

1. **Event Classification**: Binary classification may oversimplify complex events with multiple dimensions
2. **Window Selection**: The [-3, +3] day window may miss longer-term effects
3. **Sentiment Data**: GDELT coverage may be incomplete for cryptocurrency-specific news
4. **Market Conditions**: Results may be period-specific and not generalizable to bear/bull markets

### 6.2 Future Research Directions

1. **Granular Classification**: Develop multi-dimensional event taxonomies (e.g., severity, scope, surprise)
2. **Dynamic Windows**: Use event-specific windows based on information diffusion patterns
3. **Cross-Market Analysis**: Include traditional assets to assess cryptocurrency market uniqueness
4. **Machine Learning**: Apply ML techniques to identify non-linear event impact patterns
5. **High-Frequency Analysis**: Examine intraday volatility patterns around events

## 7. Conclusion

This comprehensive event study finds no evidence for differential volatility impacts between Infrastructure and Regulatory events in cryptocurrency markets. After rigorous testing with GARCH family models, meta-analysis, and multiple testing corrections, the null hypothesis of equal impacts cannot be rejected (p = 0.997).

The results suggest that cryptocurrency markets respond similarly to both technical developments and regulatory changes, with average volatility increases of approximately 0.3-0.4% during event windows. This finding challenges conventional wisdom about event-type importance and highlights the need for more nuanced approaches to understanding cryptocurrency market dynamics.

While individual cryptocurrencies show some variation (notably BNB's stronger infrastructure response), these differences are not systematic across the market and disappear after appropriate statistical corrections. The study contributes to the growing literature on cryptocurrency market efficiency and event studies, suggesting that simple event categorizations may be insufficient for predicting volatility responses in these evolving markets.

---

*Analysis completed: September 2025*
*Total Events Analyzed: 50 (27 Infrastructure, 23 Regulatory)*
*Cryptocurrencies: BTC, ETH, XRP, BNB, LTC, ADA*
*Time Period: 2019-2025*