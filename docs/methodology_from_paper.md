\## 3.1 Cryptocurrency Selection and Data



The selection of cryptocurrencies balanced statistical power, data quality, and market representativeness. Following Liu and Tsyvinski (2021) and Makarov and Schoar (2020), we prioritized data integrity over sample size. Selection criteria required: (i) continuous trading throughout January 2019–August 2025, (ii) sustained top-decile liquidity, and (iii) distinct market archetypes to capture heterogeneous event responses.



The final sample comprises six cryptocurrencies:



1\. \*\*Bitcoin (BTC)\*\* – market baseline and systematic risk factor (Bouri et al., 2017)

2\. \*\*Ethereum (ETH)\*\* – smart contract platform capturing DeFi infrastructure exposure

3\. \*\*XRP\*\* – regulatory case study given SEC litigation (2020–2025), enabling quasi-experimental identification

4\. \*\*Binance Coin (BNB)\*\* – exchange token representing centralization risks

5\. \*\*Litecoin (LTC)\*\* – control asset with high BTC correlation (0.61-0.75) for difference-in-differences estimation

6\. \*\*Cardano (ADA)\*\* – alternative proof-of-stake implementation contrasting Ethereum's model



Price and volume data are sourced from CoinGecko's institutional API, representing volume-weighted averages across major exchanges at 00:00:00 UTC. The 80-month study period encompasses complete market cycles including the 2020-2021 bull market, 2022 contagion crisis, and 2023-2025 regulatory normalization.^\[Several prominent assets were excluded: Solana (network outages), Monero (exchange delistings), Uniswap (insufficient pre-2020 history), stablecoins (price-pegging mechanisms), and meme tokens (social sentiment-driven pricing). These exclusions prioritize continuous data availability over market coverage.]



This selection provides sufficient cross-sectional variation while maintaining data quality standards essential for GARCH estimation. Daily closing prices are sourced from CoinGecko's institutional API at 00:00:00 UTC, with logarithmic returns calculated as $r\_t = ln(P\_t/P\_{t-1})$ and outliers exceeding five standard deviations winsorized.

\## 3.2 Event Selection and Classification



\### 3.2.1 Event Identification



Event identification followed a systematic protocol drawing from primary regulatory documents, exchange/network announcements, and corroborating news sources spanning January 2019–August 2025. From an initial corpus of 208 candidates, we applied three filters: (i) precise UTC timestamps, (ii) verifiable public records, and (iii) demonstrable market-wide price impact.



To address confounding from proximate events, we implemented a two-stage protocol. Events within ±3 days were consolidated if substantively related (e.g., proposal→approval), with timing anchored at first disclosure. Unrelated overlapping events were prioritized by legal finality or technical severity, with dominated events retained for robustness checks. This process yielded 50 distinct events while preserving statistical power.



\### 3.2.2 Classification Framework



Events were classified into two categories based on their market mechanism:



\*\*Infrastructure events (n=27):\*\* Incidents affecting transaction or settlement mechanics—exchange outages, chain halts, protocol exploits, consensus changes, and halvings. The classification criterion is mechanical impact on execution/settlement, regardless of predictability. Scheduled upgrades (e.g., Ethereum Merge, Bitcoin halvings) are classified as infrastructure due to their operational impact.



\*\*Regulatory events (n=23):\*\* Legal or supervisory actions that alter the informational environment while preserving trading mechanics—enforcement actions, ETF approvals, legislative frameworks. These affect valuation through legal risk and compliance cost channels rather than operational disruption.



Classification followed a decision tree: (1) If normal execution/settlement mechanisms were impaired → Infrastructure; (2) If impact operated through legal/informational channels → Regulatory. Boundary cases were resolved by proximate mechanism.



\### 3.2.3 Overlap Treatment



With \[-3,+3] event windows, any events within 6 days produce overlaps. After consolidation, three pairs required special treatment:



\*\*(A) SEC v Binance (June 5) and SEC v Coinbase (June 6, 2023):\*\* Treated as a single regulatory episode with composite dummy D\_SEC\_enforcement for \[June 2–9], recognizing coordinated enforcement action.



\*\*(B) Ethereum EIP-1559 (Aug 5) and Poly Network hack (Aug 10, 2021):\*\* Retained as separate events with overlap adjustment term for \[Aug 7–8] to prevent double-counting while preserving distinct shock identification.



\*\*(C) Bybit hack (Feb 21) and SEC-Coinbase dismissal (Feb 27, 2025):\*\* Cross-channel events handled through truncated windows—Bybit \[Feb 18–23], SEC dismissal \[Feb 27–Mar 2]—with intervening days excluded.



\### 3.2.4 Final Event Distribution



The final sample of 50 events maintains temporal spacing and category balance across the study period. Table 3.1 presents the complete event list with classifications.



\[Table 3.1 would go here with the 50 events]



This approach balances comprehensive coverage with econometric tractability, providing sufficient variation to identify differential volatility responses while maintaining clear event windows for causal inference.



\## 3.3 GDELT-Based Sentiment Proxy



\### 3.3.1 Methodological Foundation



We construct cryptocurrency sentiment indices using the Global Database of Events, Language, and Tone (GDELT), extending the news-based sentiment framework of Tetlock (2007) and Baker et al. (2016). GDELT provides standardized tone scoring across millions of global news articles, ensuring replicability. While prior studies treat cryptocurrency news as monolithic (Caferra \& Vidal-Tomás, 2021), we decompose sentiment into regulatory and infrastructure components to capture distinct market information channels.



\### 3.3.2 Index Construction



The methodology employs a three-stage process:



\*\*Stage 1: Query Specification\*\* We implement hierarchical keyword matching using GDELT's structured theme taxonomy:



\- \*\*Primary terms:\*\* 'bitcoin', 'cryptocurrency', 'ethereum' plus theme codes (e.g., 'ECON\_BITCOIN')

\- \*\*Regulatory identifiers:\*\* Policy theme codes ('EPU\_CATS\_REGULATION', 'EPU\_CATS\_FINANCIAL\_REGULATION') requiring cryptocurrency co-occurrence

\- \*\*Infrastructure markers:\*\* Crisis taxonomy codes ('ECON\_BANKRUPTCY', 'CYBER\_ATTACK', 'MANMADE\_DISASTER') with cryptocurrency context



This approach yields average weekly coverage of 26.7% for regulatory and 26.5% for infrastructure content, capturing both discrete events and persistent thematic discourse across 348 weekly observations. Weekly aggregation balances computational efficiency with analytical validity by smoothing daily noise (Huang et al., 2018).



\*\*Stage 2: Aggregation and Normalization\*\* Raw tone scores are volume-weighted:



$$S\_{t}^{GDELT} = \\frac{\\sum\_{i=1}^{N\_t} \\text{Tone}\_i \\times \\text{NumMentions}\_i}{\\sum\_{i=1}^{N\_t} \\text{NumMentions}\_i}$$



Following Manela and Moreira (2017), we apply recursive detrending via z-score transformation:



$$\\tilde{S}\_{t}^{GDELT} = \\frac{S\_{t}^{GDELT} - \\mu\_{t|t-52}}{\\sigma\_{t|t-52}}$$



Using 52-week rolling windows with 26-week initialization yields 323 usable observations, isolating abnormal sentiment from secular trends.



\*\*Stage 3: Theme Decomposition\*\* Rather than calculating separate indices from disjoint article sets, we decompose normalized aggregate sentiment by topical proportions:



$$S\_{t}^{REG} = \\tilde{S}\_{t}^{GDELT} \\times \\text{Proportion}\_{t}^{REG}$$ $$S\_{t}^{INFRA} = \\tilde{S}\_{t}^{GDELT} \\times \\text{Proportion}\_{t}^{INFRA}$$



Where proportions represent weekly article fractions matching respective keywords. This ensures complete data coverage while providing intuitive interpretation: each component represents its contribution to abnormal sentiment. Mathematical validity was verified computationally across all observations.



\### 3.3.3 Limitations



Several constraints affect the sentiment measures. GDELT's dictionary-based scoring captures journalistic framing rather than market sentiment—crisis reporting may register neutral while "justice served" narratives generate positive scores. The English-language bias underrepresents Asian market sentiment. Weekly aggregation may obscure intra-week dynamics during rapidly evolving events. The decomposition assumes sentiment scales proportionally with coverage, potentially misrepresenting events where tone and coverage diverge. Despite these limitations, temporal alignment with known events and theoretical consistency of coverage proportions support the approach's validity for capturing broad sentiment dynamics in cryptocurrency markets.



\## 3.4 Volatility Modelling Framework



\### 3.4.1 Model Specifications



We employ three nested GARCH specifications to examine cryptocurrency volatility dynamics, progressing from symmetric to asymmetric models with exogenous variables:



\*\*Model 1: GARCH(1,1) Baseline\*\* $$\\sigma^2\_t = \\omega + \\alpha\_1\\varepsilon^2\_{t-1} + \\beta\_1\\sigma^2\_{t-1}$$



This baseline specification captures volatility clustering but assumes symmetric responses to positive and negative shocks.



\*\*Model 2: TARCH(1,1)\*\* $$\\sigma^2\_t = \\omega + \\alpha\_1\\varepsilon^2\_{t-1} + \\gamma\_1\\varepsilon^2\_{t-1}I(\\varepsilon\_{t-1}<0) + \\beta\_1\\sigma^2\_{t-1}$$



The TARCH specification (Glosten et al., 1993) introduces leverage parameter γ₁ to capture asymmetric volatility responses, where I(εₜ₋₁<0) equals one for negative returns. This addresses documented asymmetries in cryptocurrency markets (Katsiampa, 2017; Cheikh et al., 2020).



\*\*Model 3: TARCH-X with Event Dummies and Sentiment\*\* $$\\sigma^2\_t = \\omega + \\alpha\_1\\varepsilon^2\_{t-1} + \\gamma\_1\\varepsilon^2\_{t-1}I(\\varepsilon\_{t-1}<0) + \\beta\_1\\sigma^2\_{t-1} + \\lambda\_1 S\_t^{REG} + \\lambda\_2 S\_t^{INFRA} + \\sum\_{j}\\delta\_j D\_{j,t}$$



The extended specification incorporates: (i) continuous sentiment proxies $S\_t^{REG}$ and $S\_t^{INFRA}$ from GDELT decomposition, and (ii) event dummy variables $D\_{j,t}$ activated during \[-3,+3] windows. This dual approach decomposes volatility into baseline dynamics, continuous sentiment effects, and discrete event shocks.



All models employ Student-t distributed innovations to accommodate heavy tails documented in cryptocurrency returns (Conrad et al., 2018). Parameters are estimated via quasi-maximum likelihood (QMLE) with robust standard errors.



\### 3.4.2 Event Window Specification



Event dummies equal one during \[t-3, t+3] windows around event dates, with special handling for overlapping events as detailed in Section 3.2.3. For infrastructure events, we test whether mechanical disruptions generate persistent volatility increases. For regulatory events, we examine whether informational shocks produce temporary or sustained effects. Primary outcomes measure average volatility change during \[t=0, t+2].



\### 3.4.3 Statistical Inference



Given non-standard distributional properties of cryptocurrency returns, we implement bootstrap inference following Pascual et al. (2006):



1\. Estimate models on original data

2\. Generate 1,000 bootstrap samples via residual resampling

3\. Re-estimate parameters for each sample

4\. Construct percentile confidence intervals



This approach preserves temporal dependence while accommodating heavy tails and potential structural breaks around events. Standard errors are clustered by event date to account for cross-sectional correlation during market stress periods.



\### 3.4.4 Model Diagnostics



Model adequacy is assessed through:



\- \*\*Ljung-Box Q-statistics\*\* on standardized and squared standardized residuals (testing for remaining autocorrelation)

\- \*\*ARCH-LM tests\*\* for residual heteroskedasticity

\- \*\*Sign bias tests\*\* (Engle \& Ng, 1993) confirming asymmetric effects are captured

\- \*\*Information criteria\*\* (AIC, BIC) for model comparison



Cross-asset effects are summarized using inverse-variance weighted averages of event coefficients. Primary outcomes focus on average conditional variance changes at t=\[0,+2], with secondary analyses examining persistence and cross-asset patterns.

\### 3.4.5 Multiple Testing Correction



With 50 events across 6 assets generating approximately 300 hypothesis tests, we apply Benjamini-Hochberg FDR correction at 10% to control Type I error while maintaining power. Results are reported both with and without adjustment.



This hierarchical approach—from symmetric baseline through asymmetric models to exogenous variable incorporation—enables systematic testing of whether: (1) cryptocurrency volatility exhibits asymmetric responses, (2) regulatory versus infrastructure events generate differential impacts, and (3) continuous sentiment provides incremental explanatory power beyond discrete events.









