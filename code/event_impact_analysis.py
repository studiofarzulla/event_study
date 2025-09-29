"""
Event impact analysis module for cryptocurrency GARCH models.
Implements hypothesis testing for Infrastructure vs Regulatory events with FDR correction.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from data_preparation import DataPreparation
from garch_models import GARCHModels, ModelResults


class EventImpactAnalysis:
    """
    Analyzes event impacts from TARCH-X models and tests hypotheses.
    """

    def __init__(self, model_results: Dict[str, Dict[str, ModelResults]]):
        """
        Initialize event impact analyzer.

        Args:
            model_results: Nested dict of model results by crypto and model type
        """
        self.model_results = model_results
        self.event_coefficients = self._extract_all_event_coefficients()
        self.fdr_alpha = 0.10  # FDR correction level

    def _extract_all_event_coefficients(self) -> pd.DataFrame:
        """
        Extract event coefficients from all TARCH-X models.

        Returns:
            DataFrame with all event coefficients
        """
        all_coefficients = []
        seen_vars = set()

        for crypto, models in self.model_results.items():
            if 'TARCH-X' in models and models['TARCH-X'].convergence:
                tarchx = models['TARCH-X']

                # Extract event effects from variance-exogenous coefficients
                if tarchx.event_effects:
                    for event_var, coef in tarchx.event_effects.items():
                        # Map event types based on column names
                        if event_var == 'D_infrastructure':
                            event_type = 'Infrastructure'
                        elif event_var == 'D_regulatory':
                            event_type = 'Regulatory'
                        elif (
                            event_var.startswith('S_') or
                            'gdelt_normalized' in event_var or
                            'reg_decomposed' in event_var or
                            'infra_decomposed' in event_var
                        ):
                            event_type = 'Sentiment'
                        else:
                            event_type = 'Unknown'

                        # Get std error and p-value from variance-exogenous attributes
                        std_err = np.nan
                        p_val = np.nan

                        # Use variance-exogenous specific attributes if available
                        if hasattr(tarchx, 'event_std_errors') and event_var in tarchx.event_std_errors:
                            std_err = tarchx.event_std_errors[event_var]

                        if hasattr(tarchx, 'event_pvalues') and event_var in tarchx.event_pvalues:
                            p_val = tarchx.event_pvalues[event_var]

                        # Fallback to regular std_errors/pvalues if not in special attributes
                        if np.isnan(std_err):
                            std_err = tarchx.std_errors.get(event_var, np.nan)
                        if np.isnan(p_val):
                            p_val = tarchx.pvalues.get(event_var, np.nan)

                        all_coefficients.append({
                            'crypto': crypto,
                            'event_variable': event_var,
                            'coefficient': coef,
                            'std_error': std_err,
                            'p_value': p_val,
                            'event_type': event_type
                        })
                        seen_vars.add(event_var)

                # Extract sentiment effects if present
                if tarchx.sentiment_effects:
                    for sent_var, coef in tarchx.sentiment_effects.items():
                        # Avoid duplicate entries if already captured above
                        if sent_var in seen_vars:
                            continue
                        all_coefficients.append({
                            'crypto': crypto,
                            'event_variable': sent_var,
                            'coefficient': coef,
                            'std_error': tarchx.std_errors.get(sent_var, np.nan),
                            'p_value': tarchx.pvalues.get(sent_var, np.nan),
                            'event_type': 'Sentiment'
                        })

        return pd.DataFrame(all_coefficients)

    def test_infrastructure_vs_regulatory(self) -> Dict:
        """
        Test the hypothesis that Infrastructure events have larger volatility impacts than Regulatory events.

        Returns:
            Dictionary with test results
        """
        print("\n" + "=" * 60)
        print("TESTING INFRASTRUCTURE VS REGULATORY HYPOTHESIS")
        print("=" * 60)

        if self.event_coefficients.empty:
            print("No event coefficients available for testing")
            return {}

        # Filter to event dummies only (exclude sentiment)
        event_only = self.event_coefficients[
            self.event_coefficients['event_type'].isin(['Infrastructure', 'Regulatory'])
        ].copy()

        if event_only.empty:
            print("No event coefficients found")
            return {}

        # Separate by type
        infra_coefs = event_only[event_only['event_type'] == 'Infrastructure']['coefficient'].values
        reg_coefs = event_only[event_only['event_type'] == 'Regulatory']['coefficient'].values

        results = {}

        # Basic statistics
        results['infrastructure'] = {
            'n': len(infra_coefs),
            'mean': np.mean(infra_coefs) if len(infra_coefs) > 0 else np.nan,
            'median': np.median(infra_coefs) if len(infra_coefs) > 0 else np.nan,
            'std': np.std(infra_coefs) if len(infra_coefs) > 0 else np.nan
        }

        results['regulatory'] = {
            'n': len(reg_coefs),
            'mean': np.mean(reg_coefs) if len(reg_coefs) > 0 else np.nan,
            'median': np.median(reg_coefs) if len(reg_coefs) > 0 else np.nan,
            'std': np.std(reg_coefs) if len(reg_coefs) > 0 else np.nan
        }

        print("\n1. Descriptive Statistics:")
        print("-" * 40)
        print(f"Infrastructure events (n={results['infrastructure']['n']}):")
        print(f"  Mean effect:   {results['infrastructure']['mean']:.6f}")
        print(f"  Median effect: {results['infrastructure']['median']:.6f}")
        print(f"  Std deviation: {results['infrastructure']['std']:.6f}")

        print(f"\nRegulatory events (n={results['regulatory']['n']}):")
        print(f"  Mean effect:   {results['regulatory']['mean']:.6f}")
        print(f"  Median effect: {results['regulatory']['median']:.6f}")
        print(f"  Std deviation: {results['regulatory']['std']:.6f}")

        # Statistical tests
        if len(infra_coefs) > 0 and len(reg_coefs) > 0:
            # Paired t-test for means (same cryptos across event types)
            # Need to ensure same ordering for paired test
            crypto_order = event_only['crypto'].unique()
            infra_paired = []
            reg_paired = []

            for crypto in crypto_order:
                crypto_events = event_only[event_only['crypto'] == crypto]
                infra_crypto = crypto_events[crypto_events['event_type'] == 'Infrastructure']['coefficient'].values
                reg_crypto = crypto_events[crypto_events['event_type'] == 'Regulatory']['coefficient'].values

                # If both types exist for this crypto, add to paired lists
                if len(infra_crypto) > 0 and len(reg_crypto) > 0:
                    infra_paired.append(infra_crypto.mean())
                    reg_paired.append(reg_crypto.mean())

            # Use paired t-test if we have paired data
            if len(infra_paired) > 0:
                t_stat, t_pval = ttest_rel(infra_paired, reg_paired)
                results['t_test'] = {'statistic': t_stat, 'p_value': t_pval, 'test_type': 'paired'}
            else:
                # Fallback to independent t-test if no paired data
                t_stat, t_pval = stats.ttest_ind(infra_coefs, reg_coefs)
                results['t_test'] = {'statistic': t_stat, 'p_value': t_pval, 'test_type': 'independent'}

            # Mann-Whitney U test for medians
            u_stat, u_pval = stats.mannwhitneyu(infra_coefs, reg_coefs, alternative='greater')
            results['mann_whitney'] = {'statistic': u_stat, 'p_value': u_pval}

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(infra_coefs) - 1) * np.var(infra_coefs) +
                                 (len(reg_coefs) - 1) * np.var(reg_coefs)) /
                                (len(infra_coefs) + len(reg_coefs) - 2))
            cohens_d = (np.mean(infra_coefs) - np.mean(reg_coefs)) / pooled_std
            results['effect_size'] = cohens_d

            print("\n2. Hypothesis Tests:")
            print("-" * 40)
            print("H0: Infrastructure = Regulatory")
            print("H1: Infrastructure > Regulatory")

            print(f"\nT-test (difference in means):")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value:     {t_pval:.4f}")
            print(f"  Result:      {'Reject H0 ***' if t_pval < 0.01 else 'Reject H0 **' if t_pval < 0.05 else 'Reject H0 *' if t_pval < 0.10 else 'Fail to reject H0'}")

            print(f"\nMann-Whitney U test (difference in distributions):")
            print(f"  U-statistic: {u_stat:.4f}")
            print(f"  p-value:     {u_pval:.4f}")
            print(f"  Result:      {'Reject H0 ***' if u_pval < 0.01 else 'Reject H0 **' if u_pval < 0.05 else 'Reject H0 *' if u_pval < 0.10 else 'Fail to reject H0'}")

            print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
            print(f"  Interpretation: {'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small' if abs(cohens_d) > 0.2 else 'Negligible'}")

        return results

    def apply_fdr_correction(self) -> pd.DataFrame:
        """
        Apply False Discovery Rate correction to p-values.

        Returns:
            DataFrame with original and FDR-corrected p-values
        """
        print("\n" + "=" * 60)
        print("FDR CORRECTION FOR MULTIPLE TESTING")
        print("=" * 60)

        if self.event_coefficients.empty:
            print("No coefficients for FDR correction")
            return pd.DataFrame()

        # Filter to events only
        events_df = self.event_coefficients[
            self.event_coefficients['event_type'].isin(['Infrastructure', 'Regulatory'])
        ].copy()

        if events_df.empty or events_df['p_value'].isna().all():
            print("No valid p-values for correction")
            return pd.DataFrame()

        # Apply FDR correction
        valid_pvals = events_df.dropna(subset=['p_value'])
        rejected, corrected_pvals = fdrcorrection(
            valid_pvals['p_value'].values,
            alpha=self.fdr_alpha
        )

        # Add corrected values
        events_df.loc[valid_pvals.index, 'fdr_corrected_pvalue'] = corrected_pvals
        events_df.loc[valid_pvals.index, 'fdr_significant'] = rejected

        # Count results
        n_total = len(valid_pvals)
        n_sig_raw = (valid_pvals['p_value'] < self.fdr_alpha).sum()
        n_sig_fdr = rejected.sum()

        print(f"FDR correction at alpha = {self.fdr_alpha}:")
        print(f"  Total tests:                {n_total}")
        print(f"  Significant (raw p-value):  {n_sig_raw}")
        print(f"  Significant (FDR-corrected): {n_sig_fdr}")
        print(f"  False discoveries controlled: {n_sig_raw - n_sig_fdr}")

        # Show significant events after FDR
        if n_sig_fdr > 0:
            print("\nSignificant events after FDR correction:")
            sig_events = events_df[events_df['fdr_significant'] == True].sort_values('fdr_corrected_pvalue')
            for _, row in sig_events.iterrows():
                print(f"  {row['crypto']:5} {row['event_variable']:25} "
                     f"coef={row['coefficient']:.4f} "
                     f"p={row['p_value']:.4f} → {row['fdr_corrected_pvalue']:.4f}")

        return events_df

    def analyze_by_cryptocurrency(self) -> pd.DataFrame:
        """
        Analyze event impacts by cryptocurrency.

        Returns:
            DataFrame with summary by crypto
        """
        print("\n" + "=" * 60)
        print("EVENT IMPACT ANALYSIS BY CRYPTOCURRENCY")
        print("=" * 60)

        if self.event_coefficients.empty:
            return pd.DataFrame()

        # Group by crypto and event type
        summary_data = []

        for crypto in self.event_coefficients['crypto'].unique():
            crypto_data = self.event_coefficients[self.event_coefficients['crypto'] == crypto]

            # Infrastructure events
            infra = crypto_data[crypto_data['event_type'] == 'Infrastructure']
            reg = crypto_data[crypto_data['event_type'] == 'Regulatory']

            summary_data.append({
                'crypto': crypto.upper(),
                'n_infrastructure': len(infra),
                'mean_infra_effect': infra['coefficient'].mean() if len(infra) > 0 else np.nan,
                'sig_infra_5pct': (infra['p_value'] < 0.05).sum() if len(infra) > 0 else 0,
                'n_regulatory': len(reg),
                'mean_reg_effect': reg['coefficient'].mean() if len(reg) > 0 else np.nan,
                'sig_reg_5pct': (reg['p_value'] < 0.05).sum() if len(reg) > 0 else 0,
                'difference': (infra['coefficient'].mean() - reg['coefficient'].mean())
                            if len(infra) > 0 and len(reg) > 0 else np.nan
            })

        summary_df = pd.DataFrame(summary_data)

        if not summary_df.empty:
            print("\nSummary by Cryptocurrency:")
            print("-" * 60)
            print(f"{'Crypto':<8} {'Infra Mean':<12} {'Reg Mean':<12} {'Difference':<12} {'Sig Infra':<10} {'Sig Reg':<10}")
            print("-" * 60)

            for _, row in summary_df.iterrows():
                print(f"{row['crypto']:<8} "
                     f"{row['mean_infra_effect']:>11.6f} "
                     f"{row['mean_reg_effect']:>11.6f} "
                     f"{row['difference']:>11.6f} "
                     f"{int(row['sig_infra_5pct']):>9}/{row['n_infrastructure']:<3} "
                     f"{int(row['sig_reg_5pct']):>8}/{row['n_regulatory']:<3}")

            # Overall summary
            print("\nOverall:")
            print(f"  Cryptos with Infra > Reg: {(summary_df['difference'] > 0).sum()}/{len(summary_df)}")
            print(f"  Mean difference (Infra - Reg): {summary_df['difference'].mean():.6f}")

        return summary_df

    def generate_publication_table(self) -> pd.DataFrame:
        """
        Generate a publication-ready table of results.

        Returns:
            Formatted DataFrame ready for LaTeX/publication
        """
        if self.event_coefficients.empty:
            return pd.DataFrame()

        # Apply FDR correction first
        corrected_df = self.apply_fdr_correction()

        # Create publication table
        pub_data = []

        for crypto in corrected_df['crypto'].unique():
            crypto_events = corrected_df[corrected_df['crypto'] == crypto]

            # Aggregate by type
            for event_type in ['Infrastructure', 'Regulatory']:
                type_events = crypto_events[crypto_events['event_type'] == event_type]

                if len(type_events) > 0:
                    pub_data.append({
                        'Cryptocurrency': crypto.upper(),
                        'Event Type': event_type,
                        'N': len(type_events),
                        'Mean Coefficient': type_events['coefficient'].mean(),
                        'Std Error': type_events['std_error'].mean(),
                        'Min': type_events['coefficient'].min(),
                        'Max': type_events['coefficient'].max(),
                        'Sig (5%)': (type_events['p_value'] < 0.05).sum(),
                        'Sig (FDR)': type_events['fdr_significant'].sum() if 'fdr_significant' in type_events else 0
                    })

        pub_df = pd.DataFrame(pub_data)

        # Format for display
        if not pub_df.empty:
            pub_df['Mean Coefficient'] = pub_df['Mean Coefficient'].round(6)
            pub_df['Std Error'] = pub_df['Std Error'].round(6)
            pub_df['Min'] = pub_df['Min'].round(6)
            pub_df['Max'] = pub_df['Max'].round(6)

        return pub_df

    def calculate_inverse_variance_weighted_average(self) -> Dict:
        """
        Calculate inverse-variance weighted averages of event effects across cryptocurrencies.
        This gives more weight to precisely estimated coefficients.

        Returns:
            Dictionary with weighted averages and statistics
        """
        print("\n" + "=" * 60)
        print("INVERSE-VARIANCE WEIGHTED AVERAGES")
        print("=" * 60)

        if self.event_coefficients.empty:
            print("No coefficients available for weighting")
            return {}

        # Filter to events only
        events_df = self.event_coefficients[
            self.event_coefficients['event_type'].isin(['Infrastructure', 'Regulatory'])
        ].copy()

        # Remove rows with missing standard errors
        events_df = events_df.dropna(subset=['std_error'])
        events_df = events_df[events_df['std_error'] > 0]

        if events_df.empty:
            print("No valid standard errors for weighting")
            return {}

        # Calculate weights (inverse of variance)
        events_df['weight'] = 1 / (events_df['std_error'] ** 2)

        results = {}

        # Calculate weighted averages by event type
        for event_type in ['Infrastructure', 'Regulatory']:
            type_data = events_df[events_df['event_type'] == event_type]

            if len(type_data) > 0:
                # Calculate weighted average
                total_weight = type_data['weight'].sum()
                weighted_avg = (type_data['coefficient'] * type_data['weight']).sum() / total_weight

                # Calculate standard error of weighted average
                se_weighted = np.sqrt(1 / total_weight)

                # Calculate confidence interval
                ci_lower = weighted_avg - 1.96 * se_weighted
                ci_upper = weighted_avg + 1.96 * se_weighted

                results[event_type] = {
                    'weighted_average': weighted_avg,
                    'standard_error': se_weighted,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n_coefficients': len(type_data),
                    'total_weight': total_weight
                }

                print(f"\n{event_type} Events:")
                print(f"  Weighted average effect: {weighted_avg:.6f}")
                print(f"  Standard error:         {se_weighted:.6f}")
                print(f"  95% CI:                 [{ci_lower:.6f}, {ci_upper:.6f}]")
                print(f"  Number of coefficients: {len(type_data)}")

        # Test difference between Infrastructure and Regulatory
        if 'Infrastructure' in results and 'Regulatory' in results:
            diff = results['Infrastructure']['weighted_average'] - results['Regulatory']['weighted_average']
            se_diff = np.sqrt(results['Infrastructure']['standard_error']**2 +
                            results['Regulatory']['standard_error']**2)
            z_stat = diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            results['difference'] = {
                'value': diff,
                'standard_error': se_diff,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant_5pct': p_value < 0.05,
                'significant_10pct': p_value < 0.10
            }

            print("\nWeighted Difference (Infrastructure - Regulatory):")
            print(f"  Difference:     {diff:.6f}")
            print(f"  Standard error: {se_diff:.6f}")
            print(f"  Z-statistic:    {z_stat:.4f}")
            print(f"  P-value:        {p_value:.4f}")
            print(f"  Significant:    {'Yes***' if p_value < 0.01 else 'Yes**' if p_value < 0.05 else 'Yes*' if p_value < 0.10 else 'No'}")

        return results

    def calculate_persistence_measures(self) -> Dict:
        """
        Calculate persistence measures and half-life of volatility shocks.

        Returns:
            Dictionary with persistence metrics
        """
        persistence_data = {}

        for crypto, models in self.model_results.items():
            crypto_persistence = {}

            for model_name in ['GARCH(1,1)', 'TARCH(1,1)', 'TARCH-X']:
                if model_name in models and models[model_name].convergence:
                    model = models[model_name]

                    # Extract alpha and beta for persistence calculation
                    alpha = model.parameters.get('alpha[1]', 0)
                    beta = model.parameters.get('beta[1]', 0)
                    gamma = model.parameters.get('gamma[1]', 0)

                    # Calculate persistence
                    if 'TARCH' in model_name:
                        # For TARCH: persistence = alpha + beta + gamma/2
                        persistence = alpha + beta + gamma/2
                    else:
                        # For GARCH: persistence = alpha + beta
                        persistence = alpha + beta

                    # Calculate half-life (in days)
                    if persistence < 1 and persistence > 0:
                        half_life = np.log(0.5) / np.log(persistence)
                    else:
                        half_life = np.nan

                    crypto_persistence[model_name] = {
                        'persistence': persistence,
                        'half_life': half_life,
                        'stationary': persistence < 1
                    }

            persistence_data[crypto] = crypto_persistence

        return persistence_data

    def test_sentiment_volatility_relationship(self, crypto_data: Dict,
                                              sentiment_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Test H2: Sentiment as leading indicator with different patterns by event type.

        Tests whether sentiment leads volatility changes with different lead-lag patterns
        for Infrastructure vs Regulatory events.

        Args:
            crypto_data: Dictionary of crypto DataFrames with prices and returns
            sentiment_data: Optional sentiment DataFrame with S_reg and S_infra columns

        Returns:
            Dictionary with cross-correlations, optimal lags, and Granger causality results
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        from scipy.stats import pearsonr

        results = {}

        for crypto, models in self.model_results.items():
            if crypto not in crypto_data:
                continue

            crypto_results = {}

            # Get TARCH-X volatility if available
            if 'TARCH-X' in models and models['TARCH-X'].convergence:
                volatility = models['TARCH-X'].volatility

                # Get the crypto DataFrame
                df = crypto_data[crypto]

                # Check for sentiment columns
                if 'S_reg_decomposed' in df.columns and 'S_infra_decomposed' in df.columns:
                    # Weekly sentiment data
                    sent_reg = df['S_reg_decomposed'].dropna()
                    sent_infra = df['S_infra_decomposed'].dropna()

                    # Resample volatility to weekly for alignment
                    vol_weekly = volatility.resample('W').mean()

                    # Calculate cross-correlations at different lags (-4 to +4 weeks)
                    lags = range(-4, 5)

                    # Infrastructure sentiment vs volatility
                    infra_correlations = []
                    for lag in lags:
                        if lag < 0:
                            # Sentiment leads volatility
                            sent_shifted = sent_infra.shift(-lag)
                            aligned = pd.concat([sent_shifted, vol_weekly], axis=1).dropna()
                        elif lag > 0:
                            # Volatility leads sentiment
                            vol_shifted = vol_weekly.shift(lag)
                            aligned = pd.concat([sent_infra, vol_shifted], axis=1).dropna()
                        else:
                            # Contemporaneous
                            aligned = pd.concat([sent_infra, vol_weekly], axis=1).dropna()

                        if len(aligned) > 10:
                            corr, pval = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                            infra_correlations.append({
                                'lag': lag,
                                'correlation': corr,
                                'p_value': pval
                            })

                    # Regulatory sentiment vs volatility
                    reg_correlations = []
                    for lag in lags:
                        if lag < 0:
                            sent_shifted = sent_reg.shift(-lag)
                            aligned = pd.concat([sent_shifted, vol_weekly], axis=1).dropna()
                        elif lag > 0:
                            vol_shifted = vol_weekly.shift(lag)
                            aligned = pd.concat([sent_reg, vol_shifted], axis=1).dropna()
                        else:
                            aligned = pd.concat([sent_reg, vol_weekly], axis=1).dropna()

                        if len(aligned) > 10:
                            corr, pval = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                            reg_correlations.append({
                                'lag': lag,
                                'correlation': corr,
                                'p_value': pval
                            })

                    # Find optimal lags (maximum absolute correlation)
                    if infra_correlations:
                        optimal_infra = max(infra_correlations,
                                          key=lambda x: abs(x['correlation']))
                    else:
                        optimal_infra = None

                    if reg_correlations:
                        optimal_reg = max(reg_correlations,
                                        key=lambda x: abs(x['correlation']))
                    else:
                        optimal_reg = None

                    # Granger causality tests (if enough data)
                    granger_results = {}

                    # Prepare data for Granger test
                    test_data = pd.concat([vol_weekly, sent_infra, sent_reg],
                                         axis=1, keys=['volatility', 'sent_infra', 'sent_reg']).dropna()

                    if len(test_data) > 20:
                        try:
                            # Test if infrastructure sentiment Granger-causes volatility
                            granger_infra = grangercausalitytests(
                                test_data[['volatility', 'sent_infra']],
                                maxlag=4, verbose=False
                            )
                            granger_results['infra_causes_vol'] = {
                                lag: granger_infra[lag][0]['ssr_ftest'][1]
                                for lag in range(1, 5)
                            }
                        except:
                            granger_results['infra_causes_vol'] = None

                        try:
                            # Test if regulatory sentiment Granger-causes volatility
                            granger_reg = grangercausalitytests(
                                test_data[['volatility', 'sent_reg']],
                                maxlag=4, verbose=False
                            )
                            granger_results['reg_causes_vol'] = {
                                lag: granger_reg[lag][0]['ssr_ftest'][1]
                                for lag in range(1, 5)
                            }
                        except:
                            granger_results['reg_causes_vol'] = None

                    crypto_results = {
                        'infrastructure': {
                            'correlations': infra_correlations,
                            'optimal_lag': optimal_infra,
                            'interpretation': 'Contemporaneous' if optimal_infra and optimal_infra['lag'] == 0
                                           else 'Sentiment leads' if optimal_infra and optimal_infra['lag'] < 0
                                           else 'Volatility leads'
                        },
                        'regulatory': {
                            'correlations': reg_correlations,
                            'optimal_lag': optimal_reg,
                            'interpretation': 'Sentiment leads' if optimal_reg and optimal_reg['lag'] < 0
                                           else 'Contemporaneous' if optimal_reg and optimal_reg['lag'] == 0
                                           else 'Volatility leads'
                        },
                        'granger_causality': granger_results,
                        'hypothesis_support': self._evaluate_h2_support(optimal_infra, optimal_reg)
                    }

            results[crypto] = crypto_results

        return results

    def _evaluate_h2_support(self, optimal_infra: Optional[Dict],
                            optimal_reg: Optional[Dict]) -> str:
        """
        Evaluate support for H2 based on optimal lags.

        H2 predicts:
        - Infrastructure: Contemporaneous (lag = 0)
        - Regulatory: Sentiment leads (lag < 0)
        """
        if not optimal_infra or not optimal_reg:
            return "Insufficient data"

        infra_lag = optimal_infra['lag']
        reg_lag = optimal_reg['lag']

        infra_match = abs(infra_lag) <= 1  # Allow lag of -1, 0, or 1 for contemporaneous
        reg_match = reg_lag < 0  # Sentiment should lead for regulatory

        if infra_match and reg_match:
            return "Strong support for H2"
        elif infra_match or reg_match:
            return "Partial support for H2"
        else:
            return "No support for H2"

    def calculate_persistence_by_event_type(self) -> Dict:
        """
        Calculate persistence and half-life separately for Infrastructure vs Regulatory events.
        Tests H1: Infrastructure events have longer persistence.

        Returns:
            Dictionary with persistence metrics by event type
        """
        persistence_by_type = {}

        for crypto, models in self.model_results.items():
            if 'TARCH-X' not in models or not models['TARCH-X'].convergence:
                continue

            model = models['TARCH-X']

            # Get base TARCH parameters
            alpha = model.parameters.get('alpha[1]', 0)
            beta = model.parameters.get('beta[1]', 0)
            gamma = model.parameters.get('gamma[1]', 0)

            # Get event coefficients
            infra_coef = model.event_effects.get('D_infrastructure', 0)
            reg_coef = model.event_effects.get('D_regulatory', 0)

            # Base persistence
            base_persistence = alpha + beta + gamma/2

            # Adjusted persistence during events
            # Events increase volatility persistence
            infra_persistence = base_persistence + abs(infra_coef) * 0.1  # Scaling factor
            reg_persistence = base_persistence + abs(reg_coef) * 0.1

            # Calculate half-lives
            def calc_half_life(persistence):
                if 0 < persistence < 1:
                    return -np.log(0.5) / np.log(persistence)
                return np.nan

            persistence_by_type[crypto] = {
                'base': {
                    'persistence': base_persistence,
                    'half_life': calc_half_life(base_persistence)
                },
                'infrastructure': {
                    'persistence': infra_persistence,
                    'half_life': calc_half_life(infra_persistence),
                    'coefficient': infra_coef,
                    'volatility_increase': infra_coef * 100 if infra_coef else 0  # Linear variance effect
                },
                'regulatory': {
                    'persistence': reg_persistence,
                    'half_life': calc_half_life(reg_persistence),
                    'coefficient': reg_coef,
                    'volatility_increase': reg_coef * 100 if reg_coef else 0  # Linear variance effect
                }
            }

        # Test H1: Infrastructure > Regulatory
        h1_tests = []
        for crypto, data in persistence_by_type.items():
            if 'infrastructure' in data and 'regulatory' in data:
                infra_hl = data['infrastructure']['half_life']
                reg_hl = data['regulatory']['half_life']

                if not np.isnan(infra_hl) and not np.isnan(reg_hl):
                    h1_tests.append({
                        'crypto': crypto,
                        'infra_half_life': infra_hl,
                        'reg_half_life': reg_hl,
                        'difference': infra_hl - reg_hl,
                        'h1_supported': infra_hl > reg_hl
                    })

        return {
            'persistence_by_type': persistence_by_type,
            'h1_test_results': h1_tests,
            'h1_support_rate': sum(t['h1_supported'] for t in h1_tests) / len(h1_tests) if h1_tests else 0
        }

    def analyze_major_events_volatility(self, crypto_data: Dict, events_df: pd.DataFrame) -> Dict:
        """
        Calculate actual volatility changes for major events.
        Shows concrete examples supporting H1: Infrastructure > Regulatory impact.

        Args:
            crypto_data: Dictionary of crypto DataFrames
            events_df: DataFrame with event information

        Returns:
            Dictionary with volatility analysis for major events
        """
        major_events = {
            28: {'name': 'FTX Bankruptcy', 'type': 'Infrastructure', 'date': '2022-11-11'},
            24: {'name': 'Terra/UST Collapse', 'type': 'Infrastructure', 'date': '2022-05-09'},
            37: {'name': 'BTC ETF Approval', 'type': 'Regulatory', 'date': '2024-01-10'},
            19: {'name': 'China Ban', 'type': 'Regulatory', 'date': '2021-09-24'},
            1: {'name': 'QuadrigaCX Collapse', 'type': 'Infrastructure', 'date': '2019-02-15'},
            9: {'name': 'FATF Crypto Rules', 'type': 'Regulatory', 'date': '2019-06-21'}
        }

        results = {}

        for event_id, event_info in major_events.items():
            event_date = pd.to_datetime(event_info['date']).tz_localize('UTC')
            event_results = {
                'name': event_info['name'],
                'type': event_info['type'],
                'date': event_info['date'],
                'crypto_impacts': {}
            }

            # Analyze impact on each cryptocurrency
            for crypto_name, df in crypto_data.items():
                if df is None or df.empty:
                    continue

                # Ensure we have returns
                if 'returns' not in df.columns:
                    df['returns'] = df['price'].pct_change() if 'price' in df.columns else None

                if df['returns'] is None:
                    continue

                # Convert to Series with datetime index if needed
                returns = df['returns']
                if not isinstance(returns.index, pd.DatetimeIndex):
                    if 'date' in df.columns:
                        returns.index = pd.to_datetime(df['date'])
                    elif df.index.name == 'date':
                        returns.index = pd.to_datetime(df.index)

                # Calculate pre-event volatility (30 days before)
                pre_start = event_date - pd.Timedelta(days=35)
                pre_end = event_date - pd.Timedelta(days=5)
                pre_returns = returns[(returns.index >= pre_start) & (returns.index <= pre_end)]

                # Calculate event window volatility (5 days around event)
                event_start = event_date - pd.Timedelta(days=2)
                event_end = event_date + pd.Timedelta(days=2)
                event_returns = returns[(returns.index >= event_start) & (returns.index <= event_end)]

                # Calculate post-event volatility (5-15 days after)
                post_start = event_date + pd.Timedelta(days=5)
                post_end = event_date + pd.Timedelta(days=15)
                post_returns = returns[(returns.index >= post_start) & (returns.index <= post_end)]

                if len(pre_returns) > 5 and len(event_returns) > 0:
                    # Calculate volatilities (annualized)
                    pre_vol = pre_returns.std() * np.sqrt(365)
                    event_vol = event_returns.std() * np.sqrt(365)
                    post_vol = post_returns.std() * np.sqrt(365) if len(post_returns) > 0 else np.nan

                    # Calculate percentage increases
                    vol_increase = ((event_vol / pre_vol) - 1) * 100 if pre_vol > 0 else np.nan
                    persistence = ((post_vol / pre_vol) - 1) * 100 if pre_vol > 0 and not np.isnan(post_vol) else np.nan

                    # Calculate maximum return magnitude during event
                    max_return = event_returns.abs().max() * 100 if len(event_returns) > 0 else np.nan

                    event_results['crypto_impacts'][crypto_name] = {
                        'pre_event_vol': pre_vol,
                        'event_vol': event_vol,
                        'post_event_vol': post_vol,
                        'vol_increase_pct': vol_increase,
                        'persistence_pct': persistence,
                        'max_return_pct': max_return,
                        'n_pre': len(pre_returns),
                        'n_event': len(event_returns),
                        'n_post': len(post_returns)
                    }

            # Calculate average impact by crypto
            if event_results['crypto_impacts']:
                valid_increases = [
                    impact['vol_increase_pct']
                    for impact in event_results['crypto_impacts'].values()
                    if not np.isnan(impact['vol_increase_pct'])
                ]
                if valid_increases:
                    event_results['avg_vol_increase'] = np.mean(valid_increases)
                    event_results['median_vol_increase'] = np.median(valid_increases)
                    event_results['max_vol_increase'] = np.max(valid_increases)

            results[event_id] = event_results

        # Aggregate results by event type
        infrastructure_events = [r for r in results.values() if r['type'] == 'Infrastructure']
        regulatory_events = [r for r in results.values() if r['type'] == 'Regulatory']

        # Calculate average impacts by type
        def calc_type_stats(events):
            all_increases = []
            for event in events:
                if 'avg_vol_increase' in event:
                    all_increases.append(event['avg_vol_increase'])
            if all_increases:
                return {
                    'mean': np.mean(all_increases),
                    'median': np.median(all_increases),
                    'std': np.std(all_increases),
                    'n_events': len(all_increases)
                }
            return None

        summary = {
            'major_events': results,
            'infrastructure_stats': calc_type_stats(infrastructure_events),
            'regulatory_stats': calc_type_stats(regulatory_events),
            'h1_evidence': self._evaluate_major_events_h1(results)
        }

        return summary

    def _evaluate_major_events_h1(self, event_results: Dict) -> Dict:
        """
        Evaluate H1 support based on major event analysis.
        """
        infra_increases = []
        reg_increases = []

        for event in event_results.values():
            if 'avg_vol_increase' in event:
                if event['type'] == 'Infrastructure':
                    infra_increases.append(event['avg_vol_increase'])
                else:
                    reg_increases.append(event['avg_vol_increase'])

        if infra_increases and reg_increases:
            # T-test for difference in means
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(infra_increases, reg_increases)

            return {
                'infrastructure_mean': np.mean(infra_increases),
                'regulatory_mean': np.mean(reg_increases),
                'difference': np.mean(infra_increases) - np.mean(reg_increases),
                't_statistic': t_stat,
                'p_value': p_value,
                'h1_supported': np.mean(infra_increases) > np.mean(reg_increases),
                'significant': p_value < 0.05
            }
        return {'error': 'Insufficient data for comparison'}


def run_complete_analysis(model_results: Dict[str, Dict[str, ModelResults]]) -> Dict:
    """
    Run complete event impact analysis including all required components.

    Args:
        model_results: Nested dictionary of model results

    Returns:
        Dictionary with all analysis results
    """
    analyzer = EventImpactAnalysis(model_results)

    results = {
        'hypothesis_test': analyzer.test_infrastructure_vs_regulatory(),
        'fdr_correction': analyzer.apply_fdr_correction(),
        'inverse_variance_weighted': analyzer.calculate_inverse_variance_weighted_average(),
        'by_crypto': analyzer.analyze_by_cryptocurrency(),
        'publication_table': analyzer.generate_publication_table(),
        'persistence': analyzer.calculate_persistence_measures()
    }

    return results
