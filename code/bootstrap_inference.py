"""
Bootstrap inference for TARCH models following Pascual et al. (2006).
Implements residual-based bootstrap for confidence intervals.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


class BootstrapInference:
    """
    Implements residual-based bootstrap for TARCH models.
    Following Pascual, Romo, and Ruiz (2006) methodology.
    """

    def __init__(self, returns: pd.Series, n_bootstrap: int = 500, seed: int = 42):
        """
        Initialize bootstrap inference.

        Args:
            returns: Series of returns for estimation
            n_bootstrap: Number of bootstrap replications
            seed: Random seed for reproducibility
        """
        self.returns = returns.dropna()
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        np.random.seed(seed)

    def residual_bootstrap_tarch(self, model_type: str = 'TARCH',
                                include_leverage: bool = True,
                                show_progress: bool = True) -> Dict:
        """
        Perform residual-based bootstrap for TARCH model.

        Args:
            model_type: 'GARCH' or 'TARCH'
            include_leverage: Whether to include leverage effect (for TARCH)
            show_progress: Show progress bar

        Returns:
            Dictionary with bootstrap results
        """
        # Step 1: Estimate original model
        print(f"Estimating original {model_type} model...")
        if model_type == 'TARCH' and include_leverage:
            original_model = arch_model(self.returns, vol='GARCH', p=1, o=1, q=1, dist='StudentsT')
        else:
            original_model = arch_model(self.returns, vol='GARCH', p=1, q=1, dist='StudentsT')

        original_fit = original_model.fit(disp='off')

        # Extract original parameters
        original_params = dict(original_fit.params)

        # Get standardized residuals and conditional volatility
        std_residuals = original_fit.resid / original_fit.conditional_volatility
        cond_vol = original_fit.conditional_volatility

        # Step 2: Bootstrap
        bootstrap_params = []
        convergence_count = 0

        iterator = tqdm(range(self.n_bootstrap), desc="Bootstrap replications") if show_progress else range(self.n_bootstrap)

        for b in iterator:
            # Resample standardized residuals with replacement
            n = len(std_residuals)
            bootstrap_indices = np.random.choice(n, size=n, replace=True)
            bootstrap_std_resid = std_residuals.iloc[bootstrap_indices].values

            # Generate bootstrap returns using original volatility structure
            bootstrap_returns = pd.Series(
                bootstrap_std_resid * cond_vol.values,
                index=self.returns.index
            )

            # Estimate model on bootstrap sample
            try:
                if model_type == 'TARCH' and include_leverage:
                    boot_model = arch_model(bootstrap_returns, vol='GARCH', p=1, o=1, q=1, dist='StudentsT')
                else:
                    boot_model = arch_model(bootstrap_returns, vol='GARCH', p=1, q=1, dist='StudentsT')

                boot_fit = boot_model.fit(disp='off', options={'maxiter': 500})

                if boot_fit.convergence_flag == 0:
                    bootstrap_params.append(dict(boot_fit.params))
                    convergence_count += 1

            except Exception:
                continue  # Skip failed bootstrap samples

        print(f"Bootstrap completed: {convergence_count}/{self.n_bootstrap} converged")

        # Step 3: Calculate confidence intervals
        confidence_intervals = self._calculate_percentile_ci(bootstrap_params, original_params)

        # Calculate bootstrap statistics
        bootstrap_stats = self._calculate_bootstrap_statistics(bootstrap_params)

        return {
            'original_params': original_params,
            'bootstrap_params': bootstrap_params,
            'confidence_intervals': confidence_intervals,
            'bootstrap_stats': bootstrap_stats,
            'convergence_rate': convergence_count / self.n_bootstrap
        }

    def _calculate_percentile_ci(self, bootstrap_params: List[Dict],
                                original_params: Dict,
                                alpha: float = 0.05) -> Dict:
        """
        Calculate percentile confidence intervals.

        Args:
            bootstrap_params: List of bootstrap parameter estimates
            original_params: Original parameter estimates
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Dictionary with confidence intervals
        """
        ci_dict = {}

        if not bootstrap_params:
            return ci_dict

        # Convert to DataFrame for easier manipulation
        params_df = pd.DataFrame(bootstrap_params)

        for param_name in original_params.keys():
            if param_name in params_df.columns:
                param_values = params_df[param_name].dropna()

                if len(param_values) > 0:
                    ci_lower = param_values.quantile(alpha / 2)
                    ci_upper = param_values.quantile(1 - alpha / 2)

                    ci_dict[param_name] = {
                        'original': original_params[param_name],
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'ci_width': ci_upper - ci_lower,
                        'bootstrap_mean': param_values.mean(),
                        'bootstrap_std': param_values.std()
                    }

        return ci_dict

    def _calculate_bootstrap_statistics(self, bootstrap_params: List[Dict]) -> Dict:
        """
        Calculate additional bootstrap statistics.

        Args:
            bootstrap_params: List of bootstrap parameter estimates

        Returns:
            Dictionary with bootstrap statistics
        """
        if not bootstrap_params:
            return {}

        params_df = pd.DataFrame(bootstrap_params)
        stats = {}

        # Focus on key parameters
        key_params = ['omega', 'alpha[1]', 'beta[1]', 'gamma[1]', 'nu']

        for param in key_params:
            if param in params_df.columns:
                values = params_df[param].dropna()
                if len(values) > 0:
                    stats[param] = {
                        'mean': values.mean(),
                        'median': values.median(),
                        'std': values.std(),
                        'skewness': values.skew(),
                        'kurtosis': values.kurtosis(),
                        'min': values.min(),
                        'max': values.max()
                    }

        # Calculate persistence (alpha + beta + gamma/2 for TARCH)
        if 'alpha[1]' in params_df.columns and 'beta[1]' in params_df.columns:
            alpha_vals = params_df['alpha[1]'].dropna()
            beta_vals = params_df['beta[1]'].dropna()

            if 'gamma[1]' in params_df.columns:
                gamma_vals = params_df['gamma[1]'].dropna()
                # Align lengths
                min_len = min(len(alpha_vals), len(beta_vals), len(gamma_vals))
                persistence = alpha_vals[:min_len] + beta_vals[:min_len] + gamma_vals[:min_len] / 2
            else:
                min_len = min(len(alpha_vals), len(beta_vals))
                persistence = alpha_vals[:min_len] + beta_vals[:min_len]

            stats['persistence'] = {
                'mean': persistence.mean(),
                'median': persistence.median(),
                'std': persistence.std(),
                'ci_lower': persistence.quantile(0.025),
                'ci_upper': persistence.quantile(0.975)
            }

        return stats

    def bootstrap_event_coefficients(self, data_with_events: pd.DataFrame,
                                   event_columns: List[str],
                                   n_bootstrap: int = 100) -> Dict:
        """
        Bootstrap confidence intervals for event coefficients.
        Implements residual bootstrap while preserving event structure.

        Args:
            data_with_events: DataFrame with returns and event dummies
            event_columns: List of event dummy column names
            n_bootstrap: Number of bootstrap replications (limited for speed)

        Returns:
            Dictionary with bootstrap results for event coefficients
        """
        print(f"\nBootstrapping event coefficients ({n_bootstrap} replications)...")

        # Import required modules here to avoid circular import
        from garch_models import GARCHModels
        import warnings
        warnings.filterwarnings('ignore')

        # Extract returns
        returns = data_with_events['returns_winsorized'].dropna()

        # Fit original model to get baseline
        print("Fitting baseline TARCH-X model...")
        estimator = GARCHModels(data_with_events, 'bootstrap')
        baseline_model = estimator.estimate_tarch_x(use_individual_events=False)

        if not baseline_model.convergence:
            return {'error': 'Baseline model failed to converge'}

        # Extract baseline coefficients
        baseline_coeffs = {}
        if baseline_model.event_effects:
            baseline_coeffs.update(baseline_model.event_effects)

        # Bootstrap replications
        bootstrap_coeffs = {col: [] for col in event_columns}
        convergence_count = 0

        for b in range(n_bootstrap):
            if b % 10 == 0:
                print(f"  Progress: {b}/{n_bootstrap} replications...")

            # Resample returns with replacement (block bootstrap for time series)
            block_size = 10  # Use blocks of 10 days
            n_blocks = len(returns) // block_size

            indices = []
            for _ in range(n_blocks):
                block_start = np.random.randint(0, len(returns) - block_size)
                indices.extend(range(block_start, block_start + block_size))

            # Create bootstrap sample
            bootstrap_data = data_with_events.iloc[indices].copy()

            # Estimate model on bootstrap sample
            try:
                boot_estimator = GARCHModels(bootstrap_data, 'bootstrap')
                boot_model = boot_estimator.estimate_tarch_x(use_individual_events=False)

                if boot_model.convergence and boot_model.event_effects:
                    for col in event_columns:
                        if col in boot_model.event_effects:
                            bootstrap_coeffs[col].append(boot_model.event_effects[col])
                    convergence_count += 1
            except:
                continue  # Skip failed bootstrap samples

        print(f"Bootstrap completed: {convergence_count}/{n_bootstrap} converged")

        # Calculate confidence intervals
        ci_results = {}
        for col in event_columns:
            if len(bootstrap_coeffs[col]) > 0:
                ci_results[col] = {
                    'baseline': baseline_coeffs.get(col, np.nan),
                    'bootstrap_mean': np.mean(bootstrap_coeffs[col]),
                    'bootstrap_std': np.std(bootstrap_coeffs[col]),
                    'ci_lower': np.percentile(bootstrap_coeffs[col], 2.5),
                    'ci_upper': np.percentile(bootstrap_coeffs[col], 97.5)
                }

        return {
            'baseline_coefficients': baseline_coeffs,
            'bootstrap_results': ci_results,
            'convergence_rate': convergence_count / n_bootstrap
        }

    def create_bootstrap_table(self, bootstrap_results: Dict) -> pd.DataFrame:
        """
        Create formatted table with original estimates and bootstrap CIs.

        Args:
            bootstrap_results: Results from residual_bootstrap_tarch

        Returns:
            DataFrame with formatted results
        """
        if 'confidence_intervals' not in bootstrap_results:
            return pd.DataFrame()

        ci_data = bootstrap_results['confidence_intervals']

        table_data = []
        for param_name, param_ci in ci_data.items():
            table_data.append({
                'Parameter': param_name,
                'Original Estimate': f"{param_ci['original']:.6f}",
                'Bootstrap Mean': f"{param_ci['bootstrap_mean']:.6f}",
                'Bootstrap Std': f"{param_ci['bootstrap_std']:.6f}",
                '95% CI Lower': f"{param_ci['ci_lower']:.6f}",
                '95% CI Upper': f"{param_ci['ci_upper']:.6f}"
            })

        return pd.DataFrame(table_data)



def run_bootstrap_analysis(returns: pd.Series, model_type: str = 'TARCH',
                         n_bootstrap: int = 500, seed: int = 42) -> Dict:
    """
    Convenience function to run bootstrap analysis.

    Args:
        returns: Series of returns
        model_type: 'GARCH' or 'TARCH'
        n_bootstrap: Number of bootstrap replications
        seed: Random seed

    Returns:
        Dictionary with bootstrap results
    """
    bootstrap = BootstrapInference(returns, n_bootstrap, seed)
    results = bootstrap.residual_bootstrap_tarch(model_type=model_type)

    # Create and print table
    table = bootstrap.create_bootstrap_table(results)
    if not table.empty:
        print("\nBootstrap Confidence Intervals:")
        print(table.to_string(index=False))

    return results
