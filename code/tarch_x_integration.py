"""
Integration Guide: Manual TARCH-X with Existing Event Study Framework
====================================================================

This module shows how to integrate the manual TARCH-X implementation
with your existing cryptocurrency event study codebase.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the manual implementation to your imports
sys.path.append(str(Path(__file__).parent))
from tarch_x_manual import estimate_tarch_x_manual, TARCHXResults
from data_preparation import DataPreparation
from garch_models import ModelResults

class EnhancedGARCHModels:
    """
    Enhanced GARCH model estimator with manual TARCH-X implementation.
    This replaces the problematic TARCH-X method in your original garch_models.py.
    """

    def __init__(self, data: pd.DataFrame, crypto: str):
        """
        Initialize enhanced GARCH model estimator.

        Args:
            data: Prepared DataFrame with returns and exogenous variables
            crypto: Cryptocurrency symbol for identification
        """
        self.data = data.copy()
        self.crypto = crypto
        self.returns = data["returns_winsorized"].dropna()

        # Store exogenous variables if present
        self.has_events = any(col.startswith("D_") for col in data.columns)
        self.has_sentiment = any(
            "gdelt" in col or "reg_decomposed" in col or "infra_decomposed" in col for col in data.columns
        )

    def estimate_tarch_x_manual(
        self, use_individual_events: bool = False, include_sentiment: bool = True
    ) -> ModelResults:
        """
        Estimate TARCH-X model using manual implementation.
        This properly implements exogenous variables in the variance equation.

        Args:
            use_individual_events: If True, use individual event dummies;
                                 if False, use aggregated type dummies
            include_sentiment: Whether to include sentiment variables

        Returns:
            ModelResults object compatible with existing analysis framework
        """
        print(f"Estimating TARCH-X (manual) for {self.crypto}...")

        # Prepare exogenous variables
        exog_vars = self._prepare_exogenous_variables(use_individual_events, include_sentiment)

        if exog_vars is None or exog_vars.empty:
            print("  [WARNING] No exogenous variables available for TARCH-X")
            return self._create_failed_result("TARCH-X")

        try:
            # Use manual TARCH-X implementation
            manual_results = estimate_tarch_x_manual(returns=self.returns, exog_vars=exog_vars, method="SLSQP")

            if not manual_results.converged:
                print("  [FAIL] Manual TARCH-X did not converge")
                return self._create_failed_result("TARCH-X")

            # Convert manual results to ModelResults format for compatibility
            model_results = ModelResults(
                model_type="TARCH-X",
                crypto=self.crypto,
                aic=manual_results.aic,
                bic=manual_results.bic,
                log_likelihood=manual_results.log_likelihood,
                parameters=manual_results.params,
                std_errors=manual_results.std_errors,
                pvalues=manual_results.pvalues,
                convergence=manual_results.converged,
                iterations=manual_results.iterations,
                volatility=manual_results.volatility,
                residuals=manual_results.residuals,
                leverage_effect=manual_results.leverage_effect,
                event_effects=manual_results.event_effects,
                sentiment_effects=manual_results.sentiment_effects,
            )

            # Add special attributes for event analysis compatibility
            model_results.event_std_errors = {
                k: manual_results.std_errors.get(k, np.nan) for k in manual_results.event_effects.keys()
            }
            model_results.event_pvalues = {
                k: manual_results.pvalues.get(k, np.nan) for k in manual_results.event_effects.keys()
            }

            print(f"  [OK] Manual TARCH-X converged in {manual_results.iterations} iterations")
            print(f"  AIC: {manual_results.aic:.2f}, BIC: {manual_results.bic:.2f}")
            print(f"  Event coefficients: {len(manual_results.event_effects)}")

            # Display key event effects
            for event, coef in manual_results.event_effects.items():
                vol_impact = coef * 100  # Linear variance effect
                p_val = manual_results.pvalues.get(event, np.nan)
                sig_stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                print(f"    {event}: {coef:+.4f}{sig_stars} ({vol_impact:+.1f}% volatility)")

            return model_results

        except Exception as e:
            print(f"  [FAIL] Manual TARCH-X estimation failed: {str(e)}")
            return self._create_failed_result("TARCH-X")

    def _prepare_exogenous_variables(self, use_individual_events: bool, include_sentiment: bool) -> pd.DataFrame:
        """
        Prepare exogenous variables for TARCH-X model.
        This is the same method from your original code.
        """
        exog_vars = []

        # Add event dummies
        if self.has_events:
            if use_individual_events:
                # Use individual event dummies
                event_cols = [
                    col for col in self.data.columns if col.startswith("D_event_") or col == "D_SEC_enforcement_2023"
                ]
                if event_cols:
                    exog_vars.extend(event_cols)
            else:
                # Use aggregated type dummies
                if "D_infrastructure" in self.data.columns:
                    exog_vars.append("D_infrastructure")
                if "D_regulatory" in self.data.columns:
                    exog_vars.append("D_regulatory")

        # Add sentiment variables
        if include_sentiment and self.has_sentiment:
            sentiment_cols = [
                col
                for col in self.data.columns
                if "gdelt_normalized" in col or "reg_decomposed" in col or "infra_decomposed" in col
            ]
            exog_vars.extend(sentiment_cols)

        if not exog_vars:
            return None

        # Return DataFrame with aligned exogenous variables
        exog_df = self.data[exog_vars].copy()
        return exog_df

    def _create_failed_result(self, model_type: str) -> ModelResults:
        """Create a failed result object for models that didn't converge."""
        return ModelResults(
            model_type=model_type,
            crypto=self.crypto,
            aic=np.nan,
            bic=np.nan,
            log_likelihood=np.nan,
            parameters={},
            std_errors={},
            pvalues={},
            convergence=False,
            iterations=0,
            volatility=pd.Series(),
            residuals=pd.Series(),
        )

def run_enhanced_analysis_example():
    """
    Example of how to use the enhanced TARCH-X implementation
    with your existing event study framework.
    """
    print("=" * 60)
    print("ENHANCED TARCH-X ANALYSIS EXAMPLE")
    print("=" * 60)

    # Step 1: Load data using your existing preparation
    data_prep = DataPreparation()  # Will use config.DATA_DIR by default

    # Test with Bitcoin
    btc_data = data_prep.prepare_crypto_data("btc", include_events=True, include_sentiment=True)

    print(f"Loaded BTC data: {btc_data.shape}")
    print(f"Event columns: {[col for col in btc_data.columns if col.startswith('D_')]}")

    # Step 2: Estimate models with enhanced estimator
    enhanced_estimator = EnhancedGARCHModels(btc_data, "btc")

    # Estimate manual TARCH-X
    tarchx_results = enhanced_estimator.estimate_tarch_x_manual(
        use_individual_events=False, include_sentiment=True  # Use aggregated event types
    )

    # Step 3: Display results
    if tarchx_results.convergence:
        print("\n" + "=" * 40)
        print("TARCH-X ESTIMATION RESULTS")
        print("=" * 40)

        print(f"Model converged: {tarchx_results.convergence}")
        print(f"AIC: {tarchx_results.aic:.2f}")
        print(f"BIC: {tarchx_results.bic:.2f}")
        print(f"Log-likelihood: {tarchx_results.log_likelihood:.2f}")

        print("\nVariance Equation Parameters:")
        for param in ["omega", "alpha[1]", "beta[1]", "gamma[1]", "nu"]:
            if param in tarchx_results.parameters:
                coef = tarchx_results.parameters[param]
                std_err = tarchx_results.std_errors.get(param, np.nan)
                p_val = tarchx_results.pvalues.get(param, np.nan)
                print(f"  {param:<10}: {coef:8.4f} ({std_err:.4f}) [p={p_val:.4f}]")

        print("\nEvent Effects:")
        if hasattr(tarchx_results, "event_effects") and tarchx_results.event_effects:
            for event, coef in tarchx_results.event_effects.items():
                p_val = tarchx_results.pvalues.get(event, np.nan)
                vol_change = coef * 100  # Linear variance effect
                stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                print(f"  {event:<20}: {coef:+8.4f}{stars} ({vol_change:+6.1f}% volatility)")

        print("\nDiagnostic Information:")
        persistence = (
            tarchx_results.parameters.get("alpha[1]", 0)
            + tarchx_results.parameters.get("beta[1]", 0)
            + tarchx_results.parameters.get("gamma[1]", 0) / 2
        )
        print(f"  Persistence (α+β+γ/2): {persistence:.4f}")
        print(f"  Leverage effect (γ): {tarchx_results.leverage_effect:.4f}")
        print(f"  Student-t df (ν): {tarchx_results.parameters.get('nu', np.nan):.2f}")

        # Calculate half-life
        if 0 < persistence < 1:
            half_life = -np.log(0.5) / np.log(persistence)
            print(f"  Half-life of shocks: {half_life:.1f} days")

    return tarchx_results

# Advantages of Manual Implementation for Your Thesis
"""
ADVANTAGES OF MANUAL TARCH-X IMPLEMENTATION:

1. **Methodological Accuracy**: 
   - Properly implements exogenous variables in variance equation
   - No approximations or workarounds needed
   - Full control over optimization process

2. **Academic Rigor**:
   - Transparent mathematical implementation
   - Proper likelihood function specification
   - Robust standard error computation via numerical Hessian

3. **Thesis Requirements**:
   - Demonstrates deep understanding of GARCH methodology
   - Shows ability to implement advanced econometric models
   - Provides clear academic contribution

4. **Flexibility**:
   - Can easily modify for different specifications
   - Can add additional exogenous variables
   - Can implement alternative distributions

5. **Reproducibility**:
   - Complete code transparency
   - No dependence on package-specific implementations
   - Easy to document in thesis methodology section

6. **Performance**:
   - Often faster than general-purpose packages
   - Can optimize for your specific use case
   - Better convergence control

USING IN YOUR THESIS:

1. Replace your existing estimate_tarch_x() method with estimate_tarch_x_manual()
2. Document the manual implementation in your methodology section
3. Emphasize this as a methodological contribution
4. Use for all your main results and robustness checks

The manual implementation gives you complete control and demonstrates
advanced econometric skills that will impress your thesis committee.
"""

if __name__ == "__main__":
    # Run the example
    run_enhanced_analysis_example()
