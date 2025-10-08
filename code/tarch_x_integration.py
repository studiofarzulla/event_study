"""
Integration Guide: Manual TARCH-X with Existing Event Study Framework
===================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the manual implementation to imports
sys.path.append(str(Path(__file__).parent))
from tarch_x_manual import estimate_tarch_x_manual, TARCHXResults
from data_preparation import DataPreparation
from garch_models import ModelResults


class EnhancedGARCHModels:
    
    def __init__(self, data: pd.DataFrame, crypto: str):
        """
        Initialize enhanced GARCH model estimator.
        
        Args:
            data: Prepared DataFrame with returns and exogenous variables
            crypto: Cryptocurrency symbol for identification
        """
        self.data = data.copy()
        self.crypto = crypto
        self.returns = data['returns_winsorized'].dropna()
        
        # Store exogenous variables if present
        self.has_events = any(col.startswith('D_') for col in data.columns)
        self.has_sentiment = any('gdelt' in col or 'reg_decomposed' in col or 'infra_decomposed' in col
                                for col in data.columns)
    
    def estimate_tarch_x_manual(self, use_individual_events: bool = False,
                               include_sentiment: bool = True) -> ModelResults:
        """
        Args:
            use_individual_events: If True, use individual event dummies;
                                 if False, use aggregated type dummies  
            include_sentiment: Whether to include sentiment variables
        """
        print(f"Estimating TARCH-X (manual) for {self.crypto}...")
        
        # Prepare exogenous variables
        exog_vars = self._prepare_exogenous_variables(use_individual_events, include_sentiment)
        
        if exog_vars is None or exog_vars.empty:
            print("  [WARNING] No exogenous variables available for TARCH-X")
            return self._create_failed_result('TARCH-X')
        
        try:
            # Use manual TARCH-X implementation
            manual_results = estimate_tarch_x_manual(
                returns=self.returns,
                exog_vars=exog_vars,
                method='SLSQP'
            )
            
            if not manual_results.converged:
                print("  [FAIL] Manual TARCH-X did not converge")
                return self._create_failed_result('TARCH-X')
            
            # Convert manual results to ModelResults format for compatibility
            model_results = ModelResults(
                model_type='TARCH-X',
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
                sentiment_effects=manual_results.sentiment_effects
            )
            
            # Add special attributes for event analysis compatibility
            model_results.event_std_errors = {k: manual_results.std_errors.get(k, np.nan) 
                                            for k in manual_results.event_effects.keys()}
            model_results.event_pvalues = {k: manual_results.pvalues.get(k, np.nan) 
                                         for k in manual_results.event_effects.keys()}
            
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
            return self._create_failed_result('TARCH-X')
    
    def _prepare_exogenous_variables(self, use_individual_events: bool,
                                   include_sentiment: bool) -> pd.DataFrame:
        """
        Prepare exogenous variables for TARCH-X model.
        """
        exog_vars = []
        
        # Add event dummies
        if self.has_events:
            if use_individual_events:
                # Use individual event dummies
                event_cols = [col for col in self.data.columns
                            if col.startswith('D_event_') or col == 'D_SEC_enforcement_2023']
                if event_cols:
                    exog_vars.extend(event_cols)
            else:
                # Use aggregated type dummies
                if 'D_infrastructure' in self.data.columns:
                    exog_vars.append('D_infrastructure')
                if 'D_regulatory' in self.data.columns:
                    exog_vars.append('D_regulatory')
        
        # Add sentiment variables
        if include_sentiment and self.has_sentiment:
            sentiment_cols = [col for col in self.data.columns
                            if 'gdelt_normalized' in col or
                               'reg_decomposed' in col or
                               'infra_decomposed' in col]
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
            residuals=pd.Series()
        )
