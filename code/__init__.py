"""
Cryptocurrency Event Study Analysis Package

This package provides comprehensive tools for analyzing cryptocurrency market volatility
in response to infrastructure and regulatory events using GARCH-family models.

Main modules:
- config: Configuration settings and paths
- data_preparation: Data loading and preprocessing
- garch_models: GARCH/TARCH model implementations
- event_impact_analysis: Event impact quantification
- hypothesis_testing_results: Statistical hypothesis testing
- publication_outputs: Publication-ready outputs generation
- robustness_checks: Model robustness validation
- coingecko_fetcher: Cryptocurrency data fetching
- bootstrap_inference: Bootstrap statistical inference
"""

__version__ = "1.0.0"
__author__ = "Murad Farzulla"
__email__ = "murad@farzulla.org"

# Import main classes for convenience - only import what's safe to avoid circular imports
# Individual modules can be imported directly if needed
