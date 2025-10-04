"""
Cryptocurrency Event Study Analysis Package

A comprehensive toolkit for analyzing the differential volatility impacts
of Infrastructure and Regulatory events on cryptocurrency markets using
GARCH models with sentiment controls.
"""

__version__ = "1.0.0"
__author__ = "Murad Farzulla"
__email__ = "murad@farzulla.org"

from .data_preparation import DataPreparation
from .garch_models import estimate_models_for_all_cryptos
from .event_impact_analysis import run_complete_analysis
from .publication_outputs import generate_publication_outputs
from .robustness_checks import run_robustness_checks
from .bootstrap_inference import run_bootstrap_analysis

__all__ = [
    'DataPreparation',
    'estimate_models_for_all_cryptos',
    'run_complete_analysis',
    'generate_publication_outputs',
    'run_robustness_checks',
    'run_bootstrap_analysis',
]
