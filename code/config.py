"""
Configuration settings for the cryptocurrency event study project.
Centralizes all paths and settings to avoid hardcoded values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", str(BASE_DIR / "outputs"))

# Specific output directories
FIGURES_DIR = Path(OUTPUTS_DIR) / "figures"
TABLES_DIR = Path(OUTPUTS_DIR) / "tables"
PUBLICATION_DIR = Path(OUTPUTS_DIR) / "publication"
ANALYSIS_RESULTS_DIR = Path(OUTPUTS_DIR) / "analysis_results"

# Create output directories if they don't exist
for directory in [FIGURES_DIR, TABLES_DIR, PUBLICATION_DIR, ANALYSIS_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_RATE_LIMIT = float(os.getenv("COINGECKO_RATE_LIMIT", "1.2"))  # seconds between requests

# Analysis parameters
DEFAULT_EVENT_WINDOW_BEFORE = int(os.getenv("EVENT_WINDOW_BEFORE", "3"))
DEFAULT_EVENT_WINDOW_AFTER = int(os.getenv("EVENT_WINDOW_AFTER", "3"))
WINSORIZATION_STD = float(os.getenv("WINSORIZATION_STD", "5"))
WINSORIZATION_WINDOW = int(os.getenv("WINSORIZATION_WINDOW", "30"))

# Model parameters
GARCH_P = int(os.getenv("GARCH_P", "1"))
GARCH_Q = int(os.getenv("GARCH_Q", "1"))
BOOTSTRAP_N_SIMULATIONS = int(os.getenv("BOOTSTRAP_N_SIMULATIONS", "1000"))

# Cryptocurrency list
CRYPTOCURRENCIES = ["btc", "eth", "xrp", "bnb", "ltc", "ada"]

# Date range for analysis
START_DATE = os.getenv("ANALYSIS_START_DATE", "2019-01-01")
END_DATE = os.getenv("ANALYSIS_END_DATE", "2025-08-31")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(BASE_DIR / "event_study.log"))


# Validation settings
def validate_config():
    """Validate that required configuration is present."""
    errors = []

    if not COINGECKO_API_KEY:
        errors.append("COINGECKO_API_KEY not set in environment variables")

    if not Path(DATA_DIR).exists():
        errors.append(f"Data directory does not exist: {DATA_DIR}")

    return errors


# Special event configurations
SPECIAL_EVENTS = {
    "SEC_TWIN_SUITS": {
        "event_ids": [31, 32],
        "composite_name": "D_SEC_enforcement_2023",
        "start_date": "2023-06-02",
        "end_date": "2023-06-09",
    },
    "EIP_POLY_OVERLAP": {
        "event_ids": [17, 18],
        "overlap_dates": ["2021-08-07", "2021-08-08"],
        "adjustment_factor": 0.5,
    },
    "BYBIT_SEC_TRUNCATION": {"bybit_id": 43, "bybit_end": "2025-02-23", "sec_id": 44, "sec_start": "2025-02-27"},
}
