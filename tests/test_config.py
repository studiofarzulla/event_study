"""
Basic configuration tests for the event study project.
"""

import os
import pytest
from pathlib import Path
import sys

# Add code directory to path
sys.path.append(str(Path(__file__).parent.parent / 'code'))

try:
    import config
except ImportError:
    config = None


class TestConfiguration:
    """Test configuration settings and validation."""
    
    def test_config_import(self):
        """Test that config module can be imported."""
        assert config is not None, "Config module should be importable"
        
    def test_base_directories_exist(self):
        """Test that required directories are defined."""
        if config:
            assert hasattr(config, 'BASE_DIR'), "BASE_DIR should be defined"
            assert hasattr(config, 'DATA_DIR'), "DATA_DIR should be defined"
            assert hasattr(config, 'OUTPUTS_DIR'), "OUTPUTS_DIR should be defined"
            
    def test_cryptocurrency_list(self):
        """Test that cryptocurrency list is properly defined."""
        if config:
            assert hasattr(config, 'CRYPTOCURRENCIES'), "CRYPTOCURRENCIES should be defined"
            cryptos = config.CRYPTOCURRENCIES
            assert isinstance(cryptos, list), "CRYPTOCURRENCIES should be a list"
            assert len(cryptos) > 0, "CRYPTOCURRENCIES should not be empty"
            assert 'btc' in cryptos, "Bitcoin should be in cryptocurrency list"
            assert 'eth' in cryptos, "Ethereum should be in cryptocurrency list"
            
    def test_model_parameters(self):
        """Test that model parameters are properly defined."""
        if config:
            assert hasattr(config, 'GARCH_P'), "GARCH_P should be defined"
            assert hasattr(config, 'GARCH_Q'), "GARCH_Q should be defined"
            assert config.GARCH_P >= 1, "GARCH_P should be at least 1"
            assert config.GARCH_Q >= 1, "GARCH_Q should be at least 1"
            
    def test_event_window_parameters(self):
        """Test that event window parameters are reasonable."""
        if config:
            assert hasattr(config, 'DEFAULT_EVENT_WINDOW_BEFORE'), "Event window before should be defined"
            assert hasattr(config, 'DEFAULT_EVENT_WINDOW_AFTER'), "Event window after should be defined"
            
            before = config.DEFAULT_EVENT_WINDOW_BEFORE
            after = config.DEFAULT_EVENT_WINDOW_AFTER
            
            assert before >= 1, "Event window before should be at least 1 day"
            assert after >= 1, "Event window after should be at least 1 day"
            assert before <= 10, "Event window before should be reasonable (≤10 days)"
            assert after <= 10, "Event window after should be reasonable (≤10 days)"
            
    @pytest.mark.skipif(config is None, reason="Config module not available")
    def test_validate_config_function(self):
        """Test the config validation function."""
        if hasattr(config, 'validate_config'):
            errors = config.validate_config()
            # We expect some errors in test environment (no API key, etc.)
            assert isinstance(errors, list), "validate_config should return a list"


class TestDataFiles:
    """Test that required data files exist and are readable."""
    
    def test_data_directory_exists(self):
        """Test that data directory exists."""
        data_dir = Path(__file__).parent.parent / 'data'
        assert data_dir.exists(), f"Data directory should exist: {data_dir}"
        
    def test_events_csv_exists(self):
        """Test that events.csv file exists."""
        events_file = Path(__file__).parent.parent / 'data' / 'events.csv'
        assert events_file.exists(), "events.csv file should exist"
        
    def test_events_csv_readable(self):
        """Test that events.csv is readable and has content."""
        events_file = Path(__file__).parent.parent / 'data' / 'events.csv'
        if events_file.exists():
            content = events_file.read_text()
            assert len(content) > 0, "events.csv should not be empty"
            assert 'Date' in content or 'date' in content, "events.csv should have date column"
            
    def test_crypto_data_files_exist(self):
        """Test that main cryptocurrency data files exist."""
        data_dir = Path(__file__).parent.parent / 'data'
        expected_files = ['btc.csv', 'eth.csv', 'xrp.csv', 'bnb.csv', 'ltc.csv', 'ada.csv']
        
        for crypto_file in expected_files:
            file_path = data_dir / crypto_file
            assert file_path.exists(), f"Cryptocurrency data file should exist: {crypto_file}"


if __name__ == "__main__":
    pytest.main([__file__])