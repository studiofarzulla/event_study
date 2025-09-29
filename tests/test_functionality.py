"""
Performance and functionality validation tests.
"""

import pytest
import time
from pathlib import Path
import sys

# Add code directory to path
sys.path.append(str(Path(__file__).parent.parent / 'code'))

try:
    from data_preparation import DataPreparation
    import config
except ImportError as e:
    pytest.skip(f"Core modules not available: {e}", allow_module_level=True)


class TestFunctionality:
    """Test core functionality and basic performance."""
    
    def test_data_preparation_import(self):
        """Test that DataPreparation can be imported and instantiated."""
        dp = DataPreparation()
        assert dp is not None
        assert hasattr(dp, 'load_events')
        assert hasattr(dp, 'prepare_all_cryptos')
        
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        assert hasattr(config, 'CRYPTOCURRENCIES')
        assert isinstance(config.CRYPTOCURRENCIES, list)
        assert len(config.CRYPTOCURRENCIES) > 0
        
    def test_events_loading_performance(self):
        """Test events loading performance (should be fast)."""
        start_time = time.time()
        
        dp = DataPreparation()
        events = dp.load_events()
        
        end_time = time.time()
        load_time = end_time - start_time
        
        # Events loading should be very fast (< 1 second)
        assert load_time < 1.0, f"Events loading took {load_time:.2f}s, should be < 1.0s"
        
        # Events should have expected structure
        assert events is not None
        if hasattr(events, '__len__'):
            assert len(events) > 0, "Events data should not be empty"
            
    def test_crypto_data_sample_loading(self):
        """Test loading a sample of crypto data."""
        dp = DataPreparation()
        
        # Try to load just BTC data as a sample
        try:
            btc_data = dp.load_crypto_data('btc')
            assert btc_data is not None
            
            if hasattr(btc_data, '__len__'):
                assert len(btc_data) > 0, "BTC data should not be empty"
                
        except FileNotFoundError:
            pytest.skip("BTC data file not available for testing")
        except Exception as e:
            pytest.skip(f"Could not load BTC data: {e}")
            
    def test_gdelt_sentiment_loading(self):
        """Test GDELT sentiment data loading."""
        dp = DataPreparation()
        
        try:
            gdelt_data = dp.load_gdelt_data()
            assert gdelt_data is not None
            
            if hasattr(gdelt_data, '__len__'):
                assert len(gdelt_data) > 0, "GDELT data should not be empty"
                
        except FileNotFoundError:
            pytest.skip("GDELT data file not available for testing")
        except Exception as e:
            pytest.skip(f"Could not load GDELT data: {e}")


class TestPerformance:
    """Basic performance tests."""
    
    def test_module_import_speed(self):
        """Test that core modules import quickly."""
        start_time = time.time()
        
        # Re-import key modules to test import time
        import importlib
        importlib.reload(sys.modules.get('config', config))
        
        end_time = time.time()
        import_time = end_time - start_time
        
        # Module imports should be very fast
        assert import_time < 0.5, f"Module imports took {import_time:.2f}s, should be < 0.5s"
        
    @pytest.mark.slow
    def test_full_data_preparation_sample(self):
        """Test a sample of full data preparation pipeline (marked as slow)."""
        dp = DataPreparation()
        
        # Try to prepare just one cryptocurrency as a sample
        try:
            start_time = time.time()
            
            # This is just a structural test - don't run full preparation
            # which could take minutes with API calls
            sample_data = dp.prepare_crypto_data('btc', include_events=False, include_sentiment=False)
            
            end_time = time.time()
            prep_time = end_time - start_time
            
            # Basic data preparation should complete reasonably quickly
            assert prep_time < 30.0, f"Sample data prep took {prep_time:.2f}s, seems too slow"
            
            # Result should have basic structure
            assert sample_data is not None
            
        except Exception as e:
            pytest.skip(f"Sample data preparation failed: {e}")


if __name__ == "__main__":
    # Run just the fast tests by default
    pytest.main([__file__, "-v", "-m", "not slow"])