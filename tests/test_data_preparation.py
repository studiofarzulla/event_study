"""
Comprehensive tests for the data preparation module.
Tests all key functionality including special event handling.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'code'))

from data_preparation import DataPreparation


class TestDataPreparation:
    """Test suite for DataPreparation class"""

    @pytest.fixture
    def data_prep(self):
        """Create DataPreparation instance with correct path"""
        return DataPreparation(data_path="../data")

    @pytest.fixture
    def sample_prices(self):
        """Create sample price series for testing"""
        dates = pd.date_range(start='2019-01-01', end='2019-01-31', freq='D', tz='UTC')
        prices = pd.Series(
            data=np.random.lognormal(8, 0.02, len(dates)),
            index=dates,
            name='price'
        )
        return prices

    def test_initialization(self, data_prep):
        """Test DataPreparation initialization"""
        assert data_prep.data_path.exists()
        assert len(data_prep.cryptocurrencies) == 6
        assert data_prep.start_date == pd.Timestamp('2019-01-01', tz='UTC')
        assert data_prep.end_date == pd.Timestamp('2025-08-31', tz='UTC')

    def test_load_crypto_prices(self, data_prep):
        """Test loading cryptocurrency price data"""
        # Test loading BTC data
        btc_data = data_prep.load_crypto_prices('btc')

        assert isinstance(btc_data, pd.DataFrame)
        assert 'price' in btc_data.columns
        assert btc_data.index.name == 'snapped_at'
        assert btc_data.index[0] >= data_prep.start_date
        assert btc_data.index[-1] <= data_prep.end_date

    def test_calculate_log_returns(self, data_prep, sample_prices):
        """Test log returns calculation"""
        returns = data_prep.calculate_log_returns(sample_prices)

        # Check basic properties
        assert len(returns) == len(sample_prices) - 1  # First value is NaN
        assert not returns.isnull().any()  # No NaN after dropping first

        # Check calculation is correct
        manual_return = np.log(sample_prices.iloc[2] / sample_prices.iloc[1]) * 100
        np.testing.assert_almost_equal(returns.iloc[1], manual_return, decimal=5)

    def test_winsorize_returns(self, data_prep):
        """Test returns winsorization"""
        # Create returns with outliers
        returns = pd.Series(
            data=np.random.normal(0, 1, 100),
            index=pd.date_range(start='2019-01-01', periods=100, freq='D')
        )
        # Add extreme outliers
        returns.iloc[50] = 20  # Large positive outlier
        returns.iloc[75] = -20  # Large negative outlier

        winsorized = data_prep.winsorize_returns(returns, window=30, n_std=5)

        # Check that outliers were clipped
        assert winsorized.iloc[50] < returns.iloc[50]
        assert winsorized.iloc[75] > returns.iloc[75]
        assert winsorized.max() < 20
        assert winsorized.min() > -20

    def test_load_events(self, data_prep):
        """Test loading event data"""
        events = data_prep.load_events()

        assert isinstance(events, pd.DataFrame)
        assert 'event_id' in events.columns
        assert 'date' in events.columns
        assert 'type' in events.columns
        assert len(events) == 50  # Total number of events

        # Check event types
        assert set(events['type'].unique()) == {'Infrastructure', 'Regulatory'}

    def test_create_event_window(self, data_prep):
        """Test event window creation"""
        event_date = pd.Timestamp('2020-01-15', tz='UTC')
        window = data_prep.create_event_window(event_date, days_before=3, days_after=3)

        assert len(window) == 7  # 3 before + event day + 3 after
        assert window[3] == event_date
        assert window[0] == event_date - timedelta(days=3)
        assert window[-1] == event_date + timedelta(days=3)

    def test_event_dummies_regular(self, data_prep):
        """Test creation of regular event dummy variables"""
        # Create date index
        dates = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D', tz='UTC')

        # Create simple event DataFrame
        events = pd.DataFrame({
            'event_id': [1],
            'date': [pd.Timestamp('2019-06-15')],
            'type': ['Infrastructure']
        })

        dummies = data_prep.create_event_dummies(dates, events)

        # Check dummy creation
        assert 'D_event_1' in dummies.columns
        assert dummies['D_event_1'].sum() == 7  # 7-day window

        # Check window dates
        event_date = pd.Timestamp('2019-06-15', tz='UTC')
        assert dummies.loc[event_date, 'D_event_1'] == 1
        assert dummies.loc[event_date - timedelta(days=3), 'D_event_1'] == 1
        assert dummies.loc[event_date + timedelta(days=3), 'D_event_1'] == 1

    def test_sec_twin_suits_handling(self, data_prep):
        """Test special handling of SEC twin suits"""
        dates = pd.date_range(start='2023-06-01', end='2023-06-15', freq='D', tz='UTC')

        events = pd.DataFrame({
            'event_id': [31, 32],
            'date': [pd.Timestamp('2023-06-05'), pd.Timestamp('2023-06-06')],
            'type': ['Regulatory', 'Regulatory']
        })

        dummies = data_prep.create_event_dummies(dates, events)

        # Check composite dummy creation
        assert 'D_SEC_enforcement_2023' in dummies.columns
        assert 'D_event_31' not in dummies.columns  # Should not exist
        assert 'D_event_32' not in dummies.columns  # Should not exist

        # Check window (June 2-9)
        assert dummies.loc[pd.Timestamp('2023-06-02', tz='UTC'), 'D_SEC_enforcement_2023'] == 1
        assert dummies.loc[pd.Timestamp('2023-06-09', tz='UTC'), 'D_SEC_enforcement_2023'] == 1
        assert dummies['D_SEC_enforcement_2023'].sum() == 8

    def test_eip_poly_overlap_handling(self, data_prep):
        """Test special handling of EIP-1559 and Poly hack overlap"""
        dates = pd.date_range(start='2021-08-01', end='2021-08-15', freq='D', tz='UTC')

        events = pd.DataFrame({
            'event_id': [17, 18],
            'date': [pd.Timestamp('2021-08-05'), pd.Timestamp('2021-08-10')],
            'type': ['Infrastructure', 'Infrastructure']
        })

        dummies = data_prep.create_event_dummies(dates, events)

        # Check individual dummies exist
        assert 'D_event_17' in dummies.columns
        assert 'D_event_18' in dummies.columns

        # Check overlap adjustment on Aug 7-8
        overlap_date = pd.Timestamp('2021-08-07', tz='UTC')
        assert dummies.loc[overlap_date, 'D_event_17'] == 0.5  # 1 - 0.5 adjustment
        assert dummies.loc[overlap_date, 'D_event_18'] == 0.5  # 1 - 0.5 adjustment

    def test_bybit_sec_truncation(self, data_prep):
        """Test truncation of Bybit and SEC dismissal windows"""
        dates = pd.date_range(start='2025-02-18', end='2025-03-03', freq='D', tz='UTC')

        events = pd.DataFrame({
            'event_id': [43, 44],
            'date': [pd.Timestamp('2025-02-21'), pd.Timestamp('2025-02-27')],
            'type': ['Infrastructure', 'Regulatory']
        })

        dummies = data_prep.create_event_dummies(dates, events)

        # Check Bybit window ends on Feb 23
        assert dummies.loc[pd.Timestamp('2025-02-23', tz='UTC'), 'D_event_43'] == 1
        assert dummies.loc[pd.Timestamp('2025-02-24', tz='UTC'), 'D_event_43'] == 0

        # Check SEC window starts on Feb 27
        assert dummies.loc[pd.Timestamp('2025-02-26', tz='UTC'), 'D_event_44'] == 0
        assert dummies.loc[pd.Timestamp('2025-02-27', tz='UTC'), 'D_event_44'] == 1

    def test_load_gdelt_sentiment(self, data_prep):
        """Test loading GDELT sentiment data"""
        sentiment = data_prep.load_gdelt_sentiment()

        assert isinstance(sentiment, pd.DataFrame)
        assert 'S_gdelt_normalized' in sentiment.columns
        assert 'S_reg_decomposed' in sentiment.columns
        assert 'S_infra_decomposed' in sentiment.columns

        # Check frequency is weekly
        date_diffs = sentiment.index[1:] - sentiment.index[:-1]
        assert all(diff.days == 7 for diff in date_diffs[:10])

    def test_merge_sentiment_data(self, data_prep):
        """Test merging sentiment with daily data"""
        # Create daily data
        daily_dates = pd.date_range(start='2019-06-01', end='2019-06-30', freq='D', tz='UTC')
        daily_data = pd.DataFrame(
            {'price': np.random.randn(len(daily_dates))},
            index=daily_dates
        )

        # Load actual sentiment data
        sentiment = data_prep.load_gdelt_sentiment()

        # Merge
        merged = data_prep.merge_sentiment_data(daily_data, sentiment)

        # Check forward-fill worked
        assert 'S_gdelt_normalized' in merged.columns
        assert not merged['S_gdelt_normalized'].isnull().all()

        # Check pre-June 2019 handling (should be 0)
        early_dates = pd.date_range(start='2019-01-01', end='2019-05-31', freq='D', tz='UTC')
        early_data = pd.DataFrame({'price': np.zeros(len(early_dates))}, index=early_dates)
        early_merged = data_prep.merge_sentiment_data(early_data, sentiment)

        if 'S_gdelt_normalized' in early_merged.columns:
            assert (early_merged['S_gdelt_normalized'] == 0).all()

    def test_prepare_crypto_data_integration(self, data_prep):
        """Integration test for complete data preparation pipeline"""
        # Test with BTC
        btc_data = data_prep.prepare_crypto_data('btc', include_events=True, include_sentiment=True)

        # Check all components are present
        assert 'price' in btc_data.columns
        assert 'returns' in btc_data.columns
        assert 'returns_winsorized' in btc_data.columns

        # Check event dummies
        event_cols = [col for col in btc_data.columns if col.startswith('D_')]
        assert len(event_cols) > 0
        assert 'D_infrastructure' in btc_data.columns
        assert 'D_regulatory' in btc_data.columns

        # Check sentiment columns
        assert 'S_gdelt_normalized' in btc_data.columns or 'S_reg_decomposed' in btc_data.columns

        # Validate data quality
        validation = data_prep.validate_data(btc_data)
        assert validation['infinite_returns'] == 0
        assert validation['return_std'] > 0

    def test_prepare_all_cryptos(self, data_prep):
        """Test preparing data for all cryptocurrencies"""
        all_data = data_prep.prepare_all_cryptos(include_events=True, include_sentiment=True)

        # Check all cryptos are present
        assert len(all_data) == 6
        assert all(crypto in all_data for crypto in data_prep.cryptocurrencies)

        # Check each crypto has consistent structure
        first_crypto_cols = all_data['btc'].columns
        for crypto, df in all_data.items():
            assert set(df.columns) == set(first_crypto_cols)

    def test_data_validation(self, data_prep):
        """Test data validation functionality"""
        # Load and prepare data
        btc_data = data_prep.prepare_crypto_data('btc')

        # Validate
        validation = data_prep.validate_data(btc_data)

        # Check validation results
        assert 'missing_values' in validation
        assert 'infinite_returns' in validation
        assert 'max_return' in validation
        assert 'min_return' in validation
        assert 'return_std' in validation

        # Returns should be reasonable after winsorization
        assert abs(validation['max_return']) < 50  # Less than 50% daily return
        assert abs(validation['min_return']) < 50


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])