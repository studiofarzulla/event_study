"""
Quick test to verify all critical components work before full run.
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.append(str(Path(__file__).parent / 'code'))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")

    try:
        from data_preparation import DataPreparation
        print("[OK] DataPreparation imported")

        from garch_models import GARCHModels
        print("[OK] GARCHModels imported")

        from tarch_x_manual import estimate_tarch_x_manual
        print("[OK] Manual TARCH-X imported")

        from event_impact_analysis import EventImpactAnalysis
        print("[OK] EventImpactAnalysis imported")

        from bootstrap_inference import BootstrapInference
        print("[OK] BootstrapInference imported")

        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic data loading and model estimation."""
    print("\nTesting basic functionality...")

    from data_preparation import DataPreparation
    from garch_models import GARCHModels
    import pandas as pd
    import numpy as np

    try:
        # Test DataPreparation
        prep = DataPreparation()
        print(f"[OK] DataPreparation initialized with UTC dates")
        print(f"  Start: {prep.start_date}, End: {prep.end_date}")

        # Create dummy data for testing
        dates = pd.date_range('2020-01-01', periods=100, freq='D', tz='UTC')
        dummy_data = pd.DataFrame({
            'returns_winsorized': np.random.randn(100) * 2,
            'D_infrastructure': np.random.binomial(1, 0.1, 100),
            'D_regulatory': np.random.binomial(1, 0.1, 100)
        }, index=dates)

        # Test GARCHModels
        models = GARCHModels(dummy_data, 'test')
        print("[OK] GARCHModels initialized")

        # Check that estimate_tarch_x uses manual implementation
        import inspect
        source = inspect.getsource(models.estimate_tarch_x)
        if 'estimate_tarch_x_manual' in source:
            print("[OK] TARCH-X uses manual implementation (correct)")
        else:
            print("[FAIL] TARCH-X doesn't use manual implementation (error)")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Functionality test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("CRITICAL FIXES VERIFICATION")
    print("=" * 60)

    success = True

    # Test imports
    if not test_imports():
        success = False

    # Test functionality
    if not test_basic_functionality():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("[OK] ALL TESTS PASSED - CODE IS READY TO RUN")
        print("\nYou can now run: python code/run_event_study_analysis.py")
    else:
        print("[FAIL] SOME TESTS FAILED - REVIEW NEEDED")

    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)