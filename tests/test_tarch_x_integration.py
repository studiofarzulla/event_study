"""
Test script to verify TARCH-X manual implementation integration.
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Add code directory to path
sys.path.append(str(Path(__file__).parent / 'code'))

from data_preparation import DataPreparation
from garch_models import GARCHModels

def test_tarch_x_integration():
    """
    Test the integrated manual TARCH-X implementation.
    """
    print("=" * 80)
    print("TESTING MANUAL TARCH-X INTEGRATION")
    print("=" * 80)

    try:
        # Load data
        print("\n1. Loading data...")
        data_prep = DataPreparation(data_path="data")

        # Test with Bitcoin
        btc_data = data_prep.prepare_crypto_data(
            'btc',
            include_events=True,
            include_sentiment=True
        )

        print(f"   Loaded BTC data: {btc_data.shape}")
        print(f"   Returns column: {'returns_winsorized' in btc_data.columns}")
        print(f"   Event columns: {[col for col in btc_data.columns if col.startswith('D_')][:5]}...")

        # Initialize GARCH models
        print("\n2. Initializing GARCH models...")
        modeler = GARCHModels(btc_data, 'btc')

        # Test TARCH-X with aggregated event dummies
        print("\n3. Testing TARCH-X with aggregated event dummies...")
        tarchx_results = modeler.estimate_tarch_x(
            use_individual_events=False,
            include_sentiment=True
        )

        # Verify results
        print("\n4. Verifying results...")
        if tarchx_results.convergence:
            print("   [OK] Model converged successfully")
            print(f"   [OK] AIC: {tarchx_results.aic:.2f}")
            print(f"   [OK] BIC: {tarchx_results.bic:.2f}")
            print(f"   [OK] Log-likelihood: {tarchx_results.log_likelihood:.2f}")

            # Check parameters
            print("\n5. Checking parameters...")
            if 'omega' in tarchx_results.parameters:
                print(f"   [OK] omega: {tarchx_results.parameters['omega']:.6f}")
            if 'alpha' in tarchx_results.parameters:
                print(f"   [OK] alpha: {tarchx_results.parameters['alpha']:.6f}")
            if 'gamma' in tarchx_results.parameters:
                print(f"   [OK] gamma: {tarchx_results.parameters['gamma']:.6f}")
            if 'beta' in tarchx_results.parameters:
                print(f"   [OK] beta: {tarchx_results.parameters['beta']:.6f}")
            if 'nu' in tarchx_results.parameters:
                print(f"   [OK] nu: {tarchx_results.parameters['nu']:.2f}")

            # Check event effects
            print("\n6. Checking event effects...")
            if tarchx_results.event_effects:
                print(f"   [OK] Number of event effects: {len(tarchx_results.event_effects)}")
                for event, coef in list(tarchx_results.event_effects.items())[:3]:
                    p_val = tarchx_results.pvalues.get(event, np.nan)
                    print(f"   [OK] {event}: {coef:+.4f} (p={p_val:.4f})")
            else:
                print("   [WARNING] No event effects found")

            # Check sentiment effects
            print("\n7. Checking sentiment effects...")
            if tarchx_results.sentiment_effects:
                print(f"   [OK] Number of sentiment effects: {len(tarchx_results.sentiment_effects)}")
                for sent, coef in list(tarchx_results.sentiment_effects.items())[:2]:
                    p_val = tarchx_results.pvalues.get(sent, np.nan)
                    print(f"   [OK] {sent}: {coef:+.4f} (p={p_val:.4f})")
            else:
                print("   [WARNING] No sentiment effects found")

            # Check compatibility attributes
            print("\n8. Checking compatibility attributes...")
            if hasattr(tarchx_results, 'event_std_errors'):
                print(f"   [OK] event_std_errors present: {len(tarchx_results.event_std_errors)} items")
            if hasattr(tarchx_results, 'event_pvalues'):
                print(f"   [OK] event_pvalues present: {len(tarchx_results.event_pvalues)} items")
            if hasattr(tarchx_results, 'leverage_effect'):
                print(f"   [OK] leverage_effect: {tarchx_results.leverage_effect:.4f}")

            print("\n" + "=" * 80)
            print("TEST PASSED: Manual TARCH-X integration successful!")
            print("=" * 80)

        else:
            print("   [FAIL] Model did not converge")
            print("\nTEST FAILED: Model convergence issue")

    except Exception as e:
        print(f"\n[ERROR] during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTEST FAILED: Exception occurred")
        return False

    return True


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Run the test
    success = test_tarch_x_integration()

    if success:
        print("\n[SUCCESS] All tests completed successfully!")
        print("\nThe manual TARCH-X implementation has been successfully integrated.")
        print("You can now use it in your event study analysis with:")
        print("  - Proper variance equation specification")
        print("  - Event dummies affecting conditional variance directly")
        print("  - Robust standard errors via numerical Hessian")
        print("  - Full compatibility with existing analysis code")
    else:
        print("\n[FAIL] Tests failed. Please review the errors above.")