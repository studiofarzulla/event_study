#!/usr/bin/env python3
"""
Quick check of outputs to verify potential issues.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

def check_outputs():
    """Check outputs for potential issues."""

    print("=" * 60)
    print("CHECKING OUTPUTS FOR POTENTIAL ISSUES")
    print("=" * 60)

    # Check saved model parameters
    param_dir = Path("outputs/analysis_results/model_parameters")

    if param_dir.exists():
        param_files = list(param_dir.glob("*.json"))
        print(f"\nFound {len(param_files)} parameter files")

        for param_file in param_files:
            with open(param_file, 'r') as f:
                params = json.load(f)

            crypto = param_file.stem.replace('_parameters', '')
            print(f"\n{crypto.upper()} Results:")

            # Check TARCH-X model
            if 'TARCH-X' in params:
                tarchx = params['TARCH-X']

                # Check convergence
                print(f"  Converged: {tarchx.get('converged', 'Unknown')}")

                # Check event effects
                if 'event_effects' in tarchx:
                    effects = tarchx['event_effects']

                    if 'D_infrastructure' in effects:
                        infra = effects['D_infrastructure']
                        print(f"  Infrastructure effect: {infra['coefficient']:+.4f} (p={infra['p_value']:.3f})")

                    if 'D_regulatory' in effects:
                        reg = effects['D_regulatory']
                        print(f"  Regulatory effect: {reg['coefficient']:+.4f} (p={reg['p_value']:.3f})")

                # Check persistence
                if 'parameters' in tarchx:
                    params_dict = tarchx['parameters']
                    alpha = params_dict.get('alpha[1]', 0) if isinstance(params_dict.get('alpha[1]'), (int, float)) else params_dict.get('alpha[1]', {}).get('estimate', 0)
                    beta = params_dict.get('beta[1]', 0) if isinstance(params_dict.get('beta[1]'), (int, float)) else params_dict.get('beta[1]', {}).get('estimate', 0)
                    gamma = params_dict.get('gamma[1]', 0) if isinstance(params_dict.get('gamma[1]'), (int, float)) else params_dict.get('gamma[1]', {}).get('estimate', 0)

                    persistence = alpha + beta + gamma/2
                    print(f"  Persistence: {persistence:.4f}")

                    if persistence > 0.99:
                        print(f"    WARNING: High persistence!")
                    elif persistence > 1.0:
                        print(f"    ERROR: Non-stationary (persistence > 1)!")

    # Check hypothesis test results
    hyp_file = Path("outputs/analysis_results/hypothesis_test_results.csv")
    if hyp_file.exists():
        print("\n" + "=" * 40)
        print("HYPOTHESIS TEST RESULTS:")
        hyp_df = pd.read_csv(hyp_file, index_col=0)
        print(hyp_df)

    # Check FDR correction
    fdr_file = Path("outputs/analysis_results/fdr_corrected_pvalues.csv")
    if fdr_file.exists():
        print("\n" + "=" * 40)
        print("FDR CORRECTED P-VALUES:")
        fdr_df = pd.read_csv(fdr_file)
        print(f"Columns: {list(fdr_df.columns)}")

        # Check for significant results
        if 'Significant' in fdr_df.columns:
            significant = fdr_df[fdr_df['Significant']]
            print(f"Significant after FDR correction: {len(significant)}")
            if len(significant) > 0:
                print(significant)
        else:
            print(f"Total tests: {len(fdr_df)}")
            print(fdr_df.head())

    print("\n" + "=" * 60)
    print("CHECK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    check_outputs()