"""
Main analysis script for cryptocurrency event study.
This runs the complete analysis pipeline from data preparation through hypothesis testing.
"""

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
from datetime import datetime
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data_preparation import DataPreparation
from garch_models import estimate_models_for_all_cryptos
from event_impact_analysis import run_complete_analysis
from publication_outputs import generate_publication_outputs
from robustness_checks import run_robustness_checks
from bootstrap_inference import run_bootstrap_analysis
import config


def save_results_to_csv(results: dict, output_dir: Path):
    """Save analysis results to CSV files for easy viewing."""

    # Save hypothesis test results
    if 'hypothesis_test' in results:
        hypothesis_df = pd.DataFrame([results['hypothesis_test']['infrastructure']], index=['Infrastructure'])
        hypothesis_df = pd.concat([hypothesis_df,
                                  pd.DataFrame([results['hypothesis_test']['regulatory']], index=['Regulatory'])])
        hypothesis_df.to_csv(output_dir / 'hypothesis_test_results.csv')

    # Save FDR correction results
    if 'fdr_correction' in results and isinstance(results['fdr_correction'], pd.DataFrame):
        results['fdr_correction'].to_csv(output_dir / 'fdr_corrected_pvalues.csv', index=False)

    # Save inverse-variance weighted results
    if 'inverse_variance_weighted' in results:
        ivw_df = pd.DataFrame(results['inverse_variance_weighted']).T
        ivw_df.to_csv(output_dir / 'inverse_variance_weighted.csv')

    # Save by-cryptocurrency analysis
    if 'by_crypto' in results and isinstance(results['by_crypto'], pd.DataFrame):
        results['by_crypto'].to_csv(output_dir / 'analysis_by_crypto.csv', index=False)

    # Save publication table
    if 'publication_table' in results and isinstance(results['publication_table'], pd.DataFrame):
        results['publication_table'].to_csv(output_dir / 'publication_table.csv', index=False)

    print(f"\nResults saved to {output_dir}")


def main(run_robustness: bool = False,
         run_bootstrap: bool = False,
         generate_publication: bool = True):
    """
    Run the complete cryptocurrency event study analysis.
    """
    print("=" * 80)
    print("CRYPTOCURRENCY EVENT STUDY - COMPLETE ANALYSIS")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Validate data files exist
    data_path = Path(config.DATA_DIR)
    required_files = ['btc.csv', 'eth.csv', 'events.csv', 'gdelt.csv']
    missing_files = []

    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
            print(f"ERROR: Required file {file} not found in {data_path}")

    if missing_files:
        print(f"\nMissing {len(missing_files)} required files. Please ensure all data files are present.")
        return None

    print("[OK] All required data files found")

    # Create output directory
    output_dir = Path(config.ANALYSIS_RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Data Preparation
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)

    data_prep = DataPreparation()  # Will use config.DATA_DIR by default

    # Prepare data for all cryptocurrencies
    all_cryptos = ['btc', 'eth', 'xrp', 'bnb', 'ltc', 'ada']
    crypto_data = {}

    for crypto in all_cryptos:
        print(f"\nPreparing {crypto.upper()}...")
        try:
            crypto_data[crypto] = data_prep.prepare_crypto_data(
                crypto,
                include_events=True,
                include_sentiment=True
            )
            print(f"  [OK] Shape: {crypto_data[crypto].shape}")
            print(f"  Returns - Mean: {crypto_data[crypto]['returns_winsorized'].mean():.4f}%, "
                  f"Std: {crypto_data[crypto]['returns_winsorized'].std():.4f}%")
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            # Continue with other cryptos even if one fails

    if not crypto_data:
        print("\nERROR: No cryptocurrency data could be loaded")
        return None

    print(f"\nSuccessfully loaded {len(crypto_data)} cryptocurrencies")

    # Step 2: GARCH Model Estimation
    print("\n" + "=" * 60)
    print("STEP 2: GARCH MODEL ESTIMATION")
    print("=" * 60)

    model_results = estimate_models_for_all_cryptos(crypto_data)

    # Count successful models
    successful_models = {}
    for crypto, models in model_results.items():
        successful_models[crypto] = {}
        for model_name, result in models.items():
            if result.convergence:
                successful_models[crypto][model_name] = True

    print("\n" + "-" * 40)
    print("Model Convergence Summary:")
    print("-" * 40)
    for crypto in model_results:
        garch_ok = model_results[crypto].get('GARCH(1,1)', {}).convergence if 'GARCH(1,1)' in model_results[crypto] else False
        tarch_ok = model_results[crypto].get('TARCH(1,1)', {}).convergence if 'TARCH(1,1)' in model_results[crypto] else False
        tarchx_ok = model_results[crypto].get('TARCH-X', {}).convergence if 'TARCH-X' in model_results[crypto] else False

        print(f"{crypto.upper():5} - GARCH: {'[OK]' if garch_ok else '[FAIL]'} | "
              f"TARCH: {'[OK]' if tarch_ok else '[FAIL]'} | "
              f"TARCH-X: {'[OK]' if tarchx_ok else '[FAIL]'}")

    # Step 3: Event Impact Analysis
    print("\n" + "=" * 60)
    print("STEP 3: EVENT IMPACT ANALYSIS")
    print("=" * 60)

    analysis_results = run_complete_analysis(model_results)

    # Step 4: Summary of Key Findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS SUMMARY")
    print("=" * 60)

    # Hypothesis test results
    if 'hypothesis_test' in analysis_results and analysis_results['hypothesis_test']:
        hyp_test = analysis_results['hypothesis_test']

        print("\n1. Infrastructure vs Regulatory Comparison:")
        print("-" * 40)

        if 'infrastructure' in hyp_test and 'regulatory' in hyp_test:
            infra_mean = hyp_test['infrastructure'].get('mean', np.nan)
            reg_mean = hyp_test['regulatory'].get('mean', np.nan)

            print(f"Infrastructure mean effect: {infra_mean:.6f}")
            print(f"Regulatory mean effect:     {reg_mean:.6f}")
            print(f"Difference:                 {infra_mean - reg_mean:.6f}")

            if 't_test' in hyp_test:
                p_val = hyp_test['t_test']['p_value']
                print(f"\nT-test p-value: {p_val:.4f}")
                print(f"Result: {'Infrastructure > Regulatory (significant)' if p_val < 0.10 else 'No significant difference'}")

    # Inverse-variance weighted results
    if 'inverse_variance_weighted' in analysis_results and analysis_results['inverse_variance_weighted']:
        ivw = analysis_results['inverse_variance_weighted']

        print("\n2. Inverse-Variance Weighted Analysis:")
        print("-" * 40)

        if 'difference' in ivw:
            diff = ivw['difference']
            print(f"Weighted difference (Infra - Reg): {diff['value']:.6f}")
            print(f"Z-statistic: {diff['z_statistic']:.4f}")
            print(f"P-value: {diff['p_value']:.4f}")
            print(f"Significant at 10%: {'Yes' if diff['significant_10pct'] else 'No'}")

    # FDR correction summary
    if 'fdr_correction' in analysis_results and isinstance(analysis_results['fdr_correction'], pd.DataFrame):
        fdr_df = analysis_results['fdr_correction']

        print("\n3. Multiple Testing Correction (FDR):")
        print("-" * 40)

        if 'fdr_significant' in fdr_df.columns:
            n_total = len(fdr_df)
            n_sig_raw = (fdr_df['p_value'] < 0.10).sum()
            n_sig_fdr = fdr_df['fdr_significant'].sum()

            print(f"Total tests: {n_total}")
            print(f"Significant before FDR: {n_sig_raw}")
            print(f"Significant after FDR:  {n_sig_fdr}")
            print(f"False discoveries controlled: {n_sig_raw - n_sig_fdr}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    save_results_to_csv(analysis_results, output_dir)

    # Save model parameters for each crypto
    model_params_dir = output_dir / 'model_parameters'
    model_params_dir.mkdir(exist_ok=True)

    for crypto, models in model_results.items():
        crypto_params = {}
        for model_name, result in models.items():
            if result.convergence:
                crypto_params[model_name] = {
                    'AIC': result.aic,
                    'BIC': result.bic,
                    'parameters': result.parameters,
                    'leverage_effect': result.leverage_effect
                }

        # Save as JSON for easy reading
        with open(model_params_dir / f'{crypto}_parameters.json', 'w') as f:
            json.dump(crypto_params, f, indent=2, default=str)

    print(f"Model parameters saved to {model_params_dir}")

    # Final conclusion
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print("\nMain Hypothesis Test Result:")
    if 'hypothesis_test' in analysis_results and 't_test' in analysis_results['hypothesis_test']:
        p_val = analysis_results['hypothesis_test']['t_test']['p_value']
        if p_val < 0.01:
            print("*** Infrastructure events have SIGNIFICANTLY LARGER volatility impacts than Regulatory events (p < 0.01) ***")
        elif p_val < 0.05:
            print("** Infrastructure events have SIGNIFICANTLY LARGER volatility impacts than Regulatory events (p < 0.05) **")
        elif p_val < 0.10:
            print("* Infrastructure events have LARGER volatility impacts than Regulatory events (p < 0.10) *")
        else:
            print("No significant difference between Infrastructure and Regulatory event impacts")

    # Step 5: Robustness Checks (Optional)
    if run_robustness:
        print("\n" + "=" * 60)
        print("STEP 5: ROBUSTNESS CHECKS")
        print("=" * 60)

        robustness_results = run_robustness_checks(
            cryptos=['btc', 'eth'],  # Limited cryptos for speed
            run_bootstrap=run_bootstrap
        )

        analysis_results['robustness'] = robustness_results

        # Save robustness results
        robustness_dir = output_dir / 'robustness'
        robustness_dir.mkdir(exist_ok=True)

        if 'placebo_test' in robustness_results:
            placebo_df = pd.DataFrame([robustness_results['placebo_test']])
            placebo_df.to_csv(robustness_dir / 'placebo_test.csv', index=False)

        if 'winsorization' in robustness_results:
            winsor_df = pd.DataFrame(robustness_results['winsorization']).T
            winsor_df.to_csv(robustness_dir / 'winsorization_comparison.csv')

        print(f"Robustness results saved to {robustness_dir}")

    # Step 6: Bootstrap Inference (Optional)
    if run_bootstrap and not run_robustness:  # If not already done in robustness
        print("\n" + "=" * 60)
        print("STEP 6: BOOTSTRAP INFERENCE")
        print("=" * 60)

        # Example with BTC
        if 'btc' in crypto_data:
            btc_returns = crypto_data['btc']['returns_winsorized'].dropna()
            bootstrap_results = run_bootstrap_analysis(
                btc_returns,
                model_type='TARCH',
                n_bootstrap=500
            )
            analysis_results['bootstrap'] = bootstrap_results

            # Save bootstrap results
            if 'confidence_intervals' in bootstrap_results:
                bootstrap_df = pd.DataFrame(bootstrap_results['confidence_intervals']).T
                bootstrap_df.to_csv(output_dir / 'bootstrap_confidence_intervals.csv')

    # Step 7: Generate Publication Outputs (Optional)
    if generate_publication:
        print("\n" + "=" * 60)
        print("STEP 7: GENERATING PUBLICATION OUTPUTS")
        print("=" * 60)

        try:
            generate_publication_outputs(model_results, analysis_results, crypto_data)
            print("Publication outputs generated successfully")
        except Exception as e:
            print(f"Error generating publication outputs: {str(e)}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return analysis_results


if __name__ == "__main__":
    # Run with default settings
    # Set run_robustness=True to include robustness checks
    # Set run_bootstrap=True to include bootstrap inference
    # Set generate_publication=True to create LaTeX tables and plots
    results = main(
        run_robustness=True,   # Enable robustness checks including event window sensitivity
        run_bootstrap=False,   # Set to True to run bootstrap inference
        generate_publication=True  # Set to True to generate publication outputs
    )
