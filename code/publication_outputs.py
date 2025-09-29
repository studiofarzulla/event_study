"""
Publication-ready outputs for cryptocurrency event study.
Generates LaTeX tables, high-quality plots, and CSV exports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

# Configure matplotlib for publication quality
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.figsize"] = (8, 6)

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


class PublicationOutputs:
    """
    Generate publication-ready outputs for the event study.
    """

    def __init__(self, model_results: Dict, analysis_results: Dict, crypto_data: Dict):
        """
        Initialize publication outputs generator.

        Args:
            model_results: Dictionary of GARCH model results by crypto
            analysis_results: Dictionary of analysis results (hypothesis tests, etc.)
            crypto_data: Dictionary of prepared data by crypto
        """
        self.model_results = model_results
        self.analysis_results = analysis_results
        self.crypto_data = crypto_data
        self.output_dir = Path("../outputs/publication")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_latex_model_comparison_table(self) -> str:
        """
        Generate LaTeX table for model comparison (AIC/BIC).

        Returns:
            LaTeX formatted table string
        """
        rows = []

        for crypto, models in self.model_results.items():
            crypto_upper = crypto.upper()

            # Get model statistics
            garch_aic = (
                models.get("GARCH(1,1)", {}).aic
                if "GARCH(1,1)" in models and models["GARCH(1,1)"].convergence
                else np.nan
            )
            garch_bic = (
                models.get("GARCH(1,1)", {}).bic
                if "GARCH(1,1)" in models and models["GARCH(1,1)"].convergence
                else np.nan
            )

            tarch_aic = (
                models.get("TARCH(1,1)", {}).aic
                if "TARCH(1,1)" in models and models["TARCH(1,1)"].convergence
                else np.nan
            )
            tarch_bic = (
                models.get("TARCH(1,1)", {}).bic
                if "TARCH(1,1)" in models and models["TARCH(1,1)"].convergence
                else np.nan
            )

            tarchx_aic = (
                models.get("TARCH-X", {}).aic if "TARCH-X" in models and models["TARCH-X"].convergence else np.nan
            )
            tarchx_bic = (
                models.get("TARCH-X", {}).bic if "TARCH-X" in models and models["TARCH-X"].convergence else np.nan
            )

            # Find best model by AIC
            min_aic = np.nanmin([garch_aic, tarch_aic, tarchx_aic])
            garch_mark = "*" if garch_aic == min_aic else ""
            tarch_mark = "*" if tarch_aic == min_aic else ""
            tarchx_mark = "*" if tarchx_aic == min_aic else ""

            rows.append(
                f"{crypto_upper} & "
                f"{garch_aic:.1f}{garch_mark} & {garch_bic:.1f} & "
                f"{tarch_aic:.1f}{tarch_mark} & {tarch_bic:.1f} & "
                f"{tarchx_aic:.1f}{tarchx_mark} & {tarchx_bic:.1f} \\\\"
            )

        # Create LaTeX table
        latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Model Comparison: AIC and BIC Statistics}
\label{tab:model_comparison}
\begin{tabular}{lcccccc}
\hline
\hline
 & \multicolumn{2}{c}{GARCH(1,1)} & \multicolumn{2}{c}{TARCH(1,1)} & \multicolumn{2}{c}{TARCH-X} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
Crypto & AIC & BIC & AIC & BIC & AIC & BIC \\
\hline
"""
        latex_table += "\n".join(rows)
        latex_table += r"""
\hline
\hline
\end{tabular}
\begin{tablenotes}
\small
\item Note: * indicates the best model by AIC for each cryptocurrency.
\item GARCH(1,1): Standard GARCH model. TARCH(1,1): Threshold ARCH with leverage effects.
\item TARCH-X: TARCH with event dummies and sentiment variables.
\end{tablenotes}
\end{table}
"""
        return latex_table

    def generate_latex_event_comparison_table(self) -> str:
        """
        Generate LaTeX table for Infrastructure vs Regulatory comparison.

        Returns:
            LaTeX formatted table string
        """
        if "hypothesis_test" not in self.analysis_results:
            return "% No hypothesis test results available"

        hyp_test = self.analysis_results["hypothesis_test"]

        # Extract statistics
        infra = hyp_test.get("infrastructure", {})
        reg = hyp_test.get("regulatory", {})

        # Format values
        infra_mean = f"{infra.get('mean', np.nan):.6f}" if "mean" in infra else "N/A"
        infra_median = f"{infra.get('median', np.nan):.6f}" if "median" in infra else "N/A"
        infra_std = f"{infra.get('std', np.nan):.6f}" if "std" in infra else "N/A"
        infra_n = infra.get("n", 0)

        reg_mean = f"{reg.get('mean', np.nan):.6f}" if "mean" in reg else "N/A"
        reg_median = f"{reg.get('median', np.nan):.6f}" if "median" in reg else "N/A"
        reg_std = f"{reg.get('std', np.nan):.6f}" if "std" in reg else "N/A"
        reg_n = reg.get("n", 0)

        # Test statistics
        t_stat = hyp_test.get("t_test", {}).get("statistic", np.nan)
        t_pval = hyp_test.get("t_test", {}).get("p_value", np.nan)
        u_stat = hyp_test.get("mann_whitney", {}).get("statistic", np.nan)
        u_pval = hyp_test.get("mann_whitney", {}).get("p_value", np.nan)
        cohens_d = hyp_test.get("effect_size", np.nan)

        latex_table = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Event Type Comparison: Infrastructure vs Regulatory}}
\label{{tab:event_comparison}}
\begin{{tabular}}{{lcc}}
\hline
\hline
 & Infrastructure & Regulatory \\
\hline
\textbf{{Descriptive Statistics}} & & \\
Number of events & {infra_n} & {reg_n} \\
Mean coefficient & {infra_mean} & {reg_mean} \\
Median coefficient & {infra_median} & {reg_median} \\
Standard deviation & {infra_std} & {reg_std} \\
\hline
\textbf{{Statistical Tests}} & \multicolumn{{2}}{{c}}{{}} \\
T-test statistic & \multicolumn{{2}}{{c}}{{{t_stat:.4f}}} \\
T-test p-value & \multicolumn{{2}}{{c}}{{{t_pval:.4f}}} \\
Mann-Whitney U statistic & \multicolumn{{2}}{{c}}{{{u_stat:.4f}}} \\
Mann-Whitney p-value & \multicolumn{{2}}{{c}}{{{u_pval:.4f}}} \\
Cohen's d (effect size) & \multicolumn{{2}}{{c}}{{{cohens_d:.4f}}} \\
\hline
\hline
\end{{tabular}}
\begin{{tablenotes}}
\small
\item Note: Tests evaluate H0: Infrastructure = Regulatory vs H1: Infrastructure > Regulatory.
\item Coefficients represent volatility impacts from TARCH-X models.
\end{{tablenotes}}
\end{{table}}
"""
        return latex_table

    def generate_latex_leverage_table(self) -> str:
        """
        Generate LaTeX table for leverage parameters across cryptocurrencies.

        Returns:
            LaTeX formatted table string
        """
        rows = []

        for crypto, models in self.model_results.items():
            crypto_upper = crypto.upper()

            # Get TARCH leverage parameters
            tarch = models.get("TARCH(1,1)")
            tarchx = models.get("TARCH-X")

            if tarch and tarch.convergence:
                gamma = tarch.leverage_effect if tarch.leverage_effect else np.nan
                pval = tarch.pvalues.get("gamma[1]", np.nan)
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                rows.append(f"{crypto_upper} & {gamma:.6f}{sig} & ({pval:.4f}) \\\\")
            else:
                rows.append(f"{crypto_upper} & N/A & N/A \\\\")

        latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Leverage Effects (Asymmetric Volatility Response)}
\label{tab:leverage}
\begin{tabular}{lcc}
\hline
\hline
Cryptocurrency & $\gamma$ & p-value \\
\hline
"""
        latex_table += "\n".join(rows)
        latex_table += r"""
\hline
\hline
\end{tabular}
\begin{tablenotes}
\small
\item Note: $\gamma$ is the leverage parameter from TARCH(1,1) models.
\item Negative values indicate higher volatility response to negative shocks.
\item *** p<0.01, ** p<0.05, * p<0.10
\end{tablenotes}
\end{table}
"""
        return latex_table

    def plot_volatility_around_events(self, major_events: Optional[List[Tuple[str, str]]] = None):
        """
        Plot volatility around major events.

        Args:
            major_events: List of (date, event_name) tuples
        """
        if major_events is None:
            major_events = [
                ("2022-11-11", "FTX Bankruptcy"),
                ("2022-05-09", "Terra/Luna Collapse"),
                ("2024-01-10", "BTC ETF Approval"),
            ]

        fig, axes = plt.subplots(len(major_events), 2, figsize=(12, 4 * len(major_events)))
        if len(major_events) == 1:
            axes = axes.reshape(1, -1)

        for idx, (event_date, event_name) in enumerate(major_events):
            event_dt = pd.Timestamp(event_date)
            # Ensure timezone awareness
            if event_dt.tz is None:
                event_dt = pd.Timestamp(event_date, tz="UTC")

            # Plot BTC and ETH volatility
            for col_idx, crypto in enumerate(["btc", "eth"]):
                ax = axes[idx, col_idx]

                if crypto in self.model_results:
                    # Get volatility from best model
                    for model_name in ["TARCH-X", "TARCH(1,1)", "GARCH(1,1)"]:
                        if model_name in self.model_results[crypto]:
                            model = self.model_results[crypto][model_name]
                            if model.convergence and len(model.volatility) > 0:
                                vol = model.volatility

                                # Get window around event
                                window_start = event_dt - timedelta(days=30)
                                window_end = event_dt + timedelta(days=30)

                                # Filter to window
                                vol_window = vol[(vol.index >= window_start) & (vol.index <= window_end)]

                                if len(vol_window) > 0:
                                    # Annualize volatility
                                    vol_annual = vol_window * np.sqrt(252)

                                    ax.plot(vol_window.index, vol_annual, label=crypto.upper(), linewidth=1.5)
                                    ax.axvline(event_dt, color="red", linestyle="--", alpha=0.7, label="Event")
                                    ax.set_title(f"{crypto.upper()} - {event_name}")
                                    ax.set_xlabel("Date")
                                    ax.set_ylabel("Annualized Volatility (%)")
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                break

        plt.tight_layout()
        plt.savefig(self.output_dir / "volatility_major_events.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Volatility plot saved to {self.output_dir / 'volatility_major_events.png'}")

    def plot_diagnostic_charts(self):
        """
        Create GARCH diagnostic plots (ACF, Q-Q plots).
        """
        from scipy import stats
        from statsmodels.graphics.tsaplots import plot_acf

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        plot_idx = 0
        for crypto in ["btc", "eth", "xrp"]:
            if crypto not in self.model_results:
                continue

            # Get best model
            model = None
            for model_name in ["TARCH-X", "TARCH(1,1)", "GARCH(1,1)"]:
                if model_name in self.model_results[crypto]:
                    if self.model_results[crypto][model_name].convergence:
                        model = self.model_results[crypto][model_name]
                        break

            if model and len(model.residuals) > 0:
                # Standardized residuals
                std_resid = model.residuals / model.volatility

                # ACF of squared standardized residuals
                ax1 = axes[0, plot_idx]
                squared_resid = std_resid**2
                plot_acf(squared_resid.dropna(), lags=20, ax=ax1)
                ax1.set_title(f"{crypto.upper()} - ACF of Squared Residuals")
                ax1.set_xlabel("Lag")

                # Q-Q plot
                ax2 = axes[1, plot_idx]
                stats.probplot(std_resid.dropna(), dist="norm", plot=ax2)
                ax2.set_title(f"{crypto.upper()} - Q-Q Plot")

                plot_idx += 1

        plt.tight_layout()
        plt.savefig(self.output_dir / "diagnostic_plots.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Diagnostic plots saved to {self.output_dir / 'diagnostic_plots.png'}")

    def plot_event_impact_comparison(self):
        """
        Create bar chart comparing Infrastructure vs Regulatory impacts with confidence intervals.
        """
        if "inverse_variance_weighted" not in self.analysis_results:
            print("No inverse-variance weighted results available for plotting")
            return

        ivw = self.analysis_results["inverse_variance_weighted"]

        if "Infrastructure" not in ivw or "Regulatory" not in ivw:
            print("Incomplete results for plotting")
            return

        # Extract data
        categories = ["Infrastructure", "Regulatory"]
        means = [ivw["Infrastructure"]["weighted_average"], ivw["Regulatory"]["weighted_average"]]
        ci_lower = [ivw["Infrastructure"]["ci_lower"], ivw["Regulatory"]["ci_lower"]]
        ci_upper = [ivw["Infrastructure"]["ci_upper"], ivw["Regulatory"]["ci_upper"]]

        # Calculate error bars
        errors = [[means[i] - ci_lower[i] for i in range(2)], [ci_upper[i] - means[i] for i in range(2)]]

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        x = np.arange(len(categories))
        width = 0.5

        bars = ax.bar(
            x,
            means,
            width,
            yerr=errors,
            capsize=10,
            color=["#e74c3c", "#3498db"],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        # Customize plot
        ax.set_ylabel("Average Volatility Impact", fontsize=12)
        ax.set_title("Event Type Comparison: Inverse-Variance Weighted Averages", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{mean:.6f}", ha="center", va="bottom", fontsize=10)

        # Add significance marker if significant
        if "difference" in ivw and ivw["difference"]["significant_10pct"]:
            y_max = max(ci_upper) * 1.1
            ax.plot([0, 1], [y_max, y_max], "k-", linewidth=1)
            sig_text = (
                "***" if ivw["difference"]["p_value"] < 0.01 else "**" if ivw["difference"]["p_value"] < 0.05 else "*"
            )
            ax.text(0.5, y_max * 1.02, sig_text, ha="center", fontsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / "event_impact_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Event impact comparison saved to {self.output_dir / 'event_impact_comparison.png'}")

    def export_all_to_csv(self):
        """
        Export all results to CSV files for thesis appendix.
        """
        csv_dir = self.output_dir / "csv_exports"
        csv_dir.mkdir(exist_ok=True)

        # 1. Model parameters for each crypto
        for crypto, models in self.model_results.items():
            params_data = []
            for model_name, model in models.items():
                if model.convergence:
                    for param_name, param_value in model.parameters.items():
                        params_data.append(
                            {
                                "model": model_name,
                                "parameter": param_name,
                                "value": param_value,
                                "std_error": model.std_errors.get(param_name, np.nan),
                                "p_value": model.pvalues.get(param_name, np.nan),
                            }
                        )

            if params_data:
                params_df = pd.DataFrame(params_data)
                params_df.to_csv(csv_dir / f"{crypto}_parameters.csv", index=False)

        # 2. Model comparison statistics
        comparison_data = []
        for crypto, models in self.model_results.items():
            for model_name, model in models.items():
                if model.convergence:
                    comparison_data.append(
                        {
                            "crypto": crypto,
                            "model": model_name,
                            "AIC": model.aic,
                            "BIC": model.bic,
                            "log_likelihood": model.log_likelihood,
                        }
                    )

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(csv_dir / "model_comparison.csv", index=False)

        # 3. Event impact analysis results
        if "fdr_correction" in self.analysis_results:
            if isinstance(self.analysis_results["fdr_correction"], pd.DataFrame):
                self.analysis_results["fdr_correction"].to_csv(csv_dir / "event_impacts_fdr.csv", index=False)

        # 4. Hypothesis test results
        if "hypothesis_test" in self.analysis_results:
            hyp_test = self.analysis_results["hypothesis_test"]
            hyp_df = pd.DataFrame(
                {
                    "Infrastructure_mean": [hyp_test.get("infrastructure", {}).get("mean", np.nan)],
                    "Infrastructure_std": [hyp_test.get("infrastructure", {}).get("std", np.nan)],
                    "Infrastructure_n": [hyp_test.get("infrastructure", {}).get("n", 0)],
                    "Regulatory_mean": [hyp_test.get("regulatory", {}).get("mean", np.nan)],
                    "Regulatory_std": [hyp_test.get("regulatory", {}).get("std", np.nan)],
                    "Regulatory_n": [hyp_test.get("regulatory", {}).get("n", 0)],
                    "t_statistic": [hyp_test.get("t_test", {}).get("statistic", np.nan)],
                    "t_pvalue": [hyp_test.get("t_test", {}).get("p_value", np.nan)],
                    "mann_whitney_statistic": [hyp_test.get("mann_whitney", {}).get("statistic", np.nan)],
                    "mann_whitney_pvalue": [hyp_test.get("mann_whitney", {}).get("p_value", np.nan)],
                    "cohens_d": [hyp_test.get("effect_size", np.nan)],
                }
            )
            hyp_df.to_csv(csv_dir / "hypothesis_test.csv", index=False)

        print(f"All results exported to {csv_dir}")

    def generate_all_outputs(self):
        """
        Generate all publication outputs.
        """
        print("\n" + "=" * 60)
        print("GENERATING PUBLICATION OUTPUTS")
        print("=" * 60)

        # LaTeX tables
        print("\n1. Generating LaTeX tables...")
        latex_dir = self.output_dir / "latex"
        latex_dir.mkdir(exist_ok=True)

        # Model comparison table
        model_comp_latex = self.generate_latex_model_comparison_table()
        with open(latex_dir / "model_comparison.tex", "w") as f:
            f.write(model_comp_latex)
        print("   - Model comparison table saved")

        # Event comparison table
        event_comp_latex = self.generate_latex_event_comparison_table()
        with open(latex_dir / "event_comparison.tex", "w") as f:
            f.write(event_comp_latex)
        print("   - Event comparison table saved")

        # Leverage parameters table
        leverage_latex = self.generate_latex_leverage_table()
        with open(latex_dir / "leverage_parameters.tex", "w") as f:
            f.write(leverage_latex)
        print("   - Leverage parameters table saved")

        # Plots
        print("\n2. Generating plots...")
        self.plot_volatility_around_events()
        self.plot_diagnostic_charts()
        self.plot_event_impact_comparison()

        # CSV exports
        print("\n3. Exporting to CSV...")
        self.export_all_to_csv()

        print(f"\nAll outputs saved to {self.output_dir}")


def generate_publication_outputs(model_results: Dict, analysis_results: Dict, crypto_data: Dict) -> None:
    """
    Convenience function to generate all publication outputs.

    Args:
        model_results: GARCH model results
        analysis_results: Event impact analysis results
        crypto_data: Prepared cryptocurrency data
    """
    publisher = PublicationOutputs(model_results, analysis_results, crypto_data)
    publisher.generate_all_outputs()
