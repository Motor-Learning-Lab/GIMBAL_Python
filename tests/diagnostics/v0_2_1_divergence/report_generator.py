"""
Report generator for v0.2.1 divergence test suite.
"""

from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def generate_report(all_results: List[Dict[str, Any]]) -> Path:
    """
    Generate a comprehensive markdown report from test results.

    Parameters
    ----------
    all_results : list
        List of all test results

    Returns
    -------
    Path
        Path to the generated report
    """
    report_path = (
        Path(__file__).parent.parent.parent.parent
        / "results"
        / "diagnostics"
        / "v0_2_1_divergence"
        / "report.md"
    )

    with open(report_path, "w") as f:
        # Header
        f.write("# GIMBAL v0.2.1 Divergence Test Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        # Count tests by group
        test_groups = {}
        total_divergences = 0
        total_samples = 0

        for result in all_results:
            test_name = result["test_name"]
            group = test_name.split(" - ")[0] if " - " in test_name else test_name

            if group not in test_groups:
                test_groups[group] = []
            test_groups[group].append(result)

            if "divergences" in result["metrics"]:
                total_divergences += result["metrics"]["divergences"]
                total_samples += result["metrics"]["total_samples"]

        f.write(f"- **Total tests run:** {len(all_results)}\n")
        f.write(f"- **Test groups:** {len(test_groups)}\n")
        f.write(f"- **Overall divergences:** {total_divergences}/{total_samples} ")
        if total_samples > 0:
            f.write(f"({total_divergences/total_samples*100:.1f}%)\n")
        else:
            f.write("(N/A)\n")
        f.write("\n")

        # Key Findings
        f.write("### Key Findings\n\n")

        # Find tests with highest/lowest divergence rates
        tests_with_divs = [r for r in all_results if "divergence_rate" in r["metrics"]]
        if tests_with_divs:
            best_test = min(
                tests_with_divs, key=lambda r: r["metrics"]["divergence_rate"]
            )
            worst_test = max(
                tests_with_divs, key=lambda r: r["metrics"]["divergence_rate"]
            )

            f.write(f"- **Lowest divergence rate:** {best_test['test_name']} ")
            f.write(f"({best_test['metrics']['divergence_rate']*100:.1f}%)\n")
            f.write(f"- **Highest divergence rate:** {worst_test['test_name']} ")
            f.write(f"({worst_test['metrics']['divergence_rate']*100:.1f}%)\n")

        f.write("\n---\n\n")

        # Detailed Results by Test Group
        f.write("## Detailed Results\n\n")

        for group_name, group_results in test_groups.items():
            f.write(f"### {group_name}\n\n")

            for result in group_results:
                f.write(f"#### {result['test_name']}\n\n")

                # Configuration
                f.write("**Configuration:**\n\n")
                f.write("```python\n")
                for key, value in result["config"].items():
                    f.write(f"{key}: {value}\n")
                f.write("```\n\n")

                # Metrics
                f.write("**Results:**\n\n")
                metrics = result["metrics"]

                if "divergences" in metrics:
                    f.write(
                        f"- **Divergences:** {metrics['divergences']}/{metrics['total_samples']} "
                    )
                    f.write(f"({metrics['divergence_rate']*100:.1f}%)\n")

                if "runtime_seconds" in metrics:
                    f.write(f"- **Runtime:** {metrics['runtime_seconds']:.2f}s\n")

                if "ess_bulk" in metrics:
                    f.write(f"- **ESS (bulk):** ")
                    ess_values = [
                        f"{k}={v:.1f}" for k, v in list(metrics["ess_bulk"].items())[:3]
                    ]
                    f.write(", ".join(ess_values))
                    if len(metrics["ess_bulk"]) > 3:
                        f.write(f", ... ({len(metrics['ess_bulk'])} total)")
                    f.write("\n")

                if "rhat_max" in metrics:
                    f.write(f"- **R-hat (max):** ")
                    rhat_values = [
                        f"{k}={v:.3f}" for k, v in list(metrics["rhat_max"].items())[:3]
                    ]
                    f.write(", ".join(rhat_values))
                    if len(metrics["rhat_max"]) > 3:
                        f.write(f", ... ({len(metrics['rhat_max'])} total)")
                    f.write("\n")

                # Reconstruction error
                if "reconstruction_error" in result:
                    rec_err = result["reconstruction_error"]
                    f.write(
                        f"- **Reconstruction error (mean):** {rec_err['mean_error']:.4f}\n"
                    )
                    f.write(
                        f"- **Reconstruction error (median):** {rec_err['median_error']:.4f}\n"
                    )
                    f.write(
                        f"- **Reconstruction error (max):** {rec_err['max_error']:.4f}\n"
                    )

                f.write("\n")

        # Diagnostic Plots
        f.write("---\n\n")
        f.write("## Diagnostic Plots\n\n")
        f.write(
            "Diagnostic plots are saved in `results/diagnostics/v0_2_1_divergence/plots/`:\n\n"
        )
        f.write(
            "- `divergence_summary.png` - Overview of divergence rates across all tests\n"
        )
        f.write("- `*_parallel.png` - Parallel coordinate plots for individual tests\n")
        f.write("- `*_pair.png` - Pair plots with divergences highlighted\n")
        f.write("\n")

        # Interpretation Guide
        f.write("---\n\n")
        f.write("## Interpretation Guide\n\n")

        f.write("### Divergence Rates\n\n")
        f.write(
            "- **< 1%:** Excellent - sampler is exploring the posterior efficiently\n"
        )
        f.write("- **1-10%:** Good - minor sampling issues but generally acceptable\n")
        f.write(
            "- **10-50%:** Problematic - significant geometry issues in posterior\n"
        )
        f.write("- **> 50%:** Critical - posterior geometry is pathological\n\n")

        f.write("### Common Causes\n\n")
        f.write(
            "- **Prior-likelihood mismatch:** Priors are on different scale than likelihood\n"
        )
        f.write("- **Funnel geometry:** Hierarchical models with varying scales\n")
        f.write("- **Stiff ODEs:** HMM dynamics creating numerical instability\n")
        f.write(
            "- **Non-identifiability:** Multiple parameter configurations explain data equally well\n\n"
        )

        f.write("### Recommended Actions\n\n")
        f.write("Based on the test results:\n\n")

        # Analyze results and provide recommendations
        hmm_off_tests = [
            r for r in all_results if not r["config"].get("use_directional_hmm", True)
        ]
        hmm_on_tests = [
            r for r in all_results if r["config"].get("use_directional_hmm", False)
        ]

        if hmm_off_tests and hmm_on_tests:
            avg_hmm_off = sum(
                r["metrics"]["divergence_rate"]
                for r in hmm_off_tests
                if "divergence_rate" in r["metrics"]
            ) / len(hmm_off_tests)
            avg_hmm_on = sum(
                r["metrics"]["divergence_rate"]
                for r in hmm_on_tests
                if "divergence_rate" in r["metrics"]
            ) / len(hmm_on_tests)

            if avg_hmm_on > avg_hmm_off * 2:
                f.write(
                    "1. **HMM Prior Issues:** The HMM prior significantly increases divergences. "
                )
                f.write("Consider:\n")
                f.write("   - Reparameterization of directional distributions\n")
                f.write("   - Adjusting kappa priors\n")
                f.write("   - Simplifying to fewer states\n\n")
            elif avg_hmm_on < avg_hmm_off * 0.5:
                f.write("1. **HMM Prior Helps:** The HMM prior reduces divergences. ")
                f.write("The base model may need stronger regularization.\n\n")
            else:
                f.write(
                    "1. **HMM Effect Neutral:** Divergences are similar with/without HMM. "
                )
                f.write("Issues likely in base model.\n\n")

        # Check state count effects
        state_tests = [r for r in all_results if "S=" in r["test_name"]]
        if len(state_tests) >= 2:
            f.write(
                "2. **State Count:** Review the S=1,2,3 results to determine optimal state count. "
            )
            f.write(
                "More states increase complexity but may better capture dynamics.\n\n"
            )

        f.write("---\n\n")
        f.write("## Next Steps\n\n")
        f.write(
            "1. Review diagnostic plots in `results/diagnostics/v0_2_1_divergence/plots/`\n"
        )
        f.write(
            "2. Identify parameters with highest divergence correlation (parallel plots)\n"
        )
        f.write("3. Implement targeted fixes based on test results\n")
        f.write("4. Re-run focused tests to verify improvements\n")
        f.write("\n")

    return report_path


if __name__ == "__main__":
    print("This module should be imported and used by test_runner.py")
