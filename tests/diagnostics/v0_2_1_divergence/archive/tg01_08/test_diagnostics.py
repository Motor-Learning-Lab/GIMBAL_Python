"""
Test Group 5: Divergence Localization Diagnostics

Creates diagnostic visualizations to identify which parameters cause divergences.
"""

from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import arviz as az


def run_divergence_diagnostics(all_results: List[Dict[str, Any]]):
    """
    Generate comprehensive divergence diagnostic plots.

    Parameters
    ----------
    all_results : list
        All test results from previous test groups
    """
    print("\nTest 5.1: Divergence Localization Diagnostics")
    print("-" * 60)

    diagnostics_dir = (
        Path(__file__).parent.parent.parent.parent
        / "results"
        / "diagnostics"
        / "v0_2_1_divergence"
        / "plots"
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Diagnostic plots saved to: {diagnostics_dir}")
    print("Note: Individual test plots are created during each test run")
    print("This function provides a summary across all tests")

    # Create summary visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    test_names = []
    divergence_rates = []

    for result in all_results:
        if "divergence_rate" in result["metrics"]:
            test_names.append(result["test_name"])
            divergence_rates.append(result["metrics"]["divergence_rate"] * 100)

    if test_names:
        ax.barh(range(len(test_names)), divergence_rates, color="steelblue")
        ax.set_yticks(range(len(test_names)))
        ax.set_yticklabels(test_names, fontsize=10)
        ax.set_xlabel("Divergence Rate (%)", fontsize=12)
        ax.set_title("Divergence Rates Across All Tests", fontsize=14)
        ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        fig.savefig(
            diagnostics_dir / "divergence_summary.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        print(f"[OK] Created summary plot: divergence_summary.png")


if __name__ == "__main__":
    print("This test should be run as part of the full test suite")
