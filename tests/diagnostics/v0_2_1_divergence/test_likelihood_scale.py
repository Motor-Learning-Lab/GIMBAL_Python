"""
Test Group 4: Likelihood Scale Comparison

Analyzes likelihood values to detect scale mismatches between priors and likelihood.
"""

import numpy as np
from typing import List, Dict, Any


def run_likelihood_scale_tests(
    all_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Analyze likelihood scales from previous test results.

    Parameters
    ----------
    all_results : list
        All test results from previous test groups

    Returns
    -------
    list
        List of analysis result dictionaries
    """
    results = []

    print("\nTest 4.1: Likelihood Scale Analysis")
    print("-" * 60)

    # Analyze log likelihood values
    for result in all_results:
        test_name = result["test_name"]
        log_likelihood = result["metrics"].get("log_likelihood")

        if log_likelihood is not None:
            # Calculate statistics
            try:
                # Get all log likelihood values
                ll_values = []
                for var_name in log_likelihood.data_vars:
                    ll_data = log_likelihood[var_name].values
                    ll_values.extend(ll_data.flatten())

                ll_values = np.array(ll_values)

                print(f"\n{test_name}:")
                print(f"  Log likelihood mean: {ll_values.mean():.2f}")
                print(f"  Log likelihood std: {ll_values.std():.2f}")
                print(
                    f"  Log likelihood range: [{ll_values.min():.2f}, {ll_values.max():.2f}]"
                )
            except Exception as e:
                print(f"\n{test_name}: Could not analyze log likelihood - {e}")

    # Create summary result
    analysis_result = {
        "test_name": "Likelihood Scale Analysis",
        "config": {"analysis_type": "likelihood_scale"},
        "metrics": {"note": "See individual test results for log likelihood values"},
    }
    results.append(analysis_result)

    return results


if __name__ == "__main__":
    # This test requires results from other tests
    print("This test should be run as part of the full test suite")
