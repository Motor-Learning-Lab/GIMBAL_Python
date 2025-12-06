"""
Test Group 7: Bone Length Variance Sensitivity

Tests sensitivity to sigma2_sigma hyperparameter.
"""

import time
from pathlib import Path
from typing import List, Dict, Any

from tests.diagnostics.v0_2_1_divergence.test_utils import (
    get_standard_synth_data,
    build_test_model,
    sample_model,
    extract_metrics,
    calculate_reconstruction_error,
    format_test_result,
)


def run_bone_length_tests() -> List[Dict[str, Any]]:
    """
    Run bone length variance sensitivity tests.

    Returns
    -------
    list
        List of test result dictionaries
    """
    results = []

    # Test different sigma2_sigma values
    test_values = [0.1, 0.2, 0.5]

    for sigma2_sigma in test_values:
        print(
            f"\nTest 7.{test_values.index(sigma2_sigma)+1}: sigma2_sigma={sigma2_sigma}"
        )
        print("-" * 60)

        synth_data = get_standard_synth_data(T=100, C=3, S=3, seed=42)

        config = {
            "use_directional_hmm": True,
            "T": 100,
            "C": 3,
            "S": 3,
            "eta2_root_sigma": 0.5,
            "sigma2_sigma": sigma2_sigma,
            "kappa_mean": 10.0,
            "kappa_sigma": 2.0,
            "draws": 10,  # Reduced for debugging
            "tune": 10,
            "chains": 1,
            "seed": 42,
        }

        print(f"Building model: HMM ON, sigma2_sigma={sigma2_sigma}")
        model = build_test_model(
            synth_data,
            use_directional_hmm=True,
            S=3,
            eta2_root_sigma=config["eta2_root_sigma"],
            sigma2_sigma=sigma2_sigma,
        )

        print("Sampling...")
        start_time = time.time()
        trace = sample_model(
            model, draws=config["draws"], tune=config["tune"], chains=config["chains"]
        )
        runtime = time.time() - start_time

        print(f"Runtime: {runtime:.2f}s")

        # Extract metrics
        metrics = extract_metrics(trace, runtime)
        print(
            f"Divergences: {metrics['divergences']}/{metrics['total_samples']} ({metrics['divergence_rate']*100:.1f}%)"
        )

        # Reconstruction error
        reconstruction_error = calculate_reconstruction_error(
            trace, synth_data["joint_positions"]
        )
        print(f"Mean reconstruction error: {reconstruction_error['mean_error']:.4f}")

        # Format result
        result = format_test_result(
            f"Bone Length Variance - sigma2_sigma={sigma2_sigma}",
            config,
            metrics,
            reconstruction_error,
        )
        results.append(result)

    return results


if __name__ == "__main__":
    results = run_bone_length_tests()
    for result in results:
        print(f"\n{result['test_name']}")
        print(
            f"  Divergences: {result['metrics']['divergences']}/{result['metrics']['total_samples']}"
        )
        print(f"  Runtime: {result['metrics']['runtime_seconds']:.2f}s")
