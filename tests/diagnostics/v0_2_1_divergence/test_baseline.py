"""
Test Group 1: Baseline Model Tests (HMM OFF)

Tests the base PyMC model without directional HMM prior to establish a baseline.
"""

import time
from pathlib import Path
from typing import List, Dict, Any

from tests.diagnostics.v0_2_1_divergence.test_utils import (
    get_standard_synth_data,
    build_test_model,
    sample_model,
    extract_metrics,
    save_diagnostic_plots,
    calculate_reconstruction_error,
    format_test_result,
)


def run_baseline_tests() -> List[Dict[str, Any]]:
    """
    Run baseline tests with HMM OFF.

    Returns
    -------
    list
        List of test result dictionaries
    """
    results = []

    # Test 1: Standard configuration
    print("\nTest 1.1: Baseline (HMM OFF) - Standard Config")
    print("-" * 60)

    synth_data = get_standard_synth_data(T=100, C=3, S=3, seed=42)

    config = {
        "use_directional_hmm": False,
        "T": 100,
        "C": 3,
        "S": 3,  # Not used, but recorded
        "eta2_root_sigma": 0.5,
        "sigma2_sigma": 0.2,
        "kappa_mean": 10.0,
        "kappa_sigma": 2.0,
        "draws": 10,  # Reduced for debugging
        "tune": 10,
        "chains": 1,
        "seed": 42,
    }

    print(f"Building model: HMM OFF, T={config['T']}, C={config['C']}")
    model = build_test_model(
        synth_data,
        use_directional_hmm=False,
        S=3,
        eta2_root_sigma=config["eta2_root_sigma"],
        sigma2_sigma=config["sigma2_sigma"],
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

    # Save diagnostics
    diagnostics_dir = (
        Path(__file__).parent.parent.parent.parent
        / "results"
        / "diagnostics"
        / "v0_2_1_divergence"
        / "plots"
    )
    save_diagnostic_plots(trace, "baseline_standard", diagnostics_dir)

    # Format result
    result = format_test_result(
        "Baseline (HMM OFF) - Standard Config", config, metrics, reconstruction_error
    )
    results.append(result)

    return results


if __name__ == "__main__":
    results = run_baseline_tests()
    for result in results:
        print(f"\n{result['test_name']}")
        print(
            f"  Divergences: {result['metrics']['divergences']}/{result['metrics']['total_samples']}"
        )
        print(f"  Runtime: {result['metrics']['runtime_seconds']:.2f}s")
