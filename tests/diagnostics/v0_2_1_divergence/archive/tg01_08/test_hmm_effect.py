"""
Test Group 2: HMM Effect Tests

Tests model with directional HMM prior enabled (S=3 states).
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add repo root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from test_utils import (
    get_standard_synth_data,
    build_test_model,
    sample_model,
    extract_metrics,
    save_diagnostic_plots,
    calculate_reconstruction_error,
    format_test_result,
)


def run_hmm_effect_tests() -> List[Dict[str, Any]]:
    """
    Run HMM effect tests with HMM ON.

    Returns
    -------
    list
        List of test result dictionaries
    """
    results = []

    # Test 2.1: HMM ON with S=3
    print("\nTest 2.1: HMM ON - S=3")
    print("-" * 60)

    synth_data = get_standard_synth_data(T=100, C=3, S=3, seed=42)

    config = {
        "use_directional_hmm": True,
        "T": 100,
        "C": 3,
        "S": 3,
        "eta2_root_sigma": 0.5,
        "sigma2_sigma": 0.2,
        "kappa_mean": 10.0,
        "kappa_sigma": 2.0,
        "draws": 10,  # Reduced for debugging
        "tune": 10,
        "chains": 1,
        "seed": 42,
    }

    print(f"Building model: HMM ON, T={config['T']}, C={config['C']}, S={config['S']}")
    model = build_test_model(
        synth_data,
        use_directional_hmm=True,
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
    save_diagnostic_plots(trace, "hmm_effect_standard", diagnostics_dir)

    # Format result
    result = format_test_result("HMM ON - S=3", config, metrics, reconstruction_error)
    results.append(result)

    return results


if __name__ == "__main__":
    results = run_hmm_effect_tests()
    for result in results:
        print(f"\n{result['test_name']}")
        print(
            f"  Divergences: {result['metrics']['divergences']}/{result['metrics']['total_samples']}"
        )
        print(f"  Runtime: {result['metrics']['runtime_seconds']:.2f}s")
