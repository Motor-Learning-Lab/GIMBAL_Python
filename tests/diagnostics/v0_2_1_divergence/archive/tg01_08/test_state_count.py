"""
Test Group 3: State Count Comparison (S=1,2,3)

Tests the model with different numbers of HMM states.
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


def run_state_count_tests() -> List[Dict[str, Any]]:
    """
    Run state count comparison tests.

    Returns
    -------
    list
        List of test result dictionaries
    """
    results = []

    for S in [1, 2, 3]:
        # For S=1, generate with S=3 but use S=1 in model
        data_S = 3 if S == 1 else S
        print(f"\nTest 3.{S}: HMM ON - S={S}")
        print("-" * 60)

        synth_data = get_standard_synth_data(T=100, C=3, S=data_S, seed=42)

        config = {
            "use_directional_hmm": True,
            "T": 100,
            "C": 3,
            "S": S,
            "eta2_root_sigma": 0.5,
            "sigma2_sigma": 0.2,
            "kappa_mean": 10.0,
            "kappa_sigma": 2.0,
            "draws": 10,  # Reduced for debugging
            "tune": 10,
            "chains": 1,
            "seed": 42,
        }

        print(f"Building model: HMM ON, T={config['T']}, C={config['C']}, S={S}")
        model = build_test_model(
            synth_data,
            use_directional_hmm=True,
            S=S,
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
        save_diagnostic_plots(trace, f"hmm_on_s{S}", diagnostics_dir)

        # Format result
        result = format_test_result(
            f"HMM ON - S={S}", config, metrics, reconstruction_error
        )
        results.append(result)

    return results


if __name__ == "__main__":
    results = run_state_count_tests()
    for result in results:
        print(f"\n{result['test_name']}")
        print(
            f"  Divergences: {result['metrics']['divergences']}/{result['metrics']['total_samples']}"
        )
        print(f"  Runtime: {result['metrics']['runtime_seconds']:.2f}s")
