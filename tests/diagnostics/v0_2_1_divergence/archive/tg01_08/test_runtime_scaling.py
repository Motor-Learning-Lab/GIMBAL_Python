"""
Test Group 9: Runtime Scaling Tests

Measures how runtime scales with T (timesteps) and S (HMM states).
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
    format_test_result,
)


def run_runtime_scaling_tests() -> List[Dict[str, Any]]:
    """
    Run runtime scaling tests.

    Returns
    -------
    list
        List of test result dictionaries
    """
    results = []

    # Test 8.1: T scaling (with reduced draws for speed)
    print("\nTest 8.1: Runtime Scaling with T")
    print("-" * 60)

    T_values = [50, 100, 150]

    for T in T_values:
        print(f"\nT={T}")
        synth_data = get_standard_synth_data(T=T, C=3, S=3, seed=42)

        config = {
            "use_directional_hmm": True,
            "T": T,
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

        model = build_test_model(
            synth_data,
            use_directional_hmm=True,
            S=3,
            eta2_root_sigma=config["eta2_root_sigma"],
            sigma2_sigma=config["sigma2_sigma"],
        )

        start_time = time.time()
        trace = sample_model(
            model, draws=config["draws"], tune=config["tune"], chains=config["chains"]
        )
        runtime = time.time() - start_time

        print(f"Runtime: {runtime:.2f}s ({runtime/T:.3f}s per timestep)")

        metrics = extract_metrics(trace, runtime)
        print(
            f"Divergences: {metrics['divergences']}/{metrics['total_samples']} ({metrics['divergence_rate']*100:.1f}%)"
        )

        result = format_test_result(f"Runtime Scaling - T={T}", config, metrics)
        results.append(result)

    # Test 8.2: S scaling
    print("\nTest 8.2: Runtime Scaling with S")
    print("-" * 60)

    S_values = [1, 2, 3, 5]

    for S in S_values:
        print(f"\nS={S}")
        # For S=1, generate with S=3 but use S=1 in model
        data_S = 3 if S == 1 else S
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

        model = build_test_model(
            synth_data,
            use_directional_hmm=True,
            S=S,
            eta2_root_sigma=config["eta2_root_sigma"],
            sigma2_sigma=config["sigma2_sigma"],
        )

        start_time = time.time()
        trace = sample_model(
            model, draws=config["draws"], tune=config["tune"], chains=config["chains"]
        )
        runtime = time.time() - start_time

        print(f"Runtime: {runtime:.2f}s")

        metrics = extract_metrics(trace, runtime)
        print(
            f"Divergences: {metrics['divergences']}/{metrics['total_samples']} ({metrics['divergence_rate']*100:.1f}%)"
        )

        result = format_test_result(f"Runtime Scaling - S={S}", config, metrics)
        results.append(result)

    return results


if __name__ == "__main__":
    results = run_runtime_scaling_tests()
    for result in results:
        print(f"\n{result['test_name']}")
        print(
            f"  Divergences: {result['metrics']['divergences']}/{result['metrics']['total_samples']}"
        )
        print(f"  Runtime: {result['metrics']['runtime_seconds']:.2f}s")
