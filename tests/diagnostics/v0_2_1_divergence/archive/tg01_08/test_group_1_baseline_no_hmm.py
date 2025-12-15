"""
Test Group 1: Baseline without HMM

Purpose:
    Determines whether divergences originate in the camera/kinematic model alone,
    independent of the HMM component.

Configuration:
    - use_directional_hmm = False
    - T = 100, C = 3, S = 3 (S not used without HMM)
    - kappa = 10.0, obs_noise_std = 0.5, occlusion_rate = 0.02
    - draws = 20, tune = 100, chains = 1
    - seed = 42

Expected outcomes:
    - High divergences → Base model has geometry issues
    - Low divergences → HMM is likely the problem

Reference: plans/v0.2.1_divergence_test_plan.md, Test Group 1
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

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
)


def test_group_1_baseline_no_hmm() -> Dict[str, Any]:
    """
    Run Test Group 1: Baseline model without HMM.

    Returns
    -------
    dict
        Complete test results including configuration, metrics, and metadata
    """
    print("\n" + "=" * 70)
    print("Test Group 1: Baseline without HMM")
    print("=" * 70)

    # Configuration from test plan
    config = {
        "test_group": 1,
        "description": "Baseline without HMM",
        "use_directional_hmm": False,
        "T": 100,
        "C": 3,
        "S": 3,  # Recorded but not used (no HMM)
        "kappa": 10.0,
        "obs_noise_std": 0.5,
        "occlusion_rate": 0.02,
        "draws": 20,
        "tune": 100,
        "chains": 1,
        "seed": 42,
        "eta2_root_sigma": 0.5,  # Default from existing code
        "sigma2_sigma": 0.2,  # Default from existing code
    }

    print(f"\nConfiguration:")
    print(f"  use_directional_hmm: {config['use_directional_hmm']}")
    print(f"  T={config['T']}, C={config['C']}, S={config['S']}")
    print(
        f"  draws={config['draws']}, tune={config['tune']}, chains={config['chains']}"
    )
    print(f"  seed={config['seed']}")

    # Generate synthetic data
    print(f"\nGenerating synthetic data...")
    synth_data = get_standard_synth_data(
        T=config["T"],
        C=config["C"],
        S=config["S"],
        seed=config["seed"],
    )
    print(f"  [OK] Generated {config['T']} timesteps, {config['C']} cameras")

    # Build model
    print(f"\nBuilding PyMC model (HMM OFF)...")
    model = build_test_model(
        synth_data,
        use_directional_hmm=False,
        S=config["S"],
        eta2_root_sigma=config["eta2_root_sigma"],
        sigma2_sigma=config["sigma2_sigma"],
    )
    print(f"  [OK] Model built successfully")

    # Sample
    print(f"\nSampling...")
    print(f"  Tuning: {config['tune']} steps")
    print(f"  Drawing: {config['draws']} samples")
    print()
    print("  NOTE: Initial gradient compilation takes ~45 seconds.")
    print("        This is normal. Please be patient...")
    print()

    start_time = time.time()
    trace = sample_model(
        model,
        draws=config["draws"],
        tune=config["tune"],
        chains=config["chains"],
    )
    runtime = time.time() - start_time

    print(f"  [OK] Sampling complete in {runtime:.2f}s")

    # Extract metrics
    print(f"\nExtracting metrics...")
    metrics = extract_metrics(trace, runtime)

    print(f"\nResults:")
    print(
        f"  Divergences: {metrics['divergences']}/{metrics['total_samples']} ({metrics['divergence_rate']*100:.1f}%)"
    )
    print(f"  Runtime: {metrics['runtime_seconds']:.2f}s")

    if "ess" in metrics:
        print(f"  ESS:")
        for param, value in metrics["ess"].items():
            print(f"    {param}: {value:.1f}")

    if "r_hat" in metrics:
        print(f"  R-hat:")
        for param, value in metrics["r_hat"].items():
            print(f"    {param}: {value:.3f}")

    # Reconstruction error
    print(f"\nCalculating reconstruction error...")
    reconstruction_error = calculate_reconstruction_error(
        trace, synth_data["joint_positions"]
    )
    print(f"  Mean reconstruction error: {reconstruction_error['mean_error']:.4f}")

    # Save diagnostic plots
    print(f"\nSaving diagnostic plots...")
    plots_dir = Path(__file__).parent / "plots" / "group_1_baseline_no_hmm"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_diagnostic_plots(trace, "group_1_baseline_no_hmm", plots_dir.parent)
    print(f"  [OK] Plots saved to {plots_dir}")

    # Assemble complete results
    results = {
        "test_group": 1,
        "description": "Baseline without HMM",
        "configuration": config,
        "metrics": {
            "divergence_count": metrics["divergences"],
            "total_samples": metrics["total_samples"],
            "divergence_rate": metrics["divergence_rate"],
            "runtime_seconds": metrics["runtime_seconds"],
            "ess": metrics.get("ess", {}),
            "r_hat": metrics.get("r_hat", {}),
        },
        "reconstruction": {
            "mean_error": reconstruction_error["mean_error"],
            "std_error": reconstruction_error.get("std_error", None),
        },
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python_version": sys.version.split()[0],
        },
    }

    # Save results to JSON
    results_file = Path(__file__).parent / "results_group_1_baseline_no_hmm.json"
    print(f"\nSaving results to {results_file}...")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] Results saved")

    print("\n" + "=" * 70)
    print("Test Group 1 Complete")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = test_group_1_baseline_no_hmm()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test: {results['description']}")
    print(
        f"Divergences: {results['metrics']['divergence_count']}/{results['metrics']['total_samples']} ({results['metrics']['divergence_rate']*100:.1f}%)"
    )
    print(f"Runtime: {results['metrics']['runtime_seconds']:.2f}s")
    print(f"Reconstruction RMSE: {results['reconstruction']['mean_error']:.4f}")
    print(f"\nResults saved to: results_group_1_baseline_no_hmm.json")
    print(f"Plots saved to: plots/group_1_baseline_no_hmm/")
    print("=" * 70)
