"""
Stage F: Prior Building for L00_minimal dataset

Build PyMC-compatible priors from empirical directional statistics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gimbal


def format_priors_for_json(prior_config: Dict) -> Dict:
    """Convert prior config to JSON-serializable format."""
    json_priors = {}
    for joint_name, params in prior_config.items():
        json_priors[joint_name] = {
            "mu_mean": (
                params["mu_mean"].tolist()
                if isinstance(params["mu_mean"], np.ndarray)
                else params["mu_mean"]
            ),
            "mu_sd": float(params["mu_sd"]),
            "kappa_mode": float(params["kappa_mode"]),
            "kappa_sd": float(params["kappa_sd"]),
        }
    return json_priors


def run_stage_f(
    stage_e_output_dir: Path, output_dir: Path, kappa_scale: float = 1.0
) -> Dict[str, Any]:
    """
    Run Stage F: Prior Building

    Parameters
    ----------
    stage_e_output_dir : Path
        Directory containing Stage E outputs (direction_stats.json)
    output_dir : Path
        Directory for outputs
    kappa_scale : float
        Scaling factor for kappa (higher = weaker priors). Default: 1.0

    Returns
    -------
    metrics : dict
        Prior configuration and metadata
    """
    print("=" * 80)
    print("STAGE F: Prior Building")
    print("=" * 80)

    # Load direction statistics from Stage E
    with open(stage_e_output_dir / "direction_stats.json") as f:
        stage_e_data = json.load(f)

    empirical_stats_raw = stage_e_data["statistics"]

    # Convert back to numpy arrays for processing
    empirical_stats = {}
    joint_names = list(empirical_stats_raw.keys())

    for joint_name, stats in empirical_stats_raw.items():
        empirical_stats[joint_name] = {
            "mu": (
                np.array(stats["mu"])
                if stats["mu"] is not None
                else np.array([np.nan, np.nan, np.nan])
            ),
            "kappa": stats["kappa"] if stats["kappa"] is not None else np.nan,
            "n_samples": stats["n_samples"],
        }

    print(f"Loaded statistics for {len(joint_names)} joints")
    print(f"Using kappa_scale = {kappa_scale}")

    # Build priors
    print("\n[1/2] Building prior configurations...")
    prior_config = gimbal.build_priors_from_statistics(
        empirical_stats, joint_names, kappa_min=0.1, kappa_scale=kappa_scale
    )

    print(f"  Built priors for {len(prior_config)} non-root joints")

    # Display priors
    print("\nPrior parameters per joint:")
    for joint_name, params in prior_config.items():
        mu_str = f"[{params['mu_mean'][0]:.3f}, {params['mu_mean'][1]:.3f}, {params['mu_mean'][2]:.3f}]"
        print(f"  {joint_name}:")
        print(f"    μ_mean = {mu_str}, μ_sd = {params['mu_sd']:.4f}")
        print(
            f"    κ_mode = {params['kappa_mode']:.2f}, κ_sd = {params['kappa_sd']:.2f}"
        )

    # Validate priors
    print("\n[2/2] Validating priors...")
    validation = {
        "all_non_root_have_priors": True,
        "reasonable_parameters": True,
        "issues": [],
    }

    # Check that all non-root joints have priors
    for joint_name in joint_names:
        if empirical_stats[joint_name]["n_samples"] > 0:  # Non-root
            if joint_name not in prior_config:
                validation["all_non_root_have_priors"] = False
                validation["issues"].append(f"{joint_name} missing from prior config")

    # Check parameter ranges
    for joint_name, params in prior_config.items():
        if params["mu_sd"] <= 0 or params["mu_sd"] > 10:
            validation["reasonable_parameters"] = False
            validation["issues"].append(
                f"{joint_name} has unusual μ_sd={params['mu_sd']:.4f}"
            )

        if params["kappa_mode"] <= 0 or params["kappa_mode"] > 1000:
            validation["reasonable_parameters"] = False
            validation["issues"].append(
                f"{joint_name} has unusual κ_mode={params['kappa_mode']:.2f}"
            )

    if validation["issues"]:
        print("  ⚠️  Validation issues:")
        for issue in validation["issues"]:
            print(f"    - {issue}")
    else:
        print("  ✓ All validations passed")

    # Compile full metrics
    full_metrics = {
        "stage": "F_prior_building",
        "config": {"kappa_scale": kappa_scale, "kappa_min": 0.1},
        "priors": format_priors_for_json(prior_config),
        "validation": validation,
        "summary": {
            "n_priors": len(prior_config),
            "mean_mu_sd": float(np.mean([p["mu_sd"] for p in prior_config.values()])),
            "mean_kappa_mode": float(
                np.mean([p["kappa_mode"] for p in prior_config.values()])
            ),
        },
    }

    # Save priors
    output_path = output_dir / "priors.json"
    with open(output_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\n✓ Stage F complete. Priors saved to {output_path}")

    return full_metrics


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    repo_root = Path(__file__).parent.parent.parent
    stage_e_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"
    output_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"

    # Use kappa_scale=1.0 as specified in Q1
    metrics = run_stage_f(stage_e_dir, output_dir, kappa_scale=1.0)

    passed = (
        metrics["validation"]["all_non_root_have_priors"]
        and metrics["validation"]["reasonable_parameters"]
    )

    if passed:
        print("\n" + "=" * 80)
        print("STAGE F: PASSED ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("STAGE F: WARNING - Validation issues detected")
        print("=" * 80)
