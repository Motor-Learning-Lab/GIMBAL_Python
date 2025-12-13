"""
Test Group 11.1: Parameter Redundancy Diagnostic (Fixed Directions/Lengths)

Purpose:
    Isolate whether redundant degrees of freedom (root motion vs directions vs
    bone lengths) significantly worsen posterior geometry by hard-fixing
    directions and lengths to ground truth.

Configuration:
    - Baseline: Full model (root RW + free directions + free lengths)
    - Variant: Fixed directions and lengths to ground truth (only root RW free)
    - T = 100, C = 3, S = 3
    - draws = 200, tune = 200, chains = 1
    - seed = 42

Expected outcomes:
    - Substantial divergence reduction when directions/lengths fixed → redundancy
      between root, directions, and lengths is a major contributor
    - Little change → redundancy is not dominant; focus on Issues #1 and #2

Reference: plans/v0.2.1_divergence_plan_2.md, Issue #3, Test Group 11.1

Implementation Checklist:
* [x] File path: tests/diagnostics/v0_2_1_divergence/test_group_11_redundancy_fixed.py
* [x] Results file: tests/diagnostics/v0_2_1_divergence/results_group_11_redundancy_fixed.json
* [x] Report file: tests/diagnostics/v0_2_1_divergence/report_group_11_redundancy_fixed.md
* [x] Uses test_utils.get_standard_synth_data(T=100, C=3, S=3, seed=42)
* [x] Uses DLT initialization via gimbal.fit_params.initialize_from_observations_dlt
* [x] Sampler via test_utils.sample_model(model, draws=200, tune=200, chains=1)
* [x] Baseline uses test_utils.build_test_model(..., use_directional_hmm=False)
* [x] Variant fixes u_all and lengths using GT-derived values
* [x] Computes u_true from ground-truth joint positions and parents
* [x] Uses lengths_true = synth_data["bone_lengths"]
* [x] No changes to gimbal/pymc_model.py
* [x] Writes JSON metrics and markdown report; does not claim issue is "solved"
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Add repo root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from test_utils import (
    get_standard_synth_data,
    build_test_model,
    sample_model,
    extract_metrics,
    save_diagnostic_plots,
)
import gimbal


def compute_ground_truth_directions(
    x_true: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """
    Compute ground-truth directional vectors from joint positions.

    Parameters
    ----------
    x_true : ndarray, shape (T, K, 3)
        Ground-truth 3D joint positions
    parents : ndarray, shape (K,)
        Parent indices for each joint (root has parent -1)

    Returns
    -------
    u_true : ndarray, shape (T, K, 3)
        Ground-truth directional unit vectors
    """
    T, K, _ = x_true.shape
    u_true = np.zeros_like(x_true)

    for k in range(K):
        p = int(parents[k])
        if p < 0:
            # Root joint - no parent, leave as zeros
            continue

        # Vector from parent to child
        v = x_true[:, k, :] - x_true[:, p, :]

        # Normalize to unit vector
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        u_true[:, k, :] = v / (norms + 1e-12)

    return u_true


def build_test_model_fixed_redundancy(
    synth_data: Dict[str, Any],
    eta2_root_sigma: float = 0.5,
) -> pm.Model:
    """
    Build PyMC model with fixed directions and lengths (only root RW free).

    This variant removes redundant degrees of freedom by fixing directional
    vectors and bone lengths to ground truth, keeping only the root random walk
    and camera likelihood.

    Parameters
    ----------
    synth_data : dict
        Synthetic data dictionary
    eta2_root_sigma : float
        Root variance hyperparameter

    Returns
    -------
    pm.Model
        PyMC model with fixed directions and lengths
    """
    # Initialize from observations using DLT
    init_result = gimbal.fit_params.initialize_from_observations_dlt(
        y_observed=synth_data["observations_uv"],
        camera_proj=synth_data["camera_matrices"],
        parents=synth_data["parents"],
    )

    # Extract dimensions and ground truth
    x_true = synth_data["joint_positions"]  # (T, K, 3)
    T, K, _ = x_true.shape
    C = synth_data["camera_matrices"].shape[0]

    # Compute ground-truth directions
    u_true = compute_ground_truth_directions(x_true, synth_data["parents"])

    # Ground-truth bone lengths
    lengths_true = synth_data["bone_lengths"]  # (K,) static reference lengths

    # Build model with fixed directions and lengths
    with pm.Model() as model:
        # =====================================================================
        # Root dynamics (same as baseline)
        # =====================================================================
        eta2_root = pm.HalfNormal(
            "eta2_root",
            sigma=eta2_root_sigma,
            initval=init_result.eta2[0],
        )

        x_root = pm.GaussianRandomWalk(
            "x_root",
            mu=0.0,
            sigma=pt.sqrt(eta2_root),
            shape=(T, 3),
            initval=init_result.x_init[:, 0, :],
        )

        # =====================================================================
        # FIXED DIRECTIONS AND LENGTHS (replaces stochastic priors)
        # =====================================================================
        u_all = pm.Data("u_all", u_true)  # (T, K, 3) fixed directions
        lengths = pm.Data("lengths", lengths_true)  # (K,) fixed bone lengths

        # =====================================================================
        # Joint positions (deterministic from fixed components)
        # =====================================================================
        x_all_list = [x_root]

        for k in range(1, K):
            parent = int(synth_data["parents"][k])

            # Extract fixed direction and length for this joint
            u_k = u_all[:, k, :]  # (T, 3)
            length_k_scalar = lengths[k]  # scalar

            # Joint position: parent + direction * length
            x_parent = x_all_list[parent]
            x_k = pm.Deterministic(
                f"x_{k}",
                x_parent + u_k * length_k_scalar,
            )
            x_all_list.append(x_k)

        # Stack all joint positions: (T, K, 3)
        x_all = pm.Deterministic("x_all", pt.stack(x_all_list, axis=1))

        # =====================================================================
        # Camera projection (same as baseline)
        # =====================================================================
        proj_param = synth_data["camera_matrices"]  # (C, 3, 4)

        # Apply projection for each camera
        y_pred_list = []
        for c in range(C):
            A_c = proj_param[c, :, :3]  # (3, 3)
            b_c = proj_param[c, :, 3]  # (3,)

            # Project: (T, K, 3) @ (3, 3)^T + (3,) -> (T, K, 3)
            uvw = pt.dot(x_all, A_c.T) + b_c[None, None, :]

            # Perspective division
            u = uvw[:, :, 0] / uvw[:, :, 2]
            v = uvw[:, :, 1] / uvw[:, :, 2]
            y_c = pt.stack([u, v], axis=-1)  # (T, K, 2)
            y_pred_list.append(y_c)

        y_pred = pm.Deterministic(
            "y_pred", pt.stack(y_pred_list, axis=0)
        )  # (C, T, K, 2)

        # =====================================================================
        # Observation likelihood (same as baseline)
        # =====================================================================
        obs_sigma = pm.HalfNormal(
            "obs_sigma",
            sigma=10.0,
            initval=init_result.obs_sigma,
        )

        pm.Normal(
            "y_obs",
            mu=y_pred,
            sigma=obs_sigma,
            observed=synth_data["observations_uv"],
        )

    return model


def run_variant_baseline(
    synth_data: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run baseline variant with free directions and lengths."""
    print("\n" + "-" * 70)
    print("Baseline Variant: Free Directions and Lengths")
    print("-" * 70)

    # Build model
    print("Building model...")
    model = build_test_model(
        synth_data,
        use_directional_hmm=False,
        S=config["S"],
        eta2_root_sigma=config["eta2_root_sigma"],
        sigma2_sigma=config["sigma2_sigma"],
    )
    print(f"  [OK] Model has {len(model.free_RVs)} free RVs")

    # Sample
    print(f"Sampling (draws={config['draws']}, tune={config['tune']})...")
    start_time = time.time()
    trace = sample_model(
        model,
        draws=config["draws"],
        tune=config["tune"],
        chains=config["chains"],
    )
    runtime = time.time() - start_time
    print(f"  [OK] Sampling completed in {runtime:.1f}s")

    # Extract metrics
    metrics = extract_metrics(trace, runtime)
    print(
        f"  Divergences: {metrics['divergences']}/{metrics['total_samples']} ({metrics['divergence_rate']:.2%})"
    )

    # Save diagnostic plots
    plots_dir = Path(__file__).parent / "plots" / "group_11"
    save_diagnostic_plots(
        trace,
        "group_11_baseline_free",
        plots_dir,
        plot_parallel=False,
        plot_pair=False,
    )

    return {
        "variant": "baseline_free_params",
        "metrics": metrics,
        "trace": trace,
    }


def run_variant_fixed(
    synth_data: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run variant with fixed directions and lengths."""
    print("\n" + "-" * 70)
    print("Diagnostic Variant: Fixed Directions and Lengths (GT)")
    print("-" * 70)

    # Build model with fixed directions/lengths
    print("Building model with fixed directions and lengths...")
    model = build_test_model_fixed_redundancy(
        synth_data,
        eta2_root_sigma=config["eta2_root_sigma"],
    )
    print(f"  [OK] Model has {len(model.free_RVs)} free RVs")

    # Sample
    print(f"Sampling (draws={config['draws']}, tune={config['tune']})...")
    start_time = time.time()
    trace = sample_model(
        model,
        draws=config["draws"],
        tune=config["tune"],
        chains=config["chains"],
    )
    runtime = time.time() - start_time
    print(f"  [OK] Sampling completed in {runtime:.1f}s")

    # Extract metrics
    metrics = extract_metrics(trace, runtime)
    print(
        f"  Divergences: {metrics['divergences']}/{metrics['total_samples']} ({metrics['divergence_rate']:.2%})"
    )

    # Save diagnostic plots
    plots_dir = Path(__file__).parent / "plots" / "group_11"
    save_diagnostic_plots(
        trace,
        "group_11_fixed_redundancy",
        plots_dir,
        plot_parallel=False,
        plot_pair=False,
    )

    return {
        "variant": "fixed_directions_lengths",
        "metrics": metrics,
        "trace": trace,
    }


def run_group_11_redundancy_fixed() -> Dict[str, Any]:
    """
    Run Test Group 11.1: Parameter redundancy diagnostic (fixed).

    Returns
    -------
    dict
        Complete test results including both variants
    """
    print("\n" + "=" * 70)
    print("Test Group 11.1: Parameter Redundancy Diagnostic (Fixed)")
    print("=" * 70)

    # Configuration (from plan specification)
    config = {
        "test_group": 11.1,
        "description": "Parameter redundancy (baseline vs fixed directions/lengths)",
        "T": 100,
        "C": 3,
        "S": 3,
        "draws": 200,
        "tune": 200,
        "chains": 1,
        "seed": 42,
        "eta2_root_sigma": 0.5,
        "sigma2_sigma": 0.2,
    }

    print(f"\nConfiguration:")
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

    # Run both variants
    results_baseline = run_variant_baseline(synth_data, config)
    results_fixed = run_variant_fixed(synth_data, config)

    # Compile results
    results = {
        "test_group": config["test_group"],
        "description": config["description"],
        "timestamp": datetime.now().isoformat(),
        "configuration": config,
        "variants": {
            "baseline": {
                "description": "Full model (root RW + free directions + free lengths)",
                "metrics": results_baseline["metrics"],
            },
            "fixed": {
                "description": "Fixed directions and lengths to ground truth",
                "metrics": results_fixed["metrics"],
            },
        },
        "comparison": {
            "divergence_reduction_factor": (
                results_baseline["metrics"]["divergences"]
                / max(results_fixed["metrics"]["divergences"], 1)
            ),
            "divergence_rate_baseline": results_baseline["metrics"]["divergence_rate"],
            "divergence_rate_fixed": results_fixed["metrics"]["divergence_rate"],
        },
    }

    # Save results
    results_path = Path(__file__).parent / "results_group_11_redundancy_fixed.json"
    with open(results_path, "w") as f:
        results_serializable = {k: v for k, v in results.items()}
        json.dump(results_serializable, f, indent=2)
    print(f"\n[OK] Results saved to: {results_path}")

    # Generate markdown report
    generate_report(results)

    return results


def generate_report(results: Dict[str, Any]):
    """Generate markdown report for Test Group 11.1."""
    report_path = Path(__file__).parent / "report_group_11_redundancy_fixed.md"

    baseline_metrics = results["variants"]["baseline"]["metrics"]
    fixed_metrics = results["variants"]["fixed"]["metrics"]
    comparison = results["comparison"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Test Group 11.1: Parameter Redundancy Diagnostic (Fixed)\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")

        f.write("## Purpose\n\n")
        f.write(
            "Isolate whether redundant degrees of freedom (root motion vs directions vs bone lengths) significantly worsen posterior geometry.\n\n"
        )

        f.write("## Configuration\n\n")
        f.write(f"- T = {results['configuration']['T']}\n")
        f.write(f"- C = {results['configuration']['C']}\n")
        f.write(
            f"- draws = {results['configuration']['draws']}, tune = {results['configuration']['tune']}\n"
        )
        f.write(f"- seed = {results['configuration']['seed']}\n\n")

        f.write("## Results\n\n")
        f.write("### Baseline: Free Directions and Lengths\n\n")
        f.write(
            f"- Divergences: {baseline_metrics['divergences']}/{baseline_metrics['total_samples']} ({baseline_metrics['divergence_rate']:.2%})\n"
        )
        f.write(f"- Runtime: {baseline_metrics['runtime_seconds']:.1f}s\n\n")

        f.write("### Variant: Fixed Directions and Lengths (GT)\n\n")
        f.write(
            f"- Divergences: {fixed_metrics['divergences']}/{fixed_metrics['total_samples']} ({fixed_metrics['divergence_rate']:.2%})\n"
        )
        f.write(f"- Runtime: {fixed_metrics['runtime_seconds']:.1f}s\n\n")

        f.write("## Comparison\n\n")
        f.write(
            f"- **Divergence reduction factor:** {comparison['divergence_reduction_factor']:.1f}×\n"
        )
        f.write(
            f"- **Baseline divergence rate:** {comparison['divergence_rate_baseline']:.2%}\n"
        )
        f.write(
            f"- **Fixed divergence rate:** {comparison['divergence_rate_fixed']:.2%}\n\n"
        )

        f.write("## Interpretation\n\n")

        # Interpretation logic
        if comparison["divergence_reduction_factor"] >= 10.0:
            f.write(
                "**Strong evidence for Issue #3:** Divergence reduction ≥10× indicates that **parameter redundancy** between root motion, directions, and lengths is a major contributor to geometric pathologies.\n\n"
            )
            f.write(
                "The multiple pathways for explaining observations (moving root vs changing directions vs adjusting lengths) create curved, partially flat manifolds that NUTS cannot navigate efficiently.\n\n"
            )
        elif comparison["divergence_reduction_factor"] >= 3.0:
            f.write(
                "**Moderate evidence for Issue #3:** Divergence reduction of 3-10× suggests redundancy contributes to geometry issues, but other factors (root RW funnel, camera conditioning) may also be significant.\n\n"
            )
        elif comparison["divergence_reduction_factor"] >= 1.5:
            f.write(
                "**Weak evidence:** Modest divergence reduction suggests redundancy has some impact, but Issues #1 (root RW) or #2 (camera conditioning) likely dominate.\n\n"
            )
        else:
            f.write(
                "**No clear evidence:** Little to no divergence reduction suggests parameter redundancy is not a major issue. Focus should remain on Issues #1 and #2.\n\n"
            )

        if fixed_metrics["divergence_rate"] > 0.05:
            f.write(
                "⚠️ Note: The fixed-redundancy variant still has >5% divergences, indicating other geometry issues (root RW funnel, camera conditioning) remain even after removing redundancy.\n\n"
            )

        f.write("---\n\n")
        f.write(
            "**Reference:** plans/v0.2.1_divergence_plan_2.md, Issue #3, Test Group 11.1\n"
        )

    print(f"[OK] Report saved to: {report_path}")


if __name__ == "__main__":
    import os

    # Fix Windows OpenMP conflict
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    results = run_group_11_redundancy_fixed()
    print("\n" + "=" * 70)
    print("Test Group 11.1 Complete")
    print("=" * 70)
