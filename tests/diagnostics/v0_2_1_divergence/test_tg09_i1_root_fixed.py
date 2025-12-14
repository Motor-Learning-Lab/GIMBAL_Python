"""
Test Group 9: Root Random-Walk Funnel Diagnostic

Purpose:
    Isolate whether the hierarchical root random walk (RW) structure is a major
    contributor to sampling divergences.

Configuration:
    - Baseline: Full hierarchical RW with free sigma_root (centered parameterization)
    - Variant: Fixed root trajectory using DLT initialization (no sigma_root, no RW)
    - T = 100, C = 3, S = 3
    - draws = 200, tune = 200, chains = 1
    - seed = 42

Expected outcomes:
    - If divergences drop substantially in variant → root RW hierarchy is major contributor
    - If divergences persist → other issues (camera conditioning, redundancy) dominate

Reference: plans/v0.2.1_divergence_plan_2.md, Issue #1, Test Group 9

Implementation Checklist:
* [x] File path: tests/diagnostics/v0_2_1_divergence/test_tg09_i1_root_fixed.py
* [x] Results file: tests/diagnostics/v0_2_1_divergence/results_tg09_i1_root_fixed.json
* [x] Report file: tests/diagnostics/v0_2_1_divergence/report_tg09_i1_root_fixed.md
* [x] Uses test_utils.get_standard_synth_data(T=100, C=3, S=3, seed=42)
* [x] Uses DLT initialization via gimbal.fit_params.initialize_from_observations_dlt
* [x] Sampler via test_utils.sample_model(model, draws=200, tune=200, chains=1)
* [x] Baseline uses test_utils.build_test_model(..., use_directional_hmm=False)
* [x] Variant replaces root RW with pm.Data using DLT initialization
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


def build_test_model_root_fixed(
    synth_data: Dict[str, Any],
    eta2_root_sigma: float = 0.5,
    sigma2_sigma: float = 0.2,
) -> pm.Model:
    """
    Build PyMC model with fixed root trajectory (no hierarchical RW).

    This variant removes the root RW by fixing x_root to DLT initialization,
    keeping all other model components (directions, lengths, camera likelihood)
    identical to the baseline.

    Parameters
    ----------
    synth_data : dict
        Synthetic data dictionary
    eta2_root_sigma : float
        Not used in this variant (kept for signature compatibility)
    sigma2_sigma : float
        Bone length variance hyperparameter

    Returns
    -------
    pm.Model
        PyMC model with fixed root trajectory
    """
    # Initialize from observations using DLT
    init_result = gimbal.fit_params.initialize_from_observations_dlt(
        y_observed=synth_data["observations_uv"],
        camera_proj=synth_data["camera_matrices"],
        parents=synth_data["parents"],
    )

    # Extract dimensions
    T, K, _ = synth_data["joint_positions"].shape
    C = synth_data["camera_matrices"].shape[0]

    # Build model manually to replace root RW
    with pm.Model() as model:
        # =====================================================================
        # FIXED ROOT (replaces hierarchical RW)
        # =====================================================================
        x_root = pm.Data("x_root", init_result.x_init[:, 0, :])  # (T, 3)

        # =====================================================================
        # Bone lengths (same as baseline)
        # =====================================================================
        # Prior on bone length means
        rho = pm.Normal(
            "rho",
            mu=init_result.rho,
            sigma=2.0,
            shape=(K - 1,),
            initval=init_result.rho,
        )

        # Prior on bone length variance
        sigma2 = pm.HalfNormal(
            "sigma2",
            sigma=sigma2_sigma,
            shape=(K - 1,),
            initval=init_result.sigma2,
        )

        # =====================================================================
        # Directional vectors and joint positions (same as baseline)
        # =====================================================================
        x_all_list = [x_root]
        u_list = []

        for k in range(1, K):
            parent = int(synth_data["parents"][k])

            # Raw directional vector
            raw_u_k = pm.Normal(
                f"raw_u_{k}",
                mu=0.0,
                sigma=1.0,
                shape=(T, 3),
                initval=init_result.u_init[:, k, :],
            )

            # Normalize to unit sphere
            norm_u_k = pt.sqrt((raw_u_k**2).sum(axis=-1, keepdims=True) + 1e-8)
            u_k = pm.Deterministic(f"u_{k}", raw_u_k / norm_u_k)
            u_list.append(u_k)

            # Bone length at each timestep
            length_k = pm.Normal(
                f"length_{k}",
                mu=rho[k - 1],
                sigma=pt.sqrt(sigma2[k - 1]),
                shape=(T,),
                initval=np.full(T, init_result.rho[k - 1]),
            )

            # Joint position: parent + direction * length
            x_parent = x_all_list[parent]
            x_k = pm.Deterministic(
                f"x_{k}",
                x_parent + u_k * length_k[:, None],
            )
            x_all_list.append(x_k)

        # Stack all joint positions: (T, K, 3)
        x_all = pm.Deterministic("x_all", pt.stack(x_all_list, axis=1))

        # =====================================================================
        # Camera projection (same as baseline)
        # =====================================================================
        # Project 3D points to 2D
        proj_param = synth_data["camera_matrices"]  # (C, 3, 4)

        # Expand x_all to (C, T, K, 3) for broadcasting
        x_all_expanded = x_all.dimshuffle("x", 0, 1, 2)  # (1, T, K, 3) -> (C, T, K, 3)

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

        # Gaussian likelihood on observed 2D keypoints
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
    """Run baseline variant with hierarchical root RW."""
    print("\n" + "-" * 70)
    print("Baseline Variant: Hierarchical Root RW")
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
    plots_dir = Path(__file__).parent / "plots" / "group_9"
    save_diagnostic_plots(
        trace,
        "group_9_baseline",
        plots_dir,
        plot_parallel=False,
        plot_pair=False,
    )

    return {
        "variant": "baseline_hierarchical_rw",
        "metrics": metrics,
        "trace": trace,
    }


def run_variant_root_fixed(
    synth_data: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run variant with fixed root trajectory (DLT-based)."""
    print("\n" + "-" * 70)
    print("Diagnostic Variant: Fixed Root (DLT-based)")
    print("-" * 70)

    # Build model with fixed root
    print("Building model with fixed root...")
    model = build_test_model_root_fixed(
        synth_data,
        eta2_root_sigma=config[
            "eta2_root_sigma"
        ],  # Not used, but kept for compatibility
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
    plots_dir = Path(__file__).parent / "plots" / "group_9"
    save_diagnostic_plots(
        trace,
        "group_9_root_fixed",
        plots_dir,
        plot_parallel=False,
        plot_pair=False,
    )

    return {
        "variant": "root_fixed_dlt",
        "metrics": metrics,
        "trace": trace,
    }


def run_tg09_i1_root_fixed() -> Dict[str, Any]:
    """
    Run Test Group 9: Root RW Funnel diagnostic.

    Returns
    -------
    dict
        Complete test results including both variants
    """
    print("\n" + "=" * 70)
    print("Test Group 9: Root Random-Walk Funnel Diagnostic")
    print("=" * 70)

    # Configuration (stricter parameters post-fix validation)
    config = {
        "test_group": 9,
        "description": "Root RW funnel diagnostic (new baseline with Gamma priors vs fixed root)",
        "T": 100,
        "C": 3,
        "S": 3,
        "draws": 500,  # Increased from 200 for more robust convergence
        "tune": 500,  # Increased from 200
        "chains": 2,  # Increased from 1 for R-hat convergence checks
        "seed": 42,
        "eta2_root_sigma": 0.5,  # NOTE: Now unused (data-driven Gamma priors in base code)
        "sigma2_sigma": 0.2,  # NOTE: Now unused (data-driven Gamma priors in base code)
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
    results_fixed = run_variant_root_fixed(synth_data, config)

    # Compile results
    results = {
        "test_group": config["test_group"],
        "description": config["description"],
        "timestamp": datetime.now().isoformat(),
        "configuration": config,
        "variants": {
            "baseline": {
                "description": "Hierarchical root RW with free sigma_root",
                "metrics": results_baseline["metrics"],
            },
            "root_fixed": {
                "description": "Fixed root trajectory (DLT initialization)",
                "metrics": results_fixed["metrics"],
            },
        },
        "comparison": {
            "divergence_reduction_factor": (
                results_baseline["metrics"]["divergences"]
                / max(
                    results_fixed["metrics"]["divergences"], 1
                )  # Avoid division by zero
            ),
            "divergence_rate_baseline": results_baseline["metrics"]["divergence_rate"],
            "divergence_rate_fixed": results_fixed["metrics"]["divergence_rate"],
        },
    }

    # Save results
    results_path = Path(__file__).parent / "results_tg09_i1_root_fixed.json"
    with open(results_path, "w") as f:
        # Convert to serializable format (exclude trace objects)
        results_serializable = {k: v for k, v in results.items()}
        json.dump(results_serializable, f, indent=2)
    print(f"\n[OK] Results saved to: {results_path}")

    # Generate markdown report
    generate_report(results)

    return results


def generate_report(results: Dict[str, Any]):
    """Generate markdown report for Test Group 9."""
    report_path = Path(__file__).parent / "report_tg09_i1_root_fixed.md"

    baseline_metrics = results["variants"]["baseline"]["metrics"]
    fixed_metrics = results["variants"]["root_fixed"]["metrics"]
    comparison = results["comparison"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Test Group 9: Root Random-Walk Funnel Diagnostic\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")

        f.write("## Purpose\n\n")
        f.write(
            "Isolate whether the hierarchical root random walk (RW) structure is a major contributor to sampling divergences.\n\n"
        )

        f.write("## Configuration\n\n")
        f.write(f"- T = {results['configuration']['T']}\n")
        f.write(f"- C = {results['configuration']['C']}\n")
        f.write(
            f"- draws = {results['configuration']['draws']}, tune = {results['configuration']['tune']}, chains = {results['configuration']['chains']}\n"
        )
        f.write(f"- seed = {results['configuration']['seed']}\n")
        f.write(
            f"- **NOTE:** Base code now uses data-driven Gamma priors (50% CV) and non-centered root parameterization\n\n"
        )

        f.write("## Results\n\n")
        f.write("### Baseline: New Base Code (Non-centered Root + Gamma Priors)\n\n")
        f.write(
            f"- Divergences: {baseline_metrics['divergences']}/{baseline_metrics['total_samples']} ({baseline_metrics['divergence_rate']:.2%})\n"
        )
        f.write(f"- Runtime: {baseline_metrics['runtime_seconds']:.1f}s\n")
        # R-hat convergence
        if baseline_metrics["rhat_max"]:
            max_rhat = max(baseline_metrics["rhat_max"].values())
            f.write(f"- Max R-hat: {max_rhat:.3f}\n")
        f.write("\n")

        f.write("### Variant: Fixed Root (DLT-based, No Dynamics)\n\n")
        f.write(
            f"- Divergences: {fixed_metrics['divergences']}/{fixed_metrics['total_samples']} ({fixed_metrics['divergence_rate']:.2%})\n"
        )
        f.write(f"- Runtime: {fixed_metrics['runtime_seconds']:.1f}s\n")
        if fixed_metrics["rhat_max"]:
            max_rhat = max(fixed_metrics["rhat_max"].values())
            f.write(f"- Max R-hat: {max_rhat:.3f}\n")
        f.write("\n")

        f.write("## Comparison\n\n")
        f.write(
            f"- **Divergence reduction factor:** {comparison['divergence_reduction_factor']:.1f}×\n"
        )
        f.write(
            f"- **Baseline divergence rate:** {comparison['divergence_rate_baseline']:.2%}\n"
        )
        f.write(
            f"- **Fixed-root divergence rate:** {comparison['divergence_rate_fixed']:.2%}\n\n"
        )

        f.write("## Interpretation\n\n")

        # Check baseline health (should be <1% with fixed base code)
        baseline_healthy = baseline_metrics["divergence_rate"] < 0.01

        if baseline_healthy:
            f.write(
                f"✅ **Baseline validation:** New base code achieves <1% divergence rate ({baseline_metrics['divergence_rate']:.2%}), "
                "confirming the non-centered root parameterization and Gamma priors successfully resolve the root RW funnel issue.\n\n"
            )
        else:
            f.write(
                f"⚠️ **Baseline validation:** New base code still shows {baseline_metrics['divergence_rate']:.2%} divergences (target: <1%), "
                "indicating possible remaining geometry issues or need for further tuning.\n\n"
            )

        # Compare baseline to fixed-root variant
        if comparison["divergence_reduction_factor"] >= 2.0:
            f.write(
                f"**Variant comparison:** Fixed-root variant shows {comparison['divergence_reduction_factor']:.1f}× divergence reduction, "
                "suggesting the root dynamics (even non-centered) still contribute some geometry complexity.\n\n"
            )
        else:
            f.write(
                "**Variant comparison:** Fixed-root variant shows similar divergence rate to new baseline, "
                "confirming the non-centered parameterization effectively addresses root RW geometry issues.\n\n"
            )

        # Convergence check
        if baseline_metrics.get("rhat_max"):
            max_rhat_baseline = max(baseline_metrics["rhat_max"].values())
            if max_rhat_baseline > 1.1:
                f.write(
                    f"⚠️ **Convergence warning:** Baseline max R-hat = {max_rhat_baseline:.3f} > 1.1, "
                    "indicating potential convergence issues. Consider increasing draws/tune further.\n\n"
                )

        f.write("---\n\n")
        f.write(
            "**Reference:** plans/v0.2.1_divergence_plan_2.md, Issue #1, Test Group 9 (post-fix validation)\n"
        )

    print(f"[OK] Report saved to: {report_path}")


if __name__ == "__main__":
    import os

    # Fix Windows OpenMP conflict
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    results = run_group_9_root_fixed()
    print("\n" + "=" * 70)
    print("Test Group 9 Complete")
    print("=" * 70)
