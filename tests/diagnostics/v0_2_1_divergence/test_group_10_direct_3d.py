"""
Test Group 10: Camera Likelihood Conditioning Diagnostic

Purpose:
    Isolate whether the camera projection layer and its likelihood scale dominate
    the pathological posterior geometry.

Configuration:
    - Baseline: Full camera likelihood (2D projections with Gaussian noise)
    - Variant: Direct 3D likelihood (Gaussian on 3D positions with tau_3d=0.02m)
    - T = 100, C = 3, S = 3
    - draws = 200, tune = 200, chains = 1
    - seed = 42

Expected outcomes:
    - If direct-3D sampling behaves well while camera sampling doesn't → camera
      conditioning (Issue #2) is a major contributor
    - If both behave poorly → skeleton hierarchy alone is problematic

Reference: plans/v0.2.1_divergence_plan_2.md, Issue #2, Test Group 10

Implementation Checklist:
* [x] File path: tests/diagnostics/v0_2_1_divergence/test_group_10_direct_3d.py
* [x] Results file: tests/diagnostics/v0_2_1_divergence/results_group_10_direct_3d.json
* [x] Report file: tests/diagnostics/v0_2_1_divergence/report_group_10_direct_3d.md
* [x] Uses test_utils.get_standard_synth_data(T=100, C=3, S=3, seed=42)
* [x] Uses DLT initialization via gimbal.fit_params.initialize_from_observations_dlt
* [x] Sampler via test_utils.sample_model(model, draws=200, tune=200, chains=1)
* [x] Baseline uses test_utils.build_test_model(..., use_directional_hmm=False)
* [x] Variant replaces camera likelihood with direct 3D Gaussian (tau_3d=0.02)
* [x] Uses x_true = synth_data["joint_positions"] with shape (T, K, 3)
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


def build_test_model_direct_3d(
    synth_data: Dict[str, Any],
    tau_3d: float = 0.02,
    eta2_root_sigma: float = 0.5,
    sigma2_sigma: float = 0.2,
) -> pm.Model:
    """
    Build PyMC model with direct 3D likelihood (no camera projection).

    This variant replaces the camera projection likelihood with a simple 3D
    Gaussian likelihood on ground-truth joint positions. The skeleton hierarchy
    (root RW, directions, lengths) remains identical to the baseline.

    Parameters
    ----------
    synth_data : dict
        Synthetic data dictionary
    tau_3d : float
        Standard deviation for 3D observation noise (meters)
    eta2_root_sigma : float
        Root variance hyperparameter
    sigma2_sigma : float
        Bone length variance hyperparameter

    Returns
    -------
    pm.Model
        PyMC model with direct 3D likelihood
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

    # Build model manually to use direct 3D likelihood
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
        # Bone lengths (same as baseline)
        # =====================================================================
        rho = pm.Normal(
            "rho",
            mu=init_result.rho,
            sigma=2.0,
            shape=(K - 1,),
            initval=init_result.rho,
        )

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
        # DIRECT 3D LIKELIHOOD (replaces camera projection)
        # =====================================================================
        pm.Normal(
            "x_obs",
            mu=x_all,
            sigma=tau_3d,
            observed=x_true,
        )

    return model


def run_variant_baseline(
    synth_data: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run baseline variant with full camera likelihood."""
    print("\n" + "-" * 70)
    print("Baseline Variant: Camera Likelihood")
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
    plots_dir = Path(__file__).parent / "plots" / "group_10"
    save_diagnostic_plots(
        trace,
        "group_10_baseline_camera",
        plots_dir,
        plot_parallel=False,
        plot_pair=False,
    )

    return {
        "variant": "camera_likelihood",
        "metrics": metrics,
        "trace": trace,
    }


def run_variant_direct_3d(
    synth_data: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run variant with direct 3D likelihood."""
    print("\n" + "-" * 70)
    print("Diagnostic Variant: Direct 3D Likelihood")
    print("-" * 70)

    # Build model with direct 3D likelihood
    print(f"Building model with direct 3D likelihood (tau_3d={config['tau_3d']}m)...")
    model = build_test_model_direct_3d(
        synth_data,
        tau_3d=config["tau_3d"],
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
    plots_dir = Path(__file__).parent / "plots" / "group_10"
    save_diagnostic_plots(
        trace,
        "group_10_direct_3d",
        plots_dir,
        plot_parallel=False,
        plot_pair=False,
    )

    return {
        "variant": "direct_3d_likelihood",
        "metrics": metrics,
        "trace": trace,
    }


def run_group_10_direct_3d() -> Dict[str, Any]:
    """
    Run Test Group 10: Camera likelihood conditioning diagnostic.

    Returns
    -------
    dict
        Complete test results including both variants
    """
    print("\n" + "=" * 70)
    print("Test Group 10: Camera Likelihood Conditioning Diagnostic")
    print("=" * 70)

    # Configuration (from plan specification)
    config = {
        "test_group": 10,
        "description": "Camera likelihood conditioning (camera vs direct-3D)",
        "T": 100,
        "C": 3,
        "S": 3,
        "draws": 200,
        "tune": 200,
        "chains": 1,
        "seed": 42,
        "tau_3d": 0.02,  # 2 cm, as specified in plan
        "eta2_root_sigma": 0.5,
        "sigma2_sigma": 0.2,
    }

    print(f"\nConfiguration:")
    print(f"  T={config['T']}, C={config['C']}, S={config['S']}")
    print(f"  tau_3d={config['tau_3d']}m (direct 3D noise)")
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
    results_direct_3d = run_variant_direct_3d(synth_data, config)

    # Compile results
    results = {
        "test_group": config["test_group"],
        "description": config["description"],
        "timestamp": datetime.now().isoformat(),
        "configuration": config,
        "variants": {
            "camera": {
                "description": "Full camera projection likelihood",
                "metrics": results_baseline["metrics"],
            },
            "direct_3d": {
                "description": f"Direct 3D Gaussian likelihood (tau={config['tau_3d']}m)",
                "metrics": results_direct_3d["metrics"],
            },
        },
        "comparison": {
            "divergence_reduction_factor": (
                results_baseline["metrics"]["divergences"]
                / max(results_direct_3d["metrics"]["divergences"], 1)
            ),
            "divergence_rate_camera": results_baseline["metrics"]["divergence_rate"],
            "divergence_rate_direct_3d": results_direct_3d["metrics"][
                "divergence_rate"
            ],
        },
    }

    # Save results
    results_path = Path(__file__).parent / "results_group_10_direct_3d.json"
    with open(results_path, "w") as f:
        results_serializable = {k: v for k, v in results.items()}
        json.dump(results_serializable, f, indent=2)
    print(f"\n[OK] Results saved to: {results_path}")

    # Generate markdown report
    generate_report(results)

    return results


def generate_report(results: Dict[str, Any]):
    """Generate markdown report for Test Group 10."""
    report_path = Path(__file__).parent / "report_group_10_direct_3d.md"

    camera_metrics = results["variants"]["camera"]["metrics"]
    direct_3d_metrics = results["variants"]["direct_3d"]["metrics"]
    comparison = results["comparison"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Test Group 10: Camera Likelihood Conditioning Diagnostic\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")

        f.write("## Purpose\n\n")
        f.write(
            "Isolate whether the camera projection layer and its likelihood scale dominate the pathological posterior geometry.\n\n"
        )

        f.write("## Configuration\n\n")
        f.write(f"- T = {results['configuration']['T']}\n")
        f.write(f"- C = {results['configuration']['C']}\n")
        f.write(f"- tau_3d = {results['configuration']['tau_3d']}m (direct 3D noise)\n")
        f.write(
            f"- draws = {results['configuration']['draws']}, tune = {results['configuration']['tune']}\n"
        )
        f.write(f"- seed = {results['configuration']['seed']}\n\n")

        f.write("## Results\n\n")
        f.write("### Baseline: Camera Projection Likelihood\n\n")
        f.write(
            f"- Divergences: {camera_metrics['divergences']}/{camera_metrics['total_samples']} ({camera_metrics['divergence_rate']:.2%})\n"
        )
        f.write(f"- Runtime: {camera_metrics['runtime_seconds']:.1f}s\n\n")

        f.write("### Variant: Direct 3D Likelihood\n\n")
        f.write(
            f"- Divergences: {direct_3d_metrics['divergences']}/{direct_3d_metrics['total_samples']} ({direct_3d_metrics['divergence_rate']:.2%})\n"
        )
        f.write(f"- Runtime: {direct_3d_metrics['runtime_seconds']:.1f}s\n\n")

        f.write("## Comparison\n\n")
        f.write(
            f"- **Divergence reduction factor:** {comparison['divergence_reduction_factor']:.1f}×\n"
        )
        f.write(
            f"- **Camera divergence rate:** {comparison['divergence_rate_camera']:.2%}\n"
        )
        f.write(
            f"- **Direct-3D divergence rate:** {comparison['divergence_rate_direct_3d']:.2%}\n\n"
        )

        f.write("## Interpretation\n\n")

        # Interpretation logic
        camera_healthy = camera_metrics["divergence_rate"] > 0.05
        direct_3d_healthy = direct_3d_metrics["divergence_rate"] <= 0.05

        if camera_healthy and not direct_3d_healthy:
            f.write(
                "**Strong evidence for Issue #2:** The direct-3D model shows healthy sampling (≤5% divergences) while the camera model remains problematic. This strongly suggests that **camera conditioning** is a major contributor to pathological geometry.\n\n"
            )
            f.write(
                "The camera projection layer or its likelihood scale induces extreme curvature that NUTS cannot traverse. Prior reparameterization alone may not fix the model until camera conditioning is addressed.\n\n"
            )
        elif camera_healthy and direct_3d_healthy:
            f.write(
                "**Both models problematic:** Both the camera and direct-3D models show >5% divergences. This suggests the **skeleton hierarchy alone** (root RW, directions, lengths) is already creating pathological geometry.\n\n"
            )
            f.write(
                "Issue #1 (root RW funnel) or Issue #3 (parameter redundancy) likely dominate over camera conditioning effects.\n\n"
            )
        elif not camera_healthy and not direct_3d_healthy:
            f.write(
                "**Both models healthy:** Both variants show ≤5% divergences, suggesting neither camera conditioning nor skeleton hierarchy are major issues in isolation.\n\n"
            )
        else:  # camera healthy but direct-3D worse (unexpected)
            f.write(
                "**Unexpected result:** The direct-3D model performs worse than the camera model. This may indicate:\n"
            )
            f.write(
                "- The tau_3d parameter is mismatched to the actual 3D noise scale\n"
            )
            f.write(
                "- The direct 3D likelihood interacts poorly with the skeleton priors\n"
            )
            f.write("- Further investigation needed\n\n")

        f.write("---\n\n")
        f.write(
            "**Reference:** plans/v0.2.1_divergence_plan_2.md, Issue #2, Test Group 10\n"
        )

    print(f"[OK] Report saved to: {report_path}")


if __name__ == "__main__":
    import os

    # Fix Windows OpenMP conflict
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    results = run_group_10_direct_3d()
    print("\n" + "=" * 70)
    print("Test Group 10 Complete")
    print("=" * 70)
