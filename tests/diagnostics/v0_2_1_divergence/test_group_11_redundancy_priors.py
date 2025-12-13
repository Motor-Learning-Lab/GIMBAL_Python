"""
Test Group 11.2: Parameter Redundancy Diagnostic (Strong GT-Based Priors)

Purpose:
    Test whether imposing strong, data-driven priors on directions and lengths
    (centered on ground truth with tight variances) reduces divergences without
    completely fixing the parameters.

Configuration:
    - Baseline: Full model (weak priors on directions + free lengths)
    - Variant: Strong GT-based priors (2% sd for lengths, tight directional priors)
    - T = 100, C = 3, S = 3
    - draws = 200, tune = 200, chains = 1
    - seed = 42

Expected outcomes:
    - Monotonic improvement (baseline → strong priors → fixed) indicates redundancy
      amplifies geometric pathologies
    - No pattern suggests redundancy is secondary to Issues #1 and #2

Reference: plans/v0.2.1_divergence_plan_2.md, Issue #3, Test Group 11.2

Implementation Checklist:
* [x] File path: tests/diagnostics/v0_2_1_divergence/test_group_11_redundancy_priors.py
* [x] Results file: tests/diagnostics/v0_2_1_divergence/results_group_11_redundancy_priors.json
* [x] Report file: tests/diagnostics/v0_2_1_divergence/report_group_11_redundancy_priors.md
* [x] Uses test_utils.get_standard_synth_data(T=100, C=3, S=3, seed=42)
* [x] Uses DLT initialization via gimbal.fit_params.initialize_from_observations_dlt
* [x] Sampler via test_utils.sample_model(model, draws=200, tune=200, chains=1)
* [x] Baseline uses test_utils.build_test_model(..., use_directional_hmm=False)
* [x] Variant uses strong GT-based priors (2% sd for lengths, sd_raw=0.05 for directions)
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
from gimbal.prior_building import get_gamma_shape_rate


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
            continue

        v = x_true[:, k, :] - x_true[:, p, :]
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        u_true[:, k, :] = v / (norms + 1e-12)

    return u_true


def build_test_model_strong_priors(
    synth_data: Dict[str, Any],
    eta2_root_sigma: float = 0.5,
    length_relative_sd: float = 0.02,  # 2% as specified
    raw_direction_sd: float = 0.05,  # Tight prior on raw vectors
) -> pm.Model:
    """
    Build PyMC model with strong GT-based priors on directions and lengths.

    This variant enforces tight priors centered on ground truth, preserving
    uncertainty but heavily discouraging large deviations.

    Parameters
    ----------
    synth_data : dict
        Synthetic data dictionary
    eta2_root_sigma : float
        Root variance hyperparameter
    length_relative_sd : float
        Relative standard deviation for bone length priors (2% = 0.02)
    raw_direction_sd : float
        Standard deviation for raw directional vector perturbations

    Returns
    -------
    pm.Model
        PyMC model with strong GT-based priors
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
    lengths_true = synth_data["bone_lengths"]  # (K,) static reference

    # Build model with strong priors
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
        # STRONG PRIORS ON BONE LENGTHS
        # =====================================================================
        # Use Gamma priors with mode = GT and sd = 2% * GT
        rho_list = []
        sigma2_list = []

        for k in range(K - 1):
            length_gt = lengths_true[
                k + 1
            ]  # Note: lengths_true is indexed by joint, not edge
            length_sd = length_relative_sd * length_gt

            # Use Gamma prior on mean bone length
            # For simplicity, use a tight Normal prior centered on GT
            rho_k = pm.Normal(
                f"rho_{k}",
                mu=length_gt,
                sigma=length_sd,
                initval=length_gt,
            )
            rho_list.append(rho_k)

            # Tight prior on bone length variance
            sigma2_k = pm.HalfNormal(
                f"sigma2_{k}",
                sigma=0.01 * length_gt,  # Very small variance
                initval=0.01 * length_gt,
            )
            sigma2_list.append(sigma2_k)

        rho = pt.stack(rho_list)
        sigma2 = pt.stack(sigma2_list)

        # =====================================================================
        # STRONG PRIORS ON DIRECTIONAL VECTORS
        # =====================================================================
        x_all_list = [x_root]
        u_list = []

        for k in range(1, K):
            parent = int(synth_data["parents"][k])

            # Compute GT raw vector (before normalization)
            u_true_k = u_true[:, k, :]  # (T, 3)
            # Assume unit length, so raw ~ u_true with small perturbations

            # Strong prior: raw vector centered on GT with tight SD
            raw_u_k = pm.Normal(
                f"raw_u_{k}",
                mu=u_true_k,
                sigma=raw_direction_sd,
                shape=(T, 3),
                initval=u_true_k,
            )

            # Normalize to unit sphere
            norm_u_k = pt.sqrt((raw_u_k**2).sum(axis=-1, keepdims=True) + 1e-8)
            u_k = pm.Deterministic(f"u_{k}", raw_u_k / norm_u_k)
            u_list.append(u_k)

            # Bone length with strong prior
            length_k = pm.Normal(
                f"length_{k}",
                mu=rho[k - 1],
                sigma=pt.sqrt(sigma2[k - 1]),
                shape=(T,),
                initval=np.full(T, lengths_true[k]),
            )

            # Joint position
            x_parent = x_all_list[parent]
            x_k = pm.Deterministic(
                f"x_{k}",
                x_parent + u_k * length_k[:, None],
            )
            x_all_list.append(x_k)

        # Stack all joint positions
        x_all = pm.Deterministic("x_all", pt.stack(x_all_list, axis=1))

        # =====================================================================
        # Camera projection (same as baseline)
        # =====================================================================
        proj_param = synth_data["camera_matrices"]

        y_pred_list = []
        for c in range(C):
            A_c = proj_param[c, :, :3]
            b_c = proj_param[c, :, 3]

            uvw = pt.dot(x_all, A_c.T) + b_c[None, None, :]

            u = uvw[:, :, 0] / uvw[:, :, 2]
            v = uvw[:, :, 1] / uvw[:, :, 2]
            y_c = pt.stack([u, v], axis=-1)
            y_pred_list.append(y_c)

        y_pred = pm.Deterministic("y_pred", pt.stack(y_pred_list, axis=0))

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
    """Run baseline variant with weak priors."""
    print("\n" + "-" * 70)
    print("Baseline Variant: Weak Priors")
    print("-" * 70)

    print("Building model...")
    model = build_test_model(
        synth_data,
        use_directional_hmm=False,
        S=config["S"],
        eta2_root_sigma=config["eta2_root_sigma"],
        sigma2_sigma=config["sigma2_sigma"],
    )
    print(f"  [OK] Model has {len(model.free_RVs)} free RVs")

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

    metrics = extract_metrics(trace, runtime)
    print(
        f"  Divergences: {metrics['divergences']}/{metrics['total_samples']} ({metrics['divergence_rate']:.2%})"
    )

    plots_dir = Path(__file__).parent / "plots" / "group_11"
    save_diagnostic_plots(
        trace,
        "group_11_baseline_weak",
        plots_dir,
        plot_parallel=False,
        plot_pair=False,
    )

    return {
        "variant": "baseline_weak_priors",
        "metrics": metrics,
        "trace": trace,
    }


def run_variant_strong_priors(
    synth_data: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run variant with strong GT-based priors."""
    print("\n" + "-" * 70)
    print("Diagnostic Variant: Strong GT-Based Priors")
    print("-" * 70)

    print("Building model with strong GT-based priors...")
    print(f"  - Length relative SD: {config['length_relative_sd']:.1%}")
    print(f"  - Raw direction SD: {config['raw_direction_sd']}")

    model = build_test_model_strong_priors(
        synth_data,
        eta2_root_sigma=config["eta2_root_sigma"],
        length_relative_sd=config["length_relative_sd"],
        raw_direction_sd=config["raw_direction_sd"],
    )
    print(f"  [OK] Model has {len(model.free_RVs)} free RVs")

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

    metrics = extract_metrics(trace, runtime)
    print(
        f"  Divergences: {metrics['divergences']}/{metrics['total_samples']} ({metrics['divergence_rate']:.2%})"
    )

    plots_dir = Path(__file__).parent / "plots" / "group_11"
    save_diagnostic_plots(
        trace,
        "group_11_strong_priors",
        plots_dir,
        plot_parallel=False,
        plot_pair=False,
    )

    return {
        "variant": "strong_gt_priors",
        "metrics": metrics,
        "trace": trace,
    }


def run_group_11_redundancy_priors() -> Dict[str, Any]:
    """
    Run Test Group 11.2: Parameter redundancy diagnostic (strong priors).

    Returns
    -------
    dict
        Complete test results including both variants
    """
    print("\n" + "=" * 70)
    print("Test Group 11.2: Parameter Redundancy Diagnostic (Strong Priors)")
    print("=" * 70)

    # Configuration
    config = {
        "test_group": 11.2,
        "description": "Parameter redundancy (baseline vs strong GT-based priors)",
        "T": 100,
        "C": 3,
        "S": 3,
        "draws": 200,
        "tune": 200,
        "chains": 1,
        "seed": 42,
        "eta2_root_sigma": 0.5,
        "sigma2_sigma": 0.2,
        "length_relative_sd": 0.02,  # 2% as specified in plan
        "raw_direction_sd": 0.05,  # Tight directional prior
    }

    print(f"\nConfiguration:")
    print(f"  T={config['T']}, C={config['C']}, S={config['S']}")
    print(f"  Strong prior settings:")
    print(f"    - Length relative SD: {config['length_relative_sd']:.1%}")
    print(f"    - Raw direction SD: {config['raw_direction_sd']}")
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
    results_strong = run_variant_strong_priors(synth_data, config)

    # Compile results
    results = {
        "test_group": config["test_group"],
        "description": config["description"],
        "timestamp": datetime.now().isoformat(),
        "configuration": config,
        "variants": {
            "baseline": {
                "description": "Weak priors on directions and lengths",
                "metrics": results_baseline["metrics"],
            },
            "strong_priors": {
                "description": "Strong GT-based priors (2% sd lengths, sd=0.05 raw directions)",
                "metrics": results_strong["metrics"],
            },
        },
        "comparison": {
            "divergence_reduction_factor": (
                results_baseline["metrics"]["divergences"]
                / max(results_strong["metrics"]["divergences"], 1)
            ),
            "divergence_rate_baseline": results_baseline["metrics"]["divergence_rate"],
            "divergence_rate_strong": results_strong["metrics"]["divergence_rate"],
        },
    }

    # Save results
    results_path = Path(__file__).parent / "results_group_11_redundancy_priors.json"
    with open(results_path, "w") as f:
        results_serializable = {k: v for k, v in results.items()}
        json.dump(results_serializable, f, indent=2)
    print(f"\n[OK] Results saved to: {results_path}")

    # Generate markdown report
    generate_report(results)

    return results


def generate_report(results: Dict[str, Any]):
    """Generate markdown report for Test Group 11.2."""
    report_path = Path(__file__).parent / "report_group_11_redundancy_priors.md"

    baseline_metrics = results["variants"]["baseline"]["metrics"]
    strong_metrics = results["variants"]["strong_priors"]["metrics"]
    comparison = results["comparison"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "# Test Group 11.2: Parameter Redundancy Diagnostic (Strong Priors)\n\n"
        )
        f.write(f"**Generated:** {results['timestamp']}\n\n")

        f.write("## Purpose\n\n")
        f.write(
            "Test whether imposing strong, data-driven priors on directions and lengths reduces divergences without completely fixing parameters.\n\n"
        )

        f.write("## Configuration\n\n")
        f.write(f"- T = {results['configuration']['T']}\n")
        f.write(f"- C = {results['configuration']['C']}\n")
        f.write(f"- Strong prior settings:\n")
        f.write(
            f"  - Length relative SD: {results['configuration']['length_relative_sd']:.1%}\n"
        )
        f.write(
            f"  - Raw direction SD: {results['configuration']['raw_direction_sd']}\n"
        )
        f.write(
            f"- draws = {results['configuration']['draws']}, tune = {results['configuration']['tune']}\n"
        )
        f.write(f"- seed = {results['configuration']['seed']}\n\n")

        f.write("## Results\n\n")
        f.write("### Baseline: Weak Priors\n\n")
        f.write(
            f"- Divergences: {baseline_metrics['divergences']}/{baseline_metrics['total_samples']} ({baseline_metrics['divergence_rate']:.2%})\n"
        )
        f.write(f"- Runtime: {baseline_metrics['runtime_seconds']:.1f}s\n\n")

        f.write("### Variant: Strong GT-Based Priors\n\n")
        f.write(
            f"- Divergences: {strong_metrics['divergences']}/{strong_metrics['total_samples']} ({strong_metrics['divergence_rate']:.2%})\n"
        )
        f.write(f"- Runtime: {strong_metrics['runtime_seconds']:.1f}s\n\n")

        f.write("## Comparison\n\n")
        f.write(
            f"- **Divergence reduction factor:** {comparison['divergence_reduction_factor']:.1f}×\n"
        )
        f.write(
            f"- **Baseline divergence rate:** {comparison['divergence_rate_baseline']:.2%}\n"
        )
        f.write(
            f"- **Strong-priors divergence rate:** {comparison['divergence_rate_strong']:.2%}\n\n"
        )

        f.write("## Interpretation\n\n")
        f.write("### Cross-Test Pattern (Test 11.1 + 11.2)\n\n")
        f.write(
            "Consider results from both Test 11.1 (fixed) and Test 11.2 (strong priors):\n\n"
        )
        f.write(
            "- **Monotonic improvement** (weak priors → strong priors → fixed) suggests parameter redundancy amplifies geometric pathologies.\n"
        )
        f.write(
            "- **No clear pattern** suggests redundancy is secondary to Issues #1 (root RW) and #2 (camera conditioning).\n\n"
        )

        f.write("### This Test (Strong Priors vs Baseline)\n\n")

        if comparison["divergence_reduction_factor"] >= 3.0:
            f.write(
                "**Moderate-to-strong evidence:** Divergence reduction ≥3× from strong priors (without full fixing) suggests that **constraining parameter redundancy** helps sampling, even when uncertainty is preserved.\n\n"
            )
        elif comparison["divergence_reduction_factor"] >= 1.5:
            f.write(
                "**Weak evidence:** Modest improvement suggests strong priors have some benefit, but redundancy is not the dominant issue.\n\n"
            )
        else:
            f.write(
                "**No clear evidence:** Little improvement from strong priors suggests redundancy is not a major factor. Focus on Issues #1 and #2.\n\n"
            )

        if strong_metrics["divergence_rate"] > 0.05:
            f.write(
                "⚠️ Note: The strong-prior variant still has >5% divergences, indicating other issues (root RW, camera conditioning) remain.\n\n"
            )

        f.write("---\n\n")
        f.write(
            "**Reference:** plans/v0.2.1_divergence_plan_2.md, Issue #3, Test Group 11.2\n"
        )

    print(f"[OK] Report saved to: {report_path}")


if __name__ == "__main__":
    import os

    # Fix Windows OpenMP conflict
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    results = run_group_11_redundancy_priors()
    print("\n" + "=" * 70)
    print("Test Group 11.2 Complete")
    print("=" * 70)
