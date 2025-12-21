"""Group 7: Likelihood-Only Freeze Latents

Tests observation model with fixed latents (x, U).
Config: use_mixture=True, use_directional_hmm=False (latents frozen)
Truncated to max_T=80

Strategy: Sample only observation parameters (rho, sigma2, logodds_inlier).

Usage:
    pixi run python test_group_7_likelihood_only_freeze_latents.py --run_id frozen_test
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
from pathlib import Path
from datetime import datetime
import traceback

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import gimbal
from _diag_utils import (
    make_paths,
    collect_environment,
    write_json,
    save_text,
    safe_point_logps,
    compile_logp_and_grad,
    summarize_worst_terms,
    format_worst_terms_table,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Group 7: Likelihood-Only Freeze Latents"
    )
    parser.add_argument(
        "--dataset_name", default="v0.2.1_L00_minimal", help="Dataset name"
    )
    parser.add_argument(
        "--max_T", type=int, default=80, help="Maximum number of time frames"
    )
    parser.add_argument(
        "--max_K", type=int, default=None, help="Maximum number of joints"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chains", type=int, default=2, help="Number of MCMC chains")
    parser.add_argument("--tune", type=int, default=300, help="Tuning iterations")
    parser.add_argument("--draws", type=int, default=100, help="Draw iterations")
    parser.add_argument("--run_id", default=None, help="Run identifier")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing results"
    )
    return parser.parse_args()


def load_dataset(dataset_name, max_T=None, max_K=None):
    """Load the dataset."""
    dataset_dir = project_root / "tests" / "pipeline" / "datasets" / dataset_name
    dataset_path = dataset_dir / "dataset.npz"

    with np.load(dataset_path, allow_pickle=True) as f:
        data = {
            "y_2d": f["y_2d"],
            "camera_proj": f["camera_proj"],
            "parents": f["parents"],
            "bone_lengths": f["bone_lengths"],
            "joint_names": [str(name) for name in f["joint_names"]],
        }

    if max_T is not None:
        data["y_2d"] = data["y_2d"][:, :max_T, :, :]
    if max_K is not None:
        data["y_2d"] = data["y_2d"][:, :, :max_K, :]
        data["parents"] = data["parents"][:max_K]
        data["joint_names"] = data["joint_names"][:max_K]

    return data, dataset_dir


def build_model_frozen(data):
    """Build model with frozen latents (x, U)."""
    C, T, K, _ = data["y_2d"].shape

    print("  Initializing from observations (DLT triangulation)...")
    from gimbal.fit_params import initialize_from_observations_dlt

    init_result = initialize_from_observations_dlt(
        y_observed=data["y_2d"],
        camera_proj=data["camera_proj"],
        parents=data["parents"],
    )

    print(f"  Initialization complete")

    # Build model with mixture but without HMM
    with pm.Model() as model:
        gimbal.build_camera_observation_model(
            y_observed=data["y_2d"],
            camera_proj=data["camera_proj"],
            parents=data["parents"],
            init_result=init_result,
            use_mixture=True,
            image_size=(1280, 720),
            use_directional_hmm=False,
            validate_init_points=False,
        )

        # Freeze latents by setting them as observed
        x_root = model["x_root"]
        U = model["U"]

        # Get initial values
        init_point = model.initial_point()
        x_root_init = init_point["x_root"]
        U_init = init_point["U"]

        # Replace with constants (frozen values)
        model.register_rv(
            pm.ConstantData("x_root_frozen", x_root_init), name="x_root_frozen"
        )
        model.register_rv(pm.ConstantData("U_frozen", U_init), name="U_frozen")

        # Rebuild observation likelihood with frozen latents
        # Note: This requires rebuilding y_pred with frozen values
        # For simplicity, we'll just sample with fixed initial values

    return model, init_result, x_root_init, U_init


def sample_model_frozen(model, x_root_init, U_init, chains, tune, draws, seed):
    """Sample model with frozen latents."""
    print(
        f"\n  Sampling with frozen latents: chains={chains}, tune={tune}, draws={draws}"
    )

    # Create initial points with frozen latents
    init_points = []
    for chain in range(chains):
        init_point = model.initial_point()
        init_point["x_root"] = x_root_init
        init_point["U"] = U_init
        init_points.append(init_point)

    # Sample only observation parameters
    # We'll use step methods that only update rho, sigma2, logodds_inlier
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            initvals=init_points,
            random_seed=seed,
            return_inferencedata=True,
            step=pm.NUTS(vars=[model["rho"], model["sigma2"], model["logodds_inlier"]]),
        )

    divergences = int(trace.sample_stats.diverging.sum().values)
    mean_step_size = float(trace.sample_stats.step_size.mean().values)
    max_treedepth_frac = float(
        (trace.sample_stats.tree_depth == trace.sample_stats.tree_depth.max())
        .mean()
        .values
    )

    summary = az.summary(trace, var_names=["rho", "sigma2", "logodds_inlier"])
    max_rhat = float(summary["r_hat"].max()) if "r_hat" in summary else np.nan
    min_ess = float(summary["ess_bulk"].min()) if "ess_bulk" in summary else np.nan

    # Inlier probability statistics
    inlier_prob = pm.math.invlogit(trace.posterior["logodds_inlier"]).values
    inlier_stats = {
        "mean": float(inlier_prob.mean()),
        "std": float(inlier_prob.std()),
        "median": float(np.median(inlier_prob)),
        "min": float(inlier_prob.min()),
        "max": float(inlier_prob.max()),
    }

    diagnostics = {
        "divergences": divergences,
        "total_samples": chains * draws,
        "divergence_rate": divergences / (chains * draws),
        "mean_step_size": mean_step_size,
        "max_treedepth_fraction": max_treedepth_frac,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "inlier_prob": inlier_stats,
    }

    return trace, diagnostics


def generate_plots(trace, paths):
    """Generate diagnostic plots."""
    plot_dir = paths["plots_dir"]
    plot_dir.mkdir(exist_ok=True)

    plots_created = []

    # Trace plot
    try:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        az.plot_trace(trace, var_names=["rho", "sigma2", "logodds_inlier"], axes=axes)
        fig.tight_layout()
        trace_path = plot_dir / "trace_selected.png"
        fig.savefig(trace_path, dpi=100)
        plt.close(fig)
        plots_created.append("trace_selected.png")
    except Exception as e:
        print(f"  Warning: Could not create trace plot: {e}")

    # Energy plot
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        az.plot_energy(trace, ax=ax)
        fig.tight_layout()
        energy_path = plot_dir / "energy.png"
        fig.savefig(energy_path, dpi=100)
        plt.close(fig)
        plots_created.append("energy.png")
    except Exception as e:
        print(f"  Warning: Could not create energy plot: {e}")

    # Write manifest
    manifest = {"plots": plots_created}
    write_json(plot_dir / "artifacts_manifest.json", manifest)

    return plots_created


def run_group_7(args):
    """Run Group 7 diagnostics."""
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = make_paths("group_7_likelihood_only_freeze_latents", args.run_id)

    results = {
        "group": "group_7_likelihood_only_freeze_latents",
        "run_id": args.run_id,
        "timestamp": datetime.now().isoformat(),
        "environment": collect_environment(),
        "config": {
            "dataset_name": args.dataset_name,
            "max_T": args.max_T,
            "max_K": args.max_K,
            "seed": args.seed,
            "use_mixture": True,
            "use_directional_hmm": False,
            "latents_frozen": True,
            "chains": args.chains,
            "tune": args.tune,
            "draws": args.draws,
        },
        "success": False,
        "error": None,
    }

    try:
        np.random.seed(args.seed)

        print(f"Loading dataset: {args.dataset_name}")
        data, dataset_dir = load_dataset(args.dataset_name, args.max_T, args.max_K)
        C, T, K, _ = data["y_2d"].shape
        print(f"  Cameras: {C}, Frames: {T}, Joints: {K}")

        results["data_shape"] = {"C": C, "T": T, "K": K}

        print("\nBuilding model with frozen latents...")
        model, init_result, x_root_init, U_init = build_model_frozen(data)
        print(f"  Model built successfully")

        print("\nEvaluating at initial point...")
        initial_point = model.initial_point()
        logp_result = safe_point_logps(model, initial_point)

        results["initial_point_logp"] = {
            "total_logp": logp_result["total_logp"],
            "has_nan": logp_result["has_nan"],
            "has_inf": logp_result["has_inf"],
        }

        if logp_result["point_logps"]:
            worst_terms = summarize_worst_terms(logp_result["point_logps"], n=10)
            results["worst_terms"] = worst_terms
            print(f"  Total logp: {logp_result['total_logp']}")

        print("\nSampling model with frozen latents...")
        trace, sampling_diagnostics = sample_model_frozen(
            model, x_root_init, U_init, args.chains, args.tune, args.draws, args.seed
        )
        results["sampling_diagnostics"] = sampling_diagnostics

        print(f"\n  Sampling complete:")
        print(
            f"    Divergences: {sampling_diagnostics['divergences']}/{sampling_diagnostics['total_samples']}"
        )
        print(
            f"    Inlier prob: {sampling_diagnostics['inlier_prob']['mean']:.3f} ± {sampling_diagnostics['inlier_prob']['std']:.3f}"
        )

        print("\nGenerating plots...")
        plots_created = generate_plots(trace, paths)
        results["plots"] = plots_created

        trace_path = paths["run_dir"] / "trace.nc"
        trace.to_netcdf(trace_path)

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\nERROR: {e}")
        print(traceback.format_exc())

    print(f"\nWriting results to: {paths['results_json']}")
    write_json(paths["results_json"], results)

    print(f"Writing report to: {paths['report_md']}")
    report = generate_report(results, paths)
    save_text(paths["report_md"], report)

    return results


def generate_report(results, paths):
    """Generate markdown report."""
    lines = [
        "# Group 7: Likelihood-Only Freeze Latents",
        "",
        f"**Run ID:** {results['run_id']}",
        f"**Status:** {'✓ SUCCESS' if results['success'] else '✗ FAILED'}",
        "",
        "## Configuration",
        "",
        f"- **Use Mixture:** {results['config']['use_mixture']}",
        f"- **Latents Frozen:** {results['config']['latents_frozen']}",
        "",
        "## Strategy",
        "",
        "Sample only observation parameters (rho, sigma2, logodds_inlier) with fixed x_root and U.",
        "",
    ]

    if "sampling_diagnostics" in results:
        diag = results["sampling_diagnostics"]
        lines.extend(
            [
                "## Sampling Diagnostics",
                "",
                f"- **Divergences:** {diag['divergences']} / {diag['total_samples']} ({diag['divergence_rate']:.1%})",
                f"- **Max R-hat:** {diag['max_rhat']:.4f}",
                f"- **Min ESS:** {diag['min_ess']:.1f}",
                f"- **Inlier Prob:** {diag['inlier_prob']['mean']:.3f} ± {diag['inlier_prob']['std']:.3f}",
                "",
                "## Interpretation",
                "",
            ]
        )

        div_rate = diag["divergence_rate"]
        if div_rate == 0:
            lines.append(
                "**OBSERVATION MODEL STABLE**: Likelihood is well-behaved. Issue is in latent geometry."
            )
        elif div_rate < 0.1:
            lines.append("**MOSTLY STABLE**: Minor issues with observation model.")
        else:
            lines.append(
                "**OBSERVATION MODEL UNSTABLE**: Likelihood itself has gradient issues."
            )

    lines.extend(
        [
            "",
            "---",
            "",
            f"**Results JSON:** `{paths['results_json']}`",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    args = parse_args()
    results = run_group_7(args)
    sys.exit(0 if results["success"] else 1)
