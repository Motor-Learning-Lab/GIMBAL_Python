"""Group 4: Sampling Mixture Only

Tests mixture observation likelihood without HMM.
Config: use_mixture=True, use_directional_hmm=False
Truncated to max_T=80

Usage:
    pixi run python test_group_4_sampling_mixture_only.py --run_id mixture_test
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
    parser = argparse.ArgumentParser(description="Group 4: Sampling Mixture Only")
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


def build_model(data):
    """Build model with mixture but no HMM."""
    C, T, K, _ = data["y_2d"].shape

    print("  Initializing from observations (DLT triangulation)...")
    from gimbal.fit_params import initialize_from_observations_dlt

    init_result = initialize_from_observations_dlt(
        y_observed=data["y_2d"],
        camera_proj=data["camera_proj"],
        parents=data["parents"],
    )

    print(f"  Initialization complete")

    # Build model with mixture
    with pm.Model() as model:
        gimbal.build_camera_observation_model(
            y_observed=data["y_2d"],
            camera_proj=data["camera_proj"],
            parents=data["parents"],
            init_result=init_result,
            use_mixture=True,  # MIXTURE ENABLED
            image_size=(1280, 720),
            use_directional_hmm=False,  # NO HMM
            validate_init_points=False,
        )

    return model, init_result


def sample_model(model, chains, tune, draws, seed):
    """Sample the model."""
    print(f"\n  Sampling: chains={chains}, tune={tune}, draws={draws}")

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            return_inferencedata=True,
        )

    divergences = int(trace.sample_stats.diverging.sum().values)
    mean_step_size = float(trace.sample_stats.step_size.mean().values)
    max_treedepth_frac = float(
        (trace.sample_stats.tree_depth == trace.sample_stats.tree_depth.max())
        .mean()
        .values
    )

    summary = az.summary(
        trace, var_names=["rho", "sigma2", "obs_sigma", "logodds_inlier"]
    )
    max_rhat = float(summary["r_hat"].max()) if "r_hat" in summary else np.nan
    min_ess = float(summary["ess_bulk"].min()) if "ess_bulk" in summary else np.nan

    # Extract mixture-specific diagnostics
    logodds_samples = trace.posterior["logodds_inlier"].values.flatten()
    inlier_prob_samples = 1 / (1 + np.exp(-logodds_samples))

    diagnostics = {
        "divergences": divergences,
        "total_samples": chains * draws,
        "divergence_rate": divergences / (chains * draws),
        "mean_step_size": mean_step_size,
        "max_treedepth_fraction": max_treedepth_frac,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "inlier_prob_posterior": {
            "mean": float(inlier_prob_samples.mean()),
            "std": float(inlier_prob_samples.std()),
            "median": float(np.median(inlier_prob_samples)),
            "min": float(inlier_prob_samples.min()),
            "max": float(inlier_prob_samples.max()),
        },
    }

    return trace, diagnostics


def generate_plots(trace, paths):
    """Generate diagnostic plots."""
    plot_dir = paths["plots_dir"]
    plot_dir.mkdir(exist_ok=True)

    plots_created = []

    # Trace plot
    try:
        fig, axes = plt.subplots(4, 2, figsize=(12, 10))
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


def run_group_4(args):
    """Run Group 4 diagnostics."""
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = make_paths("group_4_sampling_mixture_only", args.run_id)

    results = {
        "group": "group_4_sampling_mixture_only",
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

        print("\nBuilding model with mixture (no HMM)...")
        model, init_result = build_model(data)
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

        print("\nSampling model...")
        trace, sampling_diagnostics = sample_model(
            model, args.chains, args.tune, args.draws, args.seed
        )
        results["sampling_diagnostics"] = sampling_diagnostics

        print(f"\n  Sampling complete:")
        print(
            f"    Divergences: {sampling_diagnostics['divergences']}/{sampling_diagnostics['total_samples']}"
        )
        print(
            f"    Inlier prob posterior: {sampling_diagnostics['inlier_prob_posterior']['mean']:.3f} ± {sampling_diagnostics['inlier_prob_posterior']['std']:.3f}"
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
        "# Group 4: Sampling Mixture Only",
        "",
        f"**Run ID:** {results['run_id']}",
        f"**Status:** {'✓ SUCCESS' if results['success'] else '✗ FAILED'}",
        "",
        "## Configuration",
        "",
        f"- **Use Mixture:** {results['config']['use_mixture']}",
        f"- **Use Directional HMM:** {results['config']['use_directional_hmm']}",
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
                f"- **Inlier Prob (posterior):** {diag['inlier_prob_posterior']['mean']:.3f} ± {diag['inlier_prob_posterior']['std']:.3f}",
                "",
                "## Interpretation",
                "",
            ]
        )

        div_rate = diag["divergence_rate"]
        if div_rate == 0:
            lines.append(
                "**STABLE WITH MIXTURE**: Mixture model converges successfully."
            )
        elif div_rate < 0.1:
            lines.append("**MOSTLY STABLE**: Mixture introduces minor issues.")
        else:
            lines.append(
                "**UNSTABLE WITH MIXTURE**: Mixture parameterization or masking causes divergences."
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
    results = run_group_4(args)
    sys.exit(0 if results["success"] else 1)
