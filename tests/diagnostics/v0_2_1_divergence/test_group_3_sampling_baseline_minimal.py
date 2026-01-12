"""Group 3: Baseline Sampling - Smallest Model

Tests the core kinematics/likelihood without mixture or HMM.
Config: use_mixture=False, use_directional_hmm=False
Truncated to max_T=80, max_K=8 (full skeleton if fewer joints)

Usage:
    pixi run python test_group_3_sampling_baseline_minimal.py --run_id baseline_test
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
    parser = argparse.ArgumentParser(
        description="Group 3: Baseline Sampling - Minimal Model"
    )
    parser.add_argument(
        "--dataset_name", default="v0.2.1_L00_minimal", help="Dataset name"
    )
    parser.add_argument(
        "--max_T", type=int, default=80, help="Maximum number of time frames"
    )
    parser.add_argument(
        "--max_K",
        type=int,
        default=None,
        help="Maximum number of joints (None = use all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chains", type=int, default=2, help="Number of MCMC chains")
    parser.add_argument("--tune", type=int, default=300, help="Tuning iterations")
    parser.add_argument("--draws", type=int, default=100, help="Draw iterations")
    parser.add_argument(
        "--run_id", default=None, help="Run identifier (default: ISO timestamp)"
    )
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

    # Subsample if requested
    if max_T is not None:
        data["y_2d"] = data["y_2d"][:, :max_T, :, :]
    if max_K is not None:
        data["y_2d"] = data["y_2d"][:, :, :max_K, :]
        data["parents"] = data["parents"][:max_K]
        data["joint_names"] = data["joint_names"][:max_K]

    return data, dataset_dir


def build_model(data):
    """Build minimal PyMC model: no mixture, no HMM."""
    C, T, K, _ = data["y_2d"].shape

    # Initialize using library estimator
    print("  Initializing from observations (DLT triangulation)...")
    from gimbal.fit_params import initialize_from_observations_dlt

    init_result = initialize_from_observations_dlt(
        y_observed=data["y_2d"],
        camera_proj=data["camera_proj"],
        parents=data["parents"],
    )

    print(f"  Initialization complete")
    print(
        f"    Triangulation rate: {init_result.metadata.get('triangulation_rate', 'N/A'):.2%}"
    )

    # Build minimal model
    with pm.Model() as model:
        gimbal.build_camera_observation_model(
            y_observed=data["y_2d"],
            camera_proj=data["camera_proj"],
            parents=data["parents"],
            init_result=init_result,
            use_mixture=False,  # NO MIXTURE
            image_size=(1280, 720),
            use_directional_hmm=False,  # NO HMM
            validate_init_points=False,
        )

    return model, init_result


def sample_model(model, chains, tune, draws, seed):
    """Sample the model and return trace + diagnostics."""
    print(f"\n  Sampling: chains={chains}, tune={tune}, draws={draws}")

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            return_inferencedata=True,
        )

    # Extract diagnostics
    divergences = int(trace.sample_stats.diverging.sum().values)
    mean_step_size = float(trace.sample_stats.step_size.mean().values)
    max_treedepth_frac = float(
        (trace.sample_stats.tree_depth == trace.sample_stats.tree_depth.max())
        .mean()
        .values
    )

    # Compute R-hat and ESS for selected variables
    summary = az.summary(trace, var_names=["rho", "sigma2", "obs_sigma"])
    max_rhat = float(summary["r_hat"].max()) if "r_hat" in summary else np.nan
    min_ess = float(summary["ess_bulk"].min()) if "ess_bulk" in summary else np.nan

    diagnostics = {
        "divergences": divergences,
        "total_samples": chains * draws,
        "divergence_rate": divergences / (chains * draws),
        "mean_step_size": mean_step_size,
        "max_treedepth_fraction": max_treedepth_frac,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
    }

    return trace, diagnostics


def generate_plots(trace, paths):
    """Generate diagnostic plots."""
    plot_dir = paths["plots_dir"]
    plot_dir.mkdir(exist_ok=True)

    plots_created = []

    # Trace plot for selected variables
    try:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        az.plot_trace(trace, var_names=["rho", "sigma2"], axes=axes)
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

    # Divergence scatter (if any divergences)
    if trace.sample_stats.diverging.sum() > 0:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            div_mask = trace.sample_stats.diverging.values.flatten()
            # Plot first two parameters
            param_names = list(trace.posterior.data_vars)
            if len(param_names) >= 2:
                p1 = trace.posterior[param_names[0]].values.flatten()
                p2 = trace.posterior[param_names[1]].values.flatten()
                ax.scatter(
                    p1[~div_mask], p2[~div_mask], alpha=0.3, label="Normal", s=10
                )
                ax.scatter(
                    p1[div_mask],
                    p2[div_mask],
                    color="red",
                    alpha=0.8,
                    label="Divergent",
                    s=20,
                )
                ax.set_xlabel(param_names[0])
                ax.set_ylabel(param_names[1])
                ax.legend()
                fig.tight_layout()
                div_path = plot_dir / "divergence_pairs.png"
                fig.savefig(div_path, dpi=100)
                plt.close(fig)
                plots_created.append("divergence_pairs.png")
        except Exception as e:
            print(f"  Warning: Could not create divergence plot: {e}")

    # Write manifest
    manifest = {"plots": plots_created}
    write_json(plot_dir / "artifacts_manifest.json", manifest)

    return plots_created


def run_group_3(args):
    """Run Group 3 diagnostics."""
    # Setup paths
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = make_paths("group_3_sampling_baseline_minimal", args.run_id)

    # Initialize results
    results = {
        "group": "group_3_sampling_baseline_minimal",
        "run_id": args.run_id,
        "timestamp": datetime.now().isoformat(),
        "environment": collect_environment(),
        "config": {
            "dataset_name": args.dataset_name,
            "max_T": args.max_T,
            "max_K": args.max_K,
            "seed": args.seed,
            "use_mixture": False,
            "use_directional_hmm": False,
            "chains": args.chains,
            "tune": args.tune,
            "draws": args.draws,
        },
        "success": False,
        "error": None,
    }

    try:
        # Set seed
        np.random.seed(args.seed)

        # Load dataset
        print(f"Loading dataset: {args.dataset_name}")
        data, dataset_dir = load_dataset(args.dataset_name, args.max_T, args.max_K)
        C, T, K, _ = data["y_2d"].shape
        print(f"  Cameras: {C}, Frames: {T}, Joints: {K}")

        results["data_shape"] = {"C": C, "T": T, "K": K}

        # Build model
        print("\nBuilding minimal PyMC model (no mixture, no HMM)...")
        model, init_result = build_model(data)
        print(f"  Model built successfully")

        # Evaluate at initial point
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

        # Compile gradients
        print("\nCompiling gradients...")
        logp_fn, dlogp_fn = compile_logp_and_grad(model)
        grad_array = dlogp_fn(initial_point)
        grad_l2 = float(np.linalg.norm(grad_array))
        grad_linf = float(np.max(np.abs(grad_array)))

        results["gradient_stats"] = {
            "grad_norm_l2": grad_l2,
            "grad_norm_linf": grad_linf,
            "nan_count": int(np.sum(np.isnan(grad_array))),
            "inf_count": int(np.sum(np.isinf(grad_array))),
        }
        print(f"  Gradient L2 norm: {grad_l2:.2e}")
        print(f"  NaN count: {results['gradient_stats']['nan_count']}")

        # Sample
        print("\nSampling model...")
        trace, sampling_diagnostics = sample_model(
            model, args.chains, args.tune, args.draws, args.seed
        )
        results["sampling_diagnostics"] = sampling_diagnostics

        print(f"\n  Sampling complete:")
        print(
            f"    Divergences: {sampling_diagnostics['divergences']}/{sampling_diagnostics['total_samples']}"
        )
        print(f"    Mean step size: {sampling_diagnostics['mean_step_size']:.6f}")
        print(f"    Max R-hat: {sampling_diagnostics['max_rhat']:.4f}")
        print(f"    Min ESS: {sampling_diagnostics['min_ess']:.1f}")

        # Generate plots
        print("\nGenerating plots...")
        plots_created = generate_plots(trace, paths)
        results["plots"] = plots_created
        print(f"  Created {len(plots_created)} plots")

        # Save trace
        trace_path = paths["run_dir"] / "trace.nc"
        trace.to_netcdf(trace_path)
        print(f"  Trace saved to {trace_path}")

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\nERROR: {e}")
        print(traceback.format_exc())

    # Write results JSON
    print(f"\nWriting results to: {paths['results_json']}")
    write_json(paths["results_json"], results)

    # Write markdown report
    print(f"Writing report to: {paths['report_md']}")
    report = generate_report(results, paths)
    save_text(paths["report_md"], report)

    return results


def generate_report(results, paths):
    """Generate markdown report."""
    lines = [
        "# Group 3: Baseline Sampling - Minimal Model",
        "",
        f"**Run ID:** {results['run_id']}",
        f"**Timestamp:** {results['timestamp']}",
        f"**Status:** {'✓ SUCCESS' if results['success'] else '✗ FAILED'}",
        "",
        "## Configuration",
        "",
        f"- **Dataset:** {results['config']['dataset_name']}",
        f"- **Max T:** {results['config']['max_T']}",
        f"- **Max K:** {results['config']['max_K']}",
        f"- **Use Mixture:** {results['config']['use_mixture']}",
        f"- **Use Directional HMM:** {results['config']['use_directional_hmm']}",
        f"- **Chains:** {results['config']['chains']}",
        f"- **Tune:** {results['config']['tune']}",
        f"- **Draws:** {results['config']['draws']}",
        "",
    ]

    if "data_shape" in results:
        lines.extend(
            [
                "## Data Shape",
                "",
                f"- **Cameras:** {results['data_shape']['C']}",
                f"- **Frames:** {results['data_shape']['T']}",
                f"- **Joints:** {results['data_shape']['K']}",
                "",
            ]
        )

    if "initial_point_logp" in results:
        logp = results["initial_point_logp"]
        lines.extend(
            [
                "## Initial Point Log-Probability",
                "",
                f"- **Total logp:** {logp['total_logp']}",
                f"- **Has NaN:** {logp['has_nan']}",
                f"- **Has Inf:** {logp['has_inf']}",
                "",
            ]
        )

    if results.get("worst_terms"):
        lines.extend(
            [
                "## Worst 10 Log-Probability Terms",
                "",
                format_worst_terms_table(results["worst_terms"]),
                "",
            ]
        )

    if "gradient_stats" in results:
        stats = results["gradient_stats"]
        lines.extend(
            [
                "## Gradient Statistics",
                "",
                f"- **L2 Norm:** {stats['grad_norm_l2']:.6e}",
                f"- **L-infinity Norm:** {stats['grad_norm_linf']:.6e}",
                f"- **NaN Count:** {stats['nan_count']}",
                f"- **Inf Count:** {stats['inf_count']}",
                "",
            ]
        )

    if "sampling_diagnostics" in results:
        diag = results["sampling_diagnostics"]
        lines.extend(
            [
                "## Sampling Diagnostics",
                "",
                f"- **Divergences:** {diag['divergences']} / {diag['total_samples']} ({diag['divergence_rate']:.1%})",
                f"- **Mean Step Size:** {diag['mean_step_size']:.6f}",
                f"- **Max Treedepth Fraction:** {diag['max_treedepth_fraction']:.2%}",
                f"- **Max R-hat:** {diag['max_rhat']:.4f}",
                f"- **Min ESS:** {diag['min_ess']:.1f}",
                "",
            ]
        )

    if results.get("plots"):
        lines.extend(
            [
                "## Plots",
                "",
            ]
        )
        for plot in results["plots"]:
            lines.append(f"- [{plot}]({paths['plots_dir'].name}/{plot})")
        lines.append("")

    # Interpretation
    lines.extend(
        [
            "## Interpretation",
            "",
        ]
    )

    if results.get("sampling_diagnostics"):
        div_rate = results["sampling_diagnostics"]["divergence_rate"]
        max_rhat = results["sampling_diagnostics"]["max_rhat"]

        if div_rate == 0 and max_rhat < 1.05:
            interp = "**STABLE**: Baseline model without mixture or HMM converges successfully. Divergences likely originate from added complexity."
        elif div_rate < 0.1 and max_rhat < 1.1:
            interp = "**MOSTLY STABLE**: Few divergences detected. Core parameterization is reasonable but may benefit from tuning."
        else:
            interp = "**UNSTABLE**: High divergence rate indicates core kinematics/likelihood parameterization is ill-conditioned. Fundamental reparameterization needed."

        lines.append(interp)
    else:
        lines.append("Sampling did not complete successfully.")

    lines.extend(
        [
            "",
            "## Next Action",
            "",
        ]
    )

    if results.get("sampling_diagnostics"):
        div_rate = results["sampling_diagnostics"]["divergence_rate"]
        if div_rate < 0.1:
            lines.append(
                "Proceed to Group 4 (mixture only) to isolate mixture contribution."
            )
        else:
            lines.append(
                "Debug core model parameterization before testing mixture or HMM. Check bone priors, temporal dynamics, and observation likelihood scale."
            )
    else:
        lines.append("Fix build/initialization errors before sampling.")

    lines.extend(
        [
            "",
            "---",
            "",
            f"**Results JSON:** `{paths['results_json']}`",
            f"**Trace:** `{paths['run_dir']}/trace.nc`",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    args = parse_args()
    results = run_group_3(args)

    if results["success"]:
        print("\n" + "=" * 80)
        print("GROUP 3: PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("GROUP 3: FAILED")
        print("=" * 80)
        sys.exit(1)
