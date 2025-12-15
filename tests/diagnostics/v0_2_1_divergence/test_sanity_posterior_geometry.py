"""
Posterior Geometry Diagnostic Test - Group Sanity

Purpose:
    Analyze the posterior geometry of the camera observation model to understand
    what structural features cause the consistent 100% divergence rate observed
    across all model configurations.

Test Plan Reference:
    See plans/HMM 3 stage plan.md, divergence analysis section

Methodology:
    1. Build and sample from the model with reasonable tuning parameters
    2. Generate multiple diagnostic visualizations using ArviZ:
       - Trace plots (parameter evolution)
       - Energy plots (sampler diagnostics)
       - Posterior scatter plots (parameter correlations)
       - Divergence locations in parameter space
    3. Analyze what posterior geometry features cause NUTS to diverge
    4. Identify which parameters are most problematic

Expected Output:
    - Multiple PNG diagnostic plots in plots/sanity_posterior_geometry/
    - results_sanity_posterior_geometry.json with metrics
    - report_sanity_posterior_geometry.md with interpretation
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend - MUST be before pyplot import

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm

# Add repo root to path for imports
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from test_utils import get_standard_synth_data
from gimbal.fit_params import initialize_from_observations_dlt
from gimbal.pymc_model import _build_camera_observation_model_full

warnings.filterwarnings("ignore", category=UserWarning)


def run_geometry_analysis():
    """
    Run posterior geometry analysis test.

    Returns
    -------
    dict
        Metrics including divergence count, sampling time, and diagnostic info
    """

    print("=" * 70)
    print("POSTERIOR GEOMETRY ANALYSIS - Group Sanity")
    print("=" * 70)
    print()

    # Setup output directories
    test_name = "sanity_posterior_geometry"
    diagnostics_dir = Path("plots") / test_name
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Load synthetic data
    print("Loading synthetic data...")
    synth_data = get_standard_synth_data(T=100, C=3, S=3, seed=42)
    y_observed = synth_data["observations_uv"]
    camera_proj = synth_data["camera_matrices"]
    parents = synth_data["parents"]
    bone_lengths = synth_data["bone_lengths"]
    C, T, K, _ = y_observed.shape
    print(f"  Cameras: {C}, Frames: {T}, Joints: {K}")

    # Initialize model
    print("Initializing from observations via DLT...")
    init_result = initialize_from_observations_dlt(
        y_observed=y_observed,
        camera_proj=camera_proj,
        parents=parents,
    )
    print(f"  x_init shape: {init_result.x_init.shape}")
    print(f"  eta2 shape: {init_result.eta2.shape}")

    # Build model
    print("Building PyMC model...")
    with pm.Model() as model:
        _build_camera_observation_model_full(
            y_observed=y_observed,
            camera_proj=camera_proj,
            parents=parents,
            init_result=init_result,
            use_mixture=False,  # Simple Gaussian for clearer diagnostics
            prior_hyperparams={
                "eta2_root_sigma": 0.1,
                "rho_sigma": 2.0,
                "sigma2_sigma": 0.1,
            },
        )

    print(f"  Model has {len(model.free_RVs)} free random variables")
    print(f"  Variables: {[rv.name for rv in model.free_RVs]}")
    print()

    # Sample
    print("Sampling from posterior...")
    print("  Configuration: tune=500, draws=50, chains=1")
    print(
        "  Note: Small draws to complete test quickly; diagnostics focus on divergences"
    )

    try:
        import nutpie

        print("  Using nutpie sampler...")
        from gimbal.pymc_utils import compile_model_with_initialization

        compiled_model = compile_model_with_initialization(model, init_result, parents)
        trace = nutpie.sample(
            compiled_model,
            chains=1,
            tune=500,
            draws=50,
            progress_bar=True,
        )
        # Convert to arviz InferenceData
        trace = az.from_nutpie(trace)
    except (ImportError, Exception) as e:
        print(f"  Nutpie failed ({e}), falling back to PyMC sampling...")
        with model:
            trace = pm.sample(
                tune=500,
                draws=50,
                chains=1,
                target_accept=0.8,
                return_inferencedata=True,
                progressbar=True,
            )

    print()
    print("Trace summary:")
    print(f"  Total samples collected: {len(trace.posterior.draw)}")
    print(f"  Divergences: {trace.sample_stats.diverging.sum().values}")
    print(
        f"  Divergence rate: {100.0 * trace.sample_stats.diverging.sum().values / len(trace.posterior.draw):.1f}%"
    )
    print()

    # Extract metrics
    n_divergences = int(trace.sample_stats.diverging.sum().values)
    n_total = len(trace.posterior.draw)
    divergence_rate = 100.0 * n_divergences / n_total

    # Generate diagnostic plots
    print("Generating diagnostic plots...")

    # 1. Trace plots for key parameters
    print("  - Trace plots (eta2_root, obs_sigma, x_root mean)...")
    try:
        fig, ax = plt.subplots(figsize=(14, 10))
        var_names = ["eta2_root", "obs_sigma"]
        az.plot_trace(trace, var_names=var_names, ax=ax)
        plt.tight_layout()
        fig.savefig(
            diagnostics_dir / f"{test_name}_trace_key_params.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception as e:
        print(f"    Warning: Trace plot failed: {e}")

    # 2. Energy plot (NUTS-specific)
    print("  - Energy plot (HMC sampler diagnostics)...")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        az.plot_energy(trace, ax=ax)
        fig.savefig(
            diagnostics_dir / f"{test_name}_energy.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
    except Exception as e:
        print(f"    Warning: Energy plot failed: {e}")

    # 3. Parallel plot showing divergences
    print("  - Parallel coordinates plot (divergences highlighted)...")
    try:
        fig, ax = plt.subplots(figsize=(16, 10))
        az.plot_parallel(trace, var_names=["~log_likelihood"], ax=ax)
        fig.savefig(
            diagnostics_dir / f"{test_name}_parallel.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
    except Exception as e:
        print(f"    Warning: Parallel plot failed: {e}")

    # 4. Posterior scatter plots for key parameter pairs
    print("  - Posterior scatter plots (parameter correlations)...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Extract posterior samples
        posterior = trace.posterior.stack(sample=("chain", "draw"))
        eta2_root = posterior["eta2_root"].values
        obs_sigma = posterior["obs_sigma"].values
        logodds = (
            posterior["logodds_inlier"].values
            if "logodds_inlier" in posterior
            else np.zeros_like(eta2_root)
        )
        diverging = trace.sample_stats.diverging.values.flatten()

        # eta2_root vs obs_sigma
        axes[0, 0].scatter(
            eta2_root[~diverging],
            obs_sigma[~diverging],
            alpha=0.5,
            s=30,
            label="Non-divergent",
            color="blue",
        )
        axes[0, 0].scatter(
            eta2_root[diverging],
            obs_sigma[diverging],
            alpha=0.7,
            s=50,
            label="Divergent",
            color="red",
            marker="x",
        )
        axes[0, 0].set_xlabel("eta2_root")
        axes[0, 0].set_ylabel("obs_sigma")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # eta2_root over iterations
        axes[0, 1].plot(eta2_root, alpha=0.6, label="eta2_root", color="blue")
        axes[0, 1].scatter(
            np.where(diverging)[0],
            eta2_root[diverging],
            color="red",
            s=50,
            marker="x",
            label="Divergences",
            zorder=5,
        )
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("eta2_root")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # obs_sigma over iterations
        axes[1, 0].plot(obs_sigma, alpha=0.6, label="obs_sigma", color="green")
        axes[1, 0].scatter(
            np.where(diverging)[0],
            obs_sigma[diverging],
            color="red",
            s=50,
            marker="x",
            label="Divergences",
            zorder=5,
        )
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("obs_sigma")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Divergence histogram
        axes[1, 1].hist(
            [eta2_root[~diverging], eta2_root[diverging]],
            label=["Non-divergent", "Divergent"],
            bins=10,
            alpha=0.6,
            color=["blue", "red"],
        )
        axes[1, 1].set_xlabel("eta2_root")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(
            diagnostics_dir / f"{test_name}_posterior_scatter.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception as e:
        print(f"    Warning: Scatter plot failed: {e}")

    # 5. R-hat and ESS diagnostics
    print("  - R-hat and ESS diagnostics...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # R-hat
        rhat_vals = az.rhat(trace)
        rhat_data = {}
        for var in rhat_vals.data_vars:
            if rhat_vals[var].size == 1:
                rhat_data[var] = float(rhat_vals[var].values)
            else:
                rhat_data[var] = float(rhat_vals[var].mean().values)

        vars_to_plot = sorted(rhat_data.keys())[:10]  # Top 10 variables
        rhat_values = [rhat_data[v] for v in vars_to_plot]
        axes[0].barh(
            vars_to_plot,
            rhat_values,
            color=["red" if v > 1.05 else "blue" for v in rhat_values],
        )
        axes[0].axvline(1.05, color="orange", linestyle="--", label="Threshold (1.05)")
        axes[0].set_xlabel("R-hat")
        axes[0].set_title("Potential Scale Reduction Factor")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="x")

        # ESS
        ess_tail = az.ess(trace, method="tail")
        ess_data = {}
        for var in ess_tail.data_vars:
            if ess_tail[var].size == 1:
                ess_data[var] = float(ess_tail[var].values)
            else:
                ess_data[var] = float(ess_tail[var].mean().values)

        ess_values = [ess_data[v] for v in vars_to_plot if v in ess_data]
        axes[1].barh(vars_to_plot[: len(ess_values)], ess_values, color="green")
        axes[1].set_xlabel("ESS (tail)")
        axes[1].set_title("Effective Sample Size (Tail)")
        axes[1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        fig.savefig(
            diagnostics_dir / f"{test_name}_diagnostics_rhat_ess.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception as e:
        print(f"    Warning: Diagnostics plot failed: {e}")

    print()
    print("All diagnostic plots saved to:")
    print(f"  {diagnostics_dir.absolute()}")
    print()

    # Prepare results
    results = {
        "test_group": "sanity",
        "description": "Posterior geometry diagnostic analysis",
        "configuration": {
            "model": "camera_observation",
            "use_mixture": False,
            "tune": 500,
            "draws": 50,
            "chains": 1,
            "num_free_variables": len(model.free_RVs),
            "data_shape": {
                "cameras": C,
                "frames": T,
                "joints": K,
            },
        },
        "metrics": {
            "total_samples": n_total,
            "divergences": n_divergences,
            "divergence_rate_percent": divergence_rate,
            "timestamp": datetime.now().isoformat(),
        },
        "diagnostic_plots": [
            f"{test_name}_trace_key_params.png",
            f"{test_name}_energy.png",
            f"{test_name}_parallel.png",
            f"{test_name}_posterior_scatter.png",
            f"{test_name}_diagnostics_rhat_ess.png",
        ],
    }

    # Save results
    results_file = Path(__file__).parent / f"results_{test_name}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    results = run_geometry_analysis()

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Divergence Rate: {results['metrics']['divergence_rate_percent']:.1f}%")
    print()
    print("Next Steps:")
    print("  1. Review diagnostic plots in plots/sanity_posterior_geometry/")
    print("  2. Examine report_sanity_posterior_geometry.md for interpretation")
    print("  3. Consider reparameterization strategies based on findings")
    print()
