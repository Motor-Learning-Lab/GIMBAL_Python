"""Stage H: Fitting/Sampling

Runs PyMC sampling on the model built in Stage G.

Inputs:
- Stage B: y_2d_clean.npz
- Stage D: x_3d_clean.npz
- Stage F: priors.json
- L00_minimal: ground_truth.npz

Outputs:
- sampling_metrics.json
- trace.nc (InferenceData)
- trace_plots/ (convergence diagnostics)

Criteria (from Q7):
- Convergence: r_hat < 1.05 for all parameters
- No divergences
- Effective sample size > 100
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

import gimbal


def run_stage_h(dataset_dir: Path, fits_dir: Path, output_dir: Path) -> dict:
    """Run Stage H: Fitting/Sampling."""

    print("=" * 80)
    print(" " * 30 + "STAGE H: Fitting/Sampling")
    print("=" * 80)

    # Load preprocessing outputs
    print("\n[1/5] Loading preprocessing outputs...")
    data = {}

    # From dataset: Load metadata
    dataset_path = dataset_dir / "dataset.npz"
    with np.load(dataset_path, allow_pickle=True) as f:
        data["camera_proj"] = f["camera_proj"]
        data["parents"] = f["parents"]
        data["bone_lengths"] = f["bone_lengths"]
        data["joint_names"] = [
            str(name) for name in f["joint_names"]
        ]  # Convert to list of strings

    # From Stage B: cleaned 2D
    y_2d_clean_path = output_dir / "y_2d_clean.npz"
    with np.load(y_2d_clean_path) as f:
        data["y_2d_clean"] = f["y_2d_clean"]
        data["mask_clean"] = f["valid_mask_2d"]

    # From Stage D: cleaned 3D
    x_3d_clean_path = output_dir / "x_3d_clean.npz"
    with np.load(x_3d_clean_path) as f:
        data["x_3d_clean"] = f["x_3d_clean"]

    # From Stage C: original triangulation
    x_3d_triangulated_path = output_dir / "x_3d_triangulated.npz"
    with np.load(x_3d_triangulated_path) as f:
        data["x_3d_triangulated"] = f["x_3d"]

    # From Stage F: priors
    priors_path = output_dir / "priors.json"
    with open(priors_path, "r") as f:
        prior_config = json.load(f)
        data["priors"] = prior_config["priors"]

    # From dataset: ground truth for later comparison
    gt_path = dataset_dir / "dataset.npz"
    with np.load(gt_path) as f:
        data["x_gt"] = f["x_true"]  # Ground truth 3D
        data["y_gt"] = f["y_2d"]  # Noisy observations
        data["true_states"] = f.get("z_true")  # HMM states if available

    print(f"  Loaded 2D observations: {data['y_2d_clean'].shape}")
    print(f"  Cameras: {data['camera_proj'].shape[0]}")
    print(f"  Joints: {data['x_3d_clean'].shape[1]}")
    print(f"  Priors available for: {list(data['priors'].keys())}")

    # Initialize from cleaned 3D (same as Stage G)
    print("\n[2/5] Computing initialization...")
    T, K, _ = data["x_3d_clean"].shape
    x_init = data["x_3d_clean"]

    # Compute bone lengths from initial 3D
    bone_lengths_init = []
    for k in range(1, K):
        parent_idx = data["parents"][k]
        if parent_idx >= 0:
            bones = x_init[:, k, :] - x_init[:, parent_idx, :]
            lengths = np.sqrt(np.sum(bones**2, axis=1))
            bone_lengths_init.append(np.nanmean(lengths))

    rho_init = np.array(bone_lengths_init)

    from gimbal.fit_params import InitializationResult

    init_result = InitializationResult(
        x_init=x_init,
        eta2=np.array([1.0] * K),
        rho=rho_init,
        sigma2=rho_init * 0.01,
        u_init=np.zeros((T, K, 3)),
        obs_sigma=2.0,
        inlier_prob=0.95,
        metadata={"method": "stage_d_cleaned_3d", "source": "stage_d_clean_3d.py"},
    )
    print("  Initialization complete")

    # Rebuild model (same as Stage G)
    print("\n[3/5] Rebuilding PyMC model...")
    with pm.Model() as model:
        gimbal.build_camera_observation_model(
            y_observed=data["y_2d_clean"],
            camera_proj=data["camera_proj"],
            parents=data["parents"],
            init_result=init_result,
            use_mixture=True,
            image_size=(1280, 720),
            use_directional_hmm=False,
            validate_init_points=False,
        )

        # Extract variables
        U = model["U"]
        log_obs_t = model["log_obs_t"]

        # Add Stage 3 directional HMM with data-driven priors
        gimbal.add_directional_hmm_prior(
            U=U,
            log_obs_t=log_obs_t,
            S=1,  # K=1 HMM
            prior_config=data["priors"],
            joint_names=data["joint_names"],
            kappa_scale=1.0,
        )

        print("  Model rebuilt successfully")
        print(f"    Free RVs: {len(model.free_RVs)}")

    # Sample
    print("\n[4/5] Sampling...")
    print("  Configuration:")
    print("    - Draws: 200")
    print("    - Chains: 2")
    print("    - Tuning: 500")
    print("    - Sampler: PyMC default (NUTS)")

    start_time = time.time()

    try:
        with model:
            trace = pm.sample(
                draws=200,
                tune=500,
                chains=2,
                cores=2,
                return_inferencedata=True,
                progressbar=True,
                random_seed=42,
            )

        elapsed = time.time() - start_time
        print(f"\n  Sampling complete in {elapsed:.1f}s")

    except Exception as e:
        print(f"\n  Sampling failed: {e}")
        raise

    # Basic diagnostics
    print("\n[5/5] Computing diagnostics...")

    # Convergence: r_hat
    rhat_summary = az.rhat(trace)
    rhat_values = []
    for var_name in rhat_summary.data_vars:
        vals = rhat_summary[var_name].values.flatten()
        rhat_values.extend([float(v) for v in vals if not np.isnan(v)])

    max_rhat = max(rhat_values) if rhat_values else np.nan
    mean_rhat = np.mean(rhat_values) if rhat_values else np.nan
    n_rhat_high = sum(1 for r in rhat_values if r > 1.05)

    print(f"  R-hat:")
    print(f"    Max: {max_rhat:.4f}")
    print(f"    Mean: {mean_rhat:.4f}")
    print(f"    Count > 1.05: {n_rhat_high}/{len(rhat_values)}")

    # Divergences
    divergences = trace.sample_stats.diverging.values.sum()
    print(f"  Divergences: {divergences}")

    # Effective sample size
    ess_summary = az.ess(trace)
    ess_values = []
    for var_name in ess_summary.data_vars:
        vals = ess_summary[var_name].values.flatten()
        ess_values.extend([float(v) for v in vals if not np.isnan(v)])

    min_ess = min(ess_values) if ess_values else np.nan
    mean_ess = np.mean(ess_values) if ess_values else np.nan
    n_ess_low = sum(1 for e in ess_values if e < 100)

    print(f"  Effective Sample Size:")
    print(f"    Min: {min_ess:.0f}")
    print(f"    Mean: {mean_ess:.0f}")
    print(f"    Count < 100: {n_ess_low}/{len(ess_values)}")

    # Save trace
    trace_path = output_dir / "trace.nc"
    trace.to_netcdf(trace_path)
    print(f"\n  Trace saved to {trace_path}")

    # Generate trace plots
    plot_dir = output_dir / "trace_plots"
    plot_dir.mkdir(exist_ok=True)

    print(f"\n  Generating trace plots...")

    # Plot key parameters
    key_params = ["eta2_root", "rho", "sigma2", "obs_sigma", "inlier_prob"]
    for param in key_params:
        if param in trace.posterior:
            try:
                fig = az.plot_trace(trace, var_names=[param], figsize=(12, 4))
                if hasattr(fig, "suptitle"):
                    fig.suptitle(f"Trace: {param}")
                plt.tight_layout()
                plt.savefig(
                    plot_dir / f"trace_{param}.png", dpi=150, bbox_inches="tight"
                )
                plt.close("all")
            except Exception as e:
                print(f"    Warning: Could not plot {param}: {e}")

    print(f"    Saved {len(key_params)} trace plots to {plot_dir}/")

    # Compile metrics
    full_metrics = {
        "stage": "H_fitting",
        "sampling_config": {
            "draws": 200,
            "tune": 500,
            "chains": 2,
            "sampler": "PyMC_NUTS",
        },
        "convergence": {
            "max_rhat": float(max_rhat),
            "mean_rhat": float(mean_rhat),
            "n_rhat_high": int(n_rhat_high),
            "total_params": len(rhat_values),
            "divergences": int(divergences),
            "min_ess": float(min_ess),
            "mean_ess": float(mean_ess),
            "n_ess_low": int(n_ess_low),
        },
        "timing": {
            "sampling_seconds": float(elapsed),
            "draws_per_second": (
                400 / elapsed if elapsed > 0 else 0
            ),  # 200 draws * 2 chains
        },
        "criteria_Q7": {
            "rhat_pass": bool(max_rhat < 1.05 if not np.isnan(max_rhat) else False),
            "divergence_pass": bool(divergences == 0),
            "ess_pass": bool(min_ess > 100 if not np.isnan(min_ess) else False),
        },
        "timestamp": datetime.now().isoformat(),
        "outputs": {"trace": str(trace_path), "trace_plots": str(plot_dir)},
    }

    # Save metrics
    metrics_path = output_dir / "sampling_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)

    # Pass/fail
    all_pass = (
        full_metrics["criteria_Q7"]["rhat_pass"]
        and full_metrics["criteria_Q7"]["divergence_pass"]
        and full_metrics["criteria_Q7"]["ess_pass"]
    )

    print(f"\nStage H complete. Metrics saved to {metrics_path}")

    print("\n" + "=" * 80)
    if all_pass:
        print(" " * 30 + "STAGE H: PASSED")
    else:
        print(" " * 30 + "STAGE H: FAILED")
        if not full_metrics["criteria_Q7"]["rhat_pass"]:
            print("  - R-hat criterion failed (max > 1.05)")
        if not full_metrics["criteria_Q7"]["divergence_pass"]:
            print(f"  - Divergence criterion failed ({divergences} divergences)")
        if not full_metrics["criteria_Q7"]["ess_pass"]:
            print("  - ESS criterion failed (min < 100)")
    print("=" * 80)

    return full_metrics


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    dataset_dir = base_dir / "tests" / "pipeline" / "datasets" / "v0.2.1_L00_minimal"
    fits_dir = base_dir / "tests" / "pipeline" / "fits"
    output_dir = fits_dir / "v0.2.1_L00_minimal"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run stage
    metrics = run_stage_h(dataset_dir, fits_dir, output_dir)
