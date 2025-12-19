"""
Stage G: Model Building for L00_minimal dataset

Build PyMC model with K=1 HMM and data-driven priors.

NOTE: This stage reveals a current API limitation - bone lengths (rho, sigma2)
are always estimated in the current model builder. Per the API inventory,
estimate_bone_lengths parameter needs to be added but doesn't exist yet.
For L00 validation, we proceed with estimation but note this for future work.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pymc as pm

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gimbal


def load_all_preprocessing_outputs(dataset_dir: Path, fits_dir: Path) -> Dict[str, Any]:
    """Load outputs from all preprocessing stages."""
    # Original dataset
    dataset = np.load(dataset_dir / "dataset.npz", allow_pickle=True)

    # Config
    with open(dataset_dir / "config.json") as f:
        config = json.load(f)

    # Stage B: Cleaned 2D
    stage_b = np.load(fits_dir / "y_2d_clean.npz")

    # Stage D: Cleaned 3D
    stage_d = np.load(fits_dir / "x_3d_clean.npz")

    # Stage F: Priors
    with open(fits_dir / "priors.json") as f:
        priors_data = json.load(f)

    # Extract prior_config in format expected by HMM
    prior_config = {}
    for joint_name, params in priors_data["priors"].items():
        prior_config[joint_name] = {
            "mu_mean": np.array(params["mu_mean"]),
            "mu_sd": params["mu_sd"],
            "kappa_mode": params["kappa_mode"],
            "kappa_sd": params["kappa_sd"],
        }

    return {
        "config": config,
        "y_2d_clean": stage_b["y_2d_clean"],
        "camera_proj": dataset["camera_proj"],
        "parents": dataset["parents"],
        "bone_lengths": dataset["bone_lengths"],
        "joint_names": config["dataset_spec"]["skeleton"]["joint_names"],
        "prior_config": prior_config,
        "x_3d_clean": stage_d["x_3d_clean"],
    }


def run_stage_g(dataset_dir: Path, fits_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Run Stage G: Model Building

    Parameters
    ----------
    dataset_dir : Path
        Directory containing original dataset
    fits_dir : Path
        Directory containing preprocessing outputs (Stages B, D, F)
    output_dir : Path
        Directory for outputs

    Returns
    -------
    metrics : dict
        Model building metadata and configuration
    """
    print("=" * 80)
    print("STAGE G: Model Building")
    print("=" * 80)

    # Load all preprocessing data
    print("[1/4] Loading preprocessing outputs...")
    data = load_all_preprocessing_outputs(dataset_dir, fits_dir)

    print(f"  Loaded 2D observations: {data['y_2d_clean'].shape}")
    print(f"  Cameras: {data['camera_proj'].shape[0]}")
    print(f"  Joints: {len(data['joint_names'])}")
    print(f"  Priors available for: {list(data['prior_config'].keys())}")

    # Initialize from cleaned 3D (use Stage D output directly)
    print("\n[2/4] Computing initialization...")
    # Use the triangulated and cleaned 3D from Stage D
    x_init = data["x_3d_clean"]  # (T, K, 3)

    # Compute bone lengths from initial 3D
    T, K, _ = x_init.shape
    bone_lengths_init = []
    for k in range(1, K):
        parent_idx = data["parents"][k]
        if parent_idx >= 0:
            bones = x_init[:, k, :] - x_init[:, parent_idx, :]
            lengths = np.sqrt(np.sum(bones**2, axis=1))
            bone_lengths_init.append(np.nanmean(lengths))

    rho_init = np.array(bone_lengths_init)

    # Simple initialization
    from gimbal.fit_params import InitializationResult

    init_result = InitializationResult(
        x_init=x_init,
        eta2=np.array([1.0] * K),  # Temporal variances
        rho=rho_init,  # Bone lengths
        sigma2=rho_init * 0.01,  # Bone variances (1% of length)
        u_init=np.zeros((T, K, 3)),  # Will be computed by model
        obs_sigma=2.0,  # From L00 config
        inlier_prob=0.95,
        metadata={"method": "stage_d_cleaned_3d", "source": "stage_d_clean_3d.py"},
    )
    print(f"  Initialization complete")
    print(f"    Root init: {init_result.x_init[0, 0, :]}")
    print(f"    Bone lengths (rho): {init_result.rho}")
    print(f"    Obs sigma: {init_result.obs_sigma:.4f}")

    # Build PyMC model
    print("\n[3/4] Building PyMC model...")
    print("  Configuration:")
    print("    - HMM states: 1")
    print("    - Stage 3 directional HMM: enabled")
    print("    - Data-driven priors: enabled")
    print("    - Mixture likelihood: enabled")
    print("    - Cameras: FIXED (not estimated)")
    print("    - Bone lengths: ESTIMATED (API limitation - see note)")

    with pm.Model() as model:
        # Stage 1-2: Build camera observation model
        gimbal.build_camera_observation_model(
            y_observed=data["y_2d_clean"],
            camera_proj=data["camera_proj"],
            parents=data["parents"],
            init_result=init_result,
            use_mixture=True,
            image_size=(1280, 720),
            use_directional_hmm=False,  # We'll add it separately
            validate_init_points=False,
        )

        # Extract variables from model context
        U = model["U"]
        x_all = model["x_all"]
        y_pred = model["y_pred"]
        log_obs_t = model["log_obs_t"]

        # Stage 3: Add directional HMM prior with K=1
        print("\n  Adding Stage 3 directional HMM prior (K=1)...")
        hmm_result = gimbal.add_directional_hmm_prior(
            U=U,
            log_obs_t=log_obs_t,
            S=1,  # Single state for L00
            joint_names=data["joint_names"],
            name_prefix="dir_hmm",
            share_kappa_across_joints=False,
            share_kappa_across_states=False,
            kappa_scale=5.0,
            prior_config=data["prior_config"],
        )

        print(f"    HMM variables added: {list(hmm_result.keys())}")
        print(f"    Transition matrix: [[1.0]] (deterministic for K=1)")

    # Model diagnostics
    print("\n[4/4] Model diagnostics...")
    n_vars = len(model.free_RVs)
    n_obs = np.sum(~np.isnan(data["y_2d_clean"]))

    print(f"  Free RVs: {n_vars}")
    print(f"  Observed points: {n_obs}")
    print(f"  Model variables:")
    for rv in model.free_RVs[:10]:  # Show first 10
        print(f"    - {rv.name}: {rv.type}")
    if n_vars > 10:
        print(f"    ... and {n_vars - 10} more")

    # Try to compile model (check for errors)
    print("\n  Compiling model (validation check)...")
    try:
        logp = model.compile_logp()
        test_point = model.initial_point()
        logp_value = logp(test_point)
        print(f"    ✓ Model compiles successfully")
        print(f"    ✓ Log-probability at init: {logp_value:.2f}")
    except Exception as e:
        print(f"    ✗ Model compilation failed: {e}")
        raise

    # Note: PyMC models cannot be pickled directly
    # Stage H will rebuild the model from the same inputs
    print("\n  Note: Model will be rebuilt in Stage H from preprocessing outputs")

    # Compile metrics
    full_metrics = {
        "stage": "G_model_building",
        "model_config": {
            "hmm_num_states": 1,
            "use_directional_hmm": True,
            "use_mixture": True,
            "use_data_driven_priors": True,
            "estimate_cameras": False,
            "estimate_bone_lengths": True,  # API limitation
            "note": "Bone length fixing not yet implemented in API - tracked as future work",
        },
        "model_diagnostics": {
            "n_free_rvs": n_vars,
            "n_observed_points": int(n_obs),
            "logp_at_init": float(logp_value),
            "compilation_successful": True,
        },
        "initialization": {
            "method": "dlt",
            "root_pos_init": init_result.x_init[0, 0, :].tolist(),
            "bone_lengths_init": init_result.rho.tolist(),
            "obs_sigma_init": float(init_result.obs_sigma),
        },
    }

    # Save metrics
    output_path = output_dir / "model_building_metrics.json"
    with open(output_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\n✓ Stage G complete. Metrics saved to {output_path}")

    return full_metrics


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    repo_root = Path(__file__).parent.parent.parent
    dataset_dir = repo_root / "tests" / "pipeline" / "datasets" / "v0.2.1_L00_minimal"
    fits_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"
    output_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"

    metrics = run_stage_g(dataset_dir, fits_dir, output_dir)

    if metrics["model_diagnostics"]["compilation_successful"]:
        print("\n" + "=" * 80)
        print("STAGE G: PASSED ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("STAGE G: FAILED")
        print("=" * 80)
        import sys

        sys.exit(1)
