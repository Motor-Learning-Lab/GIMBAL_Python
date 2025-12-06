"""
Shared utilities for v0.2.1 divergence test suite.
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, Any, Optional


def get_standard_synth_data(T: int = 100, C: int = 3, S: int = 3, seed: int = 42):
    """
    Generate standard synthetic data for testing.

    Parameters
    ----------
    T : int
        Number of timesteps
    C : int
        Number of cameras
    S : int
        Number of HMM states (only used for generating consistent priors)
    seed : int
        Random seed

    Returns
    -------
    dict
        Synthetic data dictionary
    """
    from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence

    config = SyntheticDataConfig(
        T=T,
        C=C,
        S=S,
        kappa=10.0,
        obs_noise_std=0.5,
        occlusion_rate=0.02,
        random_seed=seed,
    )

    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

    # Return as dictionary matching expected format
    return {
        "observations_uv": data.y_observed,  # (C, T, K, 2)
        "camera_matrices": data.camera_proj,  # (C, 3, 4)
        "joint_positions": data.x_true,  # (T, K, 3)
        "joint_names": DEMO_V0_1_SKELETON.joint_names,
        "parents": DEMO_V0_1_SKELETON.parents,
        "bone_lengths": DEMO_V0_1_SKELETON.bone_lengths,
        "true_states": data.true_states,
        "config": config,
    }


def build_test_model(
    synth_data: Dict[str, Any],
    use_directional_hmm: bool,
    S: int = 3,
    eta2_root_sigma: float = 0.5,
    sigma2_sigma: float = 0.2,
    kappa_min: float = 0.1,
    kappa_scale: float = 5.0,
) -> pm.Model:
    """
    Build a PyMC model for testing.

    Parameters
    ----------
    synth_data : dict
        Synthetic data dictionary
    use_directional_hmm : bool
        Whether to use directional HMM prior
    S : int
        Number of HMM states
    eta2_root_sigma : float
        Root variance hyperparameter
    sigma2_sigma : float
        Bone length variance hyperparameter
    kappa_min : float
        Minimum kappa value
    kappa_scale : float
        Kappa scaling factor

    Returns
    -------
    pm.Model
        PyMC model
    """
    import gimbal

    # Initialize from observations
    init_result = gimbal.fit_params.initialize_from_observations_dlt(
        y_observed=synth_data["observations_uv"],
        camera_proj=synth_data["camera_matrices"],
        parents=synth_data["parents"],
    )

    # Prior hyperparams
    prior_hyperparams = {
        "eta2_root_sigma": eta2_root_sigma,
        "sigma2_sigma": sigma2_sigma,
    }

    # Build model based on whether HMM is enabled
    with pm.Model() as model:
        if use_directional_hmm:
            # Build data-driven priors for HMM
            from gimbal.triangulation import triangulate_multi_view
            from gimbal.data_cleaning import (
                clean_keypoints_2d,
                clean_keypoints_3d,
                CleaningConfig,
            )
            from gimbal.direction_statistics import compute_direction_statistics
            from gimbal.prior_building import build_priors_from_statistics

            # Clean and triangulate
            cleaning_config = CleaningConfig()
            y_clean, valid_2d_mask, _ = clean_keypoints_2d(
                synth_data["observations_uv"], synth_data["parents"], cleaning_config
            )
            x_triangulated = triangulate_multi_view(
                y_clean, synth_data["camera_matrices"]
            )
            x_clean, valid_3d_mask, use_for_stats_mask, _ = clean_keypoints_3d(
                x_triangulated, synth_data["parents"], cleaning_config
            )

            # Compute direction statistics
            dir_stats = compute_direction_statistics(
                x_clean,
                synth_data["parents"],
                use_for_stats_mask,
                synth_data["joint_names"],
            )

            # Build priors
            prior_config = build_priors_from_statistics(
                dir_stats,
                synth_data["joint_names"],
                kappa_min=kappa_min,
                kappa_scale=kappa_scale,
            )

            # Build model with HMM
            gimbal.build_camera_observation_model(
                y_observed=synth_data["observations_uv"],
                camera_proj=synth_data["camera_matrices"],
                parents=synth_data["parents"],
                init_result=init_result,
                prior_hyperparams=prior_hyperparams,
                use_directional_hmm=True,
                hmm_num_states=S,
                hmm_kwargs={
                    "joint_names": synth_data["joint_names"],
                    "prior_config": prior_config,
                },
            )
        else:
            # Build model without HMM
            gimbal.build_camera_observation_model(
                y_observed=synth_data["observations_uv"],
                camera_proj=synth_data["camera_matrices"],
                parents=synth_data["parents"],
                init_result=init_result,
                prior_hyperparams=prior_hyperparams,
                use_directional_hmm=False,
            )

    return model


def sample_model(
    model: pm.Model, draws: int = 200, tune: int = 200, chains: int = 1
) -> az.InferenceData:
    """
    Sample from a PyMC model with standardized settings.

    Parameters
    ----------
    model : pm.Model
        PyMC model
    draws : int
        Number of draws
    tune : int
        Number of tuning steps
    chains : int
        Number of chains

    Returns
    -------
    az.InferenceData
        ArviZ inference data
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True,
        )
    return trace


def extract_metrics(trace: az.InferenceData, runtime: float) -> Dict[str, Any]:
    """
    Extract key metrics from a trace.

    Parameters
    ----------
    trace : az.InferenceData
        ArviZ inference data
    runtime : float
        Runtime in seconds

    Returns
    -------
    dict
        Dictionary of metrics
    """
    # Divergences
    divergences = trace.sample_stats.diverging.sum().values
    total_samples = trace.sample_stats.diverging.size

    # ESS
    ess = az.ess(trace)
    ess_bulk = {k: v.values.mean() for k, v in ess.items() if hasattr(v, "values")}

    # R-hat
    rhat = az.rhat(trace)
    rhat_max = {k: v.values.max() for k, v in rhat.items() if hasattr(v, "values")}

    # Log likelihood
    log_likelihood = trace.log_likelihood if hasattr(trace, "log_likelihood") else None

    return {
        "divergences": int(divergences),
        "total_samples": int(total_samples),
        "divergence_rate": float(divergences / total_samples),
        "ess_bulk": ess_bulk,
        "rhat_max": rhat_max,
        "runtime_seconds": runtime,
        "log_likelihood": log_likelihood,
    }


def save_diagnostic_plots(
    trace: az.InferenceData,
    test_name: str,
    diagnostics_dir: Path,
    plot_parallel: bool = True,
    plot_pair: bool = True,
):
    """
    Save diagnostic plots for a trace.

    Parameters
    ----------
    trace : az.InferenceData
        ArviZ inference data
    test_name : str
        Name of the test
    diagnostics_dir : Path
        Directory to save plots
    plot_parallel : bool
        Whether to create parallel plot
    plot_pair : bool
        Whether to create pair plot
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Parallel plot
    if plot_parallel:
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            az.plot_parallel(trace, var_names=["~log_likelihood"], ax=ax)
            fig.savefig(
                str(diagnostics_dir / f"{test_name}_parallel.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not create parallel plot for {test_name}: {e}")

    # Pair plot with divergences
    if plot_pair:
        try:
            # Select a few key variables to plot
            var_names = ["root_position", "bone_lengths"]
            fig = az.plot_pair(
                trace, var_names=var_names, divergences=True, figsize=(12, 12)
            )
            fig.savefig(
                str(diagnostics_dir / f"{test_name}_pair.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not create pair plot for {test_name}: {e}")


def calculate_reconstruction_error(
    trace: az.InferenceData, true_positions: np.ndarray
) -> Dict[str, float]:
    """
    Calculate reconstruction error metrics.

    Parameters
    ----------
    trace : az.InferenceData
        ArviZ inference data
    true_positions : np.ndarray
        True 3D positions (T, K, 3)

    Returns
    -------
    dict
        Dictionary of error metrics
    """
    # Get posterior mean of joint_positions
    if "joint_positions" not in trace.posterior:
        return {"mean_error": np.nan, "median_error": np.nan, "max_error": np.nan}

    posterior_mean = trace.posterior.joint_positions.mean(dim=["chain", "draw"]).values

    # Calculate per-joint errors
    errors = np.linalg.norm(posterior_mean - true_positions, axis=-1)

    return {
        "mean_error": float(errors.mean()),
        "median_error": float(np.median(errors)),
        "max_error": float(errors.max()),
    }


def format_test_result(
    test_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    reconstruction_error: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Format a test result for reporting.

    Parameters
    ----------
    test_name : str
        Name of the test
    config : dict
        Test configuration
    metrics : dict
        Test metrics
    reconstruction_error : dict, optional
        Reconstruction error metrics

    Returns
    -------
    dict
        Formatted test result
    """
    result = {"test_name": test_name, "config": config, "metrics": metrics}

    if reconstruction_error is not None:
        result["reconstruction_error"] = reconstruction_error

    return result
