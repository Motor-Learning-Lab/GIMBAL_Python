"""
Minimal debug script for sampling diagnostics.

Tests the camera + kinematic model with various configurations:
- M1: no mixture, no HMM, Gamma prior on obs_sigma
- M2: mixture on, no HMM, Gamma prior on obs_sigma
- M3: mixture on, HMM on, Gamma prior (optional, once HMM is wired)

Run with: python tests/test_sampling_camera_model.py
"""

import numpy as np
import pymc as pm
import arviz as az

from gimbal import (
    DEMO_V0_1_SKELETON,
    SyntheticDataConfig,
    generate_demo_sequence,
)
from gimbal.pymc_model import _build_camera_observation_model_full
from gimbal.fit_params import initialize_from_groundtruth


def make_data():
    """Generate synthetic data with consistent seed."""
    config = SyntheticDataConfig(
        T=100,
        C=3,
        S=3,
        kappa=10.0,
        obs_noise_std=0.5,
        occlusion_rate=0.02,
        random_seed=42,
    )
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)
    return data


def make_init(data):
    """Initialize parameters from ground truth."""
    return initialize_from_groundtruth(
        x_gt=data.x_true,
        parents=DEMO_V0_1_SKELETON.parents,
        obs_noise_std=data.config.obs_noise_std,
    )


def make_prior_hyperparams(data, init_result):
    """Build data-driven prior hyperparameters."""
    return {
        "eta2_root_sigma": max(init_result.eta2[0] / 2.0, 0.5),
        "sigma2_sigma": max(init_result.sigma2.mean() * 2.0, 0.5),
        "obs_sigma_mode": max(0.1, data.config.obs_noise_std * 1.0),
        "obs_sigma_sd": max(0.1, data.config.obs_noise_std * 0.5),
    }


def run_model(label, use_mixture, use_hmm, data, init_result, prior_hyperparams):
    """Build and sample a model configuration."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    
    with pm.Model() as model:
        _build_camera_observation_model_full(
            y_observed=data.y_observed,
            camera_proj=data.camera_proj,
            parents=DEMO_V0_1_SKELETON.parents,
            init_result=init_result,
            use_mixture=use_mixture,
            use_directional_hmm=use_hmm,
            hmm_num_states=data.config.S if use_hmm else None,
            hmm_kwargs={
                "name_prefix": "dir_hmm",
                "joint_names": DEMO_V0_1_SKELETON.joint_names,
                # You can pass prior_config here later if needed
            } if use_hmm else None,
            prior_hyperparams=prior_hyperparams,
        )

        print(f"Model variables: {len(model.unobserved_RVs)}")
        print("Sampling...")
        
        trace = pm.sample(
            draws=200,
            tune=200,
            chains=1,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True,
        )

    # Divergence statistics
    div = trace.sample_stats.diverging.values[0]
    n_div = int(div.sum())
    pct_div = 100.0 * n_div / len(div)
    print(f"\nDivergences: {n_div}/{len(div)} ({pct_div:.1f}%)")

    # ESS for key variables
    print("\nEffective Sample Size (ESS):")
    for var in ["x_root", "obs_sigma", "eta2_root", "rho"]:
        if var in trace.posterior:
            ess = az.ess(trace, var_names=[var])
            vals = ess.to_array().values.flatten()
            print(f"  {var:12s}: mean={vals.mean():7.1f}, min={vals.min():7.1f}")

    # Simple root reconstruction error
    x_root_mean = trace.posterior["x_root"].values[0].mean(axis=0)
    x_root_true = data.x_true[:, 0, :]
    err = np.linalg.norm(x_root_mean - x_root_true, axis=1)
    rmse = np.sqrt((err**2).mean())
    print(f"\nRoot RMSE: {rmse:.3f}")
    
    # Posterior mean for obs_sigma
    obs_sigma_mean = trace.posterior["obs_sigma"].values[0].mean()
    print(f"obs_sigma posterior mean: {obs_sigma_mean:.3f} (true: {data.config.obs_noise_std:.3f})")

    return trace


def main():
    """Run all model tests."""
    print("Generating synthetic data...")
    data = make_data()
    
    print("\nInitializing from ground truth...")
    init_result = make_init(data)
    
    print("\nBuilding data-driven hyperparameters...")
    prior_hyperparams = make_prior_hyperparams(data, init_result)
    
    print(f"\nPrior hyperparameters:")
    for key, val in prior_hyperparams.items():
        print(f"  {key:20s}: {val:.3f}")
    
    print(f"\nInitialization values:")
    print(f"  obs_sigma: {init_result.obs_sigma:.3f}")
    print(f"  eta2[0]: {init_result.eta2[0]:.6f}")
    print(f"  sigma2 mean: {init_result.sigma2.mean():.3f}")
    
    # Test configurations
    run_model("M1: No mixture, No HMM", False, False, data, init_result, prior_hyperparams)
    run_model("M2: Mixture, No HMM", True, False, data, init_result, prior_hyperparams)
    
    # Uncomment once HMM is ready:
    # run_model("M3: Mixture + HMM", True, True, data, init_result, prior_hyperparams)
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
