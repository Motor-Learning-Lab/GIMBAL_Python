"""Demo script for v0.2.1 data-driven priors pipeline.

This script demonstrates the complete workflow:
1. Generate synthetic 2D observations
2. Clean 2D keypoints
3. Triangulate to 3D
4. Clean 3D positions
5. Compute directional statistics
6. Build prior configuration
7. Run PyMC model with data-driven priors
8. Compare ESS with v0.1 baseline
"""

import sys
from pathlib import Path

# Add parent directory to path for gimbal import
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pymc as pm

import gimbal
from gimbal import DEMO_V0_1_SKELETON
from gimbal.synthetic_data import generate_demo_sequence, SyntheticDataConfig

# =============================================================================
# 1. Generate Synthetic Data
# =============================================================================

print("=" * 70)
print("v0.2.1 Demo: Data-Driven Priors Pipeline")
print("=" * 70)

np.random.seed(42)

# Generate synthetic motion data
config = SyntheticDataConfig(
    T=100,  # 100 frames
    S=3,  # 3 states
    C=3,  # 3 cameras
    obs_noise_std=2.0,
)

print("\n[1/8] Generating synthetic data...")
data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

T, K_full = data.x_true.shape[0], data.x_true.shape[1]
print(f"  - Generated {T} frames with {config.C} cameras")
print(f"  - Skeleton: {len(DEMO_V0_1_SKELETON.joint_names)} joints")
print(f"  - 2D observations shape: {data.y_observed.shape}")

# =============================================================================
# 2. Clean 2D Keypoints (Per Camera)
# =============================================================================

print("\n[2/8] Cleaning 2D keypoints...")

cleaning_config = gimbal.CleaningConfig(
    jump_z_thresh=3.5,
    bone_z_thresh=3.5,
    max_gap=5,
    max_bad_joint_fraction=0.3,
)

keypoints_2d_clean, valid_2d_mask, summary_2d = gimbal.clean_keypoints_2d(
    data.y_observed, DEMO_V0_1_SKELETON.parents, cleaning_config
)

print(f"  - Jump outliers detected: {summary_2d['n_jump_outliers']}")
print(f"  - Bone length outliers detected: {summary_2d['n_bone_outliers']}")
print(f"  - Interpolated points: {summary_2d['n_interpolated']}")
print(f"  - Invalid frames: {summary_2d['n_invalid_frames']}")

# =============================================================================
# 3. Triangulate to 3D
# =============================================================================

print("\n[3/8] Triangulating 2D -> 3D...")

positions_3d_tri = gimbal.triangulate_multi_view(
    keypoints_2d_clean, data.camera_proj, min_cameras=2
)

print(f"  - Triangulated shape: {positions_3d_tri.shape}")
print(
    f"  - Valid 3D points: {(~np.isnan(positions_3d_tri).any(axis=-1)).sum()} / {positions_3d_tri.shape[0] * positions_3d_tri.shape[1]}"
)

# =============================================================================
# 4. Clean 3D Positions
# =============================================================================

print("\n[4/8] Cleaning 3D positions...")

positions_3d_clean, valid_3d_mask, use_for_stats_mask, summary_3d = (
    gimbal.clean_keypoints_3d(
        positions_3d_tri, DEMO_V0_1_SKELETON.parents, cleaning_config
    )
)

print(f"  - Jump outliers detected: {summary_3d['n_jump_outliers']}")
print(f"  - Bone length outliers detected: {summary_3d['n_bone_outliers']}")
print(f"  - Interpolated points: {summary_3d['n_interpolated']}")
print(f"  - Invalid frames: {summary_3d['n_invalid_frames']}")
print(f"  - Valid for statistics: {use_for_stats_mask.sum()} samples")

# =============================================================================
# 5. Compute Directional Statistics
# =============================================================================

print("\n[5/8] Computing directional statistics...")

empirical_stats = gimbal.compute_direction_statistics(
    positions_3d_clean,
    DEMO_V0_1_SKELETON.parents,
    use_for_stats_mask,
    DEMO_V0_1_SKELETON.joint_names,
    min_samples=10,
)

n_valid_stats = sum(
    1 for stats in empirical_stats.values() if not np.isnan(stats["kappa"])
)
print(f"  - Joints with valid statistics: {n_valid_stats} / {len(empirical_stats)}")

for joint_name, stats in empirical_stats.items():
    if not np.isnan(stats["kappa"]):
        print(
            f"    * {joint_name}: n={stats['n_samples']}, kappa={stats['kappa']:.2f}, "
            f"mu=[{stats['mu'][0]:.2f}, {stats['mu'][1]:.2f}, {stats['mu'][2]:.2f}]"
        )

# =============================================================================
# 6. Build Prior Configuration
# =============================================================================

print("\n[6/8] Building prior configuration...")

prior_config = gimbal.build_priors_from_statistics(
    empirical_stats,
    DEMO_V0_1_SKELETON.joint_names,
    kappa_min=0.1,
    kappa_scale=5.0,
)

print(f"  - Priors created for {len(prior_config)} joints")
for joint_name, prior in prior_config.items():
    print(
        f"    * {joint_name}: mu_sd={prior['mu_sd']:.3f}, kappa_mode={prior['kappa_mode']:.2f}"
    )

# =============================================================================
# 7. Build PyMC Model with Data-Driven Priors
# =============================================================================

print("\n[7/8] Building PyMC model with data-driven priors...")

# Initialize from triangulated positions
init_result = gimbal.fit_params.initialize_from_observations_dlt(
    y_observed=data.y_observed,
    camera_proj=data.camera_proj,
    parents=DEMO_V0_1_SKELETON.parents,
)

print("  - Initialized with DLT triangulation")
print(f"    obs_sigma: {init_result.obs_sigma:.2f}")
print(f"    inlier_prob: {init_result.inlier_prob:.3f}")

with pm.Model() as model_v0_2_1:
    # Stage 2: Camera observation model (returns model, not tuple)
    gimbal.build_camera_observation_model(
        y_observed=data.y_observed,
        camera_proj=data.camera_proj,
        parents=DEMO_V0_1_SKELETON.parents,
        init_result=init_result,
        use_directional_hmm=True,
        hmm_num_states=config.S,
        hmm_kwargs={
            "joint_names": DEMO_V0_1_SKELETON.joint_names,
            "prior_config": prior_config,
        },
    )

print(f"  - Model variables: {len(model_v0_2_1.free_RVs)}")
print("  - Model ready for sampling")

# =============================================================================
# 8. Sample and Compare with v0.1 Baseline
# =============================================================================

print("\n[8/8] Sampling (v0.2.1 with data-driven priors)...")

try:
    with model_v0_2_1:
        trace_v0_2_1 = pm.sample(
            draws=200,
            tune=200,
            chains=1,  # Use single chain to avoid Windows multiprocessing issues
            return_inferencedata=True,
            progressbar=True,
        )

    print("\n✓ Sampling completed successfully!")
    print(f"  - Draws: {len(trace_v0_2_1.posterior.draw)}")
    print(f"  - Chains: {len(trace_v0_2_1.posterior.chain)}")

    # Compute ESS for key variables
    import arviz as az

    ess = az.ess(trace_v0_2_1, var_names=["dir_hmm_mu", "dir_hmm_kappa_full"])

    print("\n  Effective Sample Size (ESS):")
    for var_name in ["dir_hmm_mu", "dir_hmm_kappa_full"]:
        if var_name in ess:
            ess_values = ess[var_name].values.flatten()
            print(
                f"    {var_name}: mean={ess_values.mean():.0f}, min={ess_values.min():.0f}"
            )

    print("\n" + "=" * 70)
    print("v0.2.1 Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Compare ESS with v0.1 baseline (uninformative priors)")
    print("  2. Visualize posterior distributions vs empirical priors")
    print("  3. Check convergence diagnostics (Rhat, divergences)")

except Exception as e:
    print(f"\n✗ Sampling failed: {e}")
    print("\nModel graph and priors are ready for inspection.")
    print("You can still explore the model structure.")

print("\n" + "=" * 70)
