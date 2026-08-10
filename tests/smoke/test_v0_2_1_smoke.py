"""Quick smoke test for v0.2.1 exports."""

import gimbal_pymc.skeleton.config as gp_config
import gimbal_pymc.skeleton.synthetic_data as gp_synth
import gimbal_pymc.cameras.triangulation as gp_tri
import gimbal_pymc.data_cleaning.cleaning as gp_clean
import gimbal_pymc.joints.statistics as gp_stats
import gimbal_pymc.priors.building as gp_priors

print("Testing v0.2.1 module imports...")

# Test exports
exports_map = {
    "triangulate_multi_view": gp_tri,
    "CleaningConfig": gp_clean,
    "clean_keypoints_2d": gp_clean,
    "clean_keypoints_3d": gp_clean,
    "compute_direction_statistics": gp_stats,
    "build_priors_from_statistics": gp_priors,
    "gamma_mode_sd_to_shape_rate": gp_priors,
}

for name, module in exports_map.items():
    if hasattr(module, name.split("_")[-1] if "_" in name else name) or hasattr(
        module, name
    ):
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name} MISSING")

# Test basic functionality
import numpy as np

print("\nTesting basic functionality...")

# Test triangulation
P = np.eye(3, 4, dtype=np.float64)
camera_proj = P[np.newaxis, :, :]
kp_2d = np.random.rand(1, 2, 1, 2)
result = gp_tri.triangulate_multi_view(kp_2d, camera_proj)
print(f"  ✓ triangulate_multi_view: output shape {result.shape}")

# Test CleaningConfig
config = gp_clean.CleaningConfig()
print(f"  ✓ CleaningConfig: jump_z_thresh={config.jump_z_thresh}")

# Test clean_keypoints_2d
kp_2d = np.random.rand(1, 10, 2, 2)
parents = np.array([-1, 0])
clean_kp, valid, summary = gp_clean.clean_keypoints_2d(kp_2d, parents, config)
print(f"  ✓ clean_keypoints_2d: processed {clean_kp.shape[1]} frames")

# Test clean_keypoints_3d
pos_3d = np.random.rand(10, 2, 3)
clean_pos, valid, use_stats, summary = gp_clean.clean_keypoints_3d(
    pos_3d, parents, config
)
print(f"  ✓ clean_keypoints_3d: {use_stats.sum()} samples for statistics")

# Test compute_direction_statistics
stats = gp_stats.compute_direction_statistics(
    clean_pos, parents, use_stats, ["root", "child"], min_samples=5
)
print(f"  ✓ compute_direction_statistics: {len(stats)} joints")

# Test build_priors_from_statistics
emp_stats = {"child": {"mu": np.array([1.0, 0.0, 0.0]), "kappa": 5.0, "n_samples": 10}}
prior_config = gp_priors.build_priors_from_statistics(emp_stats, ["root", "child"])
print(f"  ✓ build_priors_from_statistics: {len(prior_config)} joints with priors")

# Test gamma_mode_sd_to_shape_rate
shape, rate = gp_priors.gamma_mode_sd_to_shape_rate(2.0, 1.0)
print(f"  ✓ gamma_mode_sd_to_shape_rate: shape={shape:.2f}, rate={rate:.2f}")

print("\nAll smoke tests passed!")
