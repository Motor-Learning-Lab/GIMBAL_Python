"""Quick smoke test for v0.2.1 exports."""

import gimbal

print("Testing v0.2.1 module imports...")

# Test exports
exports = [
    "triangulate_multi_view",
    "CleaningConfig",
    "clean_keypoints_2d",
    "clean_keypoints_3d",
    "compute_direction_statistics",
    "build_priors_from_statistics",
    "get_gamma_shape_rate",
]

for name in exports:
    if hasattr(gimbal, name):
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
result = gimbal.triangulate_multi_view(kp_2d, camera_proj)
print(f"  ✓ triangulate_multi_view: output shape {result.shape}")

# Test CleaningConfig
config = gimbal.CleaningConfig()
print(f"  ✓ CleaningConfig: jump_z_thresh={config.jump_z_thresh}")

# Test clean_keypoints_2d
kp_2d = np.random.rand(1, 10, 2, 2)
parents = np.array([-1, 0])
clean_kp, valid, summary = gimbal.clean_keypoints_2d(kp_2d, parents, config)
print(f"  ✓ clean_keypoints_2d: processed {clean_kp.shape[1]} frames")

# Test clean_keypoints_3d
pos_3d = np.random.rand(10, 2, 3)
clean_pos, valid, use_stats, summary = gimbal.clean_keypoints_3d(pos_3d, parents, config)
print(f"  ✓ clean_keypoints_3d: {use_stats.sum()} samples for statistics")

# Test compute_direction_statistics
stats = gimbal.compute_direction_statistics(
    clean_pos, parents, use_stats, ["root", "child"], min_samples=5
)
print(f"  ✓ compute_direction_statistics: {len(stats)} joints")

# Test build_priors_from_statistics
emp_stats = {
    "child": {"mu": np.array([1.0, 0.0, 0.0]), "kappa": 5.0, "n_samples": 10}
}
prior_config = gimbal.build_priors_from_statistics(emp_stats, ["root", "child"])
print(f"  ✓ build_priors_from_statistics: {len(prior_config)} joints with priors")

# Test get_gamma_shape_rate
shape, rate = gimbal.get_gamma_shape_rate(2.0, 1.0)
print(f"  ✓ get_gamma_shape_rate: shape={shape:.2f}, rate={rate:.2f}")

print("\nAll smoke tests passed!")
