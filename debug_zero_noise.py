from gimbal import *
import numpy as np

cfg = SyntheticDataConfig(
    T=5, C=3, S=2, obs_noise_std=0.0, occlusion_rate=0.0, random_seed=42
)
d = generate_demo_sequence(DEMO_V0_1_SKELETON, cfg)

print("y shape:", d.y_observed.shape)
print("NaNs:", np.sum(np.isnan(d.y_observed)))
print("Sample y[0,0,0]:", d.y_observed[0, 0, 0, :])
print("Sample y[0,0,1]:", d.y_observed[0, 0, 1, :])

# Check if any valid points
has_valid = np.any(~np.isnan(d.y_observed))
print("Has valid points:", has_valid)

# Check camera proj
print("\nCamera proj[0]:")
print(d.camera_proj[0])
