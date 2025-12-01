"""Quick check of zero-noise triangulation"""

from gimbal import *
from gimbal.triangulation import triangulate_multi_view
import numpy as np

cfg = SyntheticDataConfig(
    T=3, C=3, S=2, obs_noise_std=0.0, occlusion_rate=0.0, random_seed=42
)
d = generate_demo_sequence(DEMO_V0_1_SKELETON, cfg)

print("x_true[0,0]:", d.x_true[0, 0, :])
print("y_observed[0,0,0]:", d.y_observed[0, 0, 0, :])
print("y_observed[1,0,0]:", d.y_observed[1, 0, 0, :])
print("y_observed[2,0,0]:", d.y_observed[2, 0, 0, :])

# Try triangulation
print("\nTriangulating...")
x_recon = triangulate_multi_view(d.y_observed, d.camera_proj, condition_threshold=1e6)

print("x_recon[0,0]:", x_recon[0, 0, :])
print("Error:", np.linalg.norm(x_recon[0, 0, :] - d.x_true[0, 0, :]))

# Check for NaNs
print("\nNaNs in x_recon:", np.sum(np.isnan(x_recon)))
