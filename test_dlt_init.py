import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve()))

import numpy as np
from gimbal.fit_params import initialize_from_observations_dlt

# Simple test case
parents = np.array([-1, 0, 1])
K = 3
T = 20
C = 3

# Create simple camera matrices
camera_proj = np.random.randn(C, 3, 4)

# Create simple observations (C, T, K, 2)
y_obs = np.random.randn(C, T, K, 2) * 100 + 320

result = initialize_from_observations_dlt(y_obs, camera_proj, parents, min_cameras=2)

print(f"x_init has NaN: {np.isnan(result.x_init).any()}")
print(f"u_init has NaN: {np.isnan(result.u_init).any()}")
print(f"eta2: {result.eta2}")
print(f"rho: {result.rho}")
print(f"sigma2: {result.sigma2}")
print(f"obs_sigma: {result.obs_sigma}")
print(f"Shape x_init: {result.x_init.shape}")
print(f"Shape u_init: {result.u_init.shape}")
print(f"\nChecking for zeros or negatives:")
print(f"  Any eta2 <= 0: {(result.eta2 <= 0).any()}")
print(f"  Any sigma2 <= 0: {(result.sigma2 <= 0).any()}")
print(f"  obs_sigma <= 0: {result.obs_sigma <= 0}")

print(f"\nChecking u_init unit vectors:")
u_norms = np.linalg.norm(result.u_init, axis=-1)
print(f"  Min norm: {u_norms.min()}")
print(f"  Max norm: {u_norms.max()}")
print(f"  Mean norm: {u_norms.mean()}")
print(f"  Any non-unit: {np.any(np.abs(u_norms - 1.0) > 0.01)}")
