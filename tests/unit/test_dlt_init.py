import numpy as np
from gimbal_pymc.priors.fit_params import initialize_from_observations_dlt
from gimbal_pymc.cameras.utils import build_projection_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Simple test case
parents = np.array([-1, 0, 1])
K = 3
T = 20
C = 3

# Create realistic camera matrices with proper projection
cameras = []
for i in range(C):
    # Create cameras at different positions
    rvec = np.array([0.0, np.pi / 6 * i, 0.0])
    tvec = np.array([200.0 * np.sin(np.pi / 3 * i), 100.0, 500.0 + 100 * i])
    P = build_projection_matrix(rvec, tvec, focal_length=800.0)
    cameras.append(P)
camera_proj = np.stack(cameras)

# Create observations from a simple skeleton motion
# Root at origin, moving in a pattern
x_true = np.zeros((T, K, 3))
for t in range(T):
    x_true[t, 0] = [np.sin(t / 5) * 50, np.cos(t / 5) * 50, 100]  # Root moves in circle
    x_true[t, 1] = x_true[t, 0] + [50, 0, 0]  # Joint 1 at fixed offset
    x_true[t, 2] = x_true[t, 1] + [0, 50, 0]  # Joint 2 at fixed offset

# Project to 2D
y_obs = np.zeros((C, T, K, 2))
for c in range(C):
    for t in range(T):
        for k in range(K):
            x_h = np.append(x_true[t, k], 1)  # Homogeneous coordinates
            p = camera_proj[c] @ x_h
            y_obs[c, t, k] = (p[:2] / p[2]) + np.random.randn(
                2
            ) * 1.0  # Add small noise

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
