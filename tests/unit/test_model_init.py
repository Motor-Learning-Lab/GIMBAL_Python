import numpy as np
import pymc as pm
import pytensor.tensor as pt

from gimbal_pymc.joints.distributions import VonMisesFisher
from gimbal_pymc.priors.fit_params import initialize_from_observations_dlt

# Set random seed for reproducibility
np.random.seed(42)

# Simple test case
parents = np.array([-1, 0, 1])
K = 3
T = 20  # Increased for better statistics
C = 3

# Create realistic camera matrices with proper projection
from gimbal_pymc.cameras.utils import build_projection_matrix

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

# Initialize
result = initialize_from_observations_dlt(y_obs, camera_proj, parents, min_cameras=2)

print("Initialization successful")
print(f"u_init shape: {result.u_init.shape}")
print(
    f"u_init[:, 0] norms (root): {np.linalg.norm(result.u_init[:, 0, :], axis=-1)[:5]}"
)
print(
    f"u_init[:, 1] norms (joint 1): {np.linalg.norm(result.u_init[:, 1, :], axis=-1)[:5]}"
)
print(
    f"u_init[:, 2] norms (joint 2): {np.linalg.norm(result.u_init[:, 2, :], axis=-1)[:5]}"
)

# Now try to build a minimal model
print("\nBuilding PyMC model...")
with pm.Model() as test_model:
    # Just test the VonMisesFisher with the DLT init
    mu_k = np.array([0.0, 0.0, 1.0])
    kappa_k = pm.Exponential("kappa_1", 1.0, initval=10.0)

    # Use joint 1's initialization (should be unit vectors)
    u_1 = VonMisesFisher(
        "u_1", mu=mu_k, kappa=kappa_k, shape=(T, 3), initval=result.u_init[:, 1, :]
    )

    print("Model built successfully")

# Try with PyMC's default sampler first
print("\nTesting PyMC default sampler...")
try:
    with test_model:
        trace = pm.sample(
            draws=10, tune=10, chains=1, random_seed=123, progressbar=False
        )
    print("PyMC sampling successful!")

except Exception as e:
    print(f"ERROR with PyMC sampler: {e}")
    import traceback

    traceback.print_exc()

# Try to compile with nutpie
print("\nTesting nutpie compilation...")
try:
    import nutpie

    compiled_model = nutpie.compile_pymc_model(test_model)
    print("Nutpie compilation successful!")

    # Try to draw a few samples with different initialization
    print("Drawing test samples with adapt_diag_grad init...")
    trace = nutpie.sample(
        compiled_model,
        chains=1,
        tune=100,
        draws=10,
        seed=123,
        init_mean="adapt_diag_grad",
    )
    print("Nutpie sampling successful!")

except Exception as e:
    print(f"ERROR with nutpie: {e}")
    import traceback

    traceback.print_exc()
