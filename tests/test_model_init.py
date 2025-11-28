import sys
from pathlib import Path

# Add parent directory to path to import gimbal
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from gimbal.pymc_distributions import VonMisesFisher
from gimbal.fit_params import initialize_from_observations_dlt

# Simple test case
parents = np.array([-1, 0, 1])
K = 3
T = 10  # Smaller for faster testing
C = 3

# Create simple camera matrices
camera_proj = np.random.randn(C, 3, 4)

# Create simple observations (C, T, K, 2)
y_obs = np.random.randn(C, T, K, 2) * 100 + 320

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
