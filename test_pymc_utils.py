"""
Test gimbal.pymc_utils functions with simple camera model.

This tests:
1. build_initial_points_for_nutpie creates correct dict structure
2. validate_initial_points catches shape mismatches
3. compile_model_with_initialization produces working compiled model
"""

import numpy as np
import pymc as pm

from gimbal.fit_params import initialize_from_groundtruth
from gimbal.pymc_utils import (
    build_initial_points_for_nutpie,
    validate_initial_points,
    compile_model_with_initialization,
)
from gimbal.pymc_distributions import VonMisesFisher

# Create simple test data from groundtruth
np.random.seed(42)
T, K = 10, 4  # 10 frames, 4 joints (1 root + 3 non-root)
parents = np.array([-1, 0, 1, 2])

# Generate simple groundtruth skeleton
x_true = np.zeros((T, K, 3))
x_true[:, 0, :] = np.random.randn(T, 3) * 50  # Root position
x_true[:, 1, :] = x_true[:, 0, :] + np.array([100, 0, 0])  # Joint 1
x_true[:, 2, :] = x_true[:, 1, :] + np.array([0, 80, 0])  # Joint 2
x_true[:, 3, :] = x_true[:, 2, :] + np.array([0, 0, 60])  # Joint 3

print("Initializing from groundtruth...")
init_result = initialize_from_groundtruth(x_true, parents)

print("Init shapes:")
print(f"  x_init: {init_result.x_init.shape}")
print(f"  u_init: {init_result.u_init.shape}")
print(f"  rho: {init_result.rho.shape}")
print(f"  eta2: {init_result.eta2.shape}")

# Build simple PyMC model
print("\nBuilding PyMC model...")
with pm.Model() as model:
    # Root temporal variance
    eta2_root = pm.Exponential("eta2_root", lam=1.0, initval=init_result.eta2[0])

    # Skeletal parameters (non-root joints) - rho and sigma2 already exclude root
    rho = pm.Exponential("rho", lam=0.1, shape=(K - 1,), initval=init_result.rho)
    sigma2 = pm.Exponential(
        "sigma2", lam=1.0, shape=(K - 1,), initval=init_result.sigma2
    )

    # Root trajectory
    x_root = pm.GaussianRandomWalk(
        "x_root",
        init_dist=pm.Normal.dist(0, 100),
        sigma=pm.math.sqrt(eta2_root),
        shape=(T, 3),
        initval=init_result.x_init[:, 0, :],
    )

    # Directional vectors (non-root joints)
    u_vecs = []
    for k in range(1, K):
        u_k = VonMisesFisher(
            f"u_{k}",
            mu=np.array([0.0, 0.0, 1.0]),
            kappa=10.0,
            shape=(T, 3),
            initval=init_result.u_init[:, k, :],
        )
        u_vecs.append(u_k)

    # Bone lengths
    for k in range(1, K):
        length_k = pm.TruncatedNormal(
            f"length_{k}",
            mu=rho[k - 1],
            sigma=pm.math.sqrt(sigma2[k - 1]),
            lower=0.0,
            shape=(T,),
            initval=np.full(T, init_result.rho[k - 1]),
        )

    # Observation noise
    obs_sigma = pm.Exponential("obs_sigma", lam=0.1, initval=5.0)

print(f"Model RVs: {list(model.named_vars.keys())}")

# Test 1: Build initial points
print("\n=== Test 1: build_initial_points_for_nutpie ===")
initial_points = build_initial_points_for_nutpie(model, init_result, parents)

print(f"Initial points keys: {list(initial_points.keys())}")
for name, value in initial_points.items():
    print(f"  {name}: shape={value.shape}, dtype={value.dtype}")

# Test 2: Validate initial points
print("\n=== Test 2: validate_initial_points ===")
try:
    validate_initial_points(model, initial_points)
    print("✓ Validation passed")
except ValueError as e:
    print(f"✗ Validation failed: {e}")

# Test 3: Test shape mismatch detection
print("\n=== Test 3: Shape mismatch detection ===")
bad_points = initial_points.copy()
bad_points["rho"] = np.array([1.0])  # Wrong shape
try:
    validate_initial_points(model, bad_points)
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly caught error: {e}")

# Test 4: Compile model with initialization
print("\n=== Test 4: compile_model_with_initialization ===")
try:
    compiled_model = compile_model_with_initialization(
        model, init_result, parents, allow_gaussian_jitter=False
    )
    print(f"✓ Compiled model type: {type(compiled_model)}")
    print(f"  Compiled model has sample method: {hasattr(compiled_model, 'sample')}")
except Exception as e:
    print(f"✗ Compilation failed: {e}")

print("\n=== All tests complete ===")
