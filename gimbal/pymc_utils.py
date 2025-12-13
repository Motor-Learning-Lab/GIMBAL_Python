"""
Utilities for integrating GIMBAL initialization with nutpie sampling.

This module provides functions to convert GIMBAL's InitializationResult into
the format required by nutpie.compile_pymc_model().

Key Design:
-----------
nutpie ignores the `init_mean` parameter in nutpie.sample(). The ONLY way to
control initialization is via `initial_points` passed to compile_pymc_model().

nutpie uses PyMC's make_initial_point_fn internally, which:
1. Takes the initial_points dict
2. Applies PyMC transformations (constrained -> unconstrained)
3. Passes the result to Rust as init_func
4. When sampling starts, calls init_func(seed) for each chain

Therefore:
- All initialization must go through initial_points in compile_pymc_model
- jitter_rvs controls which RVs get random jitter added
- default_initialization_strategy is ignored when initial_points is provided

Required RV Naming Convention:
------------------------------
This module assumes a FIXED naming scheme matching GIMBAL PyMC models:

- eta2_root: scalar, root temporal variance
- rho: (K-1,), mean bone lengths for non-root joints
- sigma2: (K-1,), bone length variances
- x0_root: (3,), initial root position (anchored to DLT estimate)
- eps_root: (T-1, 3), root increments
- x_root: (T, 3), root trajectory (Deterministic = x0 + cumsum(eps))
- u_{k}: (T, 3), directional vectors for joint k=1..K-1
- length_{k}: (T,), bone lengths over time for joint k=1..K-1
- obs_sigma: scalar, observation noise
- inlier_prob: scalar, mixture weight (for outlier models)

NO dynamic name detection - this is intentional for robustness.
"""

from __future__ import annotations

from typing import Optional, Set
import warnings

import numpy as np
import pymc as pm

from .fit_params import InitializationResult


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN values in array using linear interpolation.

    For leading/trailing NaNs, uses nearest valid value.

    Parameters
    ----------
    arr : ndarray
        Input array with potential NaN values

    Returns
    -------
    arr_filled : ndarray
        Array with NaNs interpolated
    """
    arr = arr.copy()

    # Handle different array shapes
    if arr.ndim == 1:
        # 1D array
        mask = np.isnan(arr)
        if not mask.any():
            return arr

        # Get valid indices
        valid_idx = np.where(~mask)[0]
        if len(valid_idx) == 0:
            # All NaN - use zeros
            return np.zeros_like(arr)

        # Interpolate
        arr[mask] = np.interp(np.where(mask)[0], valid_idx, arr[valid_idx])

    elif arr.ndim == 2:
        # 2D array - interpolate each column
        for col in range(arr.shape[1]):
            arr[:, col] = _interpolate_nans(arr[:, col])

    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")

    return arr


def build_initial_points_for_nutpie(
    model: pm.Model,
    init_result: InitializationResult,
    parents: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Convert GIMBAL InitializationResult to nutpie initial_points format.

    This function creates a dictionary mapping PyMC RV names to their initial
    values, suitable for passing to nutpie.compile_pymc_model().

    Parameters
    ----------
    model : pm.Model
        PyMC model with GIMBAL structure
    init_result : InitializationResult
        Output from initialize_from_groundtruth, initialize_from_observations_dlt,
        or initialize_from_observations_anipose
    parents : ndarray, shape (K,)
        Parent joint indices (-1 for root)

    Returns
    -------
    initial_points : dict[str, np.ndarray]
        Keys match PyMC RV names, values are numpy arrays with float64 dtype

    Raises
    ------
    ValueError
        If shapes don't match expected dimensions or RVs are missing from model

    Notes
    -----
    This function uses a FIXED naming convention:
    - eta2_root: scalar
    - rho: (K-1,)
    - sigma2: (K-1,)
    - x_root: (T, 3)
    - u_{k}: (T, 3) for k=1..K-1
    - length_{k}: (T,) for k=1..K-1
    - obs_sigma: scalar
    - inlier_prob: scalar (optional, for mixture models)

    Examples
    --------
    >>> from gimbal.fit_params import initialize_from_observations_dlt
    >>> init_result = initialize_from_observations_dlt(y_obs, camera_proj, parents)
    >>> initial_points = build_initial_points_for_nutpie(model, init_result, parents)
    >>> compiled = nutpie.compile_pymc_model(model, initial_points=initial_points)
    """
    T, K, _ = init_result.x_init.shape

    if len(parents) != K:
        raise ValueError(
            f"parents length ({len(parents)}) must match K ({K}) from init_result.x_init"
        )

    initial_points = {}

    # 1. Root temporal variance (scalar)
    if "eta2_root" in model.named_vars:
        initial_points["eta2_root"] = np.float64(init_result.eta2[0])

    # 2. Skeletal parameters for non-root joints (K-1,)
    if "rho" in model.named_vars:
        # rho in init_result is already (K-1,) - non-root joints only
        initial_points["rho"] = init_result.rho.astype(np.float64)

    if "sigma2" in model.named_vars:
        # sigma2 in init_result is already (K-1,) - non-root joints only
        initial_points["sigma2"] = init_result.sigma2.astype(np.float64)

    # 3. Root trajectory (now with anchored initial position)
    # x0_root: (3,) initial position
    # eps_root: (T-1, 3) increments
    if "x0_root" in model.named_vars and "eps_root" in model.named_vars:
        x_root = init_result.x_init[:, 0, :].copy()
        if np.isnan(x_root).any():
            x_root = _interpolate_nans(x_root)
            warnings.warn(
                f"x_root (for x0/eps init) contained {np.isnan(init_result.x_init[:, 0, :]).sum()} NaN values, "
                "interpolated for initialization",
                UserWarning,
            )
        # Initial position
        initial_points["x0_root"] = x_root[0, :].astype(np.float64)
        # Increments (differences)
        eps_init = np.diff(x_root, axis=0)  # (T-1, 3)
        initial_points["eps_root"] = eps_init.astype(np.float64)
    # Fallback for legacy models with direct x_root
    elif "x_root" in model.named_vars:
        x_root = init_result.x_init[:, 0, :].copy()
        if np.isnan(x_root).any():
            x_root = _interpolate_nans(x_root)
            warnings.warn(
                f"x_root contained {np.isnan(init_result.x_init[:, 0, :]).sum()} NaN values, "
                "interpolated for initialization (legacy model)",
                UserWarning,
            )
        initial_points["x_root"] = x_root.astype(np.float64)

    # 4. Raw directional vectors raw_u_k for k=1..K-1 (each is T, 3)
    # Note: u_k itself is deterministic (normalized version of raw_u_k)
    for k in range(1, K):
        rv_name = f"raw_u_{k}"
        if rv_name in model.named_vars:
            u_k = init_result.u_init[:, k, :].copy()
            # Interpolate NaN values and renormalize
            if np.isnan(u_k).any():
                u_k = _interpolate_nans(u_k)
                # Renormalize to unit vectors
                norms = np.linalg.norm(u_k, axis=1, keepdims=True)
                u_k = u_k / np.maximum(norms, 1e-8)
                warnings.warn(
                    f"raw_u_{k} contained NaN values, interpolated and renormalized",
                    UserWarning,
                )
            # Use normalized u_init as raw_u_k initialization
            # The deterministic normalization will keep them on the sphere
            initial_points[rv_name] = u_k.astype(np.float64)

    # 5. Bone lengths length_k for k=1..K-1 (each is T,)
    # Use per-frame lengths from init_result if available, else repeat mean
    for k_idx, k in enumerate(range(1, K)):
        rv_name = f"length_{k}"
        if rv_name in model.named_vars:
            # Use mean bone length as constant initialization
            # Note: init_result.rho is already (K-1,) - non-root joints only
            mean_length = float(init_result.rho[k_idx])
            initial_points[rv_name] = np.full(T, mean_length, dtype=np.float64)

    # 6. Observation noise (scalar)
    if "obs_sigma" in model.named_vars:
        if hasattr(init_result, "obs_sigma") and init_result.obs_sigma is not None:
            initial_points["obs_sigma"] = np.float64(init_result.obs_sigma)
        else:
            warnings.warn(
                "init_result.obs_sigma not available, using default 5.0 pixels",
                UserWarning,
            )
            initial_points["obs_sigma"] = np.float64(5.0)

    # 7. Inlier probability (scalar, for mixture models)
    if "inlier_prob" in model.named_vars:
        if hasattr(init_result, "inlier_prob") and init_result.inlier_prob is not None:
            initial_points["inlier_prob"] = np.float64(init_result.inlier_prob)
        else:
            warnings.warn(
                "init_result.inlier_prob not available, using default 0.9", UserWarning
            )
            initial_points["inlier_prob"] = np.float64(0.9)

    # Note: kappa parameters removed - no longer using VonMisesFisher distribution
    # Directional vectors now use Gaussian-normalize parameterization (raw_u_k)

    return initial_points


def validate_initial_points(
    model: pm.Model,
    initial_points: dict[str, np.ndarray],
) -> None:
    """
    Validate that initial_points match PyMC model RV shapes and types.

    This performs strict validation and raises ValueError on any mismatch.
    No fallbacks, no warnings - fail fast to catch errors before sampling.

    Parameters
    ----------
    model : pm.Model
        PyMC model to validate against
    initial_points : dict[str, np.ndarray]
        Initial values for model RVs

    Raises
    ------
    ValueError
        If any RV has wrong shape, wrong dtype, or is missing

    Examples
    --------
    >>> validate_initial_points(model, initial_points)  # raises on mismatch
    """
    for name, value in initial_points.items():
        if name not in model.named_vars:
            raise ValueError(
                f"RV '{name}' in initial_points not found in model. "
                f"Available RVs: {list(model.named_vars.keys())}"
            )

        rv = model.named_vars[name]
        expected_shape = tuple(rv.type.shape)
        actual_shape = value.shape

        if expected_shape != actual_shape:
            raise ValueError(
                f"RV '{name}' shape mismatch:\n"
                f"  Expected: {expected_shape}\n"
                f"  Got: {actual_shape}\n"
                f"  Value dtype: {value.dtype}"
            )

        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(
                f"RV '{name}' must have floating point dtype, got {value.dtype}"
            )

        if not np.all(np.isfinite(value)):
            raise ValueError(f"RV '{name}' contains non-finite values (NaN or Inf)")


def validate_stage2_outputs(
    model: pm.Model,
    T: int,
    K: int,
    C: int,
) -> None:
    """
    Validate Stage 2 → Stage 3 interface tensor shapes.

    This function checks that the refactored PyMC model produces all required
    outputs with correct shapes for Stage 3 HMM integration.

    Parameters
    ----------
    model : pm.Model
        PyMC model to validate
    T : int
        Number of timesteps
    K : int
        Number of joints
    C : int
        Number of cameras

    Raises
    ------
    ValueError
        If any required tensor is missing or has wrong shape

    Examples
    --------
    >>> validate_stage2_outputs(model, T=50, K=10, C=4)
    ✓ All v0.1.2 outputs have correct shapes
    """
    required_vars = {
        "U": (T, K, 3),
        "x_all": (T, K, 3),
        "y_pred": (None, T, K, 2),  # First dim can be symbolic (C) from pm.Data
        "log_obs_t": (T,),
    }

    for var_name, expected_shape in required_vars.items():
        if var_name not in model.named_vars:
            raise ValueError(
                f"Required variable '{var_name}' not found in model. "
                f"Available: {list(model.named_vars.keys())}"
            )

        var = model.named_vars[var_name]
        actual_shape = tuple(var.type.shape)

        # Check shapes, allowing None (symbolic) in expected positions
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"Variable '{var_name}' has wrong number of dimensions:\n"
                f"  Expected: {len(expected_shape)} dims {expected_shape}\n"
                f"  Got: {len(actual_shape)} dims {actual_shape}"
            )

        for i, (expected_dim, actual_dim) in enumerate(
            zip(expected_shape, actual_shape)
        ):
            # Allow None to match any value (symbolic dimension)
            if expected_dim is None:
                continue
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Variable '{var_name}' dimension {i} mismatch:\n"
                    f"  Expected: {expected_shape}\n"
                    f"  Got: {actual_shape}"
                )

    print("✓ All v0.1.2 outputs have correct shapes")


def compile_model_with_initialization(
    model: pm.Model,
    init_result: InitializationResult,
    parents: np.ndarray,
    allow_gaussian_jitter: bool = False,
) -> "nutpie.CompiledModel":
    """
    Compile PyMC model for nutpie with GIMBAL initialization.

    This is the recommended way to prepare a GIMBAL model for nutpie sampling.
    It handles:
    1. Converting InitializationResult to initial_points format
    2. Validating shapes and types
    3. Configuring jitter strategy
    4. Compiling with proper initialization strategy

    Parameters
    ----------
    model : pm.Model
        PyMC model with GIMBAL structure
    init_result : InitializationResult
        GIMBAL initialization from fit_params functions
    parents : ndarray, shape (K,)
        Parent joint indices
    allow_gaussian_jitter : bool, default=False
        If True, allow jitter on Gaussian RVs (eta2_root, rho, sigma2, obs_sigma).
        If False, no jitter on any RVs (safest for fragile distributions).

    Returns
    -------
    compiled_model : nutpie.CompiledModel
        Compiled model ready for nutpie.sample()

    Raises
    ------
    ValueError
        If initialization values don't match model structure
    ImportError
        If nutpie is not installed

    Notes
    -----
    This function:
    - Uses initial_points to control starting values (NOT init_mean in sample())
    - Sets jitter_rvs to prevent random perturbation of fragile distributions
    - Uses "support_point" strategy to respect initial_points exactly

    The fragile distributions that should NOT be jittered:
    - VonMisesFisher (u_k): requires exact unit norm
    - Mixture likelihood: sensitive to initial probabilities
    - GaussianRandomWalk (x_root): temporal correlation structure

    Examples
    --------
    >>> from gimbal.fit_params import initialize_from_observations_dlt
    >>> from gimbal.pymc_utils import compile_model_with_initialization
    >>>
    >>> # Get initialization
    >>> init_result = initialize_from_observations_dlt(y_obs, camera_proj, parents)
    >>>
    >>> # Build PyMC model
    >>> with pm.Model() as model:
    >>>     # ... define GIMBAL model ...
    >>>
    >>> # Compile with initialization
    >>> compiled = compile_model_with_initialization(model, init_result, parents)
    >>>
    >>> # Sample (initialization already handled)
    >>> trace = nutpie.sample(compiled, chains=2, tune=1000, draws=500)
    """
    try:
        import nutpie
    except ImportError:
        raise ImportError(
            "nutpie is required for this function. Install with: pip install nutpie"
        )

    # Build initial points dict
    initial_points = build_initial_points_for_nutpie(model, init_result, parents)

    # Validate before compilation
    validate_initial_points(model, initial_points)

    # Configure jitter strategy
    if allow_gaussian_jitter:
        # Only jitter Gaussian RVs - these are more robust to perturbation
        gaussian_rvs = {"eta2_root", "rho", "sigma2", "obs_sigma"}
        jitter_rvs = gaussian_rvs & set(model.named_vars.keys())
    else:
        # No jitter - safest for fragile distributions (vMF, mixture, GRW)
        jitter_rvs = set()

    # Compile model with proper initialization
    # - initial_points: our GIMBAL-computed starting values
    # - default_initialization_strategy: "support_point" respects initial_points
    # - jitter_rvs: controls which RVs get random perturbation
    compiled_model = nutpie.compile_pymc_model(
        model,
        initial_points=initial_points,
        default_initialization_strategy="support_point",
        jitter_rvs=jitter_rvs,
    )

    return compiled_model
