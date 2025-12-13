"""
PyMC model builders for GIMBAL camera observation models.

This module provides reusable functions for constructing PyMC models with
Gaussian or mixture observation likelihoods. It consolidates the common model
structure used across notebooks and provides a clean API for model building.

Key functions:
- build_camera_observation_model: Build complete PyMC model with configurable likelihood
- project_points_pytensor: PyTensor-based camera projection for 3D->2D mapping
"""

from __future__ import annotations

import warnings
from typing import Optional, Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .fit_params import InitializationResult
from .pymc_utils import (
    _interpolate_nans,
    build_initial_points_for_nutpie,
    validate_initial_points,
)


def gamma_from_mode_sd(
    mode: float | np.ndarray, sd: float | np.ndarray
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    Convert desired mode and SD of a positive quantity into Gamma(alpha, beta)
    where beta is the rate parameter (1/scale).

    mode = (alpha - 1) / beta  for alpha > 1
    var  = alpha / beta**2     and sd = sqrt(var)

    Parameters
    ----------
    mode : float or array-like
        Desired mode of the Gamma distribution (must be positive)
    sd : float or array-like
        Desired standard deviation (must be positive)

    Returns
    -------
    alpha : float or ndarray
        Shape parameter of Gamma distribution
    beta : float or ndarray
        Rate parameter (1/scale) of Gamma distribution

    Raises
    ------
    ValueError
        If mode or sd are not positive

    Notes
    -----
    If arrays are provided, operates element-wise.
    """
    mode_arr = np.atleast_1d(mode)
    sd_arr = np.atleast_1d(sd)

    if np.any(mode_arr <= 0) or np.any(sd_arr <= 0):
        raise ValueError("mode and sd for Gamma must be positive")

    target = (sd_arr**2) / (mode_arr**2)

    # Start from alpha = 2 as a reasonable guess
    alpha = np.full_like(target, 2.0)
    for _ in range(20):
        num = alpha
        den = (alpha - 1.0) ** 2
        f = num / den - target  # f(alpha) = alpha / (alpha-1)^2 - target

        # Derivative: f'(alpha)
        df = ((alpha - 1.0) ** 2 - alpha * 2.0 * (alpha - 1.0)) / (alpha - 1.0) ** 4
        # If derivative is tiny, break
        converged = np.abs(df) < 1e-8
        if np.all(converged):
            break
        alpha_new = alpha - f / df
        alpha_new = np.maximum(alpha_new, 1.01)  # keep it > 1
        alpha = alpha_new

    beta = (alpha - 1.0) / mode_arr

    # Return scalar if input was scalar
    if alpha.size == 1:
        return float(alpha[0]), float(beta[0])
    return alpha.astype(np.float64), beta.astype(np.float64)


def project_points_pytensor(
    x: pt.TensorVariable, proj: pt.TensorVariable
) -> pt.TensorVariable:
    """
    Project 3D points to 2D using PyTensor operations.

    This function performs camera projection with perspective division,
    suitable for use within PyMC models where gradients are required.

    Parameters
    ----------
    x : pytensor.tensor.TensorVariable, shape (T, K, 3)
        3D joint positions over time
    proj : pytensor.tensor.TensorVariable, shape (C, 3, 4)
        Camera projection matrices [A_c | b_c] for each camera

    Returns
    -------
    y : pytensor.tensor.TensorVariable, shape (C, T, K, 2)
        2D projected keypoint positions for each camera, time, and joint

    Notes
    -----
    The projection follows the standard pinhole camera model:
    1. Convert to homogeneous coordinates: [x, y, z, 1]
    2. Apply projection matrix: P @ [x, y, z, 1]^T = [u*w, v*w, w]^T
    3. Perspective division: (u, v) = (u*w/w, v*w/w)

    A small epsilon (1e-6) is added to w to prevent division by zero.
    """
    # Homogeneous coordinates: (T, K, 4)
    ones = pt.ones((*x.shape[:-1], 1))
    x_h = pt.concatenate([x, ones], axis=-1)

    # Project through each camera: einsum('cij,tkj->ctki', proj, x_h)
    # Result: (C, T, K, 3)
    x_cam = pt.einsum("cij,tkj->ctki", proj, x_h)

    # Perspective division
    u = x_cam[:, :, :, 0]
    v = x_cam[:, :, :, 1]
    w = pt.maximum(x_cam[:, :, :, 2], 1e-6)  # Avoid division by zero

    # Stack to (C, T, K, 2)
    y = pt.stack([u / w, v / w], axis=-1)

    return y


def build_camera_observation_model_simple(
    y_obs: np.ndarray,
    proj_param: np.ndarray,
    parents: np.ndarray,
    bone_lengths: np.ndarray,  # Not used, kept for API compatibility
    **kwargs,
):
    """
    Simplified API for build_camera_observation_model with automatic DLT initialization.

    This is a convenience wrapper for v0.2.0 demos and notebooks that automatically
    performs DLT triangulation for initialization.

    Parameters
    ----------
    y_obs : ndarray, shape (C, T, K, 2)
        Observed 2D keypoints from C cameras over T frames for K joints.
    proj_param : ndarray, shape (C, 3, 4)
        Camera projection matrices.
    parents : ndarray, shape (K,)
        Parent joint indices for skeleton tree structure.
    bone_lengths : ndarray, shape (K-1,)
        Mean bone lengths from skeleton config (not used, DLT calculates automatically).
    **kwargs
        Additional arguments passed to build_camera_observation_model.

    Returns
    -------
    tuple
        (model, U, x_all, y_pred, log_obs_t) - Same as stage 2 output in v0.1 demos
    """
    from .fit_params import initialize_from_observations_dlt

    # Perform DLT initialization (automatically calculates bone lengths)
    init_result = initialize_from_observations_dlt(
        y_observed=y_obs,
        camera_proj=proj_param,
        parents=parents,
    )

    # Call the full function with init_result (use current model context)
    model_obj = _build_camera_observation_model_full(
        y_observed=y_obs,
        camera_proj=proj_param,
        parents=parents,
        init_result=init_result,
        use_mixture=False,  # Simple Gaussian for v0.2.0 demos
        **kwargs,
    )

    # Extract key variables for v0.1 compatibility from current model
    current_model = pm.modelcontext(None)
    U = current_model["U"]
    x_all = current_model["x_all"]
    y_pred = current_model["y_pred"]
    log_obs_t = current_model["log_obs_t"]

    return model_obj, U, x_all, y_pred, log_obs_t


def _build_camera_observation_model_full(
    y_observed: np.ndarray,
    camera_proj: np.ndarray,
    parents: np.ndarray,
    init_result: InitializationResult,
    use_mixture: bool = True,
    image_size: tuple[int, int] = (640, 480),
    prior_hyperparams: Optional[dict[str, float]] = None,
    validate_init_points: bool = False,
    use_directional_hmm: bool = False,
    hmm_num_states: Optional[int] = None,
    hmm_kwargs: Optional[dict] = None,
    **kwargs,
) -> pm.Model:
    """
    Build PyMC camera observation model for GIMBAL.

    This function creates a complete PyMC model for inferring 3D skeletal motion
    from 2D multi-camera observations. The model includes:
    - Skeletal dynamics (Gaussian random walk for root, normal bone lengths)
    - Directional vectors using Gaussian-normalize parameterization
    - Camera projection to 2D keypoints
    - Configurable observation likelihood (Gaussian or mixture with outliers)

    Parameters
    ----------
    y_observed : ndarray, shape (C, T, K, 2)
        Observed 2D keypoints from C cameras over T frames for K joints.
        Can contain NaN values for occluded/missing observations.
    camera_proj : ndarray, shape (C, 3, 4)
        Camera projection matrices [A_c | b_c] for each camera.
    parents : ndarray, shape (K,)
        Parent joint indices for skeleton tree structure. Root has parent -1.
    init_result : InitializationResult
        Initialization values from initialize_from_observations_dlt() or
        initialize_from_observations_anipose(). Must contain:
        - x_init: (T, K, 3) initial 3D positions
        - eta2: (K,) temporal variances
        - rho: (K-1,) mean bone lengths
        - sigma2: (K-1,) bone length variances
        - u_init: (T, K, 3) initial direction vectors
        - obs_sigma: observation noise estimate
        - inlier_prob: inlier probability (for mixture model)
    use_mixture : bool, default=True
        If True, use mixture likelihood (Gaussian inliers + Uniform outliers).
        If False, use simple Gaussian likelihood.
    image_size : tuple[int, int], default=(640, 480)
        Image dimensions (width, height) for uniform outlier distribution.
        Required when use_mixture=True.
    prior_hyperparams : dict[str, float], optional
        **DEPRECATED.** Hyperparameters are now data-driven from DLT initialization.
        Prior hyperparameters previously controlled prior widths but are no longer used.
        The model now uses Gamma priors with mode/SD derived from init_result.
        Legacy keys (eta2_root_sigma, rho_sigma, sigma2_sigma) are ignored with a warning.
    validate_init_points : bool, default=False
        If True, validate initialization values against model structure
        (shapes, dtypes, finite values). Raises ValueError on mismatch.
        Useful for debugging initialization issues.
    use_directional_hmm : bool, default=False
        If True, add a directional HMM prior over joint directions (Stage 3).
        Requires hmm_num_states to be specified.
    hmm_num_states : int, optional
        Number of hidden states in the directional HMM. Required when
        use_directional_hmm=True.
    hmm_kwargs : dict, optional
        Additional keyword arguments for add_directional_hmm_prior():
        - 'name_prefix': str, prefix for HMM variable names (default: "dir_hmm")
        - 'share_kappa_across_joints': bool (default: False)
        - 'share_kappa_across_states': bool (default: False)
        - 'kappa_scale': float, scale for kappa prior (default: 5.0)
    **kwargs : dict
        Additional configuration options:
        - 'sigma_dir': Standard deviation for raw directional vectors (default: 1.0)

    Returns
    -------
    model : pm.Model
        PyMC model ready for sampling. Contains the following random variables:

        **Skeletal parameters:**
        - eta2_root: scalar, root temporal variance (Gamma prior, 100% CV)
        - rho: (K-1,), mean bone lengths for non-root joints (Gamma prior, 100% CV)
        - sigma2: (K-1,), bone length variances (Gamma prior, 100% CV)

        **Dynamics:**
        - x0_root: (3,), initial root position (Normal, anchored to DLT estimate)
        - eps_root: (T-1, 3), root increments (Normal with Gamma-constrained variance)
        - x_root: (T, 3), root trajectory (Deterministic = x0 + cumsum(eps))
        - raw_u_{k}: (T, 3), raw directional vectors for joint k=1..K-1
        - u_{k}: (T, 3), normalized directional vectors (Deterministic)
        - length_{k}: (T,), bone lengths over time for joint k=1..K-1
        - x_{k}: (T, 3), joint k positions (Deterministic)

        **Observation:**
        - y_pred: (C, T, K, 2), predicted 2D projections (Deterministic)
        - obs_sigma: scalar, observation noise standard deviation

        **Likelihood (Gaussian mode, use_mixture=False):**
        - y_obs: (C, T, K, 2), observed data (Normal likelihood)

        **Likelihood (Mixture mode, use_mixture=True):**
        - inlier_prob: scalar, probability of inlier observation
        - y_mixture: Potential for mixture log-likelihood

    Raises
    ------
    ValueError
        If validate_init_points=True and initialization values don't match model.

    Warnings
    --------
    UserWarning
        Issued for unrecognized keys in prior_hyperparams or kwargs.

    Examples
    --------
    >>> from gimbal.fit_params import initialize_from_observations_dlt
    >>> from gimbal.pymc_model import build_camera_observation_model
    >>>
    >>> # Initialize from observations
    >>> result = initialize_from_observations_dlt(y_obs, camera_proj, parents)
    >>>
    >>> # Build simple Gaussian model
    >>> model = build_camera_observation_model(
    ...     y_observed=y_obs,
    ...     camera_proj=camera_proj,
    ...     parents=parents,
    ...     init_result=result,
    ...     use_mixture=False
    ... )
    >>>
    >>> # Build mixture model with custom hyperparameters
    >>> model = build_camera_observation_model(
    ...     y_observed=y_obs,
    ...     camera_proj=camera_proj,
    ...     parents=parents,
    ...     init_result=result,
    ...     use_mixture=True,
    ...     image_size=(640, 480),
    ...     prior_hyperparams={
    ...         'eta2_root_sigma': 0.05,
    ...         'rho_sigma': 1.0,
    ...     },
    ...     sigma_dir=0.5
    ... )
    >>>
    >>> # Sample with nutpie
    >>> import nutpie
    >>> compiled = nutpie.compile_pymc_model(model)
    >>> trace = nutpie.sample(compiled, chains=2, tune=1000, draws=500)

    Notes
    -----
    **Gaussian-normalize parameterization:**
    Instead of sampling directional vectors directly on the unit sphere
    (which requires constrained sampling), this model samples unconstrained
    3D vectors raw_u_k ~ Normal(0, sigma_dir) and then normalizes them
    deterministically: u_k = raw_u_k / ||raw_u_k||. This improves sampler
    efficiency and stability.

    **NaN handling:**
    Missing observations (NaN values in y_observed) are automatically handled:
    - In initialization: NaN values in x_root are interpolated
    - In Gaussian mode: NaN observations are ignored by PyMC
    - In mixture mode: NaN observations are explicitly masked out

    **Nutpie compatibility:**
    The model uses exact variable naming conventions expected by
    gimbal.pymc_utils functions for nutpie sampling. All RV names are fixed
    and follow the pattern documented in build_initial_points_for_nutpie().

    See Also
    --------
    project_points_pytensor : PyTensor camera projection function
    gimbal.fit_params.initialize_from_observations_dlt : DLT initialization
    gimbal.fit_params.initialize_from_observations_anipose : Anipose initialization
    gimbal.pymc_utils.compile_model_with_initialization : Compile for nutpie
    """
    # Extract dimensions
    C, T, K, _ = y_observed.shape

    if len(parents) != K:
        raise ValueError(
            f"parents length ({len(parents)}) must match K ({K}) from y_observed shape"
        )

    # Define known hyperparameters and kwargs
    # NOTE: Skeletal variance priors (eta2_root, rho, sigma2) are now data-driven
    # from DLT initialization. Only observation noise parameters remain configurable.
    KNOWN_HYPERPARAMS = {
        "eta2_root_sigma",  # DEPRECATED: ignored with warning
        "rho_sigma",  # DEPRECATED: ignored with warning
        "sigma2_sigma",  # DEPRECATED: ignored with warning
        "obs_sigma_sigma",  # Kept for backward compatibility
        "obs_sigma_mode",  # Mode for observation noise Gamma prior
        "obs_sigma_sd",  # SD for observation noise Gamma prior
        "inlier_prob_alpha",  # DEPRECATED: kept for backward compatibility
        "inlier_prob_beta",  # DEPRECATED: kept for backward compatibility
    }
    KNOWN_KWARGS = {"sigma_dir"}

    # Warn about deprecated skeletal variance hyperparameters
    if prior_hyperparams is not None:
        deprecated_skeletal = {"eta2_root_sigma", "rho_sigma", "sigma2_sigma"}
        used_deprecated = deprecated_skeletal & set(prior_hyperparams.keys())
        if used_deprecated:
            warnings.warn(
                f"Skeletal variance hyperparameters {used_deprecated} are deprecated and ignored. "
                f"Priors are now data-driven from DLT initialization with 50% CV.",
                DeprecationWarning,
            )

        unknown_hyperparams = set(prior_hyperparams.keys()) - KNOWN_HYPERPARAMS
        if unknown_hyperparams:
            warnings.warn(
                f"Unrecognized hyperparameter(s) in prior_hyperparams will be ignored: "
                f"{unknown_hyperparams}. Known hyperparameters: {KNOWN_HYPERPARAMS}",
                UserWarning,
            )

    # Check for unrecognized kwargs
    unknown_kwargs = set(kwargs.keys()) - KNOWN_KWARGS
    if unknown_kwargs:
        warnings.warn(
            f"Unrecognized keyword argument(s) will be ignored: {unknown_kwargs}. "
            f"Known kwargs: {KNOWN_KWARGS}",
            UserWarning,
        )

    # Merge defaults with user-provided hyperparameters (only for obs_sigma)
    default_hyperparams = {
        "obs_sigma_mode": 1.0,  # Mode in pixels (tune later)
        "obs_sigma_sd": 0.5,  # SD in pixels (tune later)
    }
    hyperparams = {**default_hyperparams, **(prior_hyperparams or {})}

    # Backward compatibility: if obs_sigma_sigma is provided, ignore mode/sd
    if prior_hyperparams and "obs_sigma_sigma" in prior_hyperparams:
        warnings.warn(
            "obs_sigma_sigma is deprecated. Use obs_sigma_mode and obs_sigma_sd instead. "
            "Falling back to HalfNormal prior.",
            DeprecationWarning,
        )

    # Extract kwargs with defaults
    sigma_dir = kwargs.get("sigma_dir", 1.0)

    # Extract initialization values
    eta2_init = init_result.eta2
    rho_init = init_result.rho
    sigma2_init = init_result.sigma2
    u_init = init_result.u_init
    obs_sigma_init = init_result.obs_sigma

    # Build PyMC model using the current model context if available
    model = pm.modelcontext(None)

    with model:
        # --- Skeletal parameters (with Gamma priors for variance parameters) ---
        # Use Gamma priors derived from DLT initialization to prevent funneling
        # Floor constraints: mode >= 0.01 for reasonable physical scales
        # Relaxed coefficient of variation (100% CV) for more flexibility

        # Root temporal variance: data-driven Gamma prior with 100% CV
        eta2_root_mode = max(0.01, float(eta2_init[0]))
        eta2_root_sd = eta2_root_mode * 1.0  # 100% coefficient of variation (relaxed)
        alpha_eta2_root, beta_eta2_root = gamma_from_mode_sd(
            eta2_root_mode, eta2_root_sd
        )
        eta2_root = pm.Gamma(
            "eta2_root",
            alpha=alpha_eta2_root,
            beta=beta_eta2_root,
            initval=eta2_init[0],
        )

        # Bone length scales: data-driven Gamma priors with 100% CV
        rho_mode = np.maximum(0.01, rho_init)
        rho_sd = rho_mode * 1.0
        alpha_rho, beta_rho = gamma_from_mode_sd(rho_mode, rho_sd)
        rho = pm.Gamma(
            "rho", alpha=alpha_rho, beta=beta_rho, shape=K - 1, initval=rho_init
        )

        # Direction variances: data-driven Gamma priors with 100% CV
        sigma2_mode = np.maximum(0.01, sigma2_init)
        sigma2_sd = sigma2_mode * 1.0
        alpha_sigma2, beta_sigma2 = gamma_from_mode_sd(sigma2_mode, sigma2_sd)
        sigma2 = pm.Gamma(
            "sigma2",
            alpha=alpha_sigma2,
            beta=beta_sigma2,
            shape=K - 1,
            initval=sigma2_init,
        )

        # --- Root joint (centered GRW with anchored initial position) ---
        # CRITICAL: Must anchor x_root[0] to DLT initialization via init_dist
        # Without this, PyMC defaults to Normal(0, 100) which is catastrophically
        # wrong when camera coordinates expect root at ~1-2m scale, causing
        # extreme curvature and 100% divergences.
        # Shape: x_root will be (T, 3)

        # Handle NaN values in init_result.x_init by interpolation
        x_root_init = init_result.x_init[:, 0, :].copy()  # (T, 3)
        if np.isnan(x_root_init).any():
            x_root_init = _interpolate_nans(x_root_init)

        # Anchored random walk: initial position centered at DLT estimate
        # with 1.0m std (weakly informative in world units)
        # Use a separate RV for x0 to avoid shape issues with init_dist
        ROOT_ANCHOR_SCALE = 1.0  # meters, reasonable for skeletal tracking
        x0_root = pm.Normal(
            "x0_root",
            mu=x_root_init[0, :],
            sigma=ROOT_ANCHOR_SCALE,
            shape=(3,),
            initval=x_root_init[0, :],
        )

        # Increments: epsilon[t] ~ Normal(0, sqrt(eta2)) for t=1..T-1
        eps_root = pm.Normal(
            "eps_root",
            mu=0.0,
            sigma=pt.sqrt(eta2_root),
            shape=(T - 1, 3),
            initval=np.diff(x_root_init, axis=0),
        )

        # Build trajectory: x[t] = x0 + cumsum(eps)
        x_root = pm.Deterministic(
            "x_root",
            pt.concatenate(
                [x0_root[None, :], x0_root + pt.extra_ops.cumsum(eps_root, axis=0)],
                axis=0,
            ),
        )

        # --- Directional vectors (Gaussian-normalize parameterization) ---
        # Each u_k will be shape (T, 3), normalized to unit vectors
        u_all = []

        for k in range(1, K):
            # Sample unconstrained 3D vectors: (T, 3)
            raw_u_k = pm.Normal(
                f"raw_u_{k}",
                mu=0.0,
                sigma=sigma_dir,
                shape=(T, 3),
                initval=u_init[:, k, :],  # u_init are already normalized
            )

            # Normalize to unit length along the last axis with epsilon for stability
            norm_raw = pt.sqrt((raw_u_k**2).sum(axis=-1, keepdims=True) + 1e-8)
            u_k = pm.Deterministic(f"u_{k}", raw_u_k / norm_raw)  # (T, 3)
            u_all.append(u_k)

        # --- Child joints ---
        # Build skeleton by traversing kinematic tree
        x_joints = [x_root]  # List will contain (T, 3) for each joint

        for k_idx, k in enumerate(range(1, K)):
            parent_k = parents[k]

            # Bone lengths over time: (T,)
            length_k = pm.Normal(
                f"length_{k}",
                mu=rho[k_idx],
                sigma=pt.sqrt(sigma2[k_idx]),
                shape=T,
                initval=rho_init[k_idx] * np.ones(T),
            )

            # Child joint position: x[k] = x[parent[k]] + length[k] * u[k]
            # Shape: (T, 3) = (T, 3) + (T, 1) * (T, 3)
            x_k = pm.Deterministic(
                f"x_{k}", x_joints[parent_k] + length_k[:, None] * u_all[k_idx]
            )
            x_joints.append(x_k)

        # Stack all joint positions: (T, K, 3)
        x_all = pt.stack(x_joints, axis=1)
        pm.Deterministic("x_all", x_all)  # Expose for v0.1.3 interface

        # Stack directional vectors: (T, K, 3)
        # Root direction is zero (unused), non-root from u_all
        U = pt.stack([pt.zeros((T, 3))] + u_all, axis=1)  # (T, K, 3)
        pm.Deterministic("U", U)  # Expose for v0.1.3 interface

        # --- Camera projection ---
        proj_param = pm.Data("camera_proj", camera_proj)  # (C, 3, 4)
        y_pred = pm.Deterministic(
            "y_pred", project_points_pytensor(x_all, proj_param)
        )  # (C, T, K, 2)

        # --- Observation likelihood ---
        # Use Gamma prior for obs_sigma (or HalfNormal for backward compatibility)
        if "obs_sigma_sigma" in hyperparams and "obs_sigma_sigma" in (
            prior_hyperparams or {}
        ):
            # Backward compatibility: use HalfNormal if explicitly provided
            obs_sigma = pm.HalfNormal(
                "obs_sigma",
                sigma=hyperparams["obs_sigma_sigma"],
                initval=obs_sigma_init,
            )
        else:
            # New approach: Gamma prior with mode/SD
            mode = hyperparams["obs_sigma_mode"]
            sd = hyperparams["obs_sigma_sd"]
            alpha, beta = gamma_from_mode_sd(mode, sd)
            obs_sigma = pm.Gamma(
                "obs_sigma",
                alpha=alpha,
                beta=beta,  # rate parameter
                initval=obs_sigma_init,
            )

        if use_mixture:
            # --- Mixture likelihood with outlier detection ---
            # Compute per-timestep log-likelihood: log_obs_t shape (T,)
            inlier_prob_init = (
                init_result.inlier_prob if hasattr(init_result, "inlier_prob") else 0.9
            )
            # Use logodds parameterization instead of Beta to avoid boundary issues
            # logodds = log(p / (1-p)), sample directly on unbounded space
            logodds_init = np.log(inlier_prob_init / (1.0 - inlier_prob_init))
            logodds = pm.Normal(
                "logodds_inlier",
                mu=0.0,
                sigma=5.0,  # Prior centered at p=0.5, allows wide range
                initval=logodds_init,
            )
            inlier_prob = pm.Deterministic("inlier_prob", pm.math.sigmoid(logodds))

            # Mask out occluded observations (NaN values)
            valid_mask = ~np.isnan(y_observed)  # (C, T, K, 2)
            valid_obs_mask = (
                valid_mask[:, :, :, 0] & valid_mask[:, :, :, 1]
            )  # (C, T, K)

            # Compute log-likelihood per observation point (C, T, K)
            # Normal component (inliers) - compute for all points
            # Shape: y_pred is (C, T, K, 2), y_observed is (C, T, K, 2)
            normal_logp_per_coord = pm.logp(
                pm.Normal.dist(mu=y_pred, sigma=obs_sigma), y_observed
            )  # (C, T, K, 2)
            normal_logp_per_point = normal_logp_per_coord.sum(
                axis=-1
            )  # (C, T, K) - sum over (u,v)

            # Uniform component (outliers)
            image_width, image_height = image_size
            uniform_logp = -pt.log(float(image_width)) - pt.log(
                float(image_height)
            )  # scalar

            # Log mixture for each observation point
            log_mix_per_point = pt.logaddexp(
                pt.log(inlier_prob) + normal_logp_per_point,
                pt.log(1 - inlier_prob) + uniform_logp,
            )  # (C, T, K)

            # Apply valid mask and sum per timestep
            # Set invalid observations to zero contribution
            log_mix_masked = pt.where(
                valid_obs_mask, log_mix_per_point, 0.0
            )  # (C, T, K)

            # Sum over cameras and joints to get per-timestep likelihood: (T,)
            log_obs_t = log_mix_masked.sum(axis=(0, 2))  # Sum over C and K dimensions
            pm.Deterministic("log_obs_t", log_obs_t)  # Expose for v0.1.3 interface

            # Total likelihood
            pm.Potential("y_obs", log_obs_t.sum())

        else:
            # --- Simple Gaussian likelihood ---
            # Compute per-timestep log-likelihood: log_obs_t shape (T,)

            # Compute log-likelihood per observation: (C, T, K, 2)
            logp_per_coord = pm.logp(
                pm.Normal.dist(mu=y_pred, sigma=obs_sigma), y_observed
            )  # (C, T, K, 2)

            # Sum over (u,v) coordinates: (C, T, K)
            logp_per_point = logp_per_coord.sum(axis=-1)

            # Mask out NaN observations (PyMC logp returns -inf for NaN observations)
            # We need to explicitly handle this for per-timestep summation
            valid_mask = ~np.isnan(y_observed)  # (C, T, K, 2)
            valid_obs_mask = (
                valid_mask[:, :, :, 0] & valid_mask[:, :, :, 1]
            )  # (C, T, K)

            # Set invalid observations to zero contribution
            logp_masked = pt.where(valid_obs_mask, logp_per_point, 0.0)  # (C, T, K)

            # Sum over cameras and joints to get per-timestep likelihood: (T,)
            log_obs_t = logp_masked.sum(axis=(0, 2))  # Sum over C and K dimensions
            pm.Deterministic("log_obs_t", log_obs_t)  # Expose for v0.1.3 interface

            # Total likelihood - use Potential instead of observed RV
            pm.Potential("y_obs", log_obs_t.sum())

        # --- v0.1.3: Optional Directional HMM Prior ---
        if use_directional_hmm:
            if hmm_num_states is None:
                raise ValueError(
                    "hmm_num_states must be provided when use_directional_hmm=True"
                )

            from gimbal.hmm_directional import add_directional_hmm_prior

            _hmm_result = add_directional_hmm_prior(
                U=U,
                log_obs_t=log_obs_t,
                S=hmm_num_states,
                **(hmm_kwargs or {}),
            )

    # Optional: Validate initialization values
    if validate_init_points:
        initial_points = build_initial_points_for_nutpie(model, init_result, parents)
        validate_initial_points(model, initial_points)

    return model
