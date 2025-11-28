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


def build_camera_observation_model(
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
        Override default prior hyperparameters. Flat dictionary structure:
        - 'eta2_root_sigma': Root temporal variance prior (default: 0.1)
        - 'rho_sigma': Bone length prior std (default: 2.0)
        - 'sigma2_sigma': Bone variance prior std (default: 0.1)
        - 'obs_sigma_sigma': Observation noise prior std (default: 10.0)
        - 'inlier_prob_alpha': Inlier prob Beta prior alpha (default: 8)
        - 'inlier_prob_beta': Inlier prob Beta prior beta (default: 2)
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
        - eta2_root: scalar, root temporal variance
        - rho: (K-1,), mean bone lengths for non-root joints
        - sigma2: (K-1,), bone length variances

        **Dynamics:**
        - x_root: (T, 3), root trajectory (GaussianRandomWalk)
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
    ...         'inlier_prob_alpha': 10,
    ...         'inlier_prob_beta': 1
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
    KNOWN_HYPERPARAMS = {
        "eta2_root_sigma",
        "rho_sigma",
        "sigma2_sigma",
        "obs_sigma_sigma",
        "inlier_prob_alpha",
        "inlier_prob_beta",
    }
    KNOWN_KWARGS = {"sigma_dir"}

    # Check for unrecognized hyperparameters
    if prior_hyperparams is not None:
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

    # Merge defaults with user-provided hyperparameters
    default_hyperparams = {
        "eta2_root_sigma": 0.1,
        "rho_sigma": 2.0,
        "sigma2_sigma": 0.1,
        "obs_sigma_sigma": 10.0,
        "inlier_prob_alpha": 8,
        "inlier_prob_beta": 2,
    }
    hyperparams = {**default_hyperparams, **(prior_hyperparams or {})}

    # Extract kwargs with defaults
    sigma_dir = kwargs.get("sigma_dir", 1.0)

    # Extract initialization values
    eta2_init = init_result.eta2
    rho_init = init_result.rho
    sigma2_init = init_result.sigma2
    u_init = init_result.u_init
    obs_sigma_init = init_result.obs_sigma

    # Build PyMC model
    model = pm.Model()

    with model:
        # --- Skeletal parameters (with configurable priors) ---
        eta2_root = pm.HalfNormal(
            "eta2_root", sigma=hyperparams["eta2_root_sigma"], initval=eta2_init[0]
        )
        rho = pm.HalfNormal(
            "rho", sigma=hyperparams["rho_sigma"], shape=K - 1, initval=rho_init
        )
        sigma2 = pm.HalfNormal(
            "sigma2",
            sigma=hyperparams["sigma2_sigma"],
            shape=K - 1,
            initval=sigma2_init,
        )

        # --- Root joint (initialized from triangulated positions) ---
        # Shape: x_root will be (T, 3)
        # Handle NaN values in init_result.x_init by interpolation
        x_root_init = init_result.x_init[:, 0, :].copy()  # (T, 3)
        if np.isnan(x_root_init).any():
            x_root_init = _interpolate_nans(x_root_init)

        x_root = pm.GaussianRandomWalk(
            "x_root", mu=0, sigma=pt.sqrt(eta2_root), shape=(T, 3), initval=x_root_init
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
        obs_sigma = pm.HalfNormal(
            "obs_sigma", sigma=hyperparams["obs_sigma_sigma"], initval=obs_sigma_init
        )

        if use_mixture:
            # --- Mixture likelihood with outlier detection ---
            # Compute per-timestep log-likelihood: log_obs_t shape (T,)
            inlier_prob_init = (
                init_result.inlier_prob if hasattr(init_result, "inlier_prob") else 0.9
            )
            inlier_prob = pm.Beta(
                "inlier_prob",
                alpha=hyperparams["inlier_prob_alpha"],
                beta=hyperparams["inlier_prob_beta"],
                initval=inlier_prob_init,
            )

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
            log_obs_t = log_lik_masked.sum(axis=(0, 2))  # Sum over C and K dimensions
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
