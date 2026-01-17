"""Parameter initialization for GIMBAL.

This module provides three initialization strategies:

1. **initialize_from_groundtruth()**: Uses ground truth 3D mocap data (Section 4 of GIMBAL spec)
   - For training/validation when ground truth is available
   - Returns torch.Tensor parameters

2. **initialize_from_observations_dlt()**: Uses DLT triangulation from 2D observations
   - For real applications without ground truth
   - Robust Direct Linear Transform with SVD
   - Returns numpy arrays

3. **initialize_from_observations_anipose()**: Uses Anipose triangulation from 2D observations
   - For real applications with Anipose installed
   - RANSAC outlier rejection + bundle adjustment
   - Returns numpy arrays

All functions return InitializationResult with consistent structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, NamedTuple, Literal, Optional, Callable

import numpy as np


class InitializationResult(NamedTuple):
    """Results from parameter initialization.

    Attributes, shape (T, K, 3)
        Initial 3D joint positions
    eta2 : ndarray, shape (K,)
        Temporal variance estimates
    rho : ndarray, shape (K-1,)
        Mean bone lengths
    sigma2 : ndarray, shape (K-1,)
        Bone length variances
    u_init : ndarray, shape (T, K, 3)
        Initial direction vectors (unit vectors)
    obs_sigma : float
        Observation noise standard deviation
    inlier_prob : float
        Estimated inlier probability
    metadata : dict
        Additional information (method, success rates, etc.)
    """

    x_init: np.ndarray
    eta2: np.ndarray
    rho: np.ndarray
    sigma2: np.ndarray
    u_init: np.ndarray
    sigma2: np.ndarray | Tensor
    u_init: np.ndarray | Tensor
    obs_sigma: float
    inlier_prob: float
    metadata: dict


# =============================================================================
# Data-driven initialization from 2D observations (no ground truth required)
# =============================================================================


def _triangulate_dlt(
    y_observed: np.ndarray,
    camera_proj: np.ndarray,
    min_cameras: int = 2,
    condition_threshold: float = 1e6,
) -> np.ndarray:
    """Triangulate using Direct Linear Transform (DLT).
    
    Wrapper that calls gimbal_pymc.cameras.triangulation.triangulate_dlt.
    """
    from gimbal_pymc.cameras.triangulation import triangulate_dlt
    return triangulate_dlt(y_observed, camera_proj, min_cameras, condition_threshold)


def _triangulate_anipose(
    y_observed: np.ndarray, camera_proj: np.ndarray, **kwargs
) -> np.ndarray:
    """Triangulate using Anipose (aniposelib).
    
    Wrapper that calls gimbal_pymc.cameras.triangulation.triangulate_anipose.
    """
    from gimbal_pymc.cameras.triangulation import triangulate_anipose
    return triangulate_anipose(y_observed, camera_proj, **kwargs)


def _estimate_temporal_variances_numpy(
    x_triangulated: np.ndarray, parents: np.ndarray
) -> np.ndarray:
    """Estimate temporal variance from triangulated positions."""
    T, K, _ = x_triangulated.shape
    eta2 = np.zeros(K)

    for k in range(K):
        x_k = x_triangulated[:, k, :]
        valid_frames = ~np.any(np.isnan(x_k), axis=1)

        if valid_frames.sum() < 2:
            eta2[k] = 0.01
            continue

        x_k_valid = x_k[valid_frames]
        deltas = np.diff(x_k_valid, axis=0)
        squared_displacements = np.sum(deltas**2, axis=1)

        median_sq = np.median(squared_displacements)
        mad = np.median(np.abs(squared_displacements - median_sq))
        eta2[k] = max(1.4826 * mad, 1e-4)

    return eta2


def _estimate_skeletal_parameters_numpy(
    x_triangulated: np.ndarray, parents: np.ndarray, eps_length: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate bone lengths and variances from triangulated positions.

    Uses robust estimators (median, MAD) and enforces strict validation
    to prevent invalid initialization.

    Parameters
    ----------
    x_triangulated : ndarray, shape (T, K, 3)
        Triangulated 3D positions (may contain NaN)
    parents : ndarray, shape (K,)
        Parent indices for skeleton
    eps_length : float
        Minimum valid bone length

    Returns
    -------
    rho : ndarray, shape (K-1,)
        Median bone lengths (strictly positive)
    sigma2 : ndarray, shape (K-1,)
        Robust bone length variances (strictly positive)

    Raises
    ------
    ValueError
        If any bone has insufficient valid samples or invalid estimates
    """
    T, K, _ = x_triangulated.shape
    rho = np.zeros(K - 1)
    sigma2 = np.zeros(K - 1)

    # Absolute and relative floors
    rho_floor = 0.001  # Absolute minimum (1mm in typical units)
    rel_sigma_floor = 0.01  # 1% of bone length
    abs_sigma_floor = 1e-6  # Absolute minimum variance

    # Minimum samples required
    min_samples_abs = 10
    min_samples_frac = 0.05  # 5% of frames
    min_samples_required = max(min_samples_abs, int(min_samples_frac * T))

    diagnostics = {}

    for k_idx, k in enumerate(range(1, K)):
        parent_k = parents[k]
        x_k = x_triangulated[:, k, :]
        x_parent = x_triangulated[:, parent_k, :]

        # Find valid frames: both positions finite
        valid_positions = ~np.any(np.isnan(x_k), axis=1) & ~np.any(
            np.isnan(x_parent), axis=1
        )

        if valid_positions.sum() == 0:
            raise ValueError(
                f"Bone {k_idx} (joint {k} -> parent {parent_k}): "
                f"No valid triangulated frames. Check 2D keypoint detection "
                f"and triangulation for these joints."
            )

        # Compute bone lengths
        bone_lengths = np.linalg.norm(
            x_k[valid_positions] - x_parent[valid_positions], axis=1
        )

        # Filter for positive, finite lengths
        valid_lengths = bone_lengths[
            np.isfinite(bone_lengths) & (bone_lengths > eps_length)
        ]

        n_valid = len(valid_lengths)

        # Check minimum sample requirement
        if n_valid < min_samples_required:
            raise ValueError(
                f"Bone {k_idx} (joint {k} -> parent {parent_k}): "
                f"Insufficient valid samples ({n_valid} < {min_samples_required}). "
                f"Valid positions: {valid_positions.sum()}/{T}, "
                f"Valid lengths (>eps): {n_valid}/{valid_positions.sum()}. "
                f"Inspect triangulation quality or increase keypoint visibility."
            )

        # Robust location estimate: median
        rho_k = np.median(valid_lengths)

        # Robust dispersion estimate: MAD -> SD
        mad = np.median(np.abs(valid_lengths - rho_k))
        sigma_k = 1.4826 * mad  # MAD to SD conversion

        # Apply floors
        sigma2_floor_k = max(abs_sigma_floor, (rel_sigma_floor * rho_k) ** 2)
        sigma2_k = max(sigma_k**2, sigma2_floor_k)

        # Validation: must be strictly positive and finite
        if not (np.isfinite(rho_k) and rho_k > rho_floor):
            raise ValueError(
                f"Bone {k_idx} (joint {k} -> parent {parent_k}): "
                f"Invalid rho estimate: {rho_k:.6f} (must be finite and > {rho_floor}). "
                f"Valid samples: {n_valid}, lengths range: [{valid_lengths.min():.6f}, {valid_lengths.max():.6f}]"
            )

        if not (np.isfinite(sigma2_k) and sigma2_k > abs_sigma_floor):
            raise ValueError(
                f"Bone {k_idx} (joint {k} -> parent {parent_k}): "
                f"Invalid sigma2 estimate: {sigma2_k:.6e} (must be finite and > {abs_sigma_floor}). "
                f"MAD: {mad:.6f}, Valid samples: {n_valid}"
            )

        rho[k_idx] = rho_k
        sigma2[k_idx] = sigma2_k

        # Store diagnostics
        diagnostics[k_idx] = {
            "n_valid": n_valid,
            "rho": float(rho_k),
            "sigma2": float(sigma2_k),
            "length_range": [float(valid_lengths.min()), float(valid_lengths.max())],
        }

    return rho, sigma2


def _estimate_direction_vectors_numpy(
    x_triangulated: np.ndarray, parents: np.ndarray
) -> np.ndarray:
    """Compute initial direction vectors from triangulated positions."""
    T, K, _ = x_triangulated.shape
    u_init = np.zeros((T, K, 3))

    for k in range(1, K):
        parent_k = parents[k]

        for t in range(T):
            x_k = x_triangulated[t, k, :]
            x_parent = x_triangulated[t, parent_k, :]

            if np.any(np.isnan(x_k)) or np.any(np.isnan(x_parent)):
                u_init[t, k, :] = np.array([0.0, 0.0, 1.0])
                continue

            bone_vec = x_k - x_parent
            length = np.linalg.norm(bone_vec)

            u_init[t, k, :] = (
                bone_vec / length if length > 1e-6 else np.array([0.0, 0.0, 1.0])
            )

    return u_init


def _estimate_observation_parameters_numpy(
    y_observed: np.ndarray,
    x_triangulated: np.ndarray,
    camera_proj: np.ndarray,
    outlier_threshold_px: float = 15.0,
) -> Tuple[float, float]:
    """Estimate observation noise and inlier probability from reprojection errors."""
    from gimbal_pymc.cameras.projection import project_points_numpy

    C, T, K, _ = y_observed.shape

    # Project 3D points to 2D using numpy-based projection
    # Shape: (T, K, C, 2)
    y_reproj = project_points_numpy(x_triangulated, camera_proj)
    # Transpose to (C, T, K, 2) to match y_observed
    y_reproj = np.transpose(y_reproj, (2, 0, 1, 3))

    errors = []
    for c in range(C):
        for t in range(T):
            for k in range(K):
                if np.any(np.isnan(y_observed[c, t, k, :])) or np.any(
                    np.isnan(y_reproj[c, t, k, :])
                ):
                    continue
                error = np.linalg.norm(y_observed[c, t, k, :] - y_reproj[c, t, k, :])
                errors.append(error)

    if len(errors) == 0:
        return 2.0, 0.85

    errors = np.array(errors)
    inlier_mask = errors < outlier_threshold_px

    obs_sigma = np.std(errors[inlier_mask]) if inlier_mask.sum() > 0 else 2.0
    obs_sigma = max(obs_sigma, 0.5)
    inlier_prob = inlier_mask.sum() / len(errors)
    # Clip to avoid boundary issues in logodds transformation
    # Keep within reasonable prior support while respecting data
    inlier_prob = np.clip(inlier_prob, 0.01, 0.99)

    return obs_sigma, inlier_prob


def initialize_from_observations_dlt(
    y_observed: np.ndarray,
    camera_proj: np.ndarray,
    parents: np.ndarray,
    min_cameras: int = 2,
    outlier_threshold_px: float = 15.0,
    **kwargs,
) -> InitializationResult:
    """
    Initialize parameters from 2D observations using DLT triangulation.

    Parameters
    ----------
    y_observed : ndarray, shape (C, T, K, 2)
        2D keypoint observations from C cameras
    camera_proj : ndarray, shape (C, 3, 4)
        Camera projection matrices
    parents : ndarray, shape (K,)
        Skeleton parent indices (-1 for root)
    min_cameras : int
        Minimum cameras required for triangulation
    outlier_threshold_px : float
        Reprojection error threshold for outlier classification

    Returns
    -------
    result : InitializationResult
        All estimated parameters

    Raises
    ------
    ValueError
        If skeletal parameter estimation fails (insufficient valid data)
    """
    x_triangulated = _triangulate_dlt(y_observed, camera_proj, min_cameras=min_cameras)

    valid_tri = ~np.any(np.isnan(x_triangulated), axis=2)
    tri_rate = valid_tri.sum() / valid_tri.size

    eta2 = _estimate_temporal_variances_numpy(x_triangulated, parents)

    try:
        rho, sigma2 = _estimate_skeletal_parameters_numpy(x_triangulated, parents)
    except ValueError as e:
        # Add context and re-raise
        raise ValueError(
            f"Skeletal parameter estimation failed after DLT triangulation. "
            f"Triangulation rate: {tri_rate:.2%}. Original error: {e}"
        ) from e

    u_init = _estimate_direction_vectors_numpy(x_triangulated, parents)
    obs_sigma, inlier_prob = _estimate_observation_parameters_numpy(
        y_observed, x_triangulated, camera_proj, outlier_threshold_px
    )

    metadata = {
        "method": "dlt",
        "triangulation_rate": tri_rate,
        "min_cameras": min_cameras,
        "outlier_threshold_px": outlier_threshold_px,
    }

    return InitializationResult(
        x_init=x_triangulated,
        eta2=eta2,
        rho=rho,
        sigma2=sigma2,
        u_init=u_init,
        obs_sigma=obs_sigma,
        inlier_prob=inlier_prob,
        metadata=metadata,
    )


def initialize_from_observations_anipose(
    y_observed: np.ndarray,
    camera_proj: np.ndarray,
    parents: np.ndarray,
    min_cameras: int = 2,
    outlier_threshold_px: float = 15.0,
    **kwargs,
) -> InitializationResult:
    """
    Initialize parameters from 2D observations using Anipose triangulation.

    Requires aniposelib: pip install aniposelib

    Parameters
    ----------
    y_observed : ndarray, shape (C, T, K, 2)
        2D keypoint observations from C cameras
    camera_proj : ndarray, shape (C, 3, 4)
        Camera projection matrices
    parents : ndarray, shape (K,)
        Skeleton parent indices (-1 for root)
    min_cameras : int
        Minimum cameras required for triangulation
    outlier_threshold_px : float
        Reprojection error threshold for outlier classification

    Returns
    -------
    result : InitializationResult
        All estimated parameters
    """
    x_triangulated = _triangulate_anipose(
        y_observed, camera_proj, min_cameras=min_cameras, **kwargs
    )

    valid_tri = ~np.any(np.isnan(x_triangulated), axis=2)
    tri_rate = valid_tri.sum() / valid_tri.size

    eta2 = _estimate_temporal_variances_numpy(x_triangulated, parents)
    rho, sigma2 = _estimate_skeletal_parameters_numpy(x_triangulated, parents)
    u_init = _estimate_direction_vectors_numpy(x_triangulated, parents)
    obs_sigma, inlier_prob = _estimate_observation_parameters_numpy(
        y_observed, x_triangulated, camera_proj, outlier_threshold_px
    )

    metadata = {
        "method": "anipose",
        "triangulation_rate": tri_rate,
        "min_cameras": min_cameras,
        "outlier_threshold_px": outlier_threshold_px,
    }

    return InitializationResult(
        x_init=x_triangulated,
        eta2=eta2,
        rho=rho,
        sigma2=sigma2,
        u_init=u_init,
        obs_sigma=obs_sigma,
        inlier_prob=inlier_prob,
        metadata=metadata,
    )


def initialize_from_groundtruth(
    x_gt: np.ndarray,
    parents: np.ndarray,
    obs_noise_std: float | None = None,
) -> InitializationResult:
    """
    Initialize parameters from ground truth 3D positions.

    Useful for debugging and validation when ground truth is available.
    Uses numpy-based estimation (torch support removed in v0.2.2).

    Parameters
    ----------
    x_gt : ndarray, shape (T, K, 3)
        Ground truth 3D joint positions
    parents : ndarray, shape (K,)
        Skeleton parent indices
    obs_noise_std : float, optional
        Observation noise standard deviation from data config.
        If provided, used to set obs_sigma initialization (scaled by 1.5).

    Returns
    -------
    result : InitializationResult
        All estimated parameters
    """
    T, K, _ = x_gt.shape
    
    # Use numpy-based estimation functions
    eta2 = _estimate_temporal_variances_numpy(x_gt, parents)
    rho, sigma2 = _estimate_skeletal_parameters_numpy(x_gt, parents)
    u_init = _estimate_direction_vectors_numpy(x_gt, parents)

    metadata = {
        "method": "groundtruth",
        "triangulation_rate": 1.0,
    }

    # Set obs_sigma based on obs_noise_std if available
    if obs_noise_std is not None:
        # Start a bit over the true noise; avoids under-estimation
        obs_sigma_init = max(0.1, float(obs_noise_std) * 1.5)
    else:
        obs_sigma_init = 2.0  # fallback

    return InitializationResult(
        x_init=x_gt,
        eta2=eta2,
        rho=rho,
        sigma2=sigma2,
        u_init=u_init,
        obs_sigma=obs_sigma_init,
        inlier_prob=0.85,  # Default reasonable value
        metadata=metadata,
    )
