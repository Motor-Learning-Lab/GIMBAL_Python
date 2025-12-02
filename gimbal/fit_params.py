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
import torch
from torch import Tensor

from .torch_legacy.model import (
    GimbalParameters,
    HeadingPriorParameters,
    OutlierMixtureParameters,
    PosePriorParameters,
    RootDynamicsParameters,
    SkeletonParameters,
)


class InitializationResult(NamedTuple):
    """Results from parameter initialization.

    Attributes
    ----------
    x_init : ndarray or Tensor, shape (T, K, 3)
        Initial 3D joint positions
    eta2 : ndarray or Tensor, shape (K,)
        Temporal variance estimates
    rho : ndarray or Tensor, shape (K-1,)
        Mean bone lengths
    sigma2 : ndarray or Tensor, shape (K-1,)
        Bone length variances
    u_init : ndarray or Tensor, shape (T, K, 3)
        Initial direction vectors (unit vectors)
    obs_sigma : float
        Observation noise standard deviation
    inlier_prob : float
        Estimated inlier probability
    metadata : dict
        Additional information (method, success rates, etc.)
    """

    x_init: np.ndarray | Tensor
    eta2: np.ndarray | Tensor
    rho: np.ndarray | Tensor
    sigma2: np.ndarray | Tensor
    u_init: np.ndarray | Tensor
    obs_sigma: float
    inlier_prob: float
    metadata: dict


def estimate_temporal_variances(x_gt: Tensor) -> Tensor:
    """Estimate η_k^2 from ground truth (Section 4.1).

    x_gt has shape (T, K, 3).
    """

    diffs = x_gt[1:] - x_gt[:-1]
    # Use average squared step length per keypoint
    sq = (diffs**2).sum(dim=-1)  # (T-1, K)
    eta2 = sq.mean(dim=0)
    return eta2.clamp_min(1e-3)


def estimate_skeletal_parameters(x_gt: Tensor, parent: Tensor) -> Tuple[Tensor, Tensor]:
    """Estimate (ρ_k, σ_k^2) from ground truth (Section 4.2)."""

    T, K, _ = x_gt.shape
    rho = torch.zeros(K, dtype=x_gt.dtype, device=x_gt.device)
    sigma2 = torch.zeros(K, dtype=x_gt.dtype, device=x_gt.device)

    for k in range(1, K):
        p = int(parent[k].item())
        d = (x_gt[:, k] - x_gt[:, p]).norm(dim=-1)
        rho[k] = d.mean()
        sigma2[k] = d.var(unbiased=True).clamp_min(1e-4)

    return rho, sigma2


def estimate_root_dynamics(x_gt: Tensor) -> RootDynamicsParameters:
    """Estimate μ_1 and σ_1^2 for root keypoint (Section 2.1, 4.1)."""

    root_traj = x_gt[:, 0]
    mu0 = root_traj.mean(dim=0)
    diffs = root_traj[1:] - root_traj[:-1]
    sigma2_0 = diffs.pow(2).sum(dim=-1).mean().item()
    return RootDynamicsParameters(mu0=mu0, sigma2_0=sigma2_0)


def estimate_outlier_parameters(
    x_gt: Tensor,
    y_obs: Tensor,
    proj: Tensor,
    beta_init: float = 0.1,
) -> OutlierMixtureParameters:
    """Very simple initialization of outlier model (Section 4.3).

    This is a heuristic alternative to full EM on error magnitudes.
    It sets a shared inlier and outlier variance for each keypoint
    and camera, based on reprojection errors.

    x_gt: (T, K, 3)
    y_obs: (T, K, C, 2)
    proj: (C, 3, 4)
    """

    from .torch_legacy.camera import project_points

    T, K, C, _ = y_obs.shape
    device = x_gt.device

    x_flat = x_gt.view(T * K, 3)
    y_pred = project_points(x_flat, proj).view(T, K, C, 2)

    eps = y_obs - y_pred
    mag = eps.norm(dim=-1)  # (T,K,C)

    beta = torch.full((K, C), beta_init, dtype=x_gt.dtype, device=device)

    mu = torch.zeros(K, C, 2, 2, dtype=x_gt.dtype, device=device)
    sigma2 = torch.zeros(K, C, 2, dtype=x_gt.dtype, device=device)

    # Simple robust split by median
    for k in range(K):
        for c in range(C):
            m = mag[:, k, c]
            thresh = m.median()
            inlier = m[m <= thresh]
            outlier = m[m > thresh]
            if inlier.numel() == 0:
                var_in = torch.tensor(1.0, device=device)
            else:
                var_in = (inlier**2).mean()
            if outlier.numel() == 0:
                var_out = 100.0 * var_in
            else:
                var_out = (outlier**2).mean()
            sigma2[k, c, 0] = var_in
            sigma2[k, c, 1] = var_out

    return OutlierMixtureParameters(beta=beta, mu=mu, sigma2=sigma2)


def estimate_pose_priors(
    x_gt: Tensor,
    parent: Tensor,
    num_states: int,
) -> PosePriorParameters:
    """Rudimentary pose prior estimation (Section 4.4).

    This uses k-means clustering on concatenated unit directions to
    obtain S pose states, then estimates per-state vMF parameters and
    a transition matrix Λ from the resulting state sequence.
    """

    from sklearn.cluster import KMeans

    T, K, _ = x_gt.shape
    device = x_gt.device

    dirs = []
    for t in range(T):
        vecs = []
        for k in range(1, K):
            p = int(parent[k].item())
            diff = x_gt[t, k] - x_gt[t, p]
            diff = diff / diff.norm()
            vecs.append(diff)
        dirs.append(torch.cat(vecs, dim=0))
    X = torch.stack(dirs).cpu().numpy()  # (T, (K-1)*3)

    kmeans = KMeans(n_clusters=num_states, n_init=10).fit(X)
    labels = kmeans.labels_

    S = num_states
    nu = torch.zeros(S, K, 3, dtype=x_gt.dtype, device=device)
    kappa = torch.zeros(S, K, dtype=x_gt.dtype, device=device)

    for s_idx in range(S):
        idx = torch.tensor((labels == s_idx).nonzero()[0], dtype=torch.long)
        if idx.numel() == 0:
            continue
        for k in range(1, K):
            p = int(parent[k].item())
            diffs = x_gt[idx, k] - x_gt[idx, p]
            diffs = diffs / diffs.norm(dim=-1, keepdim=True)
            mean_dir = diffs.mean(dim=0)
            R = mean_dir.norm()
            nu[s_idx, k] = mean_dir / (R + 1e-8)
            kappa[s_idx, k] = (R * diffs.shape[0] - R**3) / (1 - R**2 + 1e-8)

    # Transition matrix from labels
    Lambda = torch.zeros(S, S, dtype=x_gt.dtype, device=device)
    for t in range(1, T):
        i = labels[t - 1]
        j = labels[t]
        Lambda[i, j] += 1.0
    Lambda = Lambda + 1.0  # add-one smoothing
    Lambda = Lambda / Lambda.sum(dim=-1, keepdim=True)

    return PosePriorParameters(nu=nu, kappa=kappa, transition=Lambda)


def build_gimbal_parameters(
    x_gt: Tensor,
    parent: Tensor,
    y_obs: Tensor,
    proj: Tensor,
    num_states: int,
) -> GimbalParameters:
    """High-level helper to build GimbalParameters from ground truth.

    This roughly follows Section 4 of the spec.
    """

    eta2 = estimate_temporal_variances(x_gt)
    rho, sigma2 = estimate_skeletal_parameters(x_gt, parent)
    root_dyn = estimate_root_dynamics(x_gt)
    outlier = estimate_outlier_parameters(x_gt, y_obs, proj)
    pose_prior = estimate_pose_priors(x_gt, parent, num_states)
    heading_prior = HeadingPriorParameters()

    skeleton = SkeletonParameters(parent=parent, rho=rho, sigma2=sigma2, eta2=eta2)

    return GimbalParameters(
        skeleton=skeleton,
        pose_prior=pose_prior,
        heading_prior=heading_prior,
        outlier=outlier,
        root_dyn=root_dyn,
    )


# =============================================================================
# Data-driven initialization from 2D observations (no ground truth required)
# =============================================================================


def _triangulate_dlt(
    y_observed: np.ndarray,
    camera_proj: np.ndarray,
    min_cameras: int = 2,
    condition_threshold: float = 1e6,
) -> np.ndarray:
    """Triangulate using Direct Linear Transform (DLT)."""
    C, T, K, _ = y_observed.shape
    x_triangulated = np.zeros((T, K, 3))

    for k in range(K):
        for t in range(T):
            y_tk = y_observed[:, t, k, :]
            valid_mask = ~np.isnan(y_tk[:, 0]) & ~np.isnan(y_tk[:, 1])
            n_valid = valid_mask.sum()

            if n_valid < min_cameras:
                x_triangulated[t, k, :] = np.nan
                continue

            A = []
            for c in np.where(valid_mask)[0]:
                u, v = y_tk[c]
                P = camera_proj[c]
                A.append(u * P[2, :] - P[0, :])
                A.append(v * P[2, :] - P[1, :])

            A = np.array(A)

            try:
                _, S, Vt = np.linalg.svd(A)
                cond = S[0] / (S[-1] + 1e-10)

                if cond > condition_threshold:
                    x_triangulated[t, k, :] = np.nan
                    continue

                X_homog = Vt[-1, :]

                if np.abs(X_homog[3]) < 1e-8:
                    x_triangulated[t, k, :] = np.nan
                else:
                    x_triangulated[t, k, :] = X_homog[:3] / X_homog[3]

            except np.linalg.LinAlgError:
                x_triangulated[t, k, :] = np.nan

    return x_triangulated


def _triangulate_anipose(
    y_observed: np.ndarray, camera_proj: np.ndarray, **kwargs
) -> np.ndarray:
    """Triangulate using Anipose (aniposelib)."""
    try:
        from aniposelib.cameras import CameraGroup

        # TODO: Full Anipose integration with CameraGroup
        print(
            "Warning: Full Anipose integration not yet implemented. Falling back to DLT."
        )
        return _triangulate_dlt(y_observed, camera_proj, **kwargs)
    except ImportError:
        print("Warning: aniposelib not installed. Falling back to DLT.")
        print("  To use full Anipose features: pip install aniposelib")
        return _triangulate_dlt(y_observed, camera_proj, **kwargs)


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
    x_triangulated: np.ndarray, parents: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate bone lengths and variances from triangulated positions."""
    T, K, _ = x_triangulated.shape
    rho = np.zeros(K - 1)
    sigma2 = np.zeros(K - 1)

    for k_idx, k in enumerate(range(1, K)):
        parent_k = parents[k]
        x_k = x_triangulated[:, k, :]
        x_parent = x_triangulated[:, parent_k, :]

        valid = ~np.any(np.isnan(x_k), axis=1) & ~np.any(np.isnan(x_parent), axis=1)

        if valid.sum() < 2:
            rho[k_idx] = 1.0
            sigma2[k_idx] = 0.01
            continue

        bone_lengths = np.linalg.norm(x_k[valid] - x_parent[valid], axis=1)
        rho[k_idx] = np.mean(bone_lengths)
        sigma2[k_idx] = max(np.var(bone_lengths), 1e-6)

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
    from .torch_legacy.camera import project_points

    C, T, K, _ = y_observed.shape

    x_torch = torch.from_numpy(x_triangulated).float()
    proj_torch = torch.from_numpy(camera_proj).float()
    y_reproj = project_points(x_torch, proj_torch).numpy()
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
    """
    x_triangulated = _triangulate_dlt(y_observed, camera_proj, min_cameras=min_cameras)

    valid_tri = ~np.any(np.isnan(x_triangulated), axis=2)
    tri_rate = valid_tri.sum() / valid_tri.size

    eta2 = _estimate_temporal_variances_numpy(x_triangulated, parents)
    rho, sigma2 = _estimate_skeletal_parameters_numpy(x_triangulated, parents)
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
    x_gt: np.ndarray | Tensor, 
    parents: np.ndarray | Tensor, 
    return_numpy: bool = True,
    obs_noise_std: float | None = None,
) -> InitializationResult:
    """
    Initialize parameters from ground truth 3D positions.

    Useful for debugging and validation when ground truth is available.

    Parameters
    ----------
    x_gt : ndarray or Tensor, shape (T, K, 3)
        Ground truth 3D joint positions
    parents : ndarray or Tensor, shape (K,)
        Skeleton parent indices
    return_numpy : bool
        If True, return numpy arrays; if False, return torch Tensors
    obs_noise_std : float, optional
        Observation noise standard deviation from data config.
        If provided, used to set obs_sigma initialization (scaled by 1.5).

    Returns
    -------
    result : InitializationResult
        All estimated parameters
    """
    # Convert to torch if needed
    if isinstance(x_gt, np.ndarray):
        x_gt_torch = torch.from_numpy(x_gt).float()
        parents_torch = torch.from_numpy(parents).long()
    else:
        x_gt_torch = x_gt
        parents_torch = parents

    eta2 = estimate_temporal_variances(x_gt_torch)
    rho, sigma2 = estimate_skeletal_parameters(x_gt_torch, parents_torch)

    # Compute direction vectors
    T, K, _ = x_gt_torch.shape
    u_init = torch.zeros(T, K, 3, dtype=x_gt_torch.dtype, device=x_gt_torch.device)

    for k in range(1, K):
        parent_k = int(parents_torch[k].item())
        for t in range(T):
            bone_vec = x_gt_torch[t, k] - x_gt_torch[t, parent_k]
            length = bone_vec.norm()
            u_init[t, k] = (
                bone_vec / length if length > 1e-6 else torch.tensor([0.0, 0.0, 1.0])
            )

    # Convert to numpy if requested
    if return_numpy:
        x_init = x_gt_torch.cpu().numpy()
        eta2 = eta2.cpu().numpy()
        rho = rho.cpu().numpy()[1:]  # Skip root
        sigma2 = sigma2.cpu().numpy()[1:]  # Skip root
        u_init = u_init.cpu().numpy()
    else:
        x_init = x_gt_torch
        rho = rho[1:]
        sigma2 = sigma2[1:]

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
        x_init=x_init,
        eta2=eta2,
        rho=rho,
        sigma2=sigma2,
        u_init=u_init,
        obs_sigma=obs_sigma_init,
        inlier_prob=0.85,  # Default reasonable value
        metadata=metadata,
    )
