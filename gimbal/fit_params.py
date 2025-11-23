"""Parameter estimation utilities for GIMBAL (Section 4 of spec).

These functions provide reasonable initial values for the model
parameters using a small ground-truth dataset of 3D joint positions
(and optionally observed 2D keypoints for the outlier model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from .model import (
    GimbalParameters,
    HeadingPriorParameters,
    OutlierMixtureParameters,
    PosePriorParameters,
    RootDynamicsParameters,
    SkeletonParameters,
)


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

    from .camera import project_points

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
