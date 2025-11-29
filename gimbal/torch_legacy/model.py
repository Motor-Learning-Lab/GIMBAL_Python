"""Core probabilistic model components for GIMBAL.

This module mirrors Sections 1–2 and 4 of `GIMBAL spec.md`.

Key responsibilities
--------------------
- Data classes for model parameters (skeleton, temporal variances,
  vMF pose priors, HMM transitions, outlier mixture parameters).
- Log-density functions for each part of the generative model:
  * Root dynamics (Eq. 2.1)
  * Hierarchical vMFG prior for non-root keypoints (Eq. 2.2)
  * Pose-state-dependent directional prior (Eq. 2.3)
  * HMM over pose states (Eq. 2.4)
  * Heading prior (Eq. 2.5)
  * Robust observation model (Eq. 2.6)

All log densities are implemented using PyTorch tensors so that
`gimbal.inference` can use automatic differentiation for HMC.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SkeletonParameters:
    """Parameters describing the kinematic tree.

    Attributes
    ----------
    parent : Tensor
        Shape (K,), integer indices with parent[0] == -1 for root.
    rho : Tensor
        Shape (K,), average distances ρ_k (Section 2.2).
    sigma2 : Tensor
        Shape (K,), variances σ_k^2 (Section 2.2).
    eta2 : Tensor
        Shape (K,), temporal variances η_k^2 (Sections 2.1 and 2.2).
    """

    parent: Tensor
    rho: Tensor
    sigma2: Tensor
    eta2: Tensor


@dataclass
class PosePriorParameters:
    """Pose-state-dependent directional prior (Section 2.3 & 2.4).

    Attributes
    ----------
    nu : Tensor
        Mean directions ν_{s,k} on S^2, shape (S, K, 3).
    kappa : Tensor
        Concentrations κ_{s,k} ≥ 0, shape (S, K).
    transition : Tensor
        HMM transition matrix Λ, shape (S, S), rows sum to 1.
    """

    nu: Tensor
    kappa: Tensor
    transition: Tensor


@dataclass
class HeadingPriorParameters:
    """Heading prior parameters (Section 2.5).

    For now this is simply a placeholder for potential non-uniform
    von Mises parameters; the spec uses a uniform prior.
    """

    mu: float = 0.0
    kappa: float = 0.0


@dataclass
class OutlierMixtureParameters:
    """Robust observation model parameters (Section 2.6).

    Attributes
    ----------
    beta : Tensor
        Outlier probabilities β_{k,c}, shape (K, C).
    mu : Tensor
        Means μ_{k,c,z}, shape (K, C, 2, 2) where z in {0,1}.
    sigma2 : Tensor
        Variances ω_{k,c,z}^2, shape (K, C, 2).
    """

    beta: Tensor
    mu: Tensor
    sigma2: Tensor


@dataclass
class RootDynamicsParameters:
    """Root keypoint dynamics (Section 2.1)."""

    mu0: Tensor  # μ_1, shape (3,)
    sigma2_0: float  # σ_1^2


@dataclass
class GimbalParameters:
    """Container for all model parameters used in inference."""

    skeleton: SkeletonParameters
    pose_prior: PosePriorParameters
    heading_prior: HeadingPriorParameters
    outlier: OutlierMixtureParameters
    root_dyn: RootDynamicsParameters


def vmf_log_norm_const(kappa: Tensor, dim: int = 3) -> Tensor:
    """Log normalization constant for vMF on S^{dim-1}.

    This uses an approximation via log of modified Bessel function of
    the first kind. For dim=3, the exact form is

        C(κ) = κ / (4π sinh κ).

    Parameters
    ----------
    kappa : Tensor
        Concentration parameter κ ≥ 0.
    dim : int, optional
        Dimension of embedding space, default 3.
    """

    if dim != 3:
        raise NotImplementedError("Only dim=3 vMF is implemented.")

    # Avoid numerical issues at κ≈0 using series expansion.
    kappa = torch.clamp(kappa, min=1e-8)
    log_c = torch.log(kappa) - torch.log(4.0 * torch.pi * torch.sinh(kappa))
    return log_c


def vmf_log_pdf(x: Tensor, mu: Tensor, kappa: Tensor) -> Tensor:
    """Log-density of vMF on S^2 (Section 2.3) up to constants.

    Implements log vMF(x | μ, κ) = κ μ^T x - log C(κ).
    Shapes broadcast as needed.
    """

    # Ensure unit vectors
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    dot = (x * mu).sum(dim=-1)
    return kappa * dot - vmf_log_norm_const(kappa)


def log_normal_isotropic(x: Tensor, mean: Tensor, sigma2: float) -> Tensor:
    """Isotropic Gaussian log-density N(x | mean, σ^2 I)."""

    d = x.shape[-1]
    diff = x - mean
    return -0.5 * (
        d * torch.log(2 * torch.pi * torch.as_tensor(sigma2))
        + (diff**2).sum(dim=-1) / sigma2
    )


def log_root_dynamics(x: Tensor, params: GimbalParameters) -> Tensor:
    """Log p(x_{1,1:T} for root) (Section 2.1).

    Parameters
    ----------
    x : Tensor
        3D positions of shape (T, K, 3).
    params : GimbalParameters
    """

    T, K, _ = x.shape
    root_idx = 0
    eta2_1 = params.skeleton.eta2[root_idx].item()

    # t = 1 prior
    lp0 = log_normal_isotropic(
        x[0, root_idx], params.root_dyn.mu0, params.root_dyn.sigma2_0
    )

    # temporal transitions
    diffs = x[1:, root_idx] - x[:-1, root_idx]
    lp_trans = log_normal_isotropic(diffs, torch.zeros_like(diffs), eta2_1).sum()
    return lp0 + lp_trans


def log_hierarchical_prior(x: Tensor, u: Tensor, params: GimbalParameters) -> Tensor:
    """Log hierarchical vMFG prior for non-root keypoints (Section 2.2).

    Combines temporal smoothness and skeletal constraint using the
    closed-form Gaussian product from the spec.

    Parameters
    ----------
    x : Tensor
        3D positions, shape (T, K, 3).
    u : Tensor
        Unit directions u_{t,k}, shape (T, K, 3), with dummy values for k=0.
    """

    T, K, _ = x.shape
    parent = params.skeleton.parent
    rho = params.skeleton.rho
    sigma2 = params.skeleton.sigma2
    eta2 = params.skeleton.eta2

    lp = torch.zeros((), dtype=x.dtype, device=x.device)
    for k in range(1, K):
        p = int(parent[k].item())
        eta2_k = eta2[k]
        sigma2_k = sigma2[k]
        alpha_k = (1.0 / eta2_k) / (1.0 / eta2_k + 1.0 / sigma2_k)
        inv_sum = 1.0 / eta2_k + 1.0 / sigma2_k
        sigma2_tilde = 1.0 / inv_sum

        for t in range(1, T):
            mu_tilde = alpha_k * x[t - 1, k] + (1.0 - alpha_k) * (
                x[t, p] + rho[k] * u[t, k]
            )
            lp = lp + log_normal_isotropic(x[t, k], mu_tilde, float(sigma2_tilde))

    return lp


def log_pose_directional_prior(
    u: Tensor, s: Tensor, h: Tensor, params: GimbalParameters
) -> Tensor:
    """Log p(u | s, h) (Section 2.3).

    Applies a heading rotation R(h_t) around z-axis to each mean
    direction ν_{s_t,k} and evaluates the vMF likelihood.
    """

    T, K, _ = u.shape
    nu = params.pose_prior.nu  # (S, K, 3)
    kappa = params.pose_prior.kappa  # (S, K)

    device = u.device
    lp = torch.zeros((), dtype=u.dtype, device=device)

    for t in range(T):
        s_t = int(s[t].item())
        h_t = h[t]
        ch, sh = torch.cos(h_t), torch.sin(h_t)
        R = torch.tensor(
            [[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]],
            dtype=u.dtype,
            device=device,
        )  # (3, 3)

        for k in range(1, K):
            mu_sk = nu[s_t, k]  # (3,)
            mu_rot = R @ mu_sk  # (3,)
            lp = lp + vmf_log_pdf(u[t, k], mu_rot, kappa[s_t, k])

    return lp


def log_pose_hmm(s: Tensor, u: Tensor, h: Tensor, params: GimbalParameters) -> Tensor:
    """Log p(s | u, h) using HMM with vMF emissions (Section 3.4).

    This computes the joint log p(s_1) + sum_t log p(s_t | s_{t-1})
    + sum_t log p(u_t | s_t, h_t).
    """

    T = s.shape[0]
    S, K, _ = params.pose_prior.nu.shape

    # Initial prior: uniform over states
    lp = -torch.log(torch.tensor(float(S), device=s.device))

    # Transition matrix
    log_trans = torch.log(params.pose_prior.transition + 1e-12)

    # Emission log-likelihoods for each time and state
    for t in range(T):
        if t > 0:
            lp = lp + log_trans[s[t - 1], s[t]]
        # Add emission term for chosen state; this reuses the vMF prior
        lp = lp + log_pose_directional_prior(
            u[t : t + 1], s[t : t + 1], h[t : t + 1], params
        )

    return lp


def log_heading_prior(h: Tensor, params: HeadingPriorParameters) -> Tensor:
    """Log p(h) (Section 2.5).

    Currently implements a uniform prior over [-π, π), i.e. constant
    log-density, which can be treated as zero for inference.
    """

    # Uniform prior: constant, can safely return zero.
    return torch.zeros((), dtype=h.dtype, device=h.device)


def log_observation_likelihood(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    proj: Tensor,
    outlier: OutlierMixtureParameters,
) -> Tensor:
    """Log p(y | x, z) with robust mixture (Section 2.6).

    Parameters
    ----------
    x : Tensor
        3D positions, shape (T, K, 3).
    y : Tensor
        2D observations, shape (T, K, C, 2).
    z : Tensor
        Outlier indicators in {0,1}, shape (T, K, C).
    proj : Tensor
        Camera parameters packed as (C, 3, 4) or similar.
        Here we assume a simple linear projective camera; see
        `gimbal.camera.project` for the actual implementation.
    outlier : OutlierMixtureParameters
    """

    from .camera import project_points

    T, K, C, _ = y.shape

    x_flat = x.view(T * K, 3)
    y_pred = project_points(x_flat, proj)  # (T*K, C, 2)
    y_pred = y_pred.view(T, K, C, 2)

    beta = outlier.beta  # (K, C)
    mu = outlier.mu  # (K, C, 2, 2)
    sigma2 = outlier.sigma2  # (K, C, 2)

    lp = torch.zeros((), dtype=x.dtype, device=x.device)
    for t in range(T):
        for k in range(K):
            for c in range(C):
                if torch.any(torch.isnan(y[t, k, c])):
                    continue
                eps = y[t, k, c] - y_pred[t, k, c]
                z_tkc = int(z[t, k, c].item())
                mu_kcz = mu[k, c, z_tkc]
                sig2_kcz = sigma2[k, c, z_tkc]
                lp = lp + log_normal_isotropic(eps, mu_kcz, float(sig2_kcz.mean()))

    return lp


def log_joint(
    x: Tensor,
    u: Tensor,
    s: Tensor,
    h: Tensor,
    z: Tensor,
    y: Tensor,
    proj: Tensor,
    params: GimbalParameters,
) -> Tensor:
    """Full unnormalized log joint log p(x,u,s,h,z,y) (Sections 2–3)."""

    lp = torch.zeros((), dtype=x.dtype, device=x.device)
    lp = lp + log_root_dynamics(x, params)
    lp = lp + log_hierarchical_prior(x, u, params)
    lp = lp + log_pose_directional_prior(u, s, h, params)
    lp = lp + log_pose_hmm(s, u, h, params)
    lp = lp + log_heading_prior(h, params.heading_prior)
    lp = lp + log_observation_likelihood(x, y, z, proj, params.outlier)
    return lp
