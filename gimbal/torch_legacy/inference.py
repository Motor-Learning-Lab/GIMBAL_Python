"""MCMC inference algorithms for GIMBAL (Section 3 of spec).

This module provides:

- Hamiltonian Monte Carlo (HMC) updates for all 3D positions x.
- Gibbs updates for u, h, s, and z.
- Utility samplers for vMF on S^2, von Mises on the circle,
  and HMM forward-filtering backward-sampling (FFBS).

We use PyTorch with automatic differentiation for the HMC
log-density gradient with respect to x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import Tensor
import time
from datetime import datetime, timedelta

from .model import (
    GimbalParameters,
    OutlierMixtureParameters,
    log_joint,
    log_observation_likelihood,
    log_pose_directional_prior,
)


# -----------------------------------------------------------------------------
# Utility distributions
# -----------------------------------------------------------------------------


def sample_von_mises(mu: Tensor, kappa: Tensor) -> Tensor:
    """Sample from von Mises on the circle (Section 3.3).

    Uses the rejection sampler of Best and Fisher (1979).
    `mu` and `kappa` are broadcastable tensors.
    Returns tensor of same shape as `mu`.
    """

    # Vectorized implementation based on PyTorch distributions is
    # possible, but here we implement a simple scalar loop for clarity.
    device = mu.device
    out = torch.empty_like(mu, device=device)

    # Flatten for simplicity
    mu_flat = mu.reshape(-1)
    kappa_flat = kappa.reshape(-1)
    out_flat = out.reshape(-1)

    for i in range(mu_flat.numel()):
        m = mu_flat[i]
        k = kappa_flat[i]
        if k < 1e-6:
            out_flat[i] = -torch.pi + 2 * torch.pi * torch.rand((), device=device)
            continue

        a = 1.0 + torch.sqrt(1.0 + 4.0 * k * k)
        b = (a - torch.sqrt(2.0 * a)) / (2.0 * k)
        r = (1.0 + b * b) / (2.0 * b)

        while True:
            u1 = torch.rand((), device=device)
            z = torch.cos(torch.pi * u1)
            f = (1.0 + r * z) / (r + z)
            c = k * (r - f)

            u2 = torch.rand((), device=device)
            if c * (2.0 - c) - u2 > 0 or torch.log(c / u2) + 1.0 - c >= 0:
                u3 = torch.rand((), device=device)
                sign = torch.where(u3 > 0.5, 1.0, -1.0)
                theta = sign * torch.acos(f) + m
                out_flat[i] = ((theta + torch.pi) % (2 * torch.pi)) - torch.pi
                break

    return out


def sample_vmf(mu: Tensor, kappa: Tensor) -> Tensor:
    """Sample from vMF on S^2 (Section 3.2).

    Implementation adapted from Wood (1994). `mu` has shape (..., 3)
    and `kappa` has shape (...,). Returns samples of shape (..., 3).
    """

    device = mu.device
    mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    kappa = kappa.clamp_min(1e-8)
    shape = mu.shape[:-1]

    b = (-2 * kappa + torch.sqrt(4 * kappa**2 + (mu.new_tensor(4.0)))) / 2.0
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + 2.0 * torch.log(1.0 - x0**2)

    def _sample_w(k: Tensor, b_: Tensor, x0_: Tensor, c_: Tensor) -> Tensor:
        while True:
            u = torch.rand(k.shape + (2,), device=device)
            z = 1.0 - torch.exp(c_ - k * u[..., 0])
            w = (1.0 + x0_ * z) / (1.0 - x0_ * z)
            u2 = u[..., 1]
            if k * w + 2.0 * torch.log(1.0 - x0_ * w) - c_ >= torch.log(u2):
                return w

    w = _sample_w(kappa, b, x0, c)

    v = torch.randn(shape + (2,), device=device)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    x = torch.empty(shape + (3,), device=device)
    x[..., 2] = w
    factor = torch.sqrt(1.0 - w**2)
    x[..., 0:2] = factor.unsqueeze(-1) * v

    # Rotate x so that mean direction is mu
    # Compute rotation matrix that maps [0,0,1] -> mu
    z_axis = torch.zeros_like(mu)
    z_axis[..., 2] = 1.0
    v_cross = torch.cross(z_axis, mu, dim=-1)
    s = v_cross.norm(dim=-1, keepdim=True)
    c_mu = (z_axis * mu).sum(dim=-1, keepdim=True)

    eye = torch.eye(3, device=device).expand(shape + (3, 3))
    vx = torch.zeros_like(eye)
    vx[..., 0, 1] = -v_cross[..., 2]
    vx[..., 0, 2] = v_cross[..., 1]
    vx[..., 1, 0] = v_cross[..., 2]
    vx[..., 1, 2] = -v_cross[..., 0]
    vx[..., 2, 0] = -v_cross[..., 1]
    vx[..., 2, 1] = v_cross[..., 0]

    R = eye + vx + vx @ vx * ((1.0 - c_mu) / (s**2 + 1e-8))
    x_rot = torch.einsum("...ij,...j->...i", R, x)
    return x_rot


# -----------------------------------------------------------------------------
# HMM forward-filtering backward-sampling (Section 3.4)
# -----------------------------------------------------------------------------


def hmm_ffbs(log_init: Tensor, log_trans: Tensor, log_lik: Tensor) -> Tensor:
    """Forward-filtering backward-sampling for discrete HMM states.

    Parameters
    ----------
    log_init : Tensor
        Log initial probabilities, shape (S,).
    log_trans : Tensor
        Log transition matrix, shape (S, S).
    log_lik : Tensor
        Emission log-likelihoods log p(y_t | s_t), shape (T, S).
    """

    T, S = log_lik.shape
    alpha = torch.empty_like(log_lik)

    alpha[0] = log_init + log_lik[0]
    alpha[0] = alpha[0] - torch.logsumexp(alpha[0], dim=-1)

    for t in range(1, T):
        alpha_prev = alpha[t - 1].unsqueeze(1)  # (S,1)
        alpha[t] = log_lik[t] + torch.logsumexp(alpha_prev + log_trans, dim=0)
        alpha[t] = alpha[t] - torch.logsumexp(alpha[t], dim=-1)

    # Backward sampling
    s = torch.empty(T, dtype=torch.long, device=log_lik.device)
    s[T - 1] = torch.distributions.Categorical(logits=alpha[T - 1]).sample()

    for t in reversed(range(T - 1)):
        log_prob = alpha[t] + log_trans[:, s[t + 1]]
        s[t] = torch.distributions.Categorical(logits=log_prob).sample()

    return s


# -----------------------------------------------------------------------------
# Gibbs updates for u, h, s, z (Sections 3.2–3.5)
# -----------------------------------------------------------------------------


def sample_u(x: Tensor, s: Tensor, h: Tensor, params: GimbalParameters) -> Tensor:
    """Sample directions u_{t,k} (Section 3.2).

    Uses the conditional vMF form with parameters (\tilde ν_{t,k},\tilde κ_{t,k}).
    """

    T, K, _ = x.shape

    parent = params.skeleton.parent
    rho = params.skeleton.rho
    sigma2 = params.skeleton.sigma2
    nu = params.pose_prior.nu
    kappa = params.pose_prior.kappa

    device = x.device
    u_new = torch.zeros_like(x, device=device)

    for t in range(T):
        s_t = int(s[t].item())
        h_t = h[t]
        ch, sh = torch.cos(h_t), torch.sin(h_t)
        R = torch.tensor(
            [[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]],
            dtype=x.dtype,
            device=device,
        )
        for k in range(1, K):
            p = int(parent[k].item())
            a = kappa[s_t, k] * (R @ nu[s_t, k]) + (rho[k] / sigma2[k]) * (
                x[t, k] - x[t, p]
            )
            kappa_tk = a.norm()
            if kappa_tk < 1e-8:
                # Nearly uniform
                v = torch.randn(3, device=device)
                u_new[t, k] = v / v.norm()
            else:
                mu_tk = a / kappa_tk
                u_new[t, k] = sample_vmf(mu_tk, kappa_tk)

    return u_new


def vector_to_angles(v: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert unit vector to azimuth and polar angles (Section 3.3).

    Returns (hat_xz, hat_xy) for each vector.
    """

    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    az = torch.atan2(y, x)  # around z-axis
    pol = torch.atan2(torch.sqrt(x**2 + y**2), z)
    return az, pol


def sample_h(u: Tensor, s: Tensor, params: GimbalParameters) -> Tensor:
    """Sample headings h_t (Section 3.3).

    Given u_{t,k} and s_t, compute von Mises parameters (θ_t, τ_t)
    and sample h_t ~ vM(θ_t, τ_t).
    """

    T, K, _ = u.shape
    nu = params.pose_prior.nu
    device = u.device

    h_new = torch.zeros(T, dtype=torch.float32, device=device)

    for t in range(T):
        s_t = int(s[t].item())
        xt = torch.zeros((), device=device)
        yt = torch.zeros((), device=device)

        for k in range(1, K):
            az_u, pol_u = vector_to_angles(u[t, k])
            az_nu, pol_nu = vector_to_angles(nu[s_t, k])
            delta = pol_u - pol_nu
            xt = xt + torch.sin(az_u) * torch.sin(az_nu) * torch.cos(delta)
            yt = yt + torch.sin(az_u) * torch.sin(az_nu) * torch.sin(delta)

        theta_t = torch.atan2(yt, xt)
        tau_t = torch.sqrt(xt**2 + yt**2)
        h_new[t] = sample_von_mises(theta_t, tau_t)

    return h_new


def sample_s(u: Tensor, h: Tensor, params: GimbalParameters) -> Tensor:
    """Sample pose state sequence s_{1:T} via FFBS (Section 3.4)."""

    T, K, _ = u.shape
    S, _, _ = params.pose_prior.nu.shape
    device = u.device

    # Emission log-likelihoods log p(u_t | s_t, h_t) for all t,s
    log_lik = torch.empty(T, S, device=device)

    for t in range(T):
        h_t = h[t : t + 1]
        u_t = u[t : t + 1]
        for s_idx in range(S):
            s_vec = torch.full((1,), s_idx, dtype=torch.long, device=device)
            log_lik[t, s_idx] = log_pose_directional_prior(u_t, s_vec, h_t, params)

    log_init = -torch.log(torch.tensor(float(S), device=device)) * torch.ones(
        S, device=device
    )
    log_trans = torch.log(params.pose_prior.transition + 1e-12)

    return hmm_ffbs(log_init, log_trans, log_lik)


def sample_z(
    x: Tensor,
    y: Tensor,
    z_old: Tensor,
    proj: Tensor,
    outlier: OutlierMixtureParameters,
) -> Tensor:
    """Sample outlier indicators z_{t,k,c} (Section 3.5)."""

    T, K, C, _ = y.shape
    device = x.device

    # Compute residuals using current x
    from .camera import project_points

    x_flat = x.view(T * K, 3)
    y_pred = project_points(x_flat, proj).view(T, K, C, 2)

    beta = outlier.beta  # (K,C)
    mu = outlier.mu  # (K,C,2,2)
    sigma2 = outlier.sigma2  # (K,C,2)

    z_new = torch.empty_like(z_old, device=device)

    for t in range(T):
        for k in range(K):
            for c in range(C):
                if torch.any(torch.isnan(y[t, k, c])):
                    z_new[t, k, c] = z_old[t, k, c]
                    continue
                eps = y[t, k, c] - y_pred[t, k, c]
                # log-likelihood ratio for z=1 vs z=0
                mu0 = mu[k, c, 0]
                mu1 = mu[k, c, 1]
                sig0 = sigma2[k, c, 0].mean()
                sig1 = sigma2[k, c, 1].mean()

                from .model import log_normal_isotropic

                ll0 = log_normal_isotropic(eps, mu0, float(sig0))
                ll1 = log_normal_isotropic(eps, mu1, float(sig1))

                logit_beta = torch.log(beta[k, c] / (1.0 - beta[k, c] + 1e-12) + 1e-12)
                logit_post = logit_beta + ll1 - ll0
                p1 = torch.sigmoid(logit_post)
                z_new[t, k, c] = torch.bernoulli(p1)

    return z_new


# -----------------------------------------------------------------------------
# HMC over x (Section 3.1)
# -----------------------------------------------------------------------------


@dataclass
class HMCConfig:
    step_size: float = 0.01
    num_steps: int = 10


def hmc_update_x(
    x: Tensor,
    u: Tensor,
    s: Tensor,
    h: Tensor,
    z: Tensor,
    y: Tensor,
    proj: Tensor,
    params: GimbalParameters,
    config: HMCConfig,
) -> Tensor:
    """Single HMC update for x (Section 3.1).

    Treats x as a flattened vector in R^{3TK} and performs L leapfrog
    steps with step size ε, using autodiff to compute ∇_x log p.
    """

    step_size = config.step_size
    num_steps = config.num_steps

    def _log_post(x_flat: Tensor) -> Tensor:
        x_shaped = x_flat.view_as(x)
        return log_joint(x_shaped, u, s, h, z, y, proj, params)

    x_flat = x.detach().clone().requires_grad_(True)
    p = torch.randn_like(x_flat)

    current_x = x_flat.clone()
    current_p = p.clone()

    # Half step for momentum
    logp = _log_post(x_flat)
    grad = torch.autograd.grad(logp, x_flat)[0]
    p = p + 0.5 * step_size * grad

    # Full steps
    for _ in range(num_steps):
        x_flat = x_flat + step_size * p
        x_flat.retain_grad()
        logp = _log_post(x_flat)
        grad = torch.autograd.grad(logp, x_flat)[0]
        if _ != num_steps - 1:
            p = p + step_size * grad

    # Final half step
    p = p + 0.5 * step_size * grad

    # Negate momentum for symmetry
    p = -p

    def _hamiltonian(xf: Tensor, pf: Tensor) -> Tensor:
        return -_log_post(xf) + 0.5 * (pf**2).sum()

    current_H = _hamiltonian(current_x, current_p)
    proposed_H = _hamiltonian(x_flat, p)
    accept_prob = torch.exp(current_H - proposed_H).clamp_max(1.0)

    if torch.rand(()) < accept_prob:
        return x_flat.view_as(x)
    else:
        return current_x.view_as(x)


# -----------------------------------------------------------------------------
# High-level MCMC driver (Section 3.6)
# -----------------------------------------------------------------------------


def run_gibbs_sampler(
    x_init: Tensor,
    u_init: Tensor,
    s_init: Tensor,
    h_init: Tensor,
    z_init: Tensor,
    y: Tensor,
    proj: Tensor,
    params: GimbalParameters,
    num_iters: int,
    hmc_config: HMCConfig | None = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run the full MCMC sampler (Section 3.6).

    Returns samples of (x, u, s, h, z) for each iteration.
    """

    if hmc_config is None:
        hmc_config = HMCConfig()

    T, K, _ = x_init.shape
    C = y.shape[2]

    x = x_init.clone()
    u = u_init.clone()
    s = s_init.clone()
    h = h_init.clone()
    z = z_init.clone()

    xs = torch.zeros((num_iters, T, K, 3), dtype=x.dtype, device=x.device)
    us = torch.zeros_like(xs)
    ss = torch.zeros((num_iters, T), dtype=torch.long, device=x.device)
    hs = torch.zeros((num_iters, T), dtype=h.dtype, device=x.device)
    zs = torch.zeros((num_iters, T, K, C), dtype=z.dtype, device=x.device)

    # Initialize timing variables
    _iter_start_time = time.perf_counter()
    _total_elapsed_time = 0.0
    _n_timed_iters = 0

    for it in range(num_iters):
        # 1) HMC for x
        x = hmc_update_x(x, u, s, h, z, y, proj, params, hmc_config)

        # 2) Gibbs for u
        u = sample_u(x, s, h, params)

        # 3) Gibbs for h
        h = sample_h(u, s, params)

        # 4) FFBS for s
        s = sample_s(u, h, params)

        # 5) Bernoulli updates for z
        z = sample_z(x, y, z, proj, params.outlier)

        xs[it] = x
        us[it] = u
        ss[it] = s
        hs[it] = h
        zs[it] = z

        now = time.perf_counter()
        wall_now = datetime.now()

        # Elapsed time for this iteration
        iter_elapsed_sec = now - _iter_start_time
        m, s = divmod(int(round(iter_elapsed_sec)), 60)
        elapsed_str = f"{m}m {s:02d}s"

        # Update running average
        _total_elapsed_time += iter_elapsed_sec
        _n_timed_iters += 1
        avg_sec = _total_elapsed_time / max(_n_timed_iters, 1)
        remaining = max(num_iters - (it + 1), 0)
        eta_clock = (wall_now + timedelta(seconds=avg_sec * remaining)).strftime(
            "%H:%M:%S"
        )

        print(
            f"Iteration {it + 1} of {num_iters} completed. Elapsed: {elapsed_str}. ETA: {eta_clock}"
        )

        # Set start time for next iteration
        _iter_start_time = time.perf_counter()

    return xs, us, ss, hs, zs
