"""
Stage 3 — Directional HMM Prior over Joint Directions

This module provides the directional HMM prior that operates on top of the
Stage 2 camera observation model. It implements:

- Canonical directions (mu) via normalized Gaussian vectors
- Concentrations (kappa) with flexible sharing options
- vMF-inspired dot-product directional emissions
- Integration with Stage 1 collapsed HMM engine
- Numerical stabilization for nutpie compatibility

The main entry point is `add_directional_hmm_prior()`, which should be called
inside a PyMC model context.
"""

import pymc as pm
import pytensor.tensor as pt
from gimbal.hmm_pytensor import collapsed_hmm_loglik


def _build_kappa(
    name_prefix: str,
    S: int,
    K: int,
    share_kappa_across_joints: bool,
    share_kappa_across_states: bool,
    kappa_scale: float,
) -> pt.TensorVariable:
    """
    Build concentration parameter kappa with flexible sharing options.

    Parameters
    ----------
    name_prefix : str
        Prefix for variable names.
    S : int
        Number of hidden states.
    K : int
        Number of joints (non-root).
    share_kappa_across_joints : bool
        If True, kappa is shared across joints.
    share_kappa_across_states : bool
        If True, kappa is shared across states.
    kappa_scale : float
        Scale parameter for HalfNormal prior.

    Returns
    -------
    kappa_full : (S, K) tensor
        Concentration parameters, broadcast to full shape.
    """
    base_name = f"{name_prefix}_kappa"

    if share_kappa_across_joints and share_kappa_across_states:
        # Single scalar concentration
        kappa = pm.HalfNormal(base_name, sigma=kappa_scale)
        kappa_full = pt.broadcast_to(kappa, (S, K))

    elif share_kappa_across_joints and not share_kappa_across_states:
        # One kappa per state: (S,)
        kappa_vec = pm.HalfNormal(base_name, sigma=kappa_scale, shape=(S,))
        kappa_full = pt.broadcast_to(kappa_vec.dimshuffle(0, "x"), (S, K))

    elif not share_kappa_across_joints and share_kappa_across_states:
        # One kappa per joint: (K,)
        kappa_vec = pm.HalfNormal(base_name, sigma=kappa_scale, shape=(K,))
        kappa_full = pt.broadcast_to(kappa_vec.dimshuffle("x", 0), (S, K))

    else:
        # Full (S, K) matrix
        kappa_full = pm.HalfNormal(base_name, sigma=kappa_scale, shape=(S, K))

    return pm.Deterministic(f"{base_name}_full", kappa_full)


def add_directional_hmm_prior(
    U: pt.TensorVariable,
    log_obs_t: pt.TensorVariable,
    S: int,
    *,
    name_prefix: str = "dir_hmm",
    share_kappa_across_joints: bool = False,
    share_kappa_across_states: bool = False,
    kappa_scale: float = 5.0,
) -> dict:
    """
    Add a directional HMM prior over U into the current PyMC model.

    This function must be called inside a `with pm.Model():` context.
    It creates canonical direction parameters, computes directional emissions,
    combines them with observation likelihoods, and calls the Stage 1 HMM engine.

    Parameters
    ----------
    U : (T, K, 3) tensor
        Unit direction vectors for all non-root joints (root row unused).
    log_obs_t : (T,) tensor
        Per-timestep observation log-likelihood from Stage 2.
    S : int
        Number of hidden states in the directional HMM.
    name_prefix : str, optional
        Prefix for variable names (e.g., "dir_hmm" → "dir_hmm_mu", etc.).
        Default: "dir_hmm".
    share_kappa_across_joints : bool, optional
        If True, `kappa` is shared across joints (shape `(S,)` broadcast to `(S, K)`).
        If False, `kappa` is joint-specific. Default: False.
    share_kappa_across_states : bool, optional
        If True, `kappa` is shared across states (shape `(K,)` broadcast to `(S, K)`).
        If False, `kappa` is state-specific. Default: False.
    kappa_scale : float, optional
        Scale parameter for HalfNormal priors on `kappa`. Default: 5.0.

    Returns
    -------
    result : dict
        A dictionary containing created variables:
        {
            "mu": mu,                            # (S, K, 3) canonical directions
            "kappa": kappa,                      # (S, K) concentrations
            "init_logits": init_logits,          # (S,) initial state logits
            "trans_logits": trans_logits,        # (S, S) transition logits
            "logp_init": logp_init_det,          # (S,) log initial probabilities
            "logp_trans": logp_trans_det,        # (S, S) log transition probabilities
            "log_dir_emit": log_dir_emit,        # (T, S) directional log emissions
            "logp_emit": logp_emit_det,          # (T, S) combined log emissions
            "hmm_loglik": hmm_loglik,            # scalar HMM log-likelihood
        }

    Notes
    -----
    - Canonical directions `mu` are parameterized as normalized Gaussian vectors,
      avoiding fragile vMF distributions.
    - Directional emissions use dot-product energy: kappa * (U · mu), summed over joints.
    - Numerical stabilization is applied via per-timestep max subtraction before
      calling the HMM engine.
    - The function adds a `pm.Potential` to the model with the HMM log-likelihood.
    """
    # Extract dimensions
    T, K, _ = U.shape

    # -------------------------------------------------------------------------
    # 1. Canonical Directions mu[s, k, :]
    # -------------------------------------------------------------------------
    mu_raw = pm.Normal(
        f"{name_prefix}_mu_raw",
        mu=0.0,
        sigma=1.0,
        shape=(S, K, 3),
    )

    # Normalize to unit vectors with epsilon for numerical stability
    norm_mu = pt.sqrt((mu_raw**2).sum(axis=-1, keepdims=True) + 1e-8)
    mu = pm.Deterministic(
        f"{name_prefix}_mu",
        mu_raw / norm_mu,
    )  # (S, K, 3)

    # -------------------------------------------------------------------------
    # 2. Concentrations kappa
    # -------------------------------------------------------------------------
    kappa = _build_kappa(
        name_prefix=name_prefix,
        S=S,
        K=K,
        share_kappa_across_joints=share_kappa_across_joints,
        share_kappa_across_states=share_kappa_across_states,
        kappa_scale=kappa_scale,
    )  # (S, K)

    # -------------------------------------------------------------------------
    # 3. Directional Log-Emission log_dir_emit[t, s]
    # -------------------------------------------------------------------------
    # Reshape for broadcasting: U[t,k,:] with mu[s,k,:]
    U_exp = U.dimshuffle(0, "x", 1, 2)  # (T, 1, K, 3)
    mu_exp = mu.dimshuffle("x", 0, 1, 2)  # (1, S, K, 3)

    # Dot-products U_tk · mu_sk
    cosine = (U_exp * mu_exp).sum(axis=-1)  # (T, S, K)

    # Apply concentration weights and sum over joints
    kappa_exp = kappa.dimshuffle("x", 0, 1)  # (1, S, K)
    log_dir_emit_raw = (kappa_exp * cosine).sum(axis=-1)  # (T, S)

    log_dir_emit = pm.Deterministic(f"{name_prefix}_log_dir_emit", log_dir_emit_raw)

    # -------------------------------------------------------------------------
    # 4. Combine with Observation Log-Likelihood
    # -------------------------------------------------------------------------
    log_obs_t_exp = log_obs_t.dimshuffle(0, "x")  # (T, 1)
    logp_emit_raw = log_dir_emit + log_obs_t_exp  # (T, S)

    # Wrap in Deterministic to keep scan gradients happy (mirrors Stage 1 pattern)
    logp_emit = pm.Deterministic(f"{name_prefix}_logp_emit", logp_emit_raw)

    # -------------------------------------------------------------------------
    # 5. HMM Parameters: Initial and Transition Log-Probabilities
    # -------------------------------------------------------------------------
    init_logits = pm.Normal(f"{name_prefix}_init_logits", 0.0, 1.0, shape=(S,))
    trans_logits = pm.Normal(f"{name_prefix}_trans_logits", 0.0, 1.0, shape=(S, S))

    # Normalize to log-probabilities
    logp_init = init_logits - pm.math.logsumexp(init_logits)
    logp_trans = trans_logits - pm.math.logsumexp(trans_logits, axis=1, keepdims=True)

    # Wrap in Deterministic for scan gradient compatibility
    logp_init_det = pm.Deterministic(f"{name_prefix}_logp_init", logp_init)
    logp_trans_det = pm.Deterministic(f"{name_prefix}_logp_trans", logp_trans)

    # -------------------------------------------------------------------------
    # 6. Numerical Stabilization and HMM Engine Call
    # -------------------------------------------------------------------------
    # Subtract per-timestep maximum for numerical stability
    max_per_t = pm.math.max(logp_emit, axis=1, keepdims=True)  # (T, 1)
    logp_emit_centered = logp_emit - max_per_t  # (T, S)

    # Sum of the constants we subtracted
    offset = max_per_t.sum()  # scalar

    # Call Stage-1 HMM engine
    hmm_ll_centered = collapsed_hmm_loglik(
        logp_emit_centered,
        logp_init_det,
        logp_trans_det,
    )

    hmm_loglik = pm.Deterministic(
        f"{name_prefix}_loglik",
        hmm_ll_centered + offset,
    )

    # Add potential to model
    pm.Potential(f"{name_prefix}_potential", hmm_loglik)

    # -------------------------------------------------------------------------
    # 7. Return Dictionary of Created Variables
    # -------------------------------------------------------------------------
    return {
        "mu": mu,
        "kappa": kappa,
        "init_logits": init_logits,
        "trans_logits": trans_logits,
        "logp_init": logp_init_det,
        "logp_trans": logp_trans_det,
        "log_dir_emit": log_dir_emit,
        "logp_emit": logp_emit,
        "hmm_loglik": hmm_loglik,
    }
