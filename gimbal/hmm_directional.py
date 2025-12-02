"""
v0.1.3 — Directional HMM Prior over Joint Directions

This module provides the directional HMM prior that operates on top of the
v0.1.2 camera observation model. It implements:

- Canonical directions (mu) via normalized Gaussian vectors
- Concentrations (kappa) with flexible sharing options
- vMF-inspired dot-product directional emissions
- Integration with v0.1.1 collapsed HMM engine
- Numerical stabilization for nutpie compatibility

The main entry point is `add_directional_hmm_prior()`, which should be called
inside a PyMC model context.
"""

import pymc as pm
import pytensor.tensor as pt
from gimbal.hmm_pytensor import collapsed_hmm_loglik


def _add_single_state_directional_prior(
    U: pt.TensorVariable,
    log_obs_t: pt.TensorVariable,
    K: int,
    *,
    joint_names: list[str] | None = None,
    name_prefix: str = "dir_hmm",
    kappa_scale: float = 5.0,
    prior_config: dict | None = None,
) -> dict:
    """
    Special case for S=1: single-state "HMM" with no transitions.

    This bypasses all the multi-state machinery to avoid PyTensor optimizer
    warnings. With S=1, there are no transitions, so the model reduces to
    per-timestep directional priors on U.

    Parameters match add_directional_hmm_prior but S is fixed to 1.
    """
    S = 1

    # -------------------------------------------------------------------------
    # 1. Canonical Direction mu (single state, so shape is just (K, 3))
    # -------------------------------------------------------------------------
    if prior_config:
        # v0.2.1 mode: Data-driven priors
        mu_raw_list = []
        for k in range(K):
            joint_name = joint_names[k]

            if joint_name in prior_config:
                joint_prior = prior_config[joint_name]
                mu_mean = joint_prior["mu_mean"]
                mu_sd = joint_prior["mu_sd"]

                mu_raw_k = pm.Normal(
                    f"{name_prefix}_mu_raw_s0_k{k}",
                    mu=mu_mean,
                    sigma=mu_sd,
                    shape=(3,),
                )
            else:
                mu_raw_k = pm.Normal(
                    f"{name_prefix}_mu_raw_s0_k{k}",
                    mu=0.0,
                    sigma=1.0,
                    shape=(3,),
                )

            mu_raw_list.append(mu_raw_k)

        mu_raw = pt.stack(mu_raw_list, axis=0)  # (K, 3)
    else:
        # v0.1 mode: Uninformative priors
        mu_raw = pm.Normal(
            f"{name_prefix}_mu_raw",
            mu=0.0,
            sigma=1.0,
            shape=(K, 3),
        )

    # Normalize to unit vectors
    norm_mu = pt.sqrt((mu_raw**2).sum(axis=-1, keepdims=True) + 1e-8)
    mu_normalized = mu_raw / norm_mu  # (K, 3)

    # Add state dimension for consistency: (1, K, 3)
    mu = pm.Deterministic(
        f"{name_prefix}_mu",
        mu_normalized.dimshuffle("x", 0, 1),
    )

    # -------------------------------------------------------------------------
    # 2. Concentrations kappa (single state, so shape is just (K,))
    # -------------------------------------------------------------------------
    if prior_config:
        # v0.2.1 mode: Data-driven kappa priors
        from gimbal.prior_building import get_gamma_shape_rate

        kappa_list = []
        for k in range(K):
            joint_name = joint_names[k]

            if joint_name in prior_config:
                joint_prior = prior_config[joint_name]
                kappa_mode = joint_prior["kappa_mode"]
                kappa_sd = joint_prior["kappa_sd"]

                shape, rate = get_gamma_shape_rate(kappa_mode, kappa_sd)
                kappa_k = pm.Gamma(
                    f"{name_prefix}_kappa_s0_k{k}",
                    alpha=shape,
                    beta=rate,
                )
            else:
                kappa_k = pm.HalfNormal(
                    f"{name_prefix}_kappa_s0_k{k}",
                    sigma=kappa_scale,
                )

            kappa_list.append(kappa_k)

        kappa_vec = pt.stack(kappa_list, axis=0)  # (K,)
    else:
        # v0.1 mode
        kappa_vec = pm.HalfNormal(
            f"{name_prefix}_kappa",
            sigma=kappa_scale,
            shape=(K,),
        )

    # Add state dimension for consistency: (1, K)
    kappa = pm.Deterministic(
        f"{name_prefix}_kappa_full",
        kappa_vec.dimshuffle("x", 0),
    )

    # -------------------------------------------------------------------------
    # 3. Directional Emissions (simplified for S=1)
    # -------------------------------------------------------------------------
    # U: (T, K, 3)
    # mu_normalized: (K, 3)
    # Compute dot products directly without extra broadcasting
    cosine = (U * mu_normalized.dimshuffle("x", 0, 1)).sum(axis=-1)  # (T, K)

    # Apply kappa and sum over joints
    log_dir_emit_raw = (kappa_vec.dimshuffle("x", 0) * cosine).sum(axis=-1)  # (T,)

    # Add state dimension: (T, 1)
    log_dir_emit = pm.Deterministic(
        f"{name_prefix}_log_dir_emit",
        log_dir_emit_raw.dimshuffle(0, "x"),
    )

    # -------------------------------------------------------------------------
    # 4. Combine with Observation Likelihood
    # -------------------------------------------------------------------------
    logp_emit_raw = log_dir_emit_raw + log_obs_t  # (T,)
    logp_emit = pm.Deterministic(
        f"{name_prefix}_logp_emit",
        logp_emit_raw.dimshuffle(0, "x"),  # (T, 1)
    )

    # -------------------------------------------------------------------------
    # 5. Transition Parameters (deterministic for S=1)
    # -------------------------------------------------------------------------
    logp_init = pt.zeros(1)
    logp_trans = pt.zeros((1, 1))

    logp_init_det = pm.Deterministic(f"{name_prefix}_logp_init", logp_init)
    logp_trans_det = pm.Deterministic(f"{name_prefix}_logp_trans", logp_trans)

    # -------------------------------------------------------------------------
    # 6. Log-Likelihood (simple sum for S=1)
    # -------------------------------------------------------------------------
    # No HMM forward algorithm needed - just sum emissions over time
    hmm_loglik = pm.Deterministic(
        f"{name_prefix}_loglik",
        logp_emit_raw.sum(),  # Sum the (T,) vector
    )

    # Add potential
    pm.Potential(f"{name_prefix}_potential", hmm_loglik)

    # -------------------------------------------------------------------------
    # 7. Return Dictionary
    # -------------------------------------------------------------------------
    return {
        "mu": mu,
        "kappa": kappa,
        "logp_init": logp_init_det,
        "logp_trans": logp_trans_det,
        "log_dir_emit": log_dir_emit,
        "logp_emit": logp_emit,
        "hmm_loglik": hmm_loglik,
        # Note: No init_logits or trans_logits for S=1
    }


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
    joint_names: list[str] | None = None,
    name_prefix: str = "dir_hmm",
    share_kappa_across_joints: bool = False,
    share_kappa_across_states: bool = False,
    kappa_scale: float = 5.0,
    prior_config: dict | None = None,
) -> dict:
    """
    Add a directional HMM prior over U into the current PyMC model.

    This function must be called inside a `with pm.Model():` context.
    It creates canonical direction parameters, computes directional emissions,
    combines them with observation likelihoods, and calls the v0.1.1 HMM engine.

    Parameters
    ----------
    U : (T, K, 3) tensor
        Unit direction vectors for all non-root joints (root row unused).
    log_obs_t : (T,) tensor
        Per-timestep observation log-likelihood from v0.1.2.
    S : int
        Number of hidden states in the directional HMM.
    joint_names : list of str, optional
        Names of all joints (length K+1, including root at index 0).
        Required when using prior_config. Default: None.
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
        Scale parameter for HalfNormal priors on `kappa` (v0.1 mode only).
        Ignored when prior_config is provided. Default: 5.0.
    prior_config : dict, optional
        Data-driven prior configuration from build_priors_from_statistics().
        When provided, completely replaces v0.1 default priors with:
        - Projected Normal for mu: Normal(mu_mean, mu_sd) then normalize
        - Gamma for kappa: Gamma(mode, sd) converted to (shape, rate)
        Structure: {'joint_name': {'mu_mean': ndarray, 'mu_sd': float,
                                     'kappa_mode': float, 'kappa_sd': float}}
        Default: None (v0.1 behavior with uninformative priors).

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
    - When prior_config is provided (v0.2.1+), it completely overrides v0.1 defaults
      with data-driven priors. No merging occurs.
    - Prior config applies per-state: each state gets the same empirical prior means,
      but samples different canonical directions from those distributions.
    """
    # Validate prior_config usage
    if prior_config is not None and joint_names is None:
        raise ValueError("joint_names must be provided when using prior_config")

    # Handle prior_config
    if prior_config is None:
        prior_config = {}

    # Extract dimensions
    # For v0.1 mode, U.shape[1] may be symbolic, but we need concrete K
    # For v0.2.1 mode with prior_config, infer K from joint_names
    if prior_config and joint_names:
        # v0.2.1: Get K from joint_names
        K = len(joint_names)
        if K < 2:
            raise ValueError(
                f"joint_names must have at least 2 elements (root + 1 joint), got {K}"
            )
    else:
        # v0.1: Try to get K from U shape
        # If U.shape[1] is symbolic, this may fail - but for testing with concrete arrays it works
        try:
            K = int(U.shape[1])
        except (TypeError, AttributeError):
            # Symbolic shape - evaluate it
            K = U.shape[1].eval() if hasattr(U.shape[1], "eval") else U.shape[1]

    # -------------------------------------------------------------------------
    # Special case: S=1 uses completely separate code path
    # -------------------------------------------------------------------------
    if S == 1:
        return _add_single_state_directional_prior(
            U=U,
            log_obs_t=log_obs_t,
            K=K,
            joint_names=joint_names,
            name_prefix=name_prefix,
            kappa_scale=kappa_scale,
            prior_config=prior_config,
        )

    # -------------------------------------------------------------------------
    # 1. Canonical Directions mu[s, k, :]
    # -------------------------------------------------------------------------
    if prior_config:
        # v0.2.1 mode: Data-driven priors with projected Normal
        # Create mu_raw with data-driven priors per joint
        mu_raw_list = []
        for s in range(S):
            mu_raw_state = []
            for k in range(K):
                joint_name = joint_names[k]  # Direct indexing, includes root at 0

                if joint_name in prior_config:
                    # Use empirical prior
                    joint_prior = prior_config[joint_name]
                    mu_mean = joint_prior["mu_mean"]  # (3,)
                    mu_sd = joint_prior["mu_sd"]  # scalar

                    mu_raw_jk = pm.Normal(
                        f"{name_prefix}_mu_raw_s{s}_k{k}",
                        mu=mu_mean,
                        sigma=mu_sd,
                        shape=(3,),
                    )
                else:
                    # Fall back to uninformative prior for this joint
                    mu_raw_jk = pm.Normal(
                        f"{name_prefix}_mu_raw_s{s}_k{k}",
                        mu=0.0,
                        sigma=1.0,
                        shape=(3,),
                    )

                mu_raw_state.append(mu_raw_jk)

            mu_raw_state_stacked = pt.stack(mu_raw_state, axis=0)  # (K, 3)
            mu_raw_list.append(mu_raw_state_stacked)

        mu_raw = pt.stack(mu_raw_list, axis=0)  # (S, K, 3)
    else:
        # v0.1 mode: Uninformative priors
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
    if prior_config:
        # v0.2.1 mode: Data-driven Gamma priors per joint
        from gimbal.prior_building import get_gamma_shape_rate

        kappa_list = []
        for s in range(S):
            kappa_state = []
            for k in range(K):
                joint_name = joint_names[k]  # Direct indexing

                if joint_name in prior_config:
                    # Use Gamma prior from empirical stats
                    joint_prior = prior_config[joint_name]
                    kappa_mode = joint_prior["kappa_mode"]
                    kappa_sd = joint_prior["kappa_sd"]

                    # Convert to shape/rate parameterization
                    shape, rate = get_gamma_shape_rate(kappa_mode, kappa_sd)

                    kappa_jk = pm.Gamma(
                        f"{name_prefix}_kappa_s{s}_k{k}",
                        alpha=shape,
                        beta=rate,
                    )
                else:
                    # Fall back to HalfNormal
                    kappa_jk = pm.HalfNormal(
                        f"{name_prefix}_kappa_s{s}_k{k}",
                        sigma=kappa_scale,
                    )

                kappa_state.append(kappa_jk)

            kappa_state_stacked = pt.stack(kappa_state, axis=0)  # (K,)
            kappa_list.append(kappa_state_stacked)

        kappa = pt.stack(kappa_list, axis=0)  # (S, K)
        kappa = pm.Deterministic(f"{name_prefix}_kappa_full", kappa)
    else:
        # v0.1 mode: Use existing flexible sharing
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

    # Wrap in Deterministic to keep scan gradients happy (mirrors v0.1.1 pattern)
    logp_emit = pm.Deterministic(f"{name_prefix}_logp_emit", logp_emit_raw)

    # -------------------------------------------------------------------------
    # 5. HMM Parameters: Initial and Transition Log-Probabilities
    # -------------------------------------------------------------------------
    if S == 1:
        # Special case: S=1 means no hidden states to infer
        # Initial state is deterministic (always state 0)
        # Transition is deterministic (always stay in state 0)
        logp_init = pt.zeros(1)
        logp_trans = pt.zeros((1, 1))

        logp_init_det = pm.Deterministic(f"{name_prefix}_logp_init", logp_init)
        logp_trans_det = pm.Deterministic(f"{name_prefix}_logp_trans", logp_trans)
    else:
        # Normal case: S > 1, sample transition parameters
        init_logits = pm.Normal(f"{name_prefix}_init_logits", 0.0, 1.0, shape=(S,))
        trans_logits = pm.Normal(f"{name_prefix}_trans_logits", 0.0, 1.0, shape=(S, S))

        # Normalize to log-probabilities
        logp_init = init_logits - pm.math.logsumexp(init_logits)
        logp_trans = trans_logits - pm.math.logsumexp(
            trans_logits, axis=1, keepdims=True
        )

        # Wrap in Deterministic for scan gradient compatibility
        logp_init_det = pm.Deterministic(f"{name_prefix}_logp_init", logp_init)
        logp_trans_det = pm.Deterministic(f"{name_prefix}_logp_trans", logp_trans)

    # -------------------------------------------------------------------------
    # 6. Numerical Stabilization and HMM Engine Call
    # -------------------------------------------------------------------------
    if S == 1:
        # Single-state HMM: no transitions, no forward recursion needed.
        # The chain is always in state 0, so the joint log-likelihood is just
        # the sum over all per-timestep emissions for that state.
        # logp_emit has shape (T, 1), we sum over both dimensions.
        hmm_loglik = pm.Deterministic(
            f"{name_prefix}_loglik",
            logp_emit.sum(),
        )
    else:
        # Multi-state HMM: use collapsed HMM engine with numerical stabilization.
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

    # Add potential to model (both S=1 and S>1 cases)
    pm.Potential(f"{name_prefix}_potential", hmm_loglik)

    # -------------------------------------------------------------------------
    # 7. Return Dictionary of Created Variables
    # -------------------------------------------------------------------------
    result = {
        "mu": mu,
        "kappa": kappa,
        "logp_init": logp_init_det,
        "logp_trans": logp_trans_det,
        "log_dir_emit": log_dir_emit,
        "logp_emit": logp_emit,
        "hmm_loglik": hmm_loglik,
    }

    # Only include init_logits and trans_logits if S > 1 (they're sampled)
    if S > 1:
        result["init_logits"] = init_logits
        result["trans_logits"] = trans_logits

    return result
