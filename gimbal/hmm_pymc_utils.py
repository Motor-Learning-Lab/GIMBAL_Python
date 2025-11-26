"""
PyMC model builders and simulation utilities for Gaussian HMMs.

Provides example implementations demonstrating how to use the collapsed HMM
with PyMC models, specifically for 1D Gaussian emissions.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from gimbal.hmm_pytensor import collapsed_hmm_loglik


def build_gaussian_hmm_model(y, S):
    """
    Build a PyMC model for a 1D Gaussian HMM with collapsed states.

    Parameters
    ----------
    y : array-like, shape (T,)
        Observed sequence
    S : int
        Number of hidden states

    Returns
    -------
    model : pm.Model
        PyMC model ready for sampling

    Notes
    -----
    This model uses:
    - State-specific means: mu[s] for each state s
    - Shared observation noise: sigma (scalar)
    - Normal(0,1) priors on init and transition logits
    """
    with pm.Model() as model:
        # Initial and transition logits
        init_logits = pm.Normal("init_logits", 0, 1, shape=S)
        trans_logits = pm.Normal("trans_logits", 0, 1, shape=(S, S))

        # Normalize to log-probabilities
        # Convert to pure tensor to avoid NominalVariable issues in scan gradients
        logp_init = pt.as_tensor_variable(init_logits) - pm.math.logsumexp(init_logits)
        logp_trans = pt.as_tensor_variable(trans_logits) - pm.math.logsumexp(
            trans_logits, axis=1, keepdims=True
        )

        # Emission parameters
        mu = pm.Normal("mu", 0, 5, shape=S)
        sigma = pm.Exponential("sigma", 1.0)

        # Construct logp_emit (T, S)
        y_t = pt.shape_padright(y, 1)  # (T, 1)
        mu_s = mu.dimshuffle("x", 0)  # (1, S)
        logp_emit_raw = pm.logp(pm.Normal.dist(mu_s, sigma), y_t)  # (T, S)

        # Wrap in Deterministic to break nominal variable chain for scan gradients
        logp_emit = pm.Deterministic("logp_emit", logp_emit_raw)
        logp_init_det = pm.Deterministic("logp_init", logp_init)
        logp_trans_det = pm.Deterministic("logp_trans", logp_trans)

        # Collapsed HMM
        hmm_ll = collapsed_hmm_loglik(logp_emit, logp_init_det, logp_trans_det)
        pm.Potential("hmm_loglik", hmm_ll)

    return model


def simulate_gaussian_hmm(
    T, S, mu_true, sigma_true, pi_true, A_true, random_state=None
):
    """
    Simulate a 1D Gaussian HMM sequence.

    Parameters
    ----------
    T : int
        Number of time steps
    S : int
        Number of states
    mu_true : array, shape (S,)
        State-specific means
    sigma_true : float
        Observation noise (shared across states)
    pi_true : array, shape (S,)
        Initial state probabilities (must sum to 1)
    A_true : array, shape (S, S)
        Transition matrix (rows must sum to 1)
    random_state : int, optional
        Random seed

    Returns
    -------
    y : array, shape (T,)
        Observed sequence
    z : array, shape (T,), dtype int
        True hidden state sequence

    Notes
    -----
    States are arbitrary labels. Permuting (mu, pi, A) gives equivalent model
    (label switching).
    """
    rng = np.random.default_rng(random_state)

    # Validate inputs
    assert np.isclose(pi_true.sum(), 1.0), "pi must sum to 1"
    assert np.allclose(A_true.sum(axis=1), 1.0), "A rows must sum to 1"

    z = np.zeros(T, dtype=int)
    y = np.zeros(T, dtype=float)

    # Initial state
    z[0] = rng.choice(S, p=pi_true)
    sigma_arr = np.broadcast_to(sigma_true, (S,))
    y[0] = rng.normal(mu_true[z[0]], sigma_arr[z[0]])

    # Transitions
    for t in range(1, T):
        z[t] = rng.choice(S, p=A_true[z[t - 1]])
        y[t] = rng.normal(mu_true[z[t]], sigma_arr[z[t]])

    return y, z
