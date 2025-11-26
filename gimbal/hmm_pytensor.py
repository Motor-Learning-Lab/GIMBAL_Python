"""
Collapsed HMM log-likelihood implementation in PyTensor.

This module provides a generic collapsed HMM implementation that computes
the marginal log-likelihood by integrating out discrete hidden states using
the forward algorithm in log-space.

Stage 1 implementation - independent of cameras, skeletons, and kinematics.
"""

import pytensor.tensor as pt
from pytensor import scan


def forward_log_prob_single(
    logp_emit: pt.TensorVariable,  # shape (T, S)
    logp_init: pt.TensorVariable,  # shape (S,)
    logp_trans: pt.TensorVariable,  # shape (S, S)
) -> pt.TensorVariable:
    """
    Compute collapsed HMM log-likelihood for one sequence using the
    forward algorithm in log-space.

    Parameters
    ----------
    logp_emit : (T, S) tensor
        Log emission probabilities: logp_emit[t, s] = log p(y_t | z_t=s, θ)
    logp_init : (S,) tensor
        Log initial state probabilities: logp_init[s] = log π_s
    logp_trans : (S, S) tensor
        Log transition probabilities: logp_trans[i, j] = log p(z_t=j | z_{t-1}=i)

    Returns
    -------
    logp : scalar tensor
        Collapsed log-likelihood: log p(y_{0:T-1} | π, A, θ)

    Notes
    -----
    Uses the forward algorithm with log-space computations for numerical stability.
    Handles T=1 edge case automatically.
    """
    # Step 1 - Initialization
    alpha_prev = logp_init + logp_emit[0]  # shape (S,)

    # Step 2 - Define one forward step
    def step(logp_emit_t, alpha_prev, logp_trans_flat):
        """
        Forward step: alpha_t[j] = logp_emit[t,j] + log sum_i exp(alpha[t-1,i] + log A[i,j])

        Parameters
        ----------
        logp_emit_t : (S,) tensor
            Log emission probabilities at time t
        alpha_prev : (S,) tensor
            Log forward probabilities at time t-1
        logp_trans_flat : (S*S,) tensor
            Flattened log transition probabilities (passed as non_sequences)

        Returns
        -------
        alpha_t : (S,) tensor
            Log forward probabilities at time t
        """
        # alpha_prev: (S,)
        # Reshape flattened transition matrix
        S = alpha_prev.shape[0]
        logp_trans_local = logp_trans_flat.reshape((S, S))

        # Broadcast alpha_prev[i] + log A[i,j] and sum over previous states
        alpha_pred = pt.logsumexp(
            alpha_prev.dimshuffle(0, "x") + logp_trans_local,
            axis=0,
        )  # shape (S,)

        alpha_t = logp_emit_t + alpha_pred
        return alpha_t

    # Step 3 - Run recursion with scan
    # Flatten logp_trans to avoid shape issues in scan's L_op
    logp_trans_flat = logp_trans.flatten()
    alpha_all, _ = scan(
        fn=step,
        sequences=[logp_emit[1:]],
        outputs_info=[alpha_prev],
        non_sequences=[logp_trans_flat],
    )  # Step 4 - Final state (T=1 handled automatically)
    alpha_last = pt.switch(
        pt.eq(logp_emit.shape[0], 1),
        alpha_prev,
        alpha_all[-1],
    )

    # Step 5 - Collapse over final states
    logp = pt.logsumexp(alpha_last)
    return logp


def collapsed_hmm_loglik(
    logp_emit: pt.TensorVariable,
    logp_init: pt.TensorVariable,
    logp_trans: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Compute collapsed HMM log-likelihood.

    This is a wrapper around forward_log_prob_single for readability
    and future extension.

    Parameters
    ----------
    logp_emit : (T, S) tensor
        Log emission probabilities
    logp_init : (S,) tensor
        Log initial state probabilities
    logp_trans : (S, S) tensor
        Log transition probabilities

    Returns
    -------
    logp : scalar tensor
        Collapsed log-likelihood
    """
    return forward_log_prob_single(logp_emit, logp_init, logp_trans)
