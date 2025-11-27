"""
Minimal Working Example for Stage 3 - Directional HMM

This script demonstrates the complete Stage 3 pipeline:
1. Generate synthetic data with state-dependent directional patterns
2. Build PyMC model with directional HMM prior
3. Run nutpie sampling
4. Visualize results

This is a reference implementation for understanding Stage 3 behavior.
"""

import numpy as np
import pymc as pm
from gimbal.hmm_directional import add_directional_hmm_prior


def generate_synthetic_directional_data(T=50, K=5, S=3, seed=42):
    """
    Generate synthetic data with S states and state-dependent directions.

    Parameters
    ----------
    T : int
        Number of time steps
    K : int
        Number of joints (excluding root)
    S : int
        Number of hidden states
    seed : int
        Random seed for reproducibility

    Returns
    -------
    U : ndarray, shape (T, K, 3)
        Unit direction vectors
    log_obs_t : ndarray, shape (T,)
        Per-timestep observation log-likelihoods
    true_states : ndarray, shape (T,)
        Ground truth state sequence (for validation)
    """
    rng = np.random.default_rng(seed)

    # Define canonical directions for each state
    # State 0: Upright posture (mostly +z)
    # State 1: Leaning forward (mostly +x)
    # State 2: Leaning sideways (mostly +y)
    canonical_mu = np.zeros((S, K, 3))
    canonical_mu[0, :, 2] = 1.0  # State 0: +z (up)
    canonical_mu[1, :, 0] = 0.7  # State 1: +x (forward)
    canonical_mu[1, :, 2] = 0.3
    canonical_mu[2, :, 1] = 0.8  # State 2: +y (sideways)
    canonical_mu[2, :, 2] = 0.2

    # Normalize
    canonical_mu /= np.linalg.norm(canonical_mu, axis=-1, keepdims=True)

    # Generate state sequence with some persistence
    # Use simple transition probabilities favoring staying in state
    trans_probs = np.array(
        [
            [0.7, 0.2, 0.1],  # From state 0
            [0.2, 0.6, 0.2],  # From state 1
            [0.1, 0.2, 0.7],  # From state 2
        ]
    )

    true_states = np.zeros(T, dtype=int)
    true_states[0] = rng.choice(S)

    for t in range(1, T):
        true_states[t] = rng.choice(S, p=trans_probs[true_states[t - 1]])

    # Generate U vectors with noise around canonical directions
    U = np.zeros((T, K, 3))
    kappa_true = 10.0  # Concentration parameter

    for t in range(T):
        s = true_states[t]
        # Add Gaussian noise and normalize
        U[t] = canonical_mu[s] + rng.normal(0, 1.0 / kappa_true, size=(K, 3))
        U[t] /= np.linalg.norm(U[t], axis=-1, keepdims=True) + 1e-8

    # Generate synthetic observation likelihoods
    # Slightly better for states that match the data
    log_obs_t = rng.normal(loc=-50.0, scale=10.0, size=(T,))

    return U, log_obs_t, true_states


def main():
    """Run the minimal Stage 3 example."""
    print("=" * 70)
    print("Stage 3 Directional HMM - Minimal Working Example")
    print("=" * 70)

    # --- 1. Generate Synthetic Data ---
    print("\n[1/4] Generating synthetic data...")
    T, K, S = 50, 5, 3
    U_data, log_obs_t_data, true_states = generate_synthetic_directional_data(
        T=T, K=K, S=S, seed=42
    )
    print(f"  T={T} timesteps, K={K} joints, S={S} states")
    print(f"  True state distribution: {np.bincount(true_states)}")

    # --- 2. Build PyMC Model with Directional HMM ---
    print("\n[2/4] Building PyMC model with directional HMM prior...")
    with pm.Model() as model:
        # Stage 2 interface: U and log_obs_t as Data
        U = pm.Data("U", U_data)
        log_obs_t = pm.Data("log_obs_t", log_obs_t_data)

        # Stage 3: Add directional HMM prior
        hmm_result = add_directional_hmm_prior(
            U=U,
            log_obs_t=log_obs_t,
            S=S,
            name_prefix="dir_hmm",
            share_kappa_across_joints=False,
            share_kappa_across_states=False,
            kappa_scale=5.0,
        )

        print(f"  Model variables:")
        print(f"    - mu: {hmm_result['mu'].eval().shape}")
        print(f"    - kappa: {hmm_result['kappa'].eval().shape}")
        print(f"    - logp_emit: {hmm_result['logp_emit'].eval().shape}")
        print(f"    - hmm_loglik: {hmm_result['hmm_loglik'].eval().shape}")

        # Check initial log-likelihood
        initial_loglik = hmm_result["hmm_loglik"].eval()
        print(f"  Initial HMM log-likelihood: {initial_loglik:.2f}")

    # --- 3. Sample with NUTS ---
    print("\n[3/4] Sampling with PyMC NUTS sampler...")
    print("  Running 2 chains, 200 tuning + 200 sampling steps...")

    try:
        with model:
            idata = pm.sample(
                draws=200,
                tune=200,
                chains=2,
                cores=2,
                progressbar=True,
            )

        print("\n  Sampling complete!")
        print(f"  Posterior shape: {idata.posterior.dims}")

        # --- 4. Basic Posterior Summaries ---
        print("\n[4/4] Posterior summaries:")

        # Extract posterior samples
        mu_samples = idata.posterior["dir_hmm_mu"].values  # (chains, draws, S, K, 3)
        kappa_samples = idata.posterior[
            "dir_hmm_kappa_full"
        ].values  # (chains, draws, S, K)

        # Compute posterior means
        mu_mean = mu_samples.mean(axis=(0, 1))  # (S, K, 3)
        kappa_mean = kappa_samples.mean(axis=(0, 1))  # (S, K)

        print(f"\n  Canonical directions (mu) - posterior mean norms by state:")
        for s in range(S):
            norms = np.linalg.norm(mu_mean[s], axis=-1)
            print(
                f"    State {s}: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}"
            )

        print(f"\n  Concentrations (kappa) - posterior mean by state:")
        for s in range(S):
            print(
                f"    State {s}: min={kappa_mean[s].min():.2f}, max={kappa_mean[s].max():.2f}, mean={kappa_mean[s].mean():.2f}"
            )

        # Transition probabilities
        trans_logits_samples = idata.posterior["dir_hmm_trans_logits"].values
        trans_probs_samples = np.exp(
            trans_logits_samples - trans_logits_samples.max(axis=-1, keepdims=True)
        )
        trans_probs_samples /= trans_probs_samples.sum(axis=-1, keepdims=True)
        trans_probs_mean = trans_probs_samples.mean(axis=(0, 1))

        print(f"\n  Transition probability matrix (posterior mean):")
        for s in range(S):
            probs_str = "  ".join([f"{p:.3f}" for p in trans_probs_mean[s]])
            print(f"    State {s} -> [{probs_str}]")

        print("\n" + "=" * 70)
        print("Stage 3 Example Complete!")
        print("=" * 70)
        print("\nNote: This is a minimal example. For production use:")
        print("  - Apply post-hoc label switching correction (Hungarian algorithm)")
        print("  - Validate convergence (R-hat, ESS)")
        print("  - Use longer chains and more draws")
        print("  - Integrate with full Stage 2 camera observation model")

    except Exception as e:
        print(f"\n  Sampling failed: {e}")
        print("\n  This may occur with complex HMM models.")
        print(
            "  The model construction was successful - sampling is optional for this demo."
        )
        print(
            "  For production use, consider simpler initialization or more informative priors."
        )


if __name__ == "__main__":
    main()
