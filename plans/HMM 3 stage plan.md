# Collapsed HMM Integration Plan — 3 Stage Overview

This document defines the end-to-end plan for integrating a **collapsed Hidden Markov Model (HMM)** into the current GIMBAL-style PyMC kinematic model. It is a clean, self-contained specification suitable for Copilot / Sonnet 4.5.

Key design principles:

* Use **PyTensor**, not JAX, for HMM log-likelihood.
* Use a **collapsed HMM**: discrete states are integrated out.
* Hidden states control **canonical directions in a global frame** (NOT parent-relative).
* Maintain current normalized Gaussian parameterization for joint directions.
* Make all probabilistic components amenable to NUTS + nutpie.

---

# Stage 1 — Generic Collapsed HMM Engine (Independent of Kinematics)

## Goal

Build and validate:

* A clean, reusable **collapsed HMM log-likelihood** implemented in PyTensor.
* A demonstration PyMC model using Gaussian emissions.
* Synthetic data generation and correctness tests.

## Deliverables

1. New file `hmm_pytensor.py` defining:

   * `forward_log_prob_single(logp_emit, logp_init, logp_trans)`
   * `collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)`
2. A simple PyMC model builder `build_gaussian_hmm_model(y, S)`.
3. Synthetic generator `simulate_gaussian_hmm(...)`.
4. Notebook: `hmm_demo_gaussians.ipynb`.
5. Tests:

   * Tiny HMM brute-force comparison (S=2, T=3).
   * T=1 edge case.
   * Finite gradients & logp.

## Status After Stage 1

You will have a drop-in library function that computes:
[ \log p(y_{0:T-1} \mid \pi, A, \theta) ]
for arbitrary emissions, with proven correctness.

---

# Stage 2 — Refactor Emission & Kinematic Pipelines (Preparation for HMM)

Stage 2 reorganizes the existing model so Stage 3 can integrate cleanly.

## Stage 2a — Factor Out Observation (Emission) Model

Extract a self-contained `project_points_pytensor(...)` and `log_obs_likelihood(...)` from the current Stage-6 model.

Deliverables:

* Functions:

  * `project_points_pytensor(x_all, camera_params)` → predicted 2D keypoints
  * `log_obs_likelihood(y_obs, y_pred, obs_sigma, inlier_prob, occlusion_mask)` → scalar log-likelihood
* Modify `pymc_model.py` so it:

  1. Builds latent 3D variables.
  2. Projects to cameras using the new function.
  3. Computes likelihood via the factored-out API.
* Confirm `demo_pymc_camera_simple` and `demo_pymc_camera_full` behave identically.

## Stage 2b — Clarify Direction Pipeline

Refactor and document the direction model:

* `raw_u[k,t,:] ~ N(0,1)`
* `u[k,t,:] = raw_u / ||raw_u||`
* Directions live in a **global frame**, not a parent-relative frame.
* Add small global regularizing priors (no HMM structure yet).

Deliverables:

* Clear, documented direction code in `pymc_model.py`.
* Combined tensor `U: (T, K, 3)`.
* Unit-norm checks and `compile_logp` sanity tests.

## Status After Stage 2

A cleanly structured, better-abstracted version of the current model, ready for state-dependent priors.

---

# Stage 3 — Full HMM Integration Over Canonical Directions

## Goal

Integrate state-specific canonical directions and spread parameters into the model using the collapsed HMM.

Hidden states represent high-level **pose regimes**, influencing the **prior over directions**. States are fully marginalized.

## Model Components

* Directions from Stage 2: `U[t, k, 3]`, unit vectors.
* For each joint `k` and state `s`:

  * Canonical direction: `mu[k, s, :]` from normalized Gaussian
  * Concentration: `kappa[k, s] ~ HalfNormal(...)`
* Emission log-likelihood from directional prior:
  [ \log p(U_t | z_t = s) = \sum_k \kappa[k,s] (u_{k,t} \cdot \mu[k,s]) ]
  Canonical directions are in **global coordinates**, not parent-relative.
* Observation log-likelihood handled via Stage-2 API.
* Total per-step, per-state logp:
  `logp_emit[t, s] = log_dir_emit[t, s] + logp_obs[t]`

## HMM Parameters

* Initial distribution: `pi[s]`
* Transition matrix: `A[i, j]`
  Built from logits:
* `init_logits[s] ~ Normal(0,1)`
* `trans_logits[s,j] ~ Normal(0,1)`
  And normalized inside the model.

## Integration Steps

1. Add canonical direction priors in `pymc_model.py`.
2. Implement helper:

   ```text
   directional_log_emissions(U, mu, kappa)  → (T, S)
   ```
3. Add observation logp: `logp_emit = log_dir_emit + logp_obs[:,None]`.
4. Apply collapsed HMM:

   ```text
   hmm_ll = collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)
   pm.Potential("directional_hmm_prior", hmm_ll)
   ```
5. Test on increasingly realistic synthetic data.

## Status After Stage 3

A fully HMM-integrated version of the GIMBAL model with state-dependent priors, robust observation model, and stable PyMC sampling.
