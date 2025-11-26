# 🚀 Collapsed HMM Integration Plan — Clean 3-Stage Overview (Rewritten 2025)

This document defines the **complete end-to-end integration plan** for adding a collapsed Hidden Markov Model (HMM) over joint-direction priors into the PyMC version of the GIMBAL-style kinematic model. It is the simplified, modernized version of the original plan, aligned with:

* Stage-1 completion report
* Updated Phase-2 specification
* Clarity > Simplicity > Heavy Engineering

It consists of **three independent, cleanly separated stages**.

---

# ✅ Stage 1 — Collapsed HMM Engine (Completed)

**Goal:** Implement and validate a reusable, numerically stable PyTensor collapsed HMM log-likelihood.

**Deliverables:**

* `hmm_pytensor.py`

  * `forward_log_prob_single`
  * `collapsed_hmm_loglik`
* Gaussian-emission demo in `hmm_pymc_utils.py`
* Notebook validation:

  * Tiny HMM brute force enumeration
  * Gradient & finite-difference checks
  * Normalization checks
  * Viterbi decoding (not in library code)

**Properties:**

* Pure PyTensor (no JAX)
* Scan-based forward algorithm
* Stable gradients
* Independent of kinematics
* Validation lives in notebooks

Stage 1 provides the **computational primitive** that Stage 3 will call.

---

# 🚧 Stage 2 — Direction & Emission Refactor (Current Work)

**Goal:** Refactor the PyMC kinematic model so that direction, projection, and likelihood pipelines are clean, readable, and exposed through a stable interface consumed by Stage 3.

This stage **does not** introduce HMM logic.
This stage **does not** alter probabilistic behavior.

### Phase 2 must output the following quantities:

| Name        | Shape          | Meaning                              |
| ----------- | -------------- | ------------------------------------ |
| `U`         | (T, K, 3)      | Unit joint directions (global frame) |
| `x_all`     | (T, K, 3)      | 3D joint positions                   |
| `y_pred`    | (C, T, K, 2)   | Projected 2D keypoints per camera    |
| `log_obs_t` | (T,) or scalar | Observation log-likelihood           |

These will be consumed unchanged by Stage 3.

### Deliverables:

* Clean `project_points_pytensor` (canonical PyTensor projector).
* Clean `build_directional_kinematics` helper (directions & 3D joint positions).
* Clean or optional `build_camera_likelihood` helper.
* A readable, linear `build_camera_observation_model`.
* Shape comments and clear documentation.
* Removal/simplification of unused options only when they obscure clarity.

### Non-goals:

* No canonical directions `mu`
* No `kappa`
* No HMM states
* No changes to Stage-1 code
* No performance tuning
* No new folders or heavy abstractions

---

# 🎯 Stage 3 — HMM Over Directional Priors (Next Phase)

**Goal:** Add state-dependent directional priors and use the Stage-1 collapsed HMM over pose regimes.

This stage uses Stage-2 outputs **without modifying** the emission model.

### Components of Stage 3:

For each joint *k* and state *s*:

* **Canonical direction**: `mu[k, s, :]` (unit vector)
* **Concentration**: `kappa[k, s] ~ HalfNormal(...)`

### Directional log-emissions:

[
\log p(U_t \mid z_t = s) = \sum_k \kappa_{k,s} (U_{t,k} \cdot \mu_{k,s})
]

### Combined emissions:

```
log_dir_emit[t, s]  # From U, mu, kappa
log_obs_t[t]        # From Stage 2
logp_emit[t, s] = log_dir_emit[t, s] + log_obs_t[t]
```

### HMM Integration:

Use Stage-1 engine:

```
hmm_ll = collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)
pm.Potential("directional_hmm_prior", hmm_ll)
```

### Non-goals of Stage 3:

* No changes to projection
* No changes to observation likelihood
* No pose dictionaries or alternative HMM formulations
* No dynamic emissions inside the scan

---

# 🧩 Complete Architecture Summary

```
Stage 1: HMM Engine (done)
    collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)
                 ↑
                 │
Stage 3: Directional HMM
    logp_emit[t,s] = log_dir_emit[t,s] + log_obs_t[t]
                 ↑                ↑
                 │                │
Stage 2: Kinematics + Emissions (current)
    U[t,k,3], x_all[t,k,3], y_pred[c,t,k,2], log_obs_t[t]
                 ↑
                 │
Base PyMC kinematic model (cleaned and documented)
```

This plan is now aligned, minimal, and ready for implementation by any engineer or by Copilot.
