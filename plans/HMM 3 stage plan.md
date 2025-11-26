# 🚀 Collapsed HMM Integration Plan — Clean 3-Stage Overview (Rewritten, Updated 2025)

This document defines the **complete end-to-end plan** for adding a collapsed Hidden Markov Model (HMM) over joint-direction priors into the PyMC kinematic model.

Updated to align with:

- Stage-1 completion report  
- Updated **Stage 2** specification  
- Clarity > Simplicity > Heavy Engineering  

---

# ✅ Stage 1 — Collapsed HMM Engine (Completed)

**Goal:** Implement and validate a numerically stable PyTensor collapsed HMM log-likelihood.

**Deliverables:**

- `hmm_pytensor.py`:  
  - `forward_log_prob_single`  
  - `collapsed_hmm_loglik`
- Gaussian-emission demo in `hmm_pymc_utils.py`

**Validation (in notebooks):**

- Tiny-HMM brute force enumeration  
- Gradient & finite-difference checks  
- Normalization tests  
- (Optional) Viterbi decoding

**Properties:**

- Pure PyTensor (`scan`, `logsumexp`)
- Gradient-safe  
- Handles edge case `T=1`  
- Validation code lives outside the library

---

# 🚧 Stage 2 — Direction & Emission Refactor (Current Work)

**Goal:** Provide a clean, readable, modular PyMC kinematic+emission model with a stable interface for Stage 3.

### Required Outputs (Stage-2 → Stage-3 contract)

| Name        | Shape        | Meaning                             |
|-------------|--------------|-------------------------------------|
| `U`         | (T, K, 3)    | Unit joint directions (global)      |
| `x_all`     | (T, K, 3)    | 3D joint positions                  |
| `y_pred`    | (C, T, K, 2) | 2D projections                      |
| `log_obs_t` | (T,)         | Per-timestep observation logp       |

These are used **unchanged** in Stage 3.

### Deliverables:

- Clean `project_points_pytensor`
- Clean `build_directional_kinematics`
- Optional `build_camera_likelihood`
- Linear, readable `build_camera_observation_model`
- Shape comments
- Removal of unused or confusing branches

### Non-Goals:

- No HMM logic  
- No canonical directions  
- No `kappa`  
- No performance optimization  
- No architectural shifts  

---

# 🎯 Stage 3 — HMM Over Directional Priors

**Goal:** Add state-dependent directional priors and apply the collapsed HMM over time.

### Components:

For each joint `k` and hidden state `s`:

- `mu[k, s, :]` — canonical direction (unit vector)  
- `kappa[k, s]` — concentration parameter (`HalfNormal` or `Gamma`)  

### Directional log-emissions:

\[
\log p(U_t \mid z_t=s) = \sum_k \kappa_{k,s} \left(U_{t,k} \cdot \mu_{k,s}\right) + \log C_3(\kappa_{k,s})
\]

where `C₃(κ)` is the vMF normalizing constant in 3D.

### Combined emissions:

```
log_dir_emit[t, s]
log_obs_t[t]
logp_emit[t, s] = log_dir_emit[t, s] + log_obs_t[t]
```

### HMM Integration:

```python
hmm_ll = collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)
pm.Potential("directional_hmm_prior", hmm_ll)
```

### Non-goals:

- No changes to projection or likelihood  
- No pose dictionaries  
- No dynamic emissions inside scan  

---

# 🧩 Architecture Summary (With Shapes)

```
Stage 1: HMM Engine (done)
    collapsed_hmm_loglik(logp_emit[T,S], logp_init[S], logp_trans[S,S])
                     ↑
                     │
Stage 3: Directional HMM
    logp_emit[t,s] = log_dir_emit[t,s] + log_obs_t[t]
                     ↑                ↑
                     │                │
Stage 2: Kinematics + Emissions
    U[T,K,3], x_all[T,K,3], y_pred[C,T,K,2], log_obs_t[T]
                     ↑
                     │
Base PyMC kinematic model (clean, documented)
```

---

# ⚠️ Known Limitations and Expected Issues

- State label switching  
- Possible local modes when κ is large  
- Root direction unused (set to zeros)  
- vMF normalization approximation may be needed  

---

# ✔ Final Output of the Full 3-Stage Plan

- Stage 1: HMM engine  
- Stage 2: Clean emissions + kinematics  
- Stage 3: Direction-HMM prior  

This revised plan is simple, modular, and consistent with the actual codebase and Stage-2 specification.
