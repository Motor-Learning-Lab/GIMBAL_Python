# Plan: Collapsed HMM Integration for GIMBAL PyMC Model

This document describes a **three-stage plan** to integrate a **collapsed Hidden Markov Model (HMM)** into the current GIMBAL-style PyMC kinematic model.

The goals:

- Keep **PyTensor** as the computational backend.
- Use **collapsed HMM** (discrete states integrated out; only continuous parameters sampled).
- Keep current **Gaussian-normalized direction priors** (no vMF).
- Make the HMM **generic** and reusable with a **general emission log-likelihood**.
- Eventually use the HMM to represent **canonical joint directions and spread** as *state-dependent*, instead of recomputing per frame.

We assume the current code layout (simplified):

- `model.py` – data classes / skeleton structure.
- `camera.py` – camera projection and related utilities.
- `pymc_model.py` – builds the PyMC model (positions, directions, robust observation model).
- `pymc_utils.py` – initialization helpers, etc.
- `inference.py` – sampling wrappers (nutpie + PyMC).
- `fit_params.py` – parameter containers/configs.
- `demo_pymc_camera_simple.html` / `demo_pymc_camera_full.html` – notebook exports showing current pipeline.

We will **add** at least one new module:

- `hmm_pytensor.py` (or similar) – generic collapsed HMM routines.


---

## Stage 1 – Generic Collapsed HMM Building Block (No Cameras, No Skeleton)

### Goal

Implement and validate a **generic collapsed HMM** component in PyTensor/PyMC, independent of 3D kinematics.

This includes:

- A **PyTensor function** that computes the **log-likelihood of a sequence** under a discrete HMM, where:
  - Emissions are provided as a **log-emission matrix** `logp_emit[t, s] = log p(y_t | z_t = s, θ)`.
  - Initial state distribution and transition matrix are represented in log-space.
  - Discrete states `z_t` are **not sampled**; they are summed out via the forward recursion.
- An idiomatic **PyMC helper** that:
  - Creates priors over HMM parameters (initial logits, transition logits, emission parameters).
  - Converts them into log-parameters.
  - Calls the PyTensor forward routine.
  - Wraps the result in a `pm.Potential("hmm_loglik", ...)`.

### Deliverables

1. `hmm_pytensor.py`:
   - `forward_log_prob_single(...)`: implements forward algorithm in log-space for one sequence.
   - `collapsed_hmm_loglik(...)`: public API for computing the collapsed HMM log-likelihood for one or multiple sequences (initially one).
2. `hmm_demo_gaussians.ipynb` (or similar) notebook:
   - Simulate data from a small HMM (e.g. 2–3 states, 1D and 2D Gaussian emissions).
   - Fit the HMM using PyMC + nutpie, using the collapsed likelihood.
   - Verify recovery of parameters (up to label-switching).
3. Optional small unit tests (if you want pytest later) for the core forward algorithm.

**No references to 3D joints, cameras, or directions yet.**


---

## Stage 2 – Factor Out Emission Functions in Current GIMBAL Model

### Goal

Refactor the current PyMC camera model into:

- A **clean separation** between:
  - **Kinematic + camera forward model** (root random walk, joint directions, bone lengths, projection).
  - **Emission likelihood** (robust mixture likelihood of 2D keypoints given projected positions).
- A **reusable emission interface** that can later be called from the HMM.

In this stage, the model still has **no discrete states**. We keep a “single state” view conceptually, but structure the code so that plugging in an HMM is straightforward.

### What changes (conceptual)

1. **Add a small emission API layer**, e.g. in `pymc_model.py` or a new module:

   - Something like:

     ```python
     def compute_projected_points(x_all, camera_params):
         # existing project_points_pytensor logic
         ...

     def log_obs_likelihood(y_obs, y_pred, obs_sigma, inlier_prob, occlusion_mask):
         # current robust mixture logp, returns scalar logp
         # will be used by both "plain" model and HMM model
         ...
     ```

   - At this stage, this API still **does not depend on discrete states**. It is the current robust likelihood.

2. **Refactor model building code** in `pymc_model.py`:

   - Ensure there is a clear “forward pass”:

     - Sample positions and directions
     - Project to cameras
     - Compute `y_pred`
     - Use `log_obs_likelihood` to add a `pm.Potential` (or specify as custom log-likelihood).

   - This will make it easy to later:
     - Expand `log_obs_likelihood` to depend on state-specific priors over directions.
     - Wrap the whole directional prior + observation model inside the HMM emission.

3. **Preserve existing notebooks**:

   - Update `demo_pymc_camera_simple.ipynb` and `demo_pymc_camera_full.ipynb` (or their `.py` equivalents) to call the new refactored API.
   - Confirm they still:
     - Build the model.
     - Sample successfully with nutpie and/or PyMC NUTS.
     - Produce similar posterior summaries as before (within Monte-Carlo noise).

### Deliverables

1. Updated `pymc_model.py` with:
   - Kinematic forward pass isolated.
   - Projected points function separated.
   - Robust observation log-likelihood factored out into a clearly named helper.

2. Updated notebooks showing:
   - **No behavior change** in the non-HMM setting.
   - All tests still pass (visual inspection and simple numeric checks).


---

## Stage 3 – Full Collapsed HMM over Canonical Directions

### Goal

Add a **state-space layer** over joint directions, similar in spirit to the original GIMBAL algorithm:

- A discrete HMM over **“pose states”**.
- Each state has its own **canonical joint directions** and **spread**.
- Temporal evolution is captured by the HMM, not by per-frame ad-hoc priors.

We still:

- Use **normalized Gaussians** for directions (no vMF).
- Keep the existing robust camera observation model.
- Use the **collapsed HMM likelihood** from Stage 1 so that:
  - Only **continuous parameters** (state-specific direction parameters, spreads, transition logits, etc.) are sampled.
  - Discrete state sequence is integrated out via the forward algorithm.

### Conceptual model

Let:

- `T` – number of time frames.
- `K` – number of joints.
- `S` – number of discrete states.
- For each joint `k` and state `s`, we have:

  - `raw_u[k, s] ~ Normal(0, 1, size=3)`, `u[k, s] = raw_u[k, s] / ||raw_u[k, s]||` – **state-specific canonical directions**.
  - Optionally, **spread** parameters `tau[k, s]` controlling the variance of directions around these canonical directions.

- For each time `t` and joint `k`, effective direction is influenced by the current hidden state `z_t` through a prior:

  - In collapsed version, we do **not** sample `z_t`. Instead, we:

    1. For each state `s`, compute:
       - The log prior over directions at time `t`, `log p(u_k(t) | z_t = s, state_parameters)` (or log prior for other relevant emitted quantities such as local coordinates).
       - The log observation likelihood under state `s`, given 2D keypoints and camera model.
       - Combine to get **per-state emission log-likelihood**:

         ```text
         logp_emit[t, s] = log p(y_t | z_t = s, θ) + log p(u(t) | z_t = s, θ)
         ```

    2. Use the collapsed HMM component from Stage 1 to sum over state sequences.

At this stage, we can implement a **simplified version** first:

- Maybe only **one or two joints** (e.g. a 3-joint chain: pelvis, shoulder, elbow).
- Low number of states (e.g. `S = 2` or `3`).
- Synthetic data where the true state sequence is simple.

Then extend to:

- Full skeleton used by your current demo.
- More states, more joints, and more realistic motion.

### Integration tasks

1. **Define HMM prior in PyMC**

   - In `pymc_model.py` or a new module, create:

     - Global transition logits `trans_logits[s_from, s_to]`.
     - Initial logits `init_logits[s]`.
     - Convert to log transition probabilities, etc.

   - Combine with state-specific directional parameters:

     - `raw_u[k, s]`, `tau[k, s]` or equivalent.

2. **Define state-conditioned emission log-likelihood**

   - For each `(t, s)`:

     - Run the existing forward kinematics (positions, projections) *once* per sample (not per state; states modify priors on directions, not the basic projection geometry).
     - Compute:

       ```text
       logp_emit[t, s] = log p(y_t | x_all[t], cameras, state s, θ)
       ```

       where “state s” typically affects:
       - Priors over the directions `u_k[t]` (and perhaps other local features).

   - Implementation detail:

     - Initially, to keep things simple, we can treat the **direction prior** at time `t` as the only state-dependent term, while the robust observation model stays the same across states. That is:

       ```text
       logp_emit[t, s] ≈ log p(y_t | x_all[t], cameras, θ)  +  log p(u(t) | state s, θ)
       ```

3. **Call collapsed HMM component**

   - Build `logp_emit` as a PyTensor tensor of shape `(T, S)` inside the PyMC model.
   - Compute:

     ```python
     from hmm_pytensor import collapsed_hmm_loglik

     hmm_ll = collapsed_hmm_loglik(
         logp_emit=logp_emit,
         logp_init=logp_init,
         logp_trans=logp_trans,
     )
     pm.Potential("hmm_loglik", hmm_ll)
     ```

4. **Testing strategy**

   - Start with a **minimal synthetic example**:
     - 1D or 2D “pose feature” per frame (e.g., a single joint’s direction in some local coordinate).
     - State-dependent Gaussians + HMM.
     - Use your collapsed HMM to recover states and parameters.
   - Then move to a **simple 3-joint synthetic skeleton**:
     - Pelvis → shoulder → elbow.
   - Finally, integrate into the full multi-camera model and adapt the demo notebooks.

### Deliverables

1. Updated `pymc_model.py` (or a new `pymc_model_hmm.py` if you prefer clear separation) with:
   - HMM parameter priors.
   - Construction of `logp_emit`.
   - Use of `collapsed_hmm_loglik`.
2. New notebook: `demo_pymc_hmm_simple.ipynb`:
   - Synthetic “pose states” with 1–3 joints.
   - Validation of state-dependent directional priors.
3. New notebook: `demo_pymc_camera_hmm_full.ipynb`:
   - Full multi-camera model with HMM over directions.
   - Demonstration that sampling remains robust with nutpie, and the HMM gives interpretable “canonical directions” as states.


---

## Summary

- **Stage 1**: Build and test a **generic collapsed HMM** in PyTensor/PyMC (no cameras).
- **Stage 2**: Refactor existing model to separate **forward kinematics** and **emission log-likelihood**.
- **Stage 3**: Add **state-dependent canonical directions** and a **collapsed HMM** over time, first in a simple synthetic setting, then in the full camera model.

Each stage is testable and useful on its own, and later stages reuse earlier code rather than rewriting it.
