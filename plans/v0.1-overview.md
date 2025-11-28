# Collapsed HMM Integration — Updated 3‑Stage Overview

This is the **high-level roadmap** for integrating a collapsed Hidden Markov Model (HMM) over joint directions into the refactored GIMBAL PyMC model. It reflects the **actual current codebase** and the finalized Stage‑3 specification.

---

## Stage 1 — Collapsed HMM Engine (✅ Complete)

**Goal:** Implement a numerically stable, gradient‑safe collapsed HMM log‑likelihood in PyTensor/PyMC, independent of cameras and kinematics.

**Core artifact:** `hmm_pytensor.py`

**API:**

* `collapsed_hmm_loglik(logp_emit, logp_init, logp_trans) -> scalar`

  * `logp_emit`: `(T, S)` — log p(y_t | z_t = s)
  * `logp_init`: `(S,)` — log initial state probabilities
  * `logp_trans`: `(S, S)` — log transition probabilities

**Properties:**

* Uses log‑space forward algorithm with `pt.scan` and `logsumexp`.
* Gradient‑friendly (nutpie and PyMC NUTS both work).
* Validated against brute‑force enumeration, T=1 edge case, and finite‑difference gradients.
* Completely decoupled from cameras, skeletons, and 3D kinematics.

Stage 1 is a **black box**: given emission log‑likelihoods and HMM parameters, it returns a scalar log‑likelihood.

---

## Stage 2 — Kinematics + Camera Emissions Refactor (✅ Complete)

**Goal:** Refactor the PyMC pose + camera model so that it exposes a clean, stable interface for Stage 3, without changing the underlying behavior.

**Core artifact:** `pymc_model.py` (+ helpers in `pymc_utils.py`)

**Key responsibilities:**

1. **3D kinematics**

   * Parameterize joint directions and bone lengths.
   * Build 3D joint positions via forward kinematics.

2. **Camera projection**

   * Project 3D joints to 2D keypoints using `project_points_pytensor`.

3. **2D observation likelihood**

   * Compute per‑timestep log‑likelihood under Gaussian or mixture models.

4. **Expose Stage‑2 → Stage‑3 interface tensors.**

**Stage‑2 → Stage‑3 contract:** inside the PyMC model, Stage 2 must produce:

* `U`: `(T, K, 3)` — unit joint direction vectors in the global frame.
* `x_all`: `(T, K, 3)` — 3D joint positions.
* `y_pred`: `(C, T, K, 2)` — projected 2D keypoints per camera.
* `log_obs_t`: `(T,)` — **per‑timestep** 2D observation log‑likelihood.

Additional details:

* Direction vectors and joint positions are normalized and constructed with small epsilons for numerical stability.
* `log_obs_t` sums log‑likelihoods over cameras and joints but **not** over time, making it suitable to combine with time‑varying HMM emissions.
* Both Gaussian and mixture likelihoods are supported and produce the same `log_obs_t` interface.

Stage 2 **does not contain any HMM logic**. It is purely:

> "joint directions + kinematics + camera + 2D likelihood → U, x_all, y_pred, log_obs_t"

---

## Stage 3 — Directional HMM Prior over Joint Directions (🚧 This Stage)

**Goal:** Add a **state‑space prior over joint directions** on top of the Stage‑2 model, using the Stage‑1 collapsed HMM. Each hidden state represents a canonical directional pattern (a pose template) with its own concentration.

**Core artifact:** `hmm_directional.py` (NEW)

**Conceptual model:**

* At each time `t`, there is a discrete state `z_t ∈ {0,…,S−1}`.
* Each state `s` defines canonical joint directions and concentration parameters:

  * `mu[s, k, 3]` — unit vector canonical direction for joint `k` in state `s`.
  * `kappa[s, k]` — concentration for joint `k` in state `s`.
* The observed directions `U[t, k, :]` from Stage 2 are more likely under the state whose `(mu, kappa)` best match them.

**Directional emission (vMF‑flavored dot‑product):**

Using dot‑product energies for numerical robustness:

* Given `U[T, K, 3]`, `mu[S, K, 3]`, `kappa[S, K]`, define:

  * `log_dir_emit[t, s] = sum_k kappa[s, k] * dot(U[t, k], mu[s, k])`

This term is vMF‑like (vMF log‑density is `κ μᵀu` + constant) but implemented with simple operations that play well with nutpie.

**Combining with Stage‑2 observation log‑likelihood:**

Stage‑2 provides `log_obs_t[T]`. Stage 3 constructs:

* `logp_emit[t, s] = log_dir_emit[t, s] + log_obs_t[t]`

This `logp_emit[T, S]` is then passed to `collapsed_hmm_loglik` along with HMM initial and transition log‑probabilities.

**Integration into the PyMC model:**

* Stage 3 is wrapped in a helper function, e.g. `add_directional_hmm_prior(U, log_obs_t, S, ...)`, which:

  * Creates `mu`, `kappa`, `init_logits`, `trans_logits`.
  * Computes `log_dir_emit`, `logp_emit` with per‑timestep numerical stabilization.
  * Calls `collapsed_hmm_loglik` and adds a `pm.Potential` to the model.
* `pymc_model.build_camera_observation_model(...)` gains an optional flag:

  * `use_directional_hmm: bool = False`
  * `hmm_num_states: int | None = None`
  * When enabled, it calls `add_directional_hmm_prior` after Stage‑2 tensors are built.

**Numerical stability:**

* `log_obs_t` can be extremely negative; before calling the HMM, Stage 3 subtracts a per‑timestep maximum from `logp_emit` and re‑adds the summed offset to the final log‑likelihood.
* This keeps dynamic ranges manageable without changing gradients.

**Label switching:**

* The HMM remains symmetric in state labels; Stage 3 does **not** attempt to break label symmetry inside the model.
* Instead, Stage 3 defines a **post‑hoc relabeling procedure**:

  * Compute a summary feature vector per state (e.g., flattened canonical pose or head height pattern).
  * Choose a reference ordering (e.g., mean over posterior draws).
  * For each draw, find the best permutation to match the reference ordering using the Hungarian algorithm.
  * Apply this permutation to all state‑indexed quantities for interpretation and plotting.

This mirrors the original GIMBAL paper’s practice of reordering states for interpretability, but in a more systematic, algorithmic way.

---

## Summary of Responsibilities

* **Stage 1** — *Time series engine*: purely mathematical collapsed HMM log‑likelihood.
* **Stage 2** — *Physical + observational layer*: from directions and skeleton to 2D keypoints and per‑timestep log‑likelihood.
* **Stage 3** — *Pose state prior*: uses Stage‑2 directions + log‑likelihood to define pose states and their dynamics via the Stage‑1 HMM engine.

Each stage is designed to be:

* **Modular** — interfaces are explicit (`logp_emit`, `U`, `log_obs_t`).
* **Stable** — numeric issues localized and handled in the relevant layer.
* **Extensible** — future work (e.g., hierarchical priors on `mu`, approximate vMF constants, pose grouping) can be added on top without breaking the 3‑stage contract.
