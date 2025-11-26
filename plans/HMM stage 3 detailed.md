# Stage 3 — Directional HMM Prior over Joint Directions (Detailed Specification)

This document specifies **Stage 3** of the GIMBAL HMM integration: adding a **state-space prior over joint directions** on top of the existing Stage‑2 PyMC model and Stage‑1 collapsed HMM engine.

It is written for Copilot/Sonnet and assumes:

* **Stage 1** (`hmm_pytensor.py`) is complete and validated.
* **Stage 2** (`pymc_model.py` + helpers) is in its **current implemented form**
  (as described in `stage2-completion-report.md`), *not* the fully helper-extracted spec.
* We are working in the **PyMC/PyTensor** branch, independent of Torch-side code.

Stage 3 must be:

* **Numerically stable** (friendly to nutpie)
* **Modular** (minimal intrusion into Stage‑2 code)
* **Label-switching aware** (with a clear post‑hoc relabeling procedure)
* **Compatible** with both **Gaussian** and **mixture** observation likelihoods.

---

## 1. Stage 3 in the 3‑Stage Pipeline

### 1.1 Inputs from Stage 1 and Stage 2

**From Stage 1** (`hmm_pytensor.py`):

* `collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)`

  * `logp_emit`: `(T, S)` — log emission probabilities
  * `logp_init`: `(S,)` — log initial state probabilities
  * `logp_trans`: `(S, S)` — log transition probabilities
  * Returns **scalar** log-likelihood.

**From Stage 2** (`pymc_model.py`):

Within the PyMC model, Stage 2 provides the following deterministic tensors:

* `U`: `(T, K, 3)` — unit direction vectors in global frame
* `x_all`: `(T, K, 3)` — 3D joint positions
* `y_pred`: `(C, T, K, 2)` — projected 2D keypoints
* `log_obs_t`: `(T,)` — **per‑timestep** observation log-likelihood (Gaussian or mixture)

These are treated as **fixed interface** for Stage 3. Stage 3 **must not modify** how Stage 2 computes these.

### 1.2 What Stage 3 Adds

Stage 3 introduces a **latent state process** over canonical joint directions:

* Discrete state at time `t`: `z_t ∈ {0,…,S−1}`
* For each state `s` and joint `k`:

  * `mu[s, k, 3]`: canonical unit direction in 3D
  * `kappa[s, k]`: concentration (how tightly directions cluster around `mu`)

Stage 3 defines a **directional emission term** from `U` and `(mu, kappa)` and combines it with `log_obs_t` to form `logp_emit[t, s]`. It then calls the Stage‑1 HMM engine.

Stage 3 **does not change**:

* Kinematic parameterization
* Camera model
* 2D observation likelihood
* Nutpie integration logic

---

## 2. Files and Public API

### 2.1 New Module: `hmm_directional.py`

Create a new module (same package as `hmm_pytensor.py`):

```text
gimbal/
  hmm_pytensor.py         # Stage 1 (existing)
  hmm_directional.py      # Stage 3 directional prior (NEW)
  pymc_model.py           # Stage 2 camera + kinematics (existing)
  pymc_utils.py           # shape validation, sampling helpers
```

#### 2.1.1 Public function

```python
def add_directional_hmm_prior(
    U: pt.TensorVariable,         # (T, K, 3) unit directions
    log_obs_t: pt.TensorVariable, # (T,) per-timestep obs logp
    S: int,
    *,
    name_prefix: str = "dir_hmm",
    share_kappa_across_joints: bool = False,
    share_kappa_across_states: bool = False,
    kappa_scale: float = 5.0,
) -> dict:
    """Add a directional HMM prior over U into the *current* PyMC model.

    This function must be called inside a `with pm.Model():` context.

    Parameters
    ----------
    U : (T, K, 3) tensor
        Unit direction vectors for all non-root joints (root row unused).
    log_obs_t : (T,) tensor
        Per-timestep observation log-likelihood from Stage 2.
    S : int
        Number of hidden states in the directional HMM.
    name_prefix : str
        Prefix for variable names (e.g., "dir_hmm" → "dir_hmm_mu", etc.).
    share_kappa_across_joints : bool
        If True, `kappa` is shared across joints (shape `(S,)`).
        If False, `kappa` is joint-specific (shape `(S, K)`).
    share_kappa_across_states : bool
        If True, `kappa` is shared across states (shape `(K,)` or `()`).
    kappa_scale : float
        Scale parameter for HalfNormal priors on `kappa`.

    Returns
    -------
    result : dict
        A dictionary containing created variables:
        {
            "mu": mu,                            # (S, K, 3)
            "kappa": kappa,                      # (S, K) or less
            "init_logits": init_logits,          # (S,)
            "trans_logits": trans_logits,        # (S, S)
            "logp_init": logp_init_det,          # (S,)
            "logp_trans": logp_trans_det,        # (S, S)
            "log_dir_emit": log_dir_emit,        # (T, S)
            "logp_emit": logp_emit_det,          # (T, S)
            "hmm_loglik": hmm_loglik,            # scalar
        }
    """
```

This function is responsible for:

1. Creating **canonical directions** `mu` and **concentrations** `kappa`.
2. Computing the **directional log emission** `log_dir_emit[t, s]`.
3. Combining with `log_obs_t[t]` into `logp_emit[t, s]`.
4. Defining `logp_init` and `logp_trans` (HMM initial and transition log-probs).
5. Calling `collapsed_hmm_loglik` with **numerical stabilization**.
6. Adding `pm.Potential(f"{name_prefix}_loglik", hmm_loglik)` to the model.

### 2.2 Integration Point in `pymc_model.py`

Modify `build_camera_observation_model(...)` to optionally include the Stage‑3 prior.

Add keyword arguments (with defaults that preserve Stage‑2 behavior):

```python
def build_camera_observation_model(
    ...,  # existing args
    use_directional_hmm: bool = False,
    hmm_num_states: int | None = None,
    hmm_kwargs: dict | None = None,
):
    """Builds the PyMC model for 3D kinematics + camera observations.

    If `use_directional_hmm` is True, also adds a directional HMM prior over
    the unit directions U, using Stage 3.
    """
```

At the end of the existing function, after `U` and `log_obs_t` are created:

```python
if use_directional_hmm:
    if hmm_num_states is None:
        raise ValueError("hmm_num_states must be provided when use_directional_hmm=True")

    from gimbal.hmm_directional import add_directional_hmm_prior

    _hmm_result = add_directional_hmm_prior(
        U=U,
        log_obs_t=log_obs_t,
        S=hmm_num_states,
        **(hmm_kwargs or {}),
    )
```

This keeps Stage 2 intact when `use_directional_hmm=False`, and makes Stage 3 an **opt‑in feature**.

---

## 3. Model Components in Stage 3

### 3.1 Canonical Directions `mu[s, k, :]`

We parameterize canonical directions as **normalized 3D vectors**. This avoids fragile vMF distributions and is friendly to nutpie.

Inside `add_directional_hmm_prior`:

```python
T, K, _ = U.shape

mu_raw = pm.Normal(
    f"{name_prefix}_mu_raw",
    mu=0.0,
    sigma=1.0,
    shape=(S, K, 3),
)

# Normalize to unit vectors with epsilon for numerical stability
norm_mu = pt.sqrt((mu_raw ** 2).sum(axis=-1, keepdims=True) + 1e-8)
mu = pm.Deterministic(
    f"{name_prefix}_mu",
    mu_raw / norm_mu,
)  # (S, K, 3)
```

Notes:

* `mu[s, k, :]` is a unit vector, but we don’t impose any further structure.
* The prior on `mu_raw` is roughly isotropic; you can later refine with hierarchical structure if needed.

### 3.2 Concentrations `kappa`

We keep `kappa` positive with a simple HalfNormal prior, and allow optional sharing across joints/states.

Define a helper to build `kappa`:

```python
def _build_kappa(name_prefix, S, K,
                 share_kappa_across_joints,
                 share_kappa_across_states,
                 kappa_scale):
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
```

Then inside `add_directional_hmm_prior`:

```python
kappa = _build_kappa(
    name_prefix,
    S=S,
    K=K,
    share_kappa_across_joints=share_kappa_across_joints,
    share_kappa_across_states=share_kappa_across_states,
    kappa_scale=kappa_scale,
)  # (S, K)
```

### 3.3 Directional Log-Emission `log_dir_emit[t, s]`

We implement a **vMF-flavored dot-product energy** that is numerically simple and nutpie-friendly.

Given:

* `U`: `(T, K, 3)`
* `mu`: `(S, K, 3)`
* `kappa`: `(S, K)`

We compute:

```python
# Reshape for broadcasting
U_exp = U.dimshuffle(0, "x", 1, 2)   # (T, 1, K, 3)
mu_exp = mu.dimshuffle("x", 0, 1, 2) # (1, S, K, 3)

# Dot-products U_tk · mu_sk
cosine = (U_exp * mu_exp).sum(axis=-1)  # (T, S, K)

# Apply concentration weights and sum over joints
kappa_exp = kappa.dimshuffle("x", 0, 1)  # (1, S, K)
log_dir_emit = (kappa_exp * cosine).sum(axis=-1)  # (T, S)

log_dir_emit = pm.Deterministic(f"{name_prefix}_log_dir_emit", log_dir_emit)
```

Notes:

* This is equivalent to a vMF log-density **up to an additive constant** in `kappa` (normalizing constant). For many purposes (directional regularization, relative weighting against `log_obs_t`) this is sufficient and simpler.
* If needed, a later extension can add an approximate `log C_3(kappa)` term, but Stage 3 **does not require** it.

### 3.4 Combining with Observation Log-Likelihood

Stage 2 provides `log_obs_t` of shape `(T,)`. We broadcast it over states:

```python
log_obs_t_exp = log_obs_t.dimshuffle(0, "x")  # (T, 1)
logp_emit_raw = log_dir_emit + log_obs_t_exp   # (T, S)

# Wrap in Deterministic to keep scan gradients happy (mirrors Stage 1 pattern)
logp_emit = pm.Deterministic(f"{name_prefix}_logp_emit", logp_emit_raw)
```

This `logp_emit` is the tensor passed to `collapsed_hmm_loglik`.

---

## 4. HMM Parameters and Numerical Stabilization

### 4.1 Initial and Transition Log-Probabilities

We adopt the same pattern as the Stage‑1 Gaussian HMM demo (`hmm_pymc_utils.py`), using logits and softmax normalization.

```python
init_logits = pm.Normal(f"{name_prefix}_init_logits", 0.0, 1.0, shape=(S,))
trans_logits = pm.Normal(f"{name_prefix}_trans_logits", 0.0, 1.0, shape=(S, S))

# Normalize to log-probabilities
logp_init = init_logits - pm.math.logsumexp(init_logits)
logp_trans = trans_logits - pm.math.logsumexp(trans_logits, axis=1, keepdims=True)

# Wrap in Deterministic for scan gradient compatibility
logp_init_det = pm.Deterministic(f"{name_prefix}_logp_init", logp_init)
logp_trans_det = pm.Deterministic(f"{name_prefix}_logp_trans", logp_trans)
```

### 4.2 Numerical Stabilization of `logp_emit`

Because `log_obs_t` can be extremely negative when summing over many joints and cameras, we stabilize `logp_emit` before calling the HMM engine by subtracting a per‑timestep maximum.

This is optional but recommended and **does not change gradients**:

```python
# logp_emit: (T, S)
max_per_t = pm.math.max(logp_emit, axis=1, keepdims=True)  # (T, 1)
logp_emit_centered = logp_emit - max_per_t                 # (T, S)

# Sum of the constants we subtracted
offset = max_per_t.sum()  # scalar

# Call Stage-1 HMM engine
from gimbal.hmm_pytensor import collapsed_hmm_loglik

hmm_ll_centered = collapsed_hmm_loglik(
    logp_emit_centered,
    logp_init_det,
    logp_trans_det,
)

hmm_loglik = pm.Deterministic(
    f"{name_prefix}_loglik",
    hmm_ll_centered + offset,
)

pm.Potential(f"{name_prefix}_potential", hmm_loglik)
```

Key points:

* Subtracting `max_per_t` per row reduces dynamic range in `logp_emit_centered`.
* Adding `offset` back preserves the true **scalar** log-likelihood.
* Gradients are unaffected by the constant offset.

---

## 5. Label Switching Mitigation (Post‑hoc Relabeling)

Stage 3 **does not break** the inherent label symmetry of the HMM. Instead, it provides a **post‑hoc relabeling algorithm** for interpretability and stable summaries.

We recommend a **Hungarian-assignment–based** procedure, implemented outside the PyMC model, e.g. in a notebook or analysis script.

### 5.1 Summary Features per State

Let `mu[s, k, :]` and `kappa[s, k]` be the state parameters from Stage 3. Define a summary feature vector per state (for each posterior draw):

```python
# Example summary: concatenation of canonical directions for selected joints
# or mean joint positions, or any feature that reflects pose.

# Pseudocode: for a given draw
v_s = flatten(mu[s, selected_joints, :])  # shape (d,)
```

Alternative / complementary features:

* Mean vertical head position or COM per state
* Average alignment of key segments (e.g., torso, legs)

All that matters is that `v_s` is consistent across draws.

### 5.2 Reference State Ordering

1. Choose a **reference estimate** per state, e.g. mean over all draws:

   ```python
   # Suppose we have samples for mu from ArviZ InferenceData
   # Compute posterior mean for each state s
   v_ref[s] = E[v_s]  # vector in R^d
   ```

2. Stack into a matrix `V_ref ∈ R^{S × d}`.

### 5.3 Per-draw Hungarian Alignment

For each posterior draw `m`:

1. Compute its state summary vectors `V_m ∈ R^{S × d}`.

2. Build a cost matrix `C_m ∈ R^{S × S}`:

   ```python
   # distance metric choice
   C_m[i, j] = ||V_m[i] - V_ref[j]||^2   # Euclidean
   # or: C_m[i, j] = 1 - cosine_similarity(V_m[i], V_ref[j])
   ```

3. Solve the assignment problem to find the **best permutation**:

   ```python
   from scipy.optimize import linear_sum_assignment

   row_ind, col_ind = linear_sum_assignment(C_m)
   # col_ind[i] gives the reference state that draw i should be mapped to
   # This corresponds to permutation π_m such that π_m(i) = col_ind[i]
   ```

4. Apply the permutation `π_m` to **all state-indexed quantities** in draw `m`:

   * `mu[s, k, :]`, `kappa[s, k]`
   * `init_logits[s]`, rows/columns of `trans_logits`
   * any state‑wise diagnostics or labels.

Optionally, iteratively update `V_ref` from relabeled samples (Stephens-style iterative relabeling). For most practical purposes, a single pass with a fixed `V_ref` is sufficient.

### 5.4 Integration in the Plan

* Stage 3 specification should include a **short, well‑documented function skeleton** (Python pseudocode) for `relabel_hmm_states` in a notebook.
* This is deliberately kept outside the core library, as it operates on posterior samples and can depend on `arviz` + `scipy`.

---

## 6. Interaction with Legacy Notebooks and vMF/Torch Implementations

The repository contains older notebooks and code paths that use:

* Explicit vMF distributions
* Torch-only GIMBAL skeleton/camera models
* Gaussian-only or camera-only demos

Because this environment cannot inspect `.ipynb` contents directly, **Copilot/Sonnet** should:

1. **Scan all `demo_*.ipynb` notebooks** (`demo_pymc_skeleton`, `demo_vmf_distribution`, `demo_pymc_camera_full`, `demo_pymc_camera_simple`, `demo_pymc_setup`, `hmm_demo_gaussians`, etc.).
2. Identify which code paths they depend on:

   * vMF distribution implementations in `pymc_distributions.py`
   * Torch-only `model.py` / `inference.py` / `camera.py`
   * Old Gaussian HMM functions in `hmm_pymc_utils.py`
3. For each notebook:

   * Decide whether it should be **kept** as a supported example or
     considered **legacy/archival**.
   * If kept:

     * Ensure imports are updated to use the **current Stage‑1/2/3 APIs**.
     * Prefer re‑implementing directional priors using the Stage‑3 dot-product HMM rather than older vMF code.

This decision and any necessary code shims can be documented in a short `NOTES.md` next to the notebooks.

---

## 7. Example: Minimal Directional HMM with Stage‑2 Interface

To give Copilot a concrete code template, Stage 3 should include (e.g., in `examples/directional_hmm_minimal.py`) a minimal runnable example that:

1. Creates synthetic `U[T, K, 3]` and `log_obs_t[T]`.
2. Builds a PyMC model using `add_directional_hmm_prior`.
3. Runs nutpie for a few short chains.
4. Applies the label-switching fix.

Sketch:

```python
import pymc as pm
import pytensor.tensor as pt
import numpy as np
from gimbal.hmm_directional import add_directional_hmm_prior

T, K, S = 50, 5, 4

# 1) Fake data
rng = np.random.default_rng(123)
U_data = rng.normal(size=(T, K, 3))
U_data /= np.linalg.norm(U_data, axis=-1, keepdims=True) + 1e-8
log_obs_t_data = rng.normal(loc=-100.0, scale=10.0, size=(T,))

with pm.Model() as model:
    # Stage 2 interface as Data inputs for this toy example
    U = pm.Data("U", U_data)            # (T, K, 3)
    log_obs_t = pm.Data("log_obs_t", log_obs_t_data)  # (T,)

    hmm_result = add_directional_hmm_prior(
        U=U,
        log_obs_t=log_obs_t,
        S=S,
        name_prefix="toy",
        share_kappa_across_joints=False,
        share_kappa_across_states=False,
        kappa_scale=5.0,
    )

    idata = pm.sample(1000, tune=500, nuts_sampler="nutpie", chains=2, cores=2)

# 4) Post-hoc relabeling would now operate on `idata`
```

This example is not intended as a scientific model; it is a **reference implementation** of the Stage‑3 logic for Copilot and for future developers.

---

## 8. Testing and Validation Plan for Stage 3

Stage 3 needs its own test coverage, complementing Stage 1 and 2.

### 8.1 Unit Tests for `hmm_directional.py`

1. **Shape tests**:

   * For a small `(T, K, S)` (e.g., `T=10, K=3, S=2`), verify that:

     * `mu` has shape `(S, K, 3)` and unit norms.
     * `kappa` has the expected shape depending on sharing flags and that `kappa_full` is `(S, K)`.
     * `log_dir_emit` and `logp_emit` are `(T, S)`.

2. **Numerical stability test**:

   * Generate extreme `log_obs_t` values (very negative numbers) and random `U`.
   * Build the model and evaluate `hmm_loglik` at the initial point.
   * Assert that `hmm_loglik` is finite (not `NaN` or `±inf`).

3. **Gradient test (coarse)**:

   * Build a small model and compile `dlogp` (PyMC’s gradient function).
   * Check that gradients wrt `init_logits`, `trans_logits`, `mu_raw`, and `kappa` are finite.

### 8.2 End-to-End Test with Simulated Data

1. Simulate a simple 2‑state directional process:

   * Two canonical direction patterns for a small skeleton.
   * Generate `U` according to those patterns with noise.
   * Generate synthetic `log_obs_t` that slightly favors the correct state.

2. Fit the Stage‑3 model:

   * Confirm that the posterior roughly recovers:

     * Distinct direction patterns per state.
     * Non-degenerate transition matrix.

3. Verify that the **label-switching fix** yields:

   * Stable posterior means for state‑specific quantities across chains.

### 8.3 Compatibility with Stage 2

Add a test that uses the **real Stage‑2 model**:

* Build `build_camera_observation_model(..., use_directional_hmm=True, hmm_num_states=S)` with a very small synthetic dataset (e.g., `T=5, K=3, C=2`).
* Validate:

  * Model builds and compiles.
  * `U`, `log_obs_t`, `logp_emit`, and `hmm_loglik` are present and finite.
  * A short nutpie run completes without errors.

---

## 9. Non-Goals and Boundaries for Stage 3

Stage 3 **deliberately does not**:

* Reintroduce full vMF priors as PyMC distributions for directions.
* Change Stage‑2 kinematics, camera, or likelihood parameterization.
* Interpret HMM states directly inside the model (all label-switching handling is post‑hoc).
* Touch Torch-only files (`model.py`, `inference.py`, Torch `camera.py`) except to keep them compiling if examples are preserved.

Future work may:

* Add approximate vMF normalizing constants `log C_3(kappa)` if needed.
* Introduce hierarchical structure over `mu` and `kappa` across joints/states.
* Add optional pose dictionaries or state grouping for interpretability.

---

This specification defines all the components required for **Stage 3**: file locations, tensor shapes, PyMC variables, HMM integration, numerical stabilization, and label-switching mitigation. It is designed so that Copilot/Sonnet can implement and extend the directional HMM prior without ambiguity while preserving the existing Stage‑1 and Stage‑2 behavior.
