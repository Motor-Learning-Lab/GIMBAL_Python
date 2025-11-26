# 📘 Phase 2 Specification: Direction & Emission Refactor for PyMC Model

*A clarity-first restructuring of `pymc_model.py` and supporting modules, sitting between Stage 1 (HMM engine) and Stage 3 (HMM over directions)*

---

This specification describes the goals, structure, responsibilities, and concrete refactoring steps for **Phase 2** of the project. It assumes:

* **Stage 1 is complete**: the collapsed HMM engine lives in `hmm_pytensor.py` and is fully validated.
* **Stage 3 will build on Phase 2**: Stage 3 will introduce canonical directions, concentration parameters, and the collapsed HMM prior over directions, but **will not modify the emission pipeline**.

Guiding principle:

> **Clarity > Simplicity > Heavy Engineering**

The aim is not to redesign the entire codebase but to make the PyMC/PyTensor model for **directions**, **3D kinematics**, **camera projection**, and **2D likelihood** easy to understand and easy to extend.

Phase 2 has two main jobs:

1. Make the current PyMC model readable and structurally clean.
2. Define a clear interface for Stage 3 to consume.

---

## 1. Interface Contract Between Stage 2 and Stage 3

By the end of Phase 2, the PyMC model must provide the following well-defined quantities:

* `U`: `(T, K, 3)` — unit direction vectors for non-root joints in the **global frame**.
* `x_all`: `(T, K, 3)` — 3D joint positions in global coordinates.
* `y_pred`: `(C, T, K, 2)` — projected 2D keypoints per camera.
* `log_obs_t` *(optional but recommended)*: `(T,)` — per-timestep observation log-likelihood, so Stage 3 can easily construct `logp_emit[t, s] = log_dir_emit[t, s] + log_obs_t[t]`.

Stage 3 will:

* Leave the **emission pipeline** (`x_all → y_pred → log_obs_t`) unchanged.
* Add canonical directions `mu[k, s]` and concentrations `kappa[k, s]`.
* Compute directional log-emissions from `U` and feed that, plus `log_obs_t`, into the **Stage-1 HMM engine**.

Phase 2 **must not** modify or depend on:

* `hmm_pytensor.py` (collapsed HMM engine)
* `hmm_pymc_utils.py` (Gaussian HMM demo)

---

## 2. Current Module Responsibilities (Baseline)

Phase 2 preserves the overall file layout and clarifies responsibilities.

### 2.1 Torch-side modules (leave mostly unchanged)

These files belong to the original GIMBAL/Torch stack and are **not** the focus of Phase 2:

* `camera.py`

  * Canonical Torch 3D→2D projection.
  * All Torch camera math lives here.

* `model.py`

  * Torch generative model:

    * Root random walk
    * Hierarchical priors
    * Directional priors / HMM priors
    * Observation likelihood (calling `camera.project_points`)

* `fit_params.py`

  * Converts measurement data into `InitializationResult`.
  * Handles DLT/Anipose/ground-truth initialization.

* `inference.py`

  * Torch MCMC: HMC, Gibbs updates, FFBS, etc.

For Phase 2, only light comment/docstring improvements or obvious dead-code removals are allowed here. No structural changes are required.

---

### 2.2 PyMC / PyTensor modules (Phase 2 focus)

These belong to the PyMC/PyTensor stack:

* `hmm_pytensor.py`

  * Stage-1 collapsed HMM engine (forward algorithm + `collapsed_hmm_loglik`).
  * **Do not modify in Phase 2.**

* `hmm_pymc_utils.py`

  * Gaussian HMM demo utilities (PyMC model + simulator).
  * May be left untouched unless small doc updates are helpful.

* `pymc_distributions.py`

  * Custom PyMC distributions (e.g., von Mises-Fisher).
  * Keep as the place for special distributions.

* `pymc_utils.py`

  * PyMC helper utilities: nutpie integration, initial point construction, shape checks.

* `pymc_model.py`

  * **Primary Phase-2 target**.
  * Contains the PyMC directional kinematics, camera projection, and observation likelihood.

Phase 2 will work entirely inside `pymc_model.py`, with optional small clarifications in `pymc_utils.py`.

---

## 3. Phase 2 Objectives (Clarity-First)

Phase 2 reorganizes the PyMC modeling code so that the following are obvious and discoverable:

1. **How 3D joint positions are built from directions and bone lengths.**
2. **How 3D positions are projected into 2D camera coordinates.**
3. **How the 2D observation likelihood is computed (Gaussian or mixture).**
4. **What tensors Stage 3 will consume.**

To accomplish this, Phase 2 introduces only:

* A small number of clearly-named **internal helper functions** within `pymc_model.py`.
* Optional pruning of unused options and arguments **when it improves clarity**.

No new top-level modules or directories will be introduced.

---

## 4. Phase 2 Deliverables

### 4.1 In `pymc_model.py`

#### 4.1.1 Single, well-structured public entrypoint

Maintain a single public model builder:

```python
def build_camera_observation_model(...):
    ...
```

This function should read like a high-level recipe:

1. Build root 3D position over time.
2. Build 3D joint positions from directions and lengths.
3. Project 3D joints to 2D keypoints via cameras.
4. Apply observation likelihood to obtain `log_obs_t` (or an equivalent scalar logp).

---

#### 4.1.2 Camera projection helper (already present)

Keep and clearly document the PyTensor projector:

```python
def project_points_pytensor(x_all, proj_param):
    """Project 3D joints x_all (T, K, 3) into 2D keypoints (C, T, K, 2).

    proj_param contains camera intrinsics/extrinsics.
    This is the PyTensor analogue of the Torch `camera.project_points`.
    """
    ...
```

This function remains the **canonical** 3D→2D projector on the PyMC side.

---

#### 4.1.3 Direction + kinematics helper

Introduce a helper that encapsulates the direction and kinematic logic:

```python
def build_directional_kinematics(
    parents,
    T,
    K,
    u_init,
    rho_init,
    sigma2_init,
    sigma_dir,
):
    """Build per-joint directions and 3D joint positions.

    Args:
        parents: array-like of length K giving the parent index for each joint.
                 parents[0] is the root and can be ignored here.
        T: int, number of time steps.
        K: int, number of joints (including root).
        u_init: (T, K, 3) or (T, K-1, 3) initial directions (global frame).
        rho_init: (K-1,) initial bone lengths for non-root joints.
        sigma2_init: (K-1,) initial bone length variances.
        sigma_dir: scalar direction prior scale.

    Returns:
        x_all: (T, K, 3) 3D joint positions in global frame.
        U:     (T, K, 3) unit direction vectors in global frame
               (root row may be unused or zeros).
    """
    ...
    return x_all, U
```

Key requirements:

* Use a single `raw_u` random variable where possible, e.g. shape `(T, K-1, 3)`.
* Normalize to unit length to obtain `U[:, 1:, :]` for non-root joints.
* Construct `x_all` by walking the kinematic tree from root according to `parents`.
* Maintain **global frame** directions; no parent-relative angles.
* Keep normalization numerically stable (add a small epsilon if needed).

This helper may be called only once, but it dramatically improves clarity: all "how do we parameterize skeleton directions and positions?" logic lives in one place.

---

#### 4.1.4 Optional observation likelihood helper

If the observation likelihood is compact and easy to read, it can remain inline.
If it is long or branching (e.g. Gaussian vs mixture), factor it into:

```python
def build_camera_likelihood(
    y_pred,
    y_observed,
    obs_sigma,
    image_size=None,
    inlier_prob_init=None,
    use_mixture=False,
):
    """Build observation likelihood.

    If use_mixture is False:
        Use simple Gaussian likelihood on 2D keypoints.

    If use_mixture is True:
        Use robust mixture (inlier + broad outlier) likelihood.
    """
    ...
    return log_obs_t
```

Expectations:

* `y_pred`: `(C, T, K, 2)` — projected 2D points.
* `y_observed`: same shape, with NaNs or masks for missing points.
* `obs_sigma`: scalar or per-joint scale parameter.
* Return:

  * Either a scalar logp (sum over time, joints, cameras), or
  * Preferably a per-timestep logp vector `log_obs_t` `(T,)` so Stage 3 can combine it with directional logp.

If mixture logic is not currently used anywhere, it may be removed or left as a clearly-marked optional branch. Do not keep complex branches that nobody uses.

---

#### 4.1.5 Add minimal, helpful shape comments

Within `pymc_model.py`, add short comments where major tensors are created:

* `x_root`: `(T, 3)`
* `raw_u`: `(T, K-1, 3)`
* `U`: `(T, K, 3)`
* `x_all`: `(T, K, 3)`
* `y_pred`: `(C, T, K, 2)`
* `log_obs_t`: `(T,)` or scalar

These comments should aid reading without adding heavy validation.

---

#### 4.1.6 Simplify or delete unused options when they obscure clarity

When reviewing `pymc_model.py`:

* If an argument to `build_camera_observation_model` is never used, remove it.
* If a code branch is never exercised (e.g., `use_mixture=True` is never set), either:

  * remove it, or
  * clearly mark it as experimental and keep it very small.
* Do **not** delete options just because they are currently unused if they clearly document a possible extension and do not clutter the main flow.

The goal is for someone reading the file to see *exactly what the current model does* without wading through dead or confusing code.

---

## 5. Non-Goals for Phase 2

Phase 2 explicitly does **not** include:

* Any use or modification of the collapsed HMM engine.
* Any change to Torch inference (`inference.py`).
* Any change to Torch model math (`model.py`).
* New directories, new packages, or sweeping architecture changes.
* Adding thick layers of input validation or type-checking.
* Performance optimization or benchmarking.

If a proposed change does not improve clarity of the **current** PyMC pipeline or the Stage-2 → Stage-3 interface, it is out of scope.

---

## 6. Detailed Refactor Plan (Step-by-Step)

### Step 1 — Add docstrings & shape comments in `pymc_model.py`

* Add concise docstrings to:

  * `project_points_pytensor`
  * `build_directional_kinematics`
  * `build_camera_likelihood` (if created)
  * `build_camera_observation_model`
* Add inline shape comments when major tensors are defined.

This step alone should already improve readability.

---

### Step 2 — Introduce `build_directional_kinematics`

1. Identify the section of `build_camera_observation_model` that:

   * Samples `raw_u` and normalizes to unit vectors.
   * Samples or defines bone lengths.
   * Builds `x_k` per joint and then `x_all`.

2. Move this logic into `build_directional_kinematics`, preserving:

   * Variable names where reasonable (`raw_u`, `U`, `x_all`).
   * Existing priors (Normal scales, initial values).

3. In `build_camera_observation_model`, replace the inlined logic with a single call:

   ```python
   x_all, U = build_directional_kinematics(
       parents=parents,
       T=T,
       K=len(parents),
       u_init=u_init,
       rho_init=rho_init,
       sigma2_init=sigma2_init,
       sigma_dir=sigma_dir,
   )
   ```

4. Ensure `U` has shape `(T, K, 3)` even if the root row is unused.

This isolates the skeleton/direction logic in one well-named helper.

---

### Step 3 — Keep or introduce `build_camera_likelihood` as needed

Inspect the current likelihood code in `build_camera_observation_model`:

* If it is short and simple (e.g. one Normal observed variable), leave it inline.
* If it has a mixture branch or complex control flow, extract it into `build_camera_likelihood`.

When extracted, ensure:

* `build_camera_likelihood` is small and focused.
* It returns either a scalar logp or `log_obs_t`.
* It is easy to see how the mixture (if any) works.

Then in `build_camera_observation_model`:

```python
log_obs_t = build_camera_likelihood(
    y_pred=y_pred,
    y_observed=y_observed,
    obs_sigma=obs_sigma,
    image_size=image_size,
    inlier_prob_init=inlier_prob_init,
    use_mixture=use_mixture,
)
```

---

### Step 4 — Clean up the signature of `build_camera_observation_model`

* Order arguments logically, for example:

  ```python
  def build_camera_observation_model(
      y_observed,
      parents,
      u_init,
      rho_init,
      sigma2_init,
      camera_proj,
      T,
      sigma_dir,
      use_mixture=False,
      image_size=None,
      inlier_prob_init=None,
  ):
      ...
  ```

* Remove arguments that are not used.

* Document:

  * expected shapes for all array parameters,
  * what the function returns (typically a `pm.Model`).

The goal is for the function signature and docstring to serve as a high-level summary of the model.

---

### Step 5 — Inline tiny helpers where appropriate

* Avoid creating helpers for truly one-line operations unless they significantly clarify intent.
* Prefer a few well-named helpers over many microscopic utilities.

This aligns with **clarity > simplicity > heavy engineering**.

---

### Step 6 — Confirm unchanged behavior

After the refactor:

1. Build a small synthetic test case (few joints, few frames, one or two cameras).
2. Run the **old** version of `build_camera_observation_model` (before refactor) and record:

   * `logp(model)` for a fixed parameter set,
   * posterior summaries for a short nutpie/NUTS run.
3. Run the **new** version and compare:

   * `logp` values should match up to floating-point noise.
   * Posterior summaries should be statistically indistinguishable.
4. Confirm variable names essential for downstream code (e.g. `y_pred`, `x_all`) are unchanged or updated consistently.

If these checks pass, Phase 2 modifications can be considered behaviorally neutral.

---

## 7. Example Final Structure of `pymc_model.py`

The final file will roughly follow this layout:

```python
# pymc_model.py

import pymc as pm
import pytensor.tensor as pt

############################################################
# 1. Camera projection (PyTensor analogue of Torch camera.py)
############################################################

def project_points_pytensor(x_all, proj_param):
    """Project 3D joints (T, K, 3) to 2D keypoints (C, T, K, 2)."""
    ...

############################################################
# 2. Direction + kinematics (clarity-first helper)
############################################################

def build_directional_kinematics(
    parents,
    T,
    K,
    u_init,
    rho_init,
    sigma2_init,
    sigma_dir,
):
    """Build directions U and joint positions x_all."""
    ...
    return x_all, U

############################################################
# 3. Optional observation likelihood helper
############################################################

def build_camera_likelihood(
    y_pred,
    y_observed,
    obs_sigma,
    image_size=None,
    inlier_prob_init=None,
    use_mixture=False,
):
    """Build observation likelihood and return log_obs_t or scalar logp."""
    ...
    return log_obs_t

############################################################
# 4. Main PyMC model (public API)
############################################################

def build_camera_observation_model(
    y_observed,
    parents,
    u_init,
    rho_init,
    sigma2_init,
    camera_proj,
    T,
    sigma_dir,
    use_mixture=False,
    image_size=None,
    inlier_prob_init=None,
):
    """Full PyMC observation model for multi-camera 3D tracking."""

    with pm.Model() as model:
        # Root dynamics
        x_root = pm.GaussianRandomWalk("x_root", ...)

        # Directions + kinematics
        x_all, U = build_directional_kinematics(
            parents, T, K=len(parents),
            u_init=u_init,
            rho_init=rho_init,
            sigma2_init=sigma2_init,
            sigma_dir=sigma_dir,
        )

        # Projection
        proj_param = pm.Data("camera_proj", camera_proj)
        y_pred = pm.Deterministic(
            "y_pred", project_points_pytensor(x_all, proj_param)
        )

        # Observation likelihood
        obs_sigma = pm.HalfNormal("obs_sigma", ...)
        log_obs_t = build_camera_likelihood(
            y_pred,
            y_observed,
            obs_sigma,
            image_size=image_size,
            inlier_prob_init=inlier_prob_init,
            use_mixture=use_mixture,
        )

    return model
```

This structure is linear, readable, and ready for Stage 3 to attach HMM priors over `U` using the Stage-1 HMM engine.

# 8. Stage 2 → Stage 3 Interface Requirements (Explicit)

To ensure Stage 3 can be implemented cleanly without refactoring Phase 2 code, Phase 2 must provide the following:

## Required Outputs

* `U`: `(T, K, 3)` — unit direction vectors in global frame.
* `x_all`: `(T, K, 3)` — 3D joint positions.
* `y_pred`: `(C, T, K, 2)` — projected 2D keypoints.
* `log_obs_t`: `(T,)` or scalar — observation log-likelihood.

Stage 3 will:

* Leave emission model (projection + likelihood) unchanged.
* Compute `log_dir_emit[t, s] = Σ_k κ[k,s] · dot(U[t,k], μ[k,s])`.
* Combine with `log_obs_t[t]` to form `logp_emit[t, s]`.
* Pass `logp_emit` to `collapsed_hmm_loglik`.

No Stage 3 components should require any modification to Phase 2 code once this interface is provided.

---

# 9. Notes on Validation and Testing

Following Stage 1 conventions:

* All validation (finite gradients, small synthetic tests, brute-force checks) should be in **notebooks**, not library code.
* Phase 2 code should:

  * Avoid unnecessary asserts.
  * Favor clear docstrings and shape comments.
  * Use `pm.Deterministic` when naming tensors improves debuggability.

---

# 10. Completion Criteria Summary

Phase 2 is complete when:

1. `pymc_model.py` uses `build_directional_kinematics` to produce `x_all` and `U`.
2. `project_points_pytensor` is the canonical projector and is cleanly documented.
3. Observation likelihood is clean, readable, and optionally factored into `build_camera_likelihood`.
4. Model-building code in `build_camera_observation_model` is linear and easy to read.
5. Stage-2 → Stage-3 interface tensors (`U`, `x_all`, `y_pred`, `log_obs_t`) are clearly defined and stable.
6. Tests show equivalence in behavior before/after the refactor.

Phase 2 produces a **clear, modular, easily extensible PyMC model** that Stage 3 can augment with canonical directions, κ parameters, and an HMM prior without any structural changes.
