# 📘 Stage 2 Specification: Direction & Emission Refactor for PyMC Model
*A clarity-first restructuring of `pymc_model.py` and supporting modules, sitting between Stage 1 (HMM engine) and Stage 3 (HMM over directions)*

---

This specification describes the goals, structure, responsibilities, and concrete refactoring steps for **Stage 2** of the project. It assumes:

- **Stage 1 is complete**: the collapsed HMM engine lives in `hmm_pytensor.py` and is fully validated.
- **Stage 3 will build on Stage 2**: Stage 3 will introduce canonical directions, concentration parameters, and the collapsed HMM prior over directions, but **will not modify the emission pipeline**.

Guiding principle:

> **Clarity > Simplicity > Heavy Engineering**

The aim is not to redesign the entire codebase but to make the PyMC/PyTensor model for **directions**, **3D kinematics**, **camera projection**, and **2D likelihood** easy to understand and easy to extend.

Stage 2 has two main jobs:

1. Make the current PyMC model readable and structurally clean.
2. Define a clear interface for Stage 3 to consume.

---

## 1. Interface Contract Between Stage 2 and Stage 3

At the end of Stage 2, the PyMC model must provide the following well-defined tensors:

- `U`: `(T, K, 3)` — unit direction vectors for non-root joints in the **global frame**.
- `x_all`: `(T, K, 3)` — 3D joint positions in global coordinates.
- `y_pred`: `(C, T, K, 2)` — projected 2D keypoints per camera.
- `log_obs_t`: `(T,)` — **per-timestep** observation log-likelihood.

Stage 3 will:

- Leave the entire **emission pipeline** (`x_all → y_pred → log_obs_t`) unchanged.
- Add canonical directions `mu[k, s]` and concentrations `kappa[k, s]`.
- Compute directional log-emissions from `U` and combine with `log_obs_t` to form `logp_emit[t, s]` for the Stage-1 HMM engine.

Stage 2 **must not** modify:

- `hmm_pytensor.py` (collapsed HMM engine)
- `hmm_pymc_utils.py` (Gaussian HMM demo)

---

## 2. Current Module Responsibilities (Baseline)

Stage 2 preserves the overall file layout and clarifies responsibilities.

### 2.1 Torch-side modules (leave mostly unchanged)

These belong to the original GIMBAL/Torch implementation and are **not** refactored in Stage 2:

- `camera.py`
- `model.py`
- `fit_params.py`
- `inference.py`

Only minimal comment or docstring cleanup is allowed here.

---

### 2.2 PyMC / PyTensor modules (Stage 2 focus)

- `hmm_pytensor.py` — **Stage 1, do not modify**
- `hmm_pymc_utils.py` — Gaussian HMM example (small doc updates allowed)
- `pymc_distributions.py` — custom PyMC distributions
- `pymc_utils.py` — helper utilities
- `pymc_model.py` — **main Stage-2 target**

---

## 3. Stage 2 Objectives (Clarity-First)

The PyMC modeling code should make it immediately clear:

1. How 3D joint positions are built from directions and bone lengths.
2. How 3D joints are projected into 2D per camera.
3. How 2D likelihoods are computed.
4. What tensors Stage 3 will consume.

To accomplish this:

- Introduce a small number of clearly named **internal helpers**.
- Remove unused options **only when doing so clarifies the flow**.
- Do **not** introduce new directories or deep reorganizations.

---

## 3.1 Current Code Audit (Required Before Refactor)

Before editing any code, create a short inventory (preferably in a notebook):

- The current projection function and its return shape.
- Current direction parameterization (`raw_u` shapes, normalization).
- Current kinematic tree traversal logic.
- Current likelihood function and mixture branch (if present).
- Variable names that Stage 3 relies on (`x_all`, `U`, `y_pred`).

This prevents accidental breakage and ensures Stage 2 faithfully restructures the **actual** model.

---

## 4. Stage 2 Deliverables

### 4.1 In `pymc_model.py`

#### 4.1.1 Single public entrypoint

Maintain:

```python
def build_camera_observation_model(...):
    ...
```

It must read like a simple recipe:

1. Build root trajectory
2. Build 3D joints via directions + lengths
3. Project joints to 2D
4. Compute per-timestep `log_obs_t`

---

#### 4.1.2 Camera projection helper

Keep and document the canonical PyTensor projector:

```python
def project_points_pytensor(x_all, proj_param):
    """Project 3D joints (T, K, 3) → 2D keypoints (C, T, K, 2)."""
```

---

#### 4.1.3 Direction + kinematics helper

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
    """Return:
    x_all: (T, K, 3) — 3D joints
    U:     (T, K, 3) — unit directions (root row = zeros or unused)
    """
```

Requirements:

- Use a single `raw_u` of shape `(T, K-1, 3)` where possible.
- Normalize with small epsilon for numerical stability.
- Define:
  ```
  x[k] = x[parent[k]] + rho[k] * U[k]
  ```
- Always provide `U` with shape `(T, K, 3)`.

---

#### 4.1.4 Optional observation likelihood helper

Create or retain a helper **only** if the likelihood is complex:

```python
def build_camera_likelihood(...):
    ...
    return log_obs_t   # (T,)
```

**Stage-2 requirement: `log_obs_t` must be `(T,)` always.**

If mixture logic is unused, prune or isolate it.

---

#### 4.1.5 Shape comments

Add comments:

- `x_root`: `(T, 3)`
- `raw_u`: `(T, K-1, 3)`
- `U`: `(T, K, 3)`
- `x_all`: `(T, K, 3)`
- `y_pred`: `(C, T, K, 2)`
- `log_obs_t`: `(T,)`

Remove references to scalar `log_obs`.

---

#### 4.1.6 Delete/simplify unused arguments

- Remove arguments not referenced in the body.
- Remove code paths not exercised.
- Keep clean docstrings.

---

## 5. Non-Goals for Stage 2

Stage 2 does **not** include:

- Any HMM logic.
- Any change to Torch model or inference.
- New directory structure.
- Heavy validation or type-checking.
- Performance optimization.

---

## 6. Detailed Refactor Plan

Steps 1–6 unchanged except with corrected assumptions that:

- `log_obs_t` is always `(T,)`
- `Phase 2` renamed to `Stage 2`

(Full step descriptions retained from original.)

---

## 7. Expected Structure of `pymc_model.py`

*(Same as original document, unchanged except for clarifications above.)*

---

## 8. Stage 2 → Stage 3 Interface (Explicit)

Stage 2 must output:

- `U`: `(T, K, 3)`
- `x_all`: `(T, K, 3)`
- `y_pred`: `(C, T, K, 2)`
- `log_obs_t`: `(T,)`

Stage 3 will then compute:

```
logp_emit[t,s] = log_dir_emit[t,s] + log_obs_t[t]
```

No Stage-2 code is modified in Stage 3.

---

## 9. Validation and Testing

- Validation lives in notebooks.
- Use `pm.Deterministic` for debugging clarity.
- Confirm gradients and log-likelihood consistency before/after refactor.

---

## 10. Completion Criteria Summary

*(Same as original, with updated `log_obs_t` shape and renamed stage.)*

Stage 2 is complete when:

- `x_all`, `U`, `y_pred`, and `log_obs_t` are correctly produced.
- Behavior matches pre-refactor version.
