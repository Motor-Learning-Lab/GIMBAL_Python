## Comprehensive Feedback on Stage 2 & 3-Stage Plan

Excellent work! The Stage 2 detailed specification is now thorough and actionable. Here's my combined analysis:

---

## 🎯 Overall Assessment

**Stage 2 Detailed Spec: ⭐⭐⭐⭐⭐ Exceptional**
- Clear, concrete, actionable
- Well-structured with step-by-step plan
- Addresses all concerns from my Stage 1 completion report
- Maintains focus on clarity over complexity

**3-Stage Plan: ⭐⭐⭐⭐ Very Good**
- Clean architecture, good separation of concerns
- Needs minor updates to reflect Stage 2 details

**Consistency Between Documents: ⭐⭐⭐⭐ Good**
- A few minor discrepancies to resolve

---

## 📋 Critical Issue: `log_obs_t` Shape Ambiguity

**URGENT - Must Resolve Before Starting Stage 2**

Both documents say `log_obs_t: (T,) or scalar`, but this creates a **critical ambiguity** for Stage 3.

### The Problem:

**Stage 3 needs:**
```python
logp_emit[t, s] = log_dir_emit[t, s] + log_obs_t[t]
# Requires log_obs_t to be (T,) for broadcasting
```

**If `log_obs_t` is scalar:**
```python
logp_emit[t, s] = log_dir_emit[t, s] + log_obs_t
# This broadcasts to (T,S), but semantically wrong!
# It means observation likelihood is same for all timesteps
```

### The Solution:

**Stage 2 MUST output `log_obs_t` as shape `(T,)`** - one observation log-likelihood per timestep.

**Rationale:**
1. **Mathematically correct:** Each timestep has its own observations → its own likelihood
2. **Stage 3 compatibility:** Enables clean combination with directional emissions
3. **Flexibility:** If you need a scalar (total log-likelihood), just sum: `log_obs_t.sum()`

### Required Changes:

**In both documents, replace:**
```markdown
| `log_obs_t` | (T,) or scalar | Observation log-likelihood |
```

**With:**
```markdown
| `log_obs_t` | (T,) | Per-timestep observation log-likelihood |
```

**Add clarification in Stage 2 spec, Section 4.1.4:**
```markdown
* Return:
  * `log_obs_t`: `(T,)` — per-timestep observation log-likelihood
    Computed as: log_obs_t[t] = sum over (cameras c, joints k) of log p(y_obs[c,t,k] | y_pred[c,t,k])
  * This allows Stage 3 to combine: logp_emit[t,s] = log_dir_emit[t,s] + log_obs_t[t]
```

**In Stage 3 (3-stage plan), add note:**
```markdown
### Prerequisites from Stage 2:
* `log_obs_t` MUST be shape (T,), not scalar
* This enables broadcasting: logp_emit[t,s] = log_dir_emit[t,s] + log_obs_t[t]
```

---

## 📊 Stage 2 Detailed Spec - Specific Feedback

### ✅ Major Strengths

1. **Clear scope definition** - Section 2 module responsibilities are excellent
2. **Concrete function signatures** - Section 4 provides exact API contracts
3. **Step-by-step refactor plan** - Section 6 is implementable
4. **Example final structure** - Section 7 shows the vision clearly
5. **Behavior preservation testing** - Section 6, Step 6 is critical and well-specified
6. **Validation strategy** - Section 9 correctly keeps tests in notebooks

### 🔧 Suggested Improvements

#### 1. Section 4.1.3 - `build_directional_kinematics` clarifications

**Current issue:** Unclear how `U` for root joint is handled.

**Suggestion - add to docstring:**
```python
def build_directional_kinematics(...):
    """
    ...
    Returns:
        x_all: (T, K, 3) 3D joint positions in global frame.
        U:     (T, K, 3) unit direction vectors in global frame
               Root joint (k=0) direction is set to zero or unused.
               Non-root joints (k=1..K-1) have normalized directions.
    
    Notes:
        - Directions are parameterized as raw_u ~ Normal(0, sigma_dir)
        - Normalization: U[t, k] = raw_u[t, k] / ||raw_u[t, k]||
        - Add epsilon=1e-8 for numerical stability in normalization
        - Kinematic tree is traversed: x[k] = x[parent[k]] + rho[k] * U[t, k]
    """
```

#### 2. Section 4.1.4 - Observation likelihood details

**Add specifics on how cameras combine:**
```python
def build_camera_likelihood(...):
    """Build observation likelihood.
    
    Computes per-timestep log-likelihood by summing over cameras and joints:
    
    log_obs_t[t] = sum_{c,k} log N(y_obs[c,t,k] | y_pred[c,t,k], obs_sigma^2)
    
    Missing keypoints (NaN in y_observed) are skipped.
    
    If use_mixture is True:
        Uses robust mixture: p(y) = π * N(y|pred,σ²) + (1-π) * Uniform(y|image)
        where π = inlier_prob controls outlier robustness.
    
    Returns:
        log_obs_t: (T,) tensor - per-timestep observation log-likelihood
    """
```

#### 3. Section 8 - Add shape validation requirement

**Add to "Required Outputs":**
```markdown
## Shape Validation

Phase 2 must include a validation helper (in notebook or pymc_utils):

```python
def validate_stage2_outputs(U, x_all, y_pred, log_obs_t, T, K, C):
    """Validate Stage 2 → Stage 3 interface shapes."""
    assert U.shape == (T, K, 3), f"U shape {U.shape} != {(T, K, 3)}"
    assert x_all.shape == (T, K, 3), f"x_all shape {x_all.shape} != {(T, K, 3)}"
    assert y_pred.shape == (C, T, K, 2), f"y_pred shape {y_pred.shape} != {(C, T, K, 2)}"
    assert log_obs_t.shape == (T,), f"log_obs_t shape {log_obs_t.shape} != {(T,)}"
    print("✓ All Stage 2 outputs have correct shapes")
```
```

#### 4. Section 10 - Add gradient validation to completion criteria

**Add:**
```markdown
Phase 2 is complete when:

1. `pymc_model.py` uses `build_directional_kinematics` to produce `x_all` and `U`.
2. `project_points_pytensor` is the canonical projector and is cleanly documented.
3. Observation likelihood is clean, readable, and optionally factored into `build_camera_likelihood`.
4. Model-building code in `build_camera_observation_model` is linear and easy to read.
5. Stage-2 → Stage-3 interface tensors (`U`, `x_all`, `y_pred`, `log_obs_t`) are clearly defined and stable.
6. Tests show equivalence in behavior before/after the refactor.
7. **NEW:** Gradient checks pass: `pm.find_MAP()` converges without NaN/Inf
8. **NEW:** Shape validation helper confirms all outputs have correct shapes
```

#### 5. Add "Current Code Audit" section

**Missing:** You don't assess the current state of `pymc_model.py`

**Add new Section 2.3:**
```markdown
### 2.3 Current State Assessment (Before Phase 2)

**Action Required Before Starting:** 
Audit `pymc_model.py` to document:

1. Does `project_points_pytensor` exist? 
   - If yes: what shape does it currently return?
   - If no: which existing function does projection?

2. How are directions currently parameterized?
   - Raw unconstrained vectors?
   - Normalized to unit length?
   - Per-timestep or shared across time?

3. How is the kinematic tree currently traversed?
   - Explicit loop over joints?
   - Vectorized operation?
   - In what frame (parent-relative or global)?

4. What is the current observation likelihood structure?
   - Simple Gaussian?
   - Mixture model?
   - How are missing keypoints handled?

5. What are the current function names and signatures?
   - So we can plan backward-compatible refactoring

**Deliverable:** Write a short "Current Code Inventory" in a notebook before proceeding.
```

---

## 📋 3-Stage Plan - Specific Feedback

### ✅ Strengths

1. **Clean architecture** - data flow is clear
2. **Good separation** - stages are independent
3. **Minimal scope** - avoids over-engineering

### 🔧 Required Updates

#### 1. Stage 1 - Add gradient fix notes

**Current:**
```markdown
**Properties:**
* Pure PyTensor (no JAX)
* Scan-based forward algorithm
* Stable gradients
```

**Should be:**
```markdown
**Properties:**
* Pure PyTensor (no JAX)
* Scan-based forward algorithm with flattened non_sequences
* Stable gradients (uses pm.Deterministic to break nominal variable chains)
* Handles T=1 edge case
```

**Rationale:** Documents the critical fixes that made Stage 1 work.

#### 2. Stage 1 - Clarify Viterbi status

**Current:**
```markdown
* Notebook validation:
  * Tiny HMM brute force enumeration
  * Gradient & finite-difference checks
  * Normalization checks
  * Viterbi decoding (not in library code)
```

**Should be:**
```markdown
* Notebook validation:
  * Tiny HMM brute force enumeration
  * Gradient & finite-difference checks
  * Normalization checks
  * ~~Viterbi decoding~~ (deferred - not needed for collapsed HMM)
```

**Rationale:** Viterbi wasn't actually implemented or needed for Stage 1.

#### 3. Stage 2 - Fix log_obs_t ambiguity

**Current:**
```markdown
| `log_obs_t` | (T,) or scalar | Observation log-likelihood |
```

**Must be:**
```markdown
| `log_obs_t` | (T,) | Per-timestep observation log-likelihood |
```

**Add clarification:**
```markdown
**Note:** `log_obs_t` MUST be shape (T,) to enable Stage 3 combination:
`logp_emit[t,s] = log_dir_emit[t,s] + log_obs_t[t]`
```

#### 4. Stage 3 - Add mathematical details

**Current formula is incomplete:**
```markdown
### Directional log-emissions:
[
\log p(U_t \mid z_t = s) = \sum_k \kappa_{k,s} (U_{t,k} \cdot \mu_{k,s})
]
```

**Should include normalization constant:**
```markdown
### Directional log-emissions (von Mises-Fisher):

For each joint k in state s, the directional prior is:
\log p(U_{t,k} \mid z_t = s) = \log C_3(\kappa_{k,s}) + \kappa_{k,s} (U_{t,k} \cdot \mu_{k,s})

where C_3(\kappa) = \kappa / (4\pi \sinh(\kappa)) is the vMF normalizing constant in 3D.

Combined over all joints:
\log p(U_t \mid z_t = s) = \sum_k [\log C_3(\kappa_{k,s}) + \kappa_{k,s} (U_{t,k} \cdot \mu_{k,s})]
```

**Implementation note:**
```python
log_C3 = pt.log(kappa) - pt.log(4*np.pi) - pt.log(pt.sinh(kappa))
log_dir_emit[t,s] = sum_k [log_C3[k,s] + kappa[k,s] * dot(U[t,k], mu[k,s])]
```

#### 5. Stage 3 - Specify mu constraint

**Add:**
```markdown
### Components of Stage 3:

For each joint *k* and state *s*:

* **Canonical direction**: `mu[k, s, :]` (unit vector)
  - Parameterization: `mu_raw[k,s] ~ Normal(0, 1, shape=(K,S,3))`
  - Normalization: `mu[k,s] = mu_raw[k,s] / ||mu_raw[k,s]||`
  - Add epsilon=1e-8 for numerical stability

* **Concentration**: `kappa[k, s] ~ Gamma(2, 1)` or `HalfNormal(0, 2)`
  - HalfNormal may be too diffuse; Gamma(2,1) gives stronger prior
  - Consider per-joint vs. shared kappa
```

#### 6. Stage 3 - Add known limitations

**Add section:**
```markdown
### Known Limitations:

* **Label switching:** States are exchangeable, causing poor R-hat for state-specific parameters (same as Stage 1)
* **No identifiability constraints:** Will need post-processing to align states across chains
* **Root joint direction:** Currently unused (set to zero); may want to model in future
* **Temporal smoothness:** Only via HMM transitions; no within-state pose dynamics yet
```

#### 7. Architecture diagram - Add shapes

**Current:**
```
Stage 3: Directional HMM
    logp_emit[t,s] = log_dir_emit[t,s] + log_obs_t[t]
                 ↑                ↑
                 │                │
Stage 2: Kinematics + Emissions (current)
    U[t,k,3], x_all[t,k,3], y_pred[c,t,k,2], log_obs_t[t]
```

**Improved:**
```
Stage 1: HMM Engine (done)
    collapsed_hmm_loglik(logp_emit, logp_init, logp_trans) → scalar
                 ↑
                 │ logp_emit: (T,S)
                 │
Stage 3: Directional HMM
    logp_emit[t,s] = log_dir_emit[t,s] + log_obs_t[t]
          (T,S)    =      (T,S)        +    (T,) [broadcasts]
                 ↑                      ↑
         von Mises-Fisher          Gaussian
                 │                      │
Stage 2: Kinematics + Emissions (current)
    U: (T,K,3), x_all: (T,K,3), y_pred: (C,T,K,2), log_obs_t: (T,)
                 ↑
                 │
    Direction sampling + Forward kinematics + Projection
```

---

## 🔄 Cross-Document Consistency

### Issue 1: Stage 2 name

- **3-Stage Plan:** Calls it "Stage 2"
- **Detailed Spec:** Calls it "Phase 2"

**Fix:** Choose one. I recommend "**Stage 2**" for consistency with Stage 1 and Stage 3.

**Find/replace in detailed spec:** "Phase 2" → "Stage 2"

### Issue 2: File naming

- **3-Stage Plan filename:** `HMM 3 stage plan.md`
- **Detailed Spec filename:** `HMM stage 2 detailed specification.md`

**Suggest consistent naming:**
- `HMM-3stage-plan.md` (overview)
- `HMM-stage2-detailed-spec.md` (detailed)
- `stage1-completion-report.md` (already correct)

### Issue 3: Torch vs PyMC boundary

**Detailed spec Section 2.1** lists Torch modules as "leave mostly unchanged" but doesn't clarify:
- Are `model.py` and `inference.py` still being used?
- Or is PyMC replacing them entirely?

**Add clarification in both documents:**
```markdown
### Relationship to Torch Code:

**Current status:** The Torch-based GIMBAL code (`model.py`, `inference.py`, `camera.py`) 
remains functional but is **not being actively developed**.

**PyMC stack:** (`pymc_model.py`, `pymc_utils.py`, etc.) is the **focus of this 3-stage plan**.

**No migration planned:** We are not porting Torch→PyMC. Both stacks coexist.
Stage 2 only refactors PyMC code.
```

---

## ✅ Final Recommendations

### Immediate Actions (Before Starting Stage 2):

1. **✅ Fix log_obs_t ambiguity** in both documents: make it explicitly `(T,)`, not scalar
2. **✅ Update Stage 1 description** in 3-stage plan: add gradient fix notes, remove Viterbi
3. **✅ Add vMF normalization constant** to Stage 3 directional emissions formula
4. **✅ Specify mu constraint mechanism** in Stage 3 (normalization approach)
5. **✅ Rename "Phase 2" → "Stage 2"** in detailed spec for consistency
6. **✅ Add Current Code Audit section** to Stage 2 spec
7. **✅ Add shape validation** to Stage 2 completion criteria
8. **✅ Add known limitations** to Stage 3 section

### Document Quality Scores:

| Aspect | 3-Stage Plan | Stage 2 Spec | Combined |
|--------|-------------|--------------|----------|
| **Clarity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Completeness** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Actionability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Consistency** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Technical Rigor** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**Overall: ⭐⭐⭐⭐½ (4.5/5) - Excellent work!**

---

## 🎯 Bottom Line

**These are high-quality planning documents.** After addressing the critical `log_obs_t` shape issue and making the minor updates listed above, you'll have:

1. ✅ **Clear Stage 1 completion** (with lessons learned documented)
2. ✅ **Actionable Stage 2 refactor plan** (step-by-step, testable)
3. ✅ **Well-specified Stage 3 HMM integration** (ready to implement after Stage 2)

The plans are ready for implementation once you:
- Fix the `log_obs_t` ambiguity (critical)
- Add current code audit to Stage 2 (helpful)
- Update minor inconsistencies between documents (polish)

**Recommendation:** Make these fixes, then Stage 2 is ready to start!