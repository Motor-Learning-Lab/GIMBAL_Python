# Stage 1 Completion Report: Collapsed HMM Engine

**Date:** November 26, 2025  
**Status:** ✅ COMPLETE

---

## Executive Summary

Stage 1 of the GIMBAL HMM integration has been successfully completed. We have implemented a fully functional collapsed HMM engine in PyTensor/PyMC that:
- Computes the marginal log-likelihood using the forward algorithm in log-space
- Successfully compiles and samples with both nutpie (fast) and PyMC NUTS (fallback)
- Passes all validation tests including brute-force verification, edge cases, and gradient correctness
- Is completely independent of cameras, skeletons, and kinematics (as specified)

---

## What Was Implemented

### 1. Core Forward Algorithm (`gimbal/hmm_pytensor.py`)

**Key Function:** `forward_log_prob_single(logp_emit, logp_init, logp_trans)`

**Implementation Details:**
- Uses PyTensor's `scan` operation for the recursive forward pass
- Computes in log-space for numerical stability using `pt.logsumexp`
- Handles T=1 edge case automatically with `pt.switch`
- Input shapes: `logp_emit (T,S)`, `logp_init (S,)`, `logp_trans (S,S)`
- Returns scalar: collapsed log-likelihood `log p(y_{0:T-1} | π, A, θ)`

**Critical Implementation Details:**
1. **Scan parameter order:** Step function signature is `step(logp_emit_t, alpha_prev, logp_trans_flat)` matching scan's convention: sequences → outputs_info → non_sequences
2. **Flattened transition matrix:** `logp_trans` is flattened before passing as `non_sequences`, then reshaped inside the step function. This was necessary to avoid PyTensor gradient computation issues.
3. **Non-sequences for gradients:** `logp_trans_flat` is passed explicitly as `non_sequences=[logp_trans_flat]` rather than captured from closure, enabling proper gradient tracking through the scan operation.

**Algorithm:**
```
1. Initialize: α₀[s] = log π[s] + log p(y₀ | s)
2. Recursion: α_t[j] = log p(y_t | j) + log Σᵢ exp(α_{t-1}[i] + log A[i,j])
3. Termination: log p(y) = log Σⱼ exp(α_T[j])
```

### 2. PyMC Model Builder (`gimbal/hmm_pymc_utils.py`)

**Key Function:** `build_gaussian_hmm_model(y, S)`

**Model Structure:**
- **Priors:**
  - `init_logits ~ Normal(0, 1)` shape (S,) - unnormalized initial state logits
  - `trans_logits ~ Normal(0, 1)` shape (S,S) - unnormalized transition logits
  - `mu ~ Normal(0, 5)` shape (S,) - state-specific emission means
  - `sigma ~ Exponential(1)` - shared observation noise (scalar)

- **Transformations:**
  - `logp_init = init_logits - logsumexp(init_logits)` - normalized log initial probabilities
  - `logp_trans = trans_logits - logsumexp(trans_logits, axis=1)` - normalized log transition probabilities (rows sum to 1 in probability space)
  - All intermediate variables wrapped in `pm.Deterministic` to break nominal variable chains for scan gradient compatibility

- **Likelihood:**
  - `logp_emit[t,s] = logp(Normal(mu[s], sigma), y[t])` - emission log-probabilities (T,S)
  - `hmm_loglik = collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)` - marginal likelihood via forward algorithm
  - `pm.Potential("hmm_loglik", hmm_loglik)` - add to model log-probability

**Critical Fix for Scan Gradients:**
```python
# Wrap in Deterministic to break nominal variable chain for scan gradients
logp_emit = pm.Deterministic("logp_emit", logp_emit_raw)
logp_init_det = pm.Deterministic("logp_init", logp_init)
logp_trans_det = pm.Deterministic("logp_trans", logp_trans)
```
This was essential to prevent `'NominalVariable' object has no attribute 'shape'` errors during gradient computation.

### 3. Data Simulator (`gimbal/hmm_pymc_utils.py`)

**Key Function:** `simulate_gaussian_hmm(T, S, mu_true, sigma_true, pi_true, A_true)`

**Features:**
- Generates synthetic 1D Gaussian HMM sequences
- Validates inputs (π sums to 1, A rows sum to 1)
- Returns both observations `y` and true hidden states `z`
- Used for validation and testing

### 4. Comprehensive Validation Suite (`test_hmm_stage1.py`)

**All Tests PASSING ✅**

1. **Brute-Force Verification** (`test_tiny_hmm_brute_force`)
   - S=2 states, T=3 timesteps
   - Enumerates all 8 possible state sequences
   - Compares sum of path probabilities to forward algorithm result
   - **Result:** Match within 1e-10 (exact agreement)

2. **T=1 Edge Case** (`test_t1_edge_case`)
   - Verifies single-timestep behavior
   - Tests that `pt.switch` correctly handles T=1
   - **Result:** PASSED

3. **Gradient Correctness** (`test_gradient_correctness`)
   - Finite difference validation (ε=1e-5)
   - Tests gradients w.r.t. `init_logits` and `trans_logits`
   - Tolerance: 1e-4 (0.01%)
   - **Result:** PASSED (max relative error ~1e-6)

4. **Normalization Checks** (`test_normalization_checks`)
   - Verifies probability distributions sum to 1
   - Checks initial distribution and transition matrix rows
   - **Result:** PASSED (errors < machine epsilon)

### 5. Demo Notebook (`notebook/hmm_demo_gaussians.ipynb`)

**Structure:** 7 sections demonstrating complete workflow

1. Simulate synthetic 3-state Gaussian HMM (T=100, S=3)
2. Build PyMC model
3. Sample with nutpie (fast numba-compiled sampler)
4. Compute diagnostics (ESS, R-hat)
5. Evaluate parameter recovery
6. Generate trace plots
7. Plot posterior distributions with true values

**Sampling Results:**
- ✅ **nutpie compilation:** SUCCESS (no more gradient errors!)
- ✅ **Sampling:** 4 chains × 1400 draws (400 tuning + 1000 sampling)
- ✅ **Divergences:** 0 across all chains
- ⚠️ **Convergence for mu:** Poor (R-hat=1.73-2.41, ESS=5-6) - **EXPECTED due to label switching**
- ✅ **Convergence for sigma:** Perfect (R-hat=1.00, ESS=7827)

---

## Technical Challenges Overcome

### Challenge 1: Scan Gradient Computation Error

**Problem:** `AttributeError: 'NominalVariable' object has no attribute 'shape'` in `scan.op.py:2912`

**Root Cause:** PyTensor's scan `L_op` (gradient method) tried to access `.shape` on variables derived from PyMC random variables, which use `NominalVariable` types internally that don't expose `.shape` in the expected format.

**Solution (Two-Part Fix):**
1. **Flatten transition matrix:** Pass `logp_trans.flatten()` as `non_sequences`, reshape inside step function
2. **Break nominal variable chains:** Wrap `logp_emit`, `logp_init`, `logp_trans` in `pm.Deterministic()` before passing to scan

This combination successfully resolved the gradient computation issue, enabling both nutpie and PyMC NUTS sampling.

### Challenge 2: Scan Parameter Order

**Problem:** Initial implementation had wrong parameter order in step function

**Solution:** Followed scan convention strictly:
- Sequences first: `logp_emit_t`
- Outputs from previous iteration: `alpha_prev`
- Non-sequences last: `logp_trans_flat`

### Challenge 3: Label Switching in HMM

**Problem:** Poor convergence diagnostics for `mu` parameters (high R-hat, low ESS)

**Root Cause:** This is NOT a bug! HMM states are exchangeable - permuting state labels (and corresponding parameters) gives identical likelihood. Different chains explore different labelings, causing apparent non-convergence.

**Evidence:**
- Sigma (shared parameter) converges perfectly: R-hat=1.00, ESS=7827
- Mu parameters show label switching: R-hat=1.73-2.41, ESS=5-6
- Zero divergences indicate sampler is working correctly
- This is a well-known issue in HMM inference

**Status:** Expected behavior, will be addressed in Stage 3 with identifiability constraints

---

## Validation of Completion Criteria

Per the Stage 1 specification, the following criteria have been met:

✅ **Implementation Complete:**
- Forward algorithm in PyTensor with scan
- Gaussian emission model
- PyMC integration with Potential
- Validation test suite

✅ **Code Quality:**
- All unit tests passing
- Comprehensive docstrings
- Type hints on function signatures
- Clean separation from camera/skeleton/kinematics code

✅ **Numerical Verification:**
- Brute-force test matches to machine precision
- Gradient correctness validated with finite differences
- T=1 edge case handled correctly
- Probability normalizations verified

✅ **Sampling Success:**
- Model compiles without errors (both nutpie and PyMC)
- Samples without divergences
- Sigma parameter shows excellent convergence (ESS > 7000)
- Label switching for mu is expected and documented

✅ **Documentation:**
- Demo notebook with full workflow
- Comprehensive test suite
- Inline code documentation
- This completion report

---

## Stage 1 Deliverables

### Files Created/Modified:

1. **`gimbal/hmm_pytensor.py`** (NEW)
   - `forward_log_prob_single()` - core forward algorithm
   - `collapsed_hmm_loglik()` - wrapper for readability

2. **`gimbal/hmm_pymc_utils.py`** (NEW)
   - `build_gaussian_hmm_model()` - PyMC model constructor
   - `simulate_gaussian_hmm()` - data generator for testing

3. **`test_hmm_stage1.py`** (NEW)
   - Comprehensive validation suite (4 tests, all passing)

4. **`notebook/hmm_demo_gaussians.ipynb`** (NEW)
   - Interactive demonstration of Stage 1 HMM
   - Successfully samples with nutpie
   - Visualizes results and diagnostics

5. **`plans/stage1-completion-report.md`** (THIS FILE)
   - Complete documentation of Stage 1 work

---

## Known Limitations & Future Work

### Current Limitations:

1. **Label Switching:** State identifiability issue causes poor convergence diagnostics for state-specific parameters. This is mathematically expected and not a bug.

2. **Single Emission Type:** Only 1D Gaussian emissions implemented. This is intentional for Stage 1 - more complex emissions come in Stage 2.

3. **No State Recovery:** Current implementation marginalizes out states (collapsed HMM). Viterbi decoding or forward-backward for state inference not yet implemented.

4. **Scan Performance:** Scan adds overhead compared to vectorized operations, but enables variable-length sequences and gradient computation.

### Not Addressed (By Design):

- Camera geometry integration (Stage 2)
- Skeleton/joint structure (Stage 2)
- Kinematic constraints (Stage 2)
- 3D pose estimation (Stage 2)
- Identifiability constraints for label switching (Stage 3)
- State inference/decoding (Stage 3)

---

## Recommendations for Stage 2

### Stage 2 Overview (From Original Plan):

Stage 2 should integrate the collapsed HMM engine with GIMBAL's camera-skeleton infrastructure to enable multi-view 3D pose tracking.

### Key Components Needed:

#### 1. **Camera-Aware Emission Model**

**Current:** Simple 1D Gaussian emissions `p(y_t | z_t, θ)`

**Needed for Stage 2:**
- Multi-camera observations: `y_t = {y_t^(c)}` for cameras c=1..C
- 3D joint positions: `x_t ∈ ℝ^(J×3)` (J joints, 3D coordinates)
- Camera projections: `p(y_t^(c) | x_t, camera_c)`
- Conditional independence across cameras: `p(y_t | x_t) = ∏_c p(y_t^(c) | x_t, camera_c)`

**Design Questions:**
- How to represent joint positions? (single vector vs. structured)
- What emission distribution? (Gaussian, Student-t, mixture?)
- How to handle occlusions/missing data?
- How to integrate existing GIMBAL camera code?

#### 2. **Skeleton Structure Integration**

**Current:** No skeleton structure, just abstract states

**Needed for Stage 2:**
- Joint hierarchy (parent-child relationships)
- Bone lengths (potentially as parameters)
- Joint angle representations (Euler angles, rotation matrices, quaternions?)
- Forward kinematics: joint angles → 3D positions

**Design Questions:**
- How to parameterize state space? (joint angles vs. positions)
- Fixed vs. learned bone lengths?
- How to incorporate existing skeleton definitions?
- Kinematic constraints (joint limits, bone length constraints)?

#### 3. **State Space Design for 3D Pose**

**Current:** Discrete states `z_t ∈ {1..S}` with Gaussian emissions

**Options for Stage 2:**

**Option A: Discrete Pose States**
- Each state s corresponds to a "canonical" 3D pose
- State-specific joint positions: `μ_s ∈ ℝ^(J×3)`
- Emission: `p(y_t | z_t=s) = ∏_c p(y_t^(c) | π(μ_s), camera_c)` where π is projection
- **Pros:** Maintains collapsed HMM structure, fast forward algorithm
- **Cons:** Limited pose variability, may need many states

**Option B: Continuous Latent Poses within States**
- Each state s defines a distribution over poses: `p(x_t | z_t=s)`
- Emission: `p(y_t | x_t, z_t=s)` 
- Need to marginalize: `p(y_t | z_t=s) = ∫ p(y_t | x_t) p(x_t | z_t=s) dx_t`
- **Pros:** More flexible, captures pose variability
- **Cons:** Intractable integral, need approximations (Laplace, quadrature, variational)

**Option C: Hybrid Approach**
- States represent activity/behavior modes (walking, reaching, etc.)
- Within each state, pose follows kinematic prior: `p(x_t | z_t=s, x_{t-1})`
- Smooth pose dynamics within states
- **Pros:** Interpretable states, realistic dynamics
- **Cons:** More complex, need pose dynamics model

**Recommendation:** Start with Option A (discrete pose states) to maintain Stage 1's collapsed HMM structure, then explore Option C if more flexibility is needed.

#### 4. **Camera Model Integration**

**Existing GIMBAL Code to Leverage:**
- `gimbal/camera.py` - camera intrinsics/extrinsics
- Projection functions: 3D → 2D
- Distortion models

**Integration Points:**
- Use existing camera objects in emission computation
- Project state-specific 3D poses to 2D keypoints
- Compute likelihood of observed vs. projected keypoints

**Design Questions:**
- How to pass camera parameters to PyMC model?
- Fixed vs. estimated camera parameters?
- How to handle camera calibration uncertainty?

#### 5. **Emission Likelihood Computation**

**Current:** `pm.logp(pm.Normal.dist(mu, sigma), y)` - simple univariate

**Needed for Stage 2:**
```python
def compute_emission_logp(x_3d, y_obs, cameras, sigma):
    """
    Compute log p(y_obs | x_3d) for multi-view observations.
    
    x_3d: (J, 3) - 3D joint positions
    y_obs: list of (K, 2) - observed 2D keypoints per camera
    cameras: list of camera objects
    sigma: observation noise
    """
    logp_total = 0
    for c, (y_c, camera) in enumerate(zip(y_obs, cameras)):
        # Project 3D pose to 2D using camera
        y_pred_c = camera.project(x_3d)  # (J, 2)
        
        # Compute likelihood (Gaussian noise on 2D coords)
        # Handle missing/occluded joints
        logp_c = compute_2d_likelihood(y_c, y_pred_c, sigma)
        logp_total += logp_c
    
    return logp_total
```

**Design Challenges:**
- Missing/occluded joints (masking, mixture with uniform background)
- Outliers/misdetections (robust likelihood, e.g., Student-t)
- Vectorization for multiple states/timesteps
- Gradients through projection (PyTensor implementation needed)

#### 6. **Modified Forward Algorithm**

**Current:** Operates on pre-computed `logp_emit` matrix (T, S)

**Needed for Stage 2:**
- On-the-fly emission computation during forward pass
- Potentially: `logp_emit` function of state-specific parameters
- Maintain differentiability for PyMC

**Two Approaches:**

**Approach A: Pre-compute Emission Matrix**
```python
# For each state s, compute emissions for all timesteps
logp_emit = pt.zeros((T, S))
for s in range(S):
    x_3d_s = state_poses[s]  # (J, 3)
    for t in range(T):
        logp_emit[t, s] = compute_emission_logp(x_3d_s, y_obs[t], cameras, sigma)
```
- **Pros:** Reuses Stage 1 forward algorithm unchanged
- **Cons:** Large memory for big T or S, rigid structure

**Approach B: Dynamic Emission Computation**
Modify forward algorithm to compute emissions on-demand within scan loop.
- **Pros:** Memory efficient, more flexible
- **Cons:** More complex scan function, potential performance issues

**Recommendation:** Start with Approach A to minimize changes to tested forward algorithm.

#### 7. **Testing & Validation Strategy**

**Unit Tests:**
- Camera projection functions (3D → 2D)
- Emission likelihood computation (known pose, known observations)
- Forward algorithm with camera emissions (synthetic data)
- Gradient checks for camera parameters

**Integration Tests:**
- End-to-end: synthetic 3D poses → multi-view 2D obs → HMM inference
- Parameter recovery: known poses/transitions → infer from projections
- Multi-camera consistency checks

**Validation Data:**
- Synthetic: Known 3D poses, project to 2D with known cameras
- Simple motions: Single joint movements, controlled trajectories
- Mocap data (if available): Ground truth 3D poses + cameras

#### 8. **Performance Considerations**

**Potential Bottlenecks:**
- Emission computation (T × S × C × J projections)
- Forward algorithm with large S (many pose states)
- Gradient computation through projections
- Scan overhead for long sequences

**Optimization Strategies:**
- Vectorize projection operations
- Cache repeated computations
- Use nutpie for fast sampling (already working!)
- Consider JAX backend for auto-parallelization
- Profile before optimizing

---

## Open Questions for Stage 2 Planning

### 1. State Space Design
- **Q:** How many discrete states S? Few (behavior modes) vs. many (pose dictionary)?
- **Q:** How to initialize state-specific poses? K-means on data? Manual specification?
- **Q:** Should bone lengths be state-specific or shared?

### 2. Emission Model
- **Q:** Gaussian vs. Student-t vs. mixture for observation noise?
- **Q:** Isotropic vs. per-joint vs. per-coordinate noise variance?
- **Q:** How to handle missing keypoints? (marginalize, impute, robust likelihood)
- **Q:** Detection confidence scores - incorporate or ignore?

### 3. Priors & Regularization
- **Q:** Priors on 3D poses? (kinematic constraints, anatomical limits)
- **Q:** Temporal smoothness? (already have transition matrix, but what about pose smoothness within states?)
- **Q:** Regularization for identifiability? (reference frames, anchor joints)

### 4. Camera Integration
- **Q:** Use existing `gimbal.camera` module as-is or modify?
- **Q:** Fixed cameras or estimate extrinsics?
- **Q:** How to pass camera data to PyMC? (shared variables, constants, parameters)

### 5. Computational Strategy
- **Q:** Pre-compute emissions vs. dynamic computation?
- **Q:** How to handle large T (hundreds of frames)? (minibatching, hierarchical)
- **Q:** Target inference speed? (real-time, near-real-time, offline)

### 6. Validation & Metrics
- **Q:** What defines "success" for Stage 2? (reprojection error, 3D pose error, tracking consistency)
- **Q:** What datasets/scenarios to validate on?
- **Q:** How to evaluate with label switching still present?

---

## Suggested Next Steps

### Immediate (Before Stage 2 Implementation):

1. **Review Stage 2 Plan Specification**
   - Revisit original 3-stage plan document
   - Identify any constraints or requirements not addressed above
   - Clarify scope: minimal viable vs. full-featured

2. **Assess Existing GIMBAL Code**
   - Audit `gimbal/camera.py` - what's available?
   - Check skeleton/joint definitions
   - Review any existing projection/forward kinematics code
   - Identify reusable components

3. **Make Design Decisions** (answer open questions above)
   - State space: discrete poses, how many, how defined?
   - Emission model: likelihood function, noise model
   - Camera handling: fixed vs. estimated, data flow
   - Computational approach: pre-compute vs. dynamic

4. **Create Synthetic Test Case**
   - Simple skeleton (e.g., 3 joints: root, mid, end)
   - 2-3 cameras with known positions
   - 2-3 discrete poses (e.g., arm up, arm down, arm forward)
   - Generate synthetic 2D observations
   - Use as development/validation target

### Stage 2 Implementation Phases:

**Phase 1: Camera-Aware Emissions (No Skeleton Yet)**
- Implement 3D point → 2D projection in PyTensor
- Compute emission likelihood for known 3D positions
- Test: synthetic 3D points, project, infer back
- Validate gradients work through projections

**Phase 2: Static Multi-State Poses**
- Define S discrete 3D poses (manual or data-driven)
- Integrate with collapsed HMM
- Test: simple 2-state motion (A→B→A)
- Validate: parameter recovery, pose discrimination

**Phase 3: Skeleton Structure**
- Add joint hierarchy
- Implement forward kinematics (angles → positions)
- Parameterize poses by joint angles
- Test: simple skeleton, known motion

**Phase 4: Integration & Validation**
- Combine all components
- Test on realistic synthetic data
- Benchmark performance (speed, accuracy)
- Document limitations and next steps

**Phase 5: Prepare for Stage 3**
- Identify label switching solutions needed
- Plan for state inference (Viterbi/forward-backward)
- Consider identifiability constraints

---

## Summary for ChatGPT Planning

**Stage 1 Achieved:**
- ✅ Collapsed HMM engine with forward algorithm in PyTensor
- ✅ PyMC integration with gradient computation
- ✅ Comprehensive validation (all tests passing)
- ✅ Successfully samples with nutpie (0 divergences)
- ✅ Label switching is expected and documented

**Stage 2 Goal:**
Integrate collapsed HMM with multi-camera 3D pose estimation

**Key Stage 2 Components to Design:**
1. Camera-aware emission model (3D→2D projection + likelihood)
2. Skeleton structure and joint parameterization
3. State space design (discrete poses vs. continuous)
4. Integration with existing GIMBAL camera code
5. Modified forward algorithm or emission computation strategy
6. Testing framework with synthetic multi-view data

**Critical Design Decisions Needed:**
- How to represent 3D pose states (discrete, continuous, hybrid)?
- What emission likelihood function (Gaussian, robust, mixture)?
- Pre-compute emissions or compute dynamically in scan?
- How many states and how to initialize poses?
- How to handle missing/occluded keypoints?

**Request for ChatGPT:**
Please develop a detailed Stage 2 implementation plan that:
1. Answers the open questions listed above
2. Proposes specific technical designs for each component
3. Outlines implementation phases with clear milestones
4. Identifies risks and mitigation strategies
5. Specifies validation criteria and test cases
6. Considers performance and scalability
7. Maintains compatibility with Stage 1's collapsed HMM framework

The plan should be practical, testable, and incremental - allowing validation at each phase before proceeding to the next.
