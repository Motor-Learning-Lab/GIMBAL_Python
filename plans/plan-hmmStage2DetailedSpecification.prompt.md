# HMM Stage 2 Detailed Specification

## 1. Overview

**Goal**: Refactor the current single-function `build_gimbal_pymc_model` in `pymc_model.py` into smaller, testable components that separate directional kinematics, camera likelihood, and overall model assembly. This modularization enables seamless integration with the Stage 1 collapsed HMM engine in Stage 3.

**Context**: Stage 1 completed the HMM engine (`hmm_pytensor.py`) which requires specific inputs (`log_obs_t`, `logp_emit`, `logp_init`, `logp_trans`). Stage 2 prepares the camera model to provide `log_obs_t` by extracting and documenting its likelihood computation logic.

**Key Constraint**: Stage 2 must NOT change model behavior - this is a pure refactoring for readability and testability. The existing model's predictions and sampling behavior should remain identical.

## 2. Interface Contract with Stage 1

The Stage 1 HMM engine (`collapsed_hmm_loglik` in `hmm_pytensor.py`) requires:

**Inputs**:
- `log_obs_t`: (T,) or scalar - log-likelihood of observations at each timestep
- `logp_emit`: (S, T) - log emission probabilities for S states across T timesteps  
- `logp_init`: (S,) - log initial state probabilities
- `logp_trans`: (S, S) - log transition probabilities

**Output**:
- Scalar log-likelihood value suitable for `pm.Potential("hmm_prior", ...)`

**Critical Note**: Stage 2 must produce `log_obs_t` in a format that Stage 3 can combine with `logp_emit`.

## 3. Interface Contract with Stage 3

Stage 3 will combine directional kinematics priors (from Stage 2) with HMM temporal priors (from Stage 1):

**From Stage 2 to Stage 3**:
- `U`: (T, M, 3) - directional predictions at each frame
- `x_all`: Any shape - biomechanical state variables (joint angles, etc.)
- `y_pred`: (T, N_cams, M, 2) - predicted camera observations
- `log_obs_t`: (T,) or scalar - camera likelihood at each timestep

**Stage 3 Combination**:
```python
# Directional emissions from vMF distributions
logp_emit[t, s] = log_dir_emit[t, s] + log_obs_t[t]  # Broadcasting required

# Then apply HMM
hmm_loglik = collapsed_hmm_loglik(log_obs_t, logp_emit, logp_init, logp_trans)
```

**Key Design Decision**: `log_obs_t` must represent the total camera likelihood per timestep, while `logp_emit` will add the directional kinematic prior.

## 4. Module Responsibilities

### 4.1 `pymc_model.py` (After Refactor)

**Primary Function**: `build_gimbal_pymc_model`
- Orchestrates overall model construction
- Manages priors on global parameters (camera positions, orientations, etc.)
- Delegates to helper functions
- Assembles final likelihood

**Helper Functions** (to be extracted):

1. `build_directional_kinematics`
   - **Inputs**: Biomechanical parameters, frame count T
   - **Outputs**: `U` (T, M, 3), `x_all` (biomechanical states)
   - **Purpose**: Predict 3D unit vectors for M markers at T frames

2. `build_camera_likelihood`
   - **Inputs**: `U`, camera parameters, observed keypoints `y_obs`
   - **Outputs**: `y_pred` (predictions), `log_obs_t` (per-frame log-likelihood)
   - **Purpose**: Project 3D directions to 2D, compute observation likelihood

### 4.2 `camera.py` (May need minor updates)

- Already contains `project_points_to_cameras` and other camera utilities
- May add helper for computing per-frame likelihood if needed
- No major changes expected

### 4.3 `model.py` (Read-only for Stage 2)

- Contains biomechanical forward kinematics
- Used by directional kinematics builder
- No changes in Stage 2

## 5. Current Code Audit

**File**: `pymc_model.py`
**Current Function**: `build_gimbal_pymc_model` (~200-300 lines, estimated)

**Suspected Structure** (to be verified):
1. Camera parameter priors (positions, orientations, intrinsics)
2. Biomechanical parameter priors (joint angles, lengths, etc.)
3. Forward kinematics to get 3D marker positions
4. Projection to 2D camera views
5. Likelihood computation with observation noise

**Extraction Targets**:
- Steps 3-4: Directional kinematics (3D unit vectors)
- Steps 4-5: Camera likelihood (projection + noise model)

## 6. Stage 2 Deliverables

### Core Functions

1. **`build_directional_kinematics`**
   ```python
   def build_directional_kinematics(
       bio_params: Dict,  # Biomechanical parameters (already sampled PyMC RVs)
       T: int,            # Number of timesteps
       model_config: Dict # Configuration (e.g., marker indices, skeleton structure)
   ) -> Tuple[TensorVariable, Any]:
       """
       Build directional kinematics component.
       
       Returns:
           U: (T, M, 3) - Unit vectors for M markers at T timesteps
           x_all: Biomechanical state variables (for diagnostics)
       """
       # Use model.py functions to compute 3D marker positions
       # Normalize to unit vectors
       return U, x_all
   ```

2. **`build_camera_likelihood`**
   ```python
   def build_camera_likelihood(
       U: TensorVariable,           # (T, M, 3) directional predictions
       camera_params: Dict,          # Camera parameters (PyMC RVs)
       y_obs: np.ndarray,           # (T, N_cams, M, 2) observed keypoints
       obs_noise_params: Dict       # Observation noise parameters
   ) -> Tuple[TensorVariable, TensorVariable]:
       """
       Build camera observation likelihood.
       
       Returns:
           y_pred: (T, N_cams, M, 2) - Predicted 2D keypoints
           log_obs_t: (T,) or scalar - Log-likelihood at each timestep
       """
       # Project U to camera views using camera.py functions
       # Compute likelihood with noise model
       # Sum over cameras and markers per timestep
       return y_pred, log_obs_t
   ```

3. **Refactored `build_gimbal_pymc_model`**
   ```python
   def build_gimbal_pymc_model(
       y_obs: np.ndarray,
       config: Dict,
       name: str = "gimbal_model"
   ) -> pm.Model:
       """
       Build complete GIMBAL PyMC model (refactored).
       
       Same signature and behavior as current implementation.
       """
       with pm.Model(name=name) as model:
           # 1. Camera parameter priors
           camera_params = _build_camera_priors(config)
           
           # 2. Biomechanical parameter priors
           bio_params = _build_biomechanical_priors(config)
           
           # 3. Directional kinematics
           U, x_all = build_directional_kinematics(bio_params, T, config)
           
           # 4. Camera likelihood
           y_pred, log_obs_t = build_camera_likelihood(
               U, camera_params, y_obs, config["obs_noise"]
           )
           
           # 5. Store for later access
           pm.Deterministic("U_pred", U)
           pm.Deterministic("y_pred", y_pred)
           
           return model
   ```

### Documentation

- **Docstrings**: Every function must have clear docstrings with:
  - Purpose
  - Input shapes and meanings
  - Output shapes and meanings
  - Example usage if non-trivial

- **Shape Comments**: Inline comments documenting tensor shapes at key steps

- **Module-level Documentation**: Update `pymc_model.py` header with refactoring rationale

## 7. Step-by-Step Refactoring Plan

### Phase 1: Preparation & Audit

1. **Read Current Code**: Carefully read `build_gimbal_pymc_model` in `pymc_model.py`
2. **Document Structure**: Add inline comments describing what each section does
3. **Identify Boundaries**: Mark exact lines for kinematics vs. camera likelihood split
4. **Create Test Case**: Save a minimal example that can verify behavior equivalence

### Phase 2: Extract Directional Kinematics

5. **Create Function Stub**: Add `build_directional_kinematics` with docstring
6. **Move Code**: Transfer biomechanical forward kinematics logic
7. **Update Main Function**: Replace moved code with function call
8. **Test Equivalence**: Verify model predictions unchanged (mean, std of posterior samples)

### Phase 3: Extract Camera Likelihood

9. **Create Function Stub**: Add `build_camera_likelihood` with docstring
10. **Move Code**: Transfer projection and likelihood computation
11. **Update Main Function**: Replace moved code with function call
12. **Test Equivalence**: Again verify predictions unchanged

### Phase 4: Documentation & Polish

13. **Add Shape Comments**: Document tensor shapes throughout
14. **Write Examples**: Create simple example showing new API
15. **Update README**: Note refactoring completed (if README exists)
16. **Final Validation**: Run full test suite if available

## 8. Validation & Testing

### Behavior Equivalence Tests

**Goal**: Prove refactored model produces identical predictions

**Method**:
1. Load same synthetic dataset in both versions
2. Sample with same random seed
3. Compare posterior means/stds for key parameters
4. Verify predictions `y_pred` are identical (within numerical tolerance)

**Acceptance**: All values match within 1e-6 relative error

### Shape Validation

**Tests to Add**:
```python
def test_directional_kinematics_shapes():
    """Verify U is (T, M, 3) and properly normalized."""
    U, x_all = build_directional_kinematics(...)
    assert U.shape == (T, M, 3)
    assert pt.allclose(pt.sum(U**2, axis=-1), 1.0)  # Unit vectors

def test_camera_likelihood_shapes():
    """Verify y_pred and log_obs_t have correct shapes."""
    y_pred, log_obs_t = build_camera_likelihood(...)
    assert y_pred.shape == (T, N_cams, M, 2)
    assert log_obs_t.shape == (T,) or log_obs_t.ndim == 0  # (T,) or scalar
```

### Integration Validation

**Stage 2 → Stage 3 Interface Test**:
```python
def test_stage2_stage3_interface():
    """Verify Stage 2 outputs can be used by Stage 3."""
    # Build Stage 2 model
    model = build_gimbal_pymc_model(y_obs, config)
    
    # Extract required outputs
    U = model["U_pred"]
    log_obs_t = model["log_obs_t"]  # Must be accessible
    
    # Verify shapes compatible with Stage 3
    assert U.shape[0] == log_obs_t.shape[0]  # Both have T timesteps
    
    # Mock Stage 3 combination
    S = 5  # Number of HMM states
    logp_emit = pt.zeros((S, T))  # Mock directional emissions
    combined = logp_emit + log_obs_t  # Should broadcast correctly
    assert combined.shape == (S, T)
```

## 9. Completion Criteria

Stage 2 is complete when:

1. ✅ `build_directional_kinematics` exists with full docstring
2. ✅ `build_camera_likelihood` exists with full docstring
3. ✅ `build_gimbal_pymc_model` refactored to use helper functions
4. ✅ All shape comments added
5. ✅ Behavior equivalence test passes (predictions unchanged)
6. ✅ Shape validation tests pass
7. ✅ `log_obs_t` is accessible and properly shaped for Stage 3
8. ✅ No sampling errors (model compiles and samples as before)
9. ✅ Documentation updated

## 10. Stage 3 Preview

Once Stage 2 is complete, Stage 3 will:

1. **Add Directional Emissions**:
   ```python
   # In Stage 3 model builder
   logp_emit = build_vmf_emissions(U, mu_states, kappa_states)  # (S, T)
   ```

2. **Combine Likelihoods**:
   ```python
   # Replace direct camera likelihood with HMM-enhanced version
   full_logp = collapsed_hmm_loglik(log_obs_t, logp_emit, logp_init, logp_trans)
   pm.Potential("full_likelihood", full_logp)
   ```

3. **Validate**:
   - Confirm HMM affects posterior distributions
   - Check temporal smoothness improved
   - Verify directional priors guide marker predictions

**Key Stage 2 Contribution**: By exposing `log_obs_t` cleanly, Stage 3 can seamlessly add HMM priors without re-engineering the camera model.
