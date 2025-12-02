# GIMBAL Tests

This directory contains all test files for the GIMBAL project.

## Test Organization

### v0.1.1 (Collapsed HMM Engine)
- **`test_hmm_v0_1_1.py`** - Comprehensive tests for the collapsed HMM forward algorithm
  - Brute-force verification against manual enumeration
  - Edge case testing (T=1, extreme values)
  - Gradient validation with finite differences
  - Normalization checks

### v0.1.2 (Camera Observation Model)
- **`test_hmm_v0_1_2.py`** - Tests for the camera observation model refactoring
  - Shape validation for U, x_all, y_pred, log_obs_t
  - Mixture likelihood mode testing
  - Gaussian likelihood mode testing
  - PyMC compilation and sampling validation

### v0.1.3 (Directional HMM Prior)
- **`test_v0_1_3_directional_hmm.py`** - Comprehensive test suite for directional HMM (380 lines, 6 tests)
  - Kappa sharing options (4 configurations)
  - Shape validation for all tensors
  - Numerical stability with extreme log_obs_t values
  - Gradient computation without errors
  - LogP normalization properties
  - Integration with v0.1.2 camera model
- **`test_hmm_v0_1_3.py`** - Minimal working example demonstrating full v0.1.3 pipeline (195 lines)
- **`run_v0_1_3_tests.py`** - Simple test runner for v0.1.3 tests without pytest

### v0.2.1 (Data-Driven Priors & Sampling Diagnostics)
- **`test_sampling_camera_model.py`** - Minimal sampling diagnostic script for camera model
  - Tests Gamma prior on obs_sigma (mode/SD parameterization)
  - Compares mixture vs non-mixture likelihoods
  - Provides divergence, ESS, and RMSE metrics
  - Isolates camera/kinematic issues from HMM
  - Usage: `python tests/test_sampling_camera_model.py`
  - See detailed documentation at end of this file

### Initialization & Utilities
- **`test_dlt_init.py`** - Tests for DLT-based initialization
- **`test_model_init.py`** - Tests for model initialization functions
- **`test_pymc_utils.py`** - Tests for PyMC utility functions

## Running Tests

### Run All v0.1.3 Tests
```bash
# From repository root
python tests/run_v0_1_3_tests.py

# Or from tests directory
cd tests
python run_v0_1_3_tests.py
```

### Run Individual Test Files
```bash
# From repository root
python tests/test_hmm_v0_1_1.py
python tests/test_hmm_v0_1_2.py
python tests/test_hmm_v0_1_3.py
python tests/test_v0_1_3_directional_hmm.py
```

### Run with Pytest (if installed)
```bash
pytest tests/
pytest tests/test_v0_1_3_directional_hmm.py -v
```

## Test Status

All tests passing as of v0.1.3 completion (November 2025):
- ✅ v0.1.1 HMM engine tests: 5/5 passing
- ✅ v0.1.2 camera model tests: 4/4 passing
- ✅ v0.1.3 directional HMM tests: 6/6 passing

## Coverage

Tests cover:
- Core HMM forward algorithm (log-space)
- Camera projection and kinematics
- Directional emissions with vMF-inspired parameterization
- Numerical stability and gradient computation
- Integration between all three phases
- Initialization from observations (DLT, Anipose fallbacks)

## Notes

- Tests use synthetic data with known ground truth for validation
- Numerical tolerances are set conservatively for robust CI/CD
- Some tests may require specific dependencies (PyMC, nutpie, etc.)
- See individual test files for detailed documentation

---

## Detailed: test_sampling_camera_model.py

### Purpose
Minimal script for testing the camera + kinematic observation model in isolation, with focus on the new Gamma prior for obs_sigma.

### Configurations Tested

**M1: No mixture, No HMM**
- Simple Gaussian likelihood
- No outlier detection
- Gamma prior on obs_sigma
- Baseline for convergence

**M2: Mixture, No HMM**
- Mixture likelihood (inliers + uniform outliers)
- Robust to occlusions
- Gamma prior on obs_sigma
- Production camera model

**M3: Mixture + HMM** (optional, currently commented out)
- Full v0.2.1 model with directional HMM prior
- Test only after M1/M2 work well

### Output Metrics

For each model:
- **Divergences**: Count and percentage (target: <5%)
- **ESS**: Effective Sample Size for x_root, obs_sigma, eta2_root, rho (target: >100)
- **Root RMSE**: Reconstruction error vs ground truth (target: <1.0)
- **obs_sigma posterior**: Compare to true value (0.5)

### Typical Runtime
- M1: ~2-3 minutes
- M2: ~3-4 minutes
- M3: ~5-7 minutes (when enabled)
- Total: ~10 minutes

### Interpreting Results

**Good Results:**
- Divergences < 5%
- ESS > 100 for all variables
- Root RMSE < 1.0
- obs_sigma posterior ≈ 0.5 (true value)

**Warning Signs:**
- Divergences 5-10%
- ESS 50-100 (marginal)
- Root RMSE 1-2
- obs_sigma posterior off by >50%

**Bad Results:**
- Divergences > 10%
- ESS < 50 (inefficient sampling)
- Root RMSE > 2
- obs_sigma posterior way off

### Next Steps Based on Results

1. **If M1/M2 look good** (low divergences, good ESS):
   - Uncomment M3 and test with HMM
   - Proceed to full notebook pipeline

2. **If M1/M2 have issues**:
   - Adjust hyperparameters in `make_prior_hyperparams()`
   - Try non-centered parameterization for eta2_root, sigma2
   - Check initialization scaling

3. **If M3 adds divergences**:
   - Issue is in HMM, not camera model
   - Focus on HMM parameterization/scaling
   - See `../notebook/demo_v0_2_1_data_driven_priors.ipynb`

### Related Files
- `../notebook/demo_v0_2_1_data_driven_priors.ipynb`: Full pipeline demo
- `../gimbal/pymc_model.py`: Model building with Gamma prior
- `../gimbal/fit_params.py`: Initialization with obs_noise_std
- `../IMPLEMENTATION_SUMMARY_v0_2_1.md`: Detailed change log
