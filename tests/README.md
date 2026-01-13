# Tests Directory

Organized test suite for GIMBAL

**Prerequisites**: Tests assume the package is installed in editable mode:
```powershell
pixi install         # Install dependencies
pixi run install-dev # Install gimbal_pymc package
```

## Structure

- **`test_stage*.py`**: Core pipeline stages (Stage 1: HMM, Stage 2: Camera model)
- **`unit/`**: Unit tests for individual modules (initialization, utilities)
- **`integration/`**: Integration tests for multi-module features (Stage 3, data-driven priors)
- **`smoke/`**: Quick smoke tests for rapid validation
- **`pipeline/`**: End-to-end pipeline tests (stages A-J, synthetic data generation)
- **`diagnostics/`**: Comprehensive diagnostic test suites (divergence analysis)

## Running Tests

### Quick Validation
```powershell
# Quick smoke tests (30 seconds)
pixi run test-smoke

# Core HMM stages (10 minutes)
pixi run test-stages

# Integration features (10 minutes)
pixi run test-integration
```

### Comprehensive Testing
```powershell
# All tests (45 minutes)
pixi run test-all

# Specific suites
pixi run test-unit          # Unit tests only
pixi run test-pipeline      # Synthetic data pipeline
```

### Diagnostics
```powershell
# Run divergence diagnostic suite
cd tests/diagnostics/v0_2_1_divergence
pixi run python test_runner.py
```

**Important**: Test outputs (reports, plots) go in `results/` directory, NOT in `tests/`.

## Test Files

### Core Pipeline Stages
- **`test_stage1_collapsed_hmm.py`** - Stage 1: Collapsed HMM forward algorithm
  - Brute-force verification, edge cases, gradient validation
- **`test_stage2_camera_model.py`** - Stage 2: Camera observation model
  - Shape validation, mixture/Gaussian modes, compilation tests

### Integration Tests
- **`integration/test_stage3_directional_hmm.py`** - Stage 3: Directional HMM prior
  - Kappa sharing, numerical stability, gradient computation
- **`integration/test_data_driven_priors.py`** - Data-driven priors pipeline
  - Triangulation, cleaning, statistics, prior building

### Unit Tests
- **`unit/test_dlt_*.py`** - DLT triangulation tests
- **`unit/test_model_init.py`** - Initialization tests
- **`unit/test_pymc_utils.py`** - Utility function tests

### Smoke Tests
- **`smoke/test_api_exports_smoke.py`** - Quick API export validation
- **`smoke/test_full_pipeline_smoke.py`** - Full pipeline build test

### Pipeline Tests
- **`pipeline/test_synthetic_data_generator.py`** - Synthetic dataset validation
- **`pipeline/stage_*.py`** - Individual pipeline stages (A-J)
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
