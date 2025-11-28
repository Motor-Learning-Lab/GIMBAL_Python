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
