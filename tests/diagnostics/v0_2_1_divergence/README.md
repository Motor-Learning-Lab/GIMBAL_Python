# GIMBAL v0.2.1 Divergence Test Suite

This directory contains a comprehensive test battery for diagnosing and analyzing divergences in GIMBAL PyMC models.

## Overview

The test suite systematically evaluates the v0.2.1 PyMC model with and without the directional HMM prior to identify the source of divergences during NUTS sampling.

## Test Groups

1. **Baseline Tests** (`test_baseline.py`) - HMM OFF
   - Establishes baseline divergence rates without the HMM prior
   
2. **HMM Effect Tests** (`test_hmm_effect.py`) - HMM ON with S=3
   - Measures the effect of enabling the directional HMM prior
   
3. **State Count Tests** (`test_state_count.py`) - S=1, 2, 3
   - Tests how the number of HMM states affects divergence rates
   - Includes special S=1 single-state configuration
   
4. **Likelihood Scale Tests** (`test_likelihood_scale.py`)
   - Analyzes log likelihood values to detect scale mismatches
   
5. **Divergence Diagnostics** (`test_diagnostics.py`)
   - Creates diagnostic visualizations (parallel plots, pair plots)
   
6. **Root Variance Sensitivity** (`test_root_variance.py`)
   - Tests sensitivity to `eta2_root_sigma` hyperparameter
   
7. **Bone Length Variance Sensitivity** (`test_bone_length_variance.py`)
   - Tests sensitivity to `sigma2_sigma` hyperparameter
   
8. **Runtime Scaling** (`test_runtime_scaling.py`)
   - Measures how runtime scales with T (timesteps) and S (states)

## Running the Tests

### Run All Tests

```powershell
cd c:\Repositories\GIMBAL_Python
python tests\v0_2_1_divergence_tests\test_runner.py
```

### Run Individual Test Groups

```powershell
python tests\v0_2_1_divergence_tests\test_baseline.py
python tests\v0_2_1_divergence_tests\test_hmm_effect.py
python tests\v0_2_1_divergence_tests\test_state_count.py
# etc.
```

## Output

### Report

The test suite generates a comprehensive markdown report at:
```
tests/v0.2.1-divergence-report.md
```

The report includes:
- Executive summary with key findings
- Detailed results for each test
- Diagnostic plot references
- Interpretation guide
- Recommended actions

### Diagnostic Plots

Visualizations are saved to:
```
tests/v0.2.1-diagnostics/
```

Includes:
- `divergence_summary.png` - Bar chart of divergence rates across all tests
- `*_parallel.png` - Parallel coordinate plots for individual tests
- `*_pair.png` - Pair plots with divergences highlighted

## Test Configuration

All tests use standardized synthetic data:
- **T**: 100 timesteps
- **C**: 3 cameras
- **S**: 3 states (or 1, 2, 3 for state count tests)
- **kappa**: 10.0 (directional concentration)
- **obs_noise_std**: 0.5 (2D observation noise)
- **occlusion_rate**: 0.02 (2% occlusion rate)
- **seed**: 42 (reproducibility)

MCMC sampling parameters:
- **draws**: 200 (100 for runtime scaling tests)
- **tune**: 200 (100 for runtime scaling tests)
- **chains**: 1
- **target_accept**: 0.95

## Interpretation

### Divergence Rate Thresholds

- **< 1%**: Excellent - sampler exploring efficiently
- **1-10%**: Good - minor issues but acceptable
- **10-50%**: Problematic - significant geometry issues
- **> 50%**: Critical - pathological posterior geometry

### Common Causes

- **Prior-likelihood mismatch**: Priors on different scale than likelihood
- **Funnel geometry**: Hierarchical models with varying scales
- **Stiff dynamics**: HMM dynamics creating numerical instability
- **Non-identifiability**: Multiple parameter configurations equally valid

## Architecture

```
v0_2_1_divergence_tests/
├── __init__.py                      # Package initialization
├── test_runner.py                   # Main orchestrator
├── test_utils.py                    # Shared utilities
├── report_generator.py              # Markdown report generation
├── test_baseline.py                 # Group 1: HMM OFF
├── test_hmm_effect.py               # Group 2: HMM ON
├── test_state_count.py              # Group 3: S=1,2,3
├── test_likelihood_scale.py         # Group 4: Likelihood analysis
├── test_diagnostics.py              # Group 5: Visualizations
├── test_root_variance.py            # Group 6: eta2_root_sigma
├── test_bone_length_variance.py     # Group 7: sigma2_sigma
└── test_runtime_scaling.py          # Group 8: T and S scaling
```

## Dependencies

- pymc >= 5.26.1
- arviz
- numpy
- matplotlib
- gimbal (local package)

## Notes

- Tests are designed to run sequentially (not in parallel)
- Each test group is independent and can be run separately
- Some tests (runtime scaling) use reduced draws for speed
- All tests share the same synthetic data configuration for consistency
