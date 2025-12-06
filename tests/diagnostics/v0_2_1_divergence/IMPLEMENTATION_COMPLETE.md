# GIMBAL v0.2.1 Divergence Test Suite - Implementation Complete

**Date**: December 6, 2025

## Overview

A comprehensive, automated test battery has been created to systematically diagnose divergence issues in the GIMBAL v0.2.1 PyMC models. The suite consists of 8 test groups that isolate different aspects of the model.

## What Was Created

### Test Infrastructure

1. **`test_runner.py`** - Main orchestrator that runs all tests sequentially
2. **`test_utils.py`** - Shared utilities for data generation, model building, sampling, and metrics extraction
3. **`report_generator.py`** - Automated markdown report generation with interpretation guide

### Test Groups (8 total)

1. **`test_baseline.py`** - Baseline model (HMM OFF)
2. **`test_hmm_effect.py`** - HMM enabled (S=3)
3. **`test_state_count.py`** - State count comparison (S=1, 2, 3)
4. **`test_likelihood_scale.py`** - Likelihood scale analysis
5. **`test_diagnostics.py`** - Divergence localization visualizations
6. **`test_root_variance.py`** - eta2_root_sigma sensitivity (0.1, 0.5, 1.0)
7. **`test_bone_length_variance.py`** - sigma2_sigma sensitivity (0.1, 0.2, 0.5)
8. **`test_runtime_scaling.py`** - Runtime scaling with T and S

### Documentation

- **`README.md`** - Comprehensive suite documentation
- **`QUICKSTART.md`** - Quick start guide with pixi commands

## Test Configuration

All tests use standardized synthetic data:
- **T**: 100 timesteps
- **C**: 3 cameras  
- **S**: 3 states (varies for state count tests)
- **kappa**: 10.0
- **obs_noise_std**: 0.5
- **occlusion_rate**: 0.02
- **seed**: 42

MCMC parameters:
- **draws**: 200 (100 for runtime tests)
- **tune**: 200 (100 for runtime tests)
- **chains**: 1
- **target_accept**: 0.95

## How to Run

### Full Test Suite (recommended)

```powershell
pixi run python tests/v0_2_1_divergence_tests/test_runner.py
```

**Expected runtime**: 30-60 minutes depending on hardware

### Individual Test Groups

```powershell
# Baseline (HMM OFF)
pixi run python tests/v0_2_1_divergence_tests/test_baseline.py

# HMM Effect (HMM ON)
pixi run python tests/v0_2_1_divergence_tests/test_hmm_effect.py

# State count comparison
pixi run python tests/v0_2_1_divergence_tests/test_state_count.py

# Variance sensitivity
pixi run python tests/v0_2_1_divergence_tests/test_root_variance.py
pixi run python tests/v0_2_1_divergence_tests/test_bone_length_variance.py

# Runtime scaling
pixi run python tests/v0_2_1_divergence_tests/test_runtime_scaling.py
```

## Output

### Comprehensive Report
- **Location**: `tests/v0.2.1-divergence-report.md`
- **Contents**:
  - Executive summary with key findings
  - Detailed results for each test
  - Divergence rates, ESS, R-hat, runtime
  - Reconstruction error metrics
  - Interpretation guide
  - Recommended actions

### Diagnostic Plots
- **Location**: `tests/v0.2.1-diagnostics/`
- **Files**:
  - `divergence_summary.png` - Bar chart of all test results
  - `*_parallel.png` - Parallel coordinate plots per test
  - `*_pair.png` - Pair plots with divergences highlighted

## What Each Test Measures

### Test 1: Baseline (HMM OFF)
Establishes baseline divergence rate without HMM prior. If baseline has high divergences, issues are in the base model, not the HMM.

### Test 2: HMM Effect (HMM ON)
Measures impact of enabling HMM prior. Compare with baseline to isolate HMM-specific issues.

### Test 3: State Count (S=1, 2, 3)
- **S=1**: Single-state model (uses special code path, no collapsed HMM)
- **S=2**: Minimal multi-state configuration
- **S=3**: Standard configuration

Identifies whether divergences increase with state count complexity.

### Test 4: Likelihood Scale
Analyzes log likelihood magnitudes to detect prior-likelihood scale mismatches.

### Test 5: Divergence Diagnostics
Creates visualizations showing which parameters correlate with divergences.

### Test 6: Root Variance Sensitivity
Tests eta2_root_sigma values: 0.1, 0.5, 1.0
Identifies if root motion prior is too tight/loose.

### Test 7: Bone Length Variance Sensitivity
Tests sigma2_sigma values: 0.1, 0.2, 0.5
Identifies if bone length prior is too tight/loose.

### Test 8: Runtime Scaling
- **T scaling**: 50, 100, 150 timesteps
- **S scaling**: 1, 2, 3, 5 states

Measures computational cost and checks if divergences worsen with scale.

## Interpreting Results

### Divergence Rate Thresholds
- **< 1%**: ✅ Excellent - sampling working well
- **1-10%**: ✓ Good - minor issues
- **10-50%**: ⚠️ Problematic - needs attention
- **> 50%**: ❌ Critical - requires fixes

### Key Comparisons

**Baseline vs HMM ON:**
- If HMM >> Baseline: HMM prior causing issues
- If HMM << Baseline: HMM prior helping
- If HMM ≈ Baseline: Issues in base model

**State Count Progression:**
- If S=1 OK but S=2,3 bad: Multi-state dynamics problematic
- If all similar: State count not the issue

**Variance Sensitivity:**
- If lower values better: Priors too weak
- If higher values better: Priors too strong

## Next Steps After Running

1. **Review the report**: `tests/v0.2.1-divergence-report.md`
2. **Examine diagnostic plots**: `tests/v0.2.1-diagnostics/`
3. **Identify patterns**:
   - Which tests have lowest divergences?
   - Which parameters correlate with divergences (parallel plots)?
   - Does HMM help or hurt?
4. **Implement fixes** based on findings
5. **Re-run specific tests** to verify improvements

## Files Created

```
tests/v0_2_1_divergence_tests/
├── __init__.py                      # Package initialization
├── README.md                        # Full documentation
├── QUICKSTART.md                    # Quick start guide
├── test_runner.py                   # Main orchestrator (300 lines)
├── test_utils.py                    # Shared utilities (320 lines)
├── report_generator.py              # Report generation (250 lines)
├── test_baseline.py                 # Test Group 1 (75 lines)
├── test_hmm_effect.py               # Test Group 2 (75 lines)
├── test_state_count.py              # Test Group 3 (95 lines)
├── test_likelihood_scale.py         # Test Group 4 (50 lines)
├── test_diagnostics.py              # Test Group 5 (60 lines)
├── test_root_variance.py            # Test Group 6 (85 lines)
├── test_bone_length_variance.py     # Test Group 7 (85 lines)
└── test_runtime_scaling.py          # Test Group 8 (140 lines)

Total: ~1,630 lines of production code
```

## Status

✅ **Implementation Complete**
- All 8 test groups implemented
- Shared utilities created
- Report generation automated
- Documentation written
- Import verification successful

⏳ **Ready to Run**
- Test suite has not been executed yet
- Estimated runtime: 30-60 minutes for full suite
- Will generate report and diagnostic plots automatically

## Command to Start

```powershell
cd c:\Repositories\GIMBAL_Python
pixi run python tests/v0_2_1_divergence_tests/test_runner.py
```

The suite will run all tests, generate the report, and save diagnostic plots. Progress will be printed to console throughout execution.
