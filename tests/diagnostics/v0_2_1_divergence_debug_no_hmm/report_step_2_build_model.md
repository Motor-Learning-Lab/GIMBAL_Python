# Debug Step 2: Test Model Building in Isolation

**Date:** December 11, 2025  
**Status:** ✅ PASSED

---

## Purpose

Test if the PyMC model can be built WITHOUT sampling to isolate whether the issue is in model specification or during gradient compilation for sampling.

---

## Hypothesis

If the model builds successfully but crashes during sampling, the issue is likely in:
- Gradient compilation for specific parameters
- Sampling initialization
- PyTensor graph optimization during sampling

If the model fails to build, the issue is in model specification itself.

---

## Results

### 1. Data Generation

**Status:** ✅ PASS  
Successfully generated synthetic data using standard test utilities.

### 2. Model Building (HMM OFF)

**Status:** ✅ PASS  
Model built successfully with `use_directional_hmm=False`.

**Key Finding:** The model specification itself is valid. The crash must occur during gradient compilation or sampling initialization.

### 3. Model Structure Inspection

**Status:** ✅ PASS  
**Free RVs:** 16 parameters

**Variables:**
- `eta2_root`: Root variance (1 param)
- `rho`: Bone stiffness (5 params, one per bone)
- `sigma2`: Bone length variance (5 params, one per bone)
- `x_root`: Root positions over time (100 timesteps × 3D = 300 params)
- `raw_u_1` through `raw_u_5`: Raw directional vectors for 5 bones (5 × 100 × 3 = 1500 params)
- `length_1` through `length_5`: Bone lengths over time (5 × 100 = 500 params)
- `obs_sigma`: Observation noise (1 param)
- `logodds_inlier`: Inlier probability (1 param)

**Total parameters (without HMM):** 16 variable groups

### 4. Initial Point (Test Point)

**Status:** ✅ PASS  
Successfully obtained initial test point with all 16 variables.

**Sample initial values:**
- `eta2_root_log__`: 3.70 (log-transformed)
- `rho_log__`: [2.14, 2.52] (log-transformed)
- `sigma2_log__`: [2.18, 2.75] (log-transformed)

All initial values are finite and reasonable.

---

## Conclusion

**The model builds successfully.** The issue is NOT in model specification.

The crash observed in Test Group 1 must occur during:
1. **Gradient compilation** - PyTensor compiling gradient functions
2. **Sampling initialization** - Computing initial log probability or gradients
3. **First sampling step** - Attempting to take first NUTS step

---

## Next Steps

Proceed to **Step 3: Test Initialization** to check if we can:
1. Get the model log probability at the initial point
2. Manually compile and evaluate gradients
3. Identify which specific parameter causes gradient compilation to fail

---

## Files Generated

- `debug_step_2_build_model.py` - Test script
- `results_step_2_build_model.json` - Detailed results
- `report_step_2_build_model.md` - This report
