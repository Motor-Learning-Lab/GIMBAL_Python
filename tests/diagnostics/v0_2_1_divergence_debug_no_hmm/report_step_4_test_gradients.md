# Debug Step 4: Test Gradient Compilation

**Date:** December 11, 2025  
**Status:** ✅ PASSED (with important finding)

---

## Purpose

Test gradient compilation, which was suspected to be where the crash occurs. The NUTS sampler requires gradients of the log probability for Hamiltonian dynamics.

---

## Results

### 1. Model Building

**Status:** ✅ PASS  
Model built successfully.

### 2. Initial Point

**Status:** ✅ PASS  
Initial point obtained successfully.

### 3. Gradient Compilation

**Status:** ✅ PASS  
**Compilation Time:** 44.73 seconds

**CRITICAL FINDING:** The gradient function DOES compile successfully, but it takes almost 45 seconds. This is unexpectedly slow and may have appeared to be a "hang" in previous tests.

### 4. Gradient Evaluation

**Status:** ✅ PASS  
**Evaluation Time:** 2.06 seconds  
**Number of Gradient Arrays:** 2313  
**All gradients finite:** Yes (no NaN or inf)

---

## Analysis

### Why Did This Seem Like a Crash?

The 45-second compilation time is unusually long and could easily be mistaken for:
1. **A hang** - No progress indicator during compilation
2. **Infinite recursion** - PyTensor doing extensive graph optimization
3. **A crash** - If user/system interrupts during this time

### Why Is Compilation So Slow?

With HMM OFF, the model has:
- **16 variable groups**
- **2313 gradient arrays** (one per transformed parameter element)
- **100 timesteps** of high-dimensional variables
- **Complex likelihood** with camera projection and occlusion model

PyTensor must:
1. Build computation graph for log probability
2. Compute symbolic gradients for all 2313 parameters
3. Optimize the gradient graph
4. Compile to efficient C code

This is computationally expensive, especially for the directional unit vectors (`raw_u_1` through `raw_u_5`) which require:
- Normalization constraints
- Transformations for sampling on unit sphere
- Gradients through these transformations

### Comparison: What About With HMM?

When HMM is ON, the model has:
- **Additional 26 HMM parameters**
- **More structure** (hidden states provide regularization)
- **Different gradient patterns** (discrete states vs continuous)

It's possible the HMM structure actually **simplifies** gradient computation or PyTensor's optimizer handles it better.

---

## Implications for Test Group 1

The original Test Group 1 crash was likely:
1. **User interruption** during long gradient compilation
2. **System timeout** if patience ran out
3. **Memory pressure** during compilation (though gradients evaluated fine)

**The model CAN work**, it just needs:
- Patience during initial compilation (~45s)
- Adequate memory
- Clear progress indication

---

## Next Steps

### Option A: Test Actual Sampling (Step 5)
Now that we know gradients compile, try `pm.sample()` with:
- Small number of samples (draws=20)
- Progress bar to show it's working
- Monitor if sampling completes successfully

### Option B: Compare with HMM (Step 6)
Build the same model with HMM ON and compare:
- Gradient compilation time
- Number of parameters
- Whether HMM structure improves compilation

### Recommendation
Proceed with **Step 5: Test Sampling** to confirm the model can actually run end-to-end.

---

## Files Generated

- `debug_step_4_test_gradients.py` - Test script
- `results_step_4_test_gradients.json` - Detailed results
- `report_step_4_test_gradients.md` - This report
