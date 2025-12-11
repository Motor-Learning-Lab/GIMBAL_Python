# Debug Step 3: Test Initialization and Log Probability

**Date:** December 11, 2025  
**Status:** ✅ PASSED

---

## Purpose

Test if we can evaluate the model's log probability at the initial point. This is a critical step that `pm.sample()` performs before gradient compilation.

---

## Results

### 1. Model Building

**Status:** ✅ PASS  
Model built successfully with `use_directional_hmm=False`.

### 2. Initial Point

**Status:** ✅ PASS  
**Variables:** 16  
Successfully obtained initial point for all parameters.

### 3. Initial Values Check

**Status:** ✅ PASS  
All initial values are finite (no NaN or inf values).

### 4. Log Probability Compilation

**Status:** ✅ PASS  
Successfully compiled log probability function.

**Key Finding:** The log probability function CAN be compiled. This is different from gradient compilation.

### 5. Log Probability Evaluation

**Status:** ✅ PASS  
**Log Probability at Initial Point:** -32499.28

The log probability is finite and reasonable. This confirms:
- Model specification is correct
- Initialization is valid
- Log probability can be computed

---

## Conclusion

**All initialization checks passed.**

The model:
- ✅ Builds successfully
- ✅ Has valid initial point
- ✅ Can compile log probability function
- ✅ Evaluates to finite log probability

**Critical Insight:** The crash must occur during **gradient compilation**, not during model building or log probability evaluation. This is the next step that `pm.sample()` performs.

---

## Next Steps

Proceed to **Step 4: Test Gradient Compilation** to:
1. Manually compile the gradient function (dlogp/dx)
2. Evaluate gradients at the initial point
3. Test gradients for each parameter group individually
4. Identify which specific parameter causes the infinite recursion in PyTensor

This is where we expect to encounter the crash that blocks Test Group 1.

---

## Files Generated

- `debug_step_3_test_initialization.py` - Test script
- `results_step_3_test_initialization.json` - Detailed results
- `report_step_3_test_initialization.md` - This report
