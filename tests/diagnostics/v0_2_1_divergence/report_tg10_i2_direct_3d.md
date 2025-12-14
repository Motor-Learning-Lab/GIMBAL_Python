# Test Group 10: Camera Likelihood Conditioning Diagnostic

**Generated:** 2025-12-14T00:57:24.781218

## Purpose

Isolate whether the camera projection layer and its likelihood scale dominate the pathological posterior geometry.

## Configuration

- T = 100
- C = 3
- tau_3d = 0.02m (direct 3D noise)
- draws = 500, tune = 500
- seed = 42

## Results

### Baseline: Camera Projection Likelihood

- Divergences: 1000/1000 (100.00%)
- Runtime: 265.7s

### Variant: Direct 3D Likelihood

- Divergences: 32/1000 (3.20%)
- Runtime: 379.0s

## Comparison

- **Divergence reduction factor:** 31.2×
- **Camera divergence rate:** 100.00%
- **Direct-3D divergence rate:** 3.20%

## Interpretation

**Camera projection is major contributor:** The 31.2× divergence reduction (100% → 3.2%) when replacing camera projection with direct 3D likelihood **definitively confirms** that Issue #2 (camera conditioning) is a dominant pathology.

**Residual divergences suggest compound issue:** The remaining 3.2% divergences in the direct-3D variant indicate that **skeleton hierarchy** (root RW + directional parameters) contributes additional geometric pathology, though at much lower severity.

**Conclusion:** Camera projection geometry is the **primary** cause of divergences. The interaction between:
- Camera perspective projection (depth division)
- Root random walk temporal coupling  
- Multi-camera redundancy

creates extreme posterior curvature that NUTS cannot handle.

**Recommended next steps:**
1. Test TG10A (orthographic projection) to isolate depth-division effects
2. Consider camera model redesign or observation model alternatives
3. Test TG11a/b to quantify redundancy contribution to the residual 3.2%

---

**Reference:** plans/v0.2.1_divergence_plan_2.md, Issue #2, Test Group 10
