# Test Group 11.1: Parameter Redundancy Diagnostic (Fixed)

**Generated:** 2025-12-14T01:15:15.878563

## Purpose

Isolate whether redundant degrees of freedom (root motion vs directions vs bone lengths) significantly worsen posterior geometry.

## Configuration

- T = 100
- C = 3
- draws = 500, tune = 500
- seed = 42

## Results

### Baseline: Free Directions and Lengths

- Divergences: 1000/1000 (100.00%)
- Runtime: 55.7s

### Variant: Fixed Directions and Lengths (GT)

- Divergences: 0/1000 (0.00%)
- Runtime: 514.3s

## Comparison

- **Divergence reduction factor:** 1000.0×
- **Baseline divergence rate:** 100.00%
- **Fixed divergence rate:** 0.00%

## Interpretation

**Strong evidence for Issue #3:** Divergence reduction ≥10× indicates that **parameter redundancy** between root motion, directions, and lengths is a major contributor to geometric pathologies.

The multiple pathways for explaining observations (moving root vs changing directions vs adjusting lengths) create curved, partially flat manifolds that NUTS cannot navigate efficiently.

---

**Reference:** plans/v0.2.1_divergence_plan_2.md, Issue #3, Test Group 11.1
