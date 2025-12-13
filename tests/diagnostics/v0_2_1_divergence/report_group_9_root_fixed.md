# Test Group 9: Root Random-Walk Funnel Diagnostic

**Generated:** 2025-12-13T13:13:03.056344

## Purpose

Isolate whether the hierarchical root random walk (RW) structure is a major contributor to sampling divergences.

## Configuration

- T = 100
- C = 3
- draws = 500, tune = 500, chains = 2
- seed = 42
- **NOTE:** Base code now uses data-driven Gamma priors (50% CV) and non-centered root parameterization

## Results

### Baseline: New Base Code (Non-centered Root + Gamma Priors)

- Divergences: 1000/1000 (100.00%)
- Runtime: 52.6s
- Max R-hat: 14462920.057

### Variant: Fixed Root (DLT-based, No Dynamics)

- Divergences: 1/1000 (0.10%)
- Runtime: 395.9s
- Max R-hat: 1.897

## Comparison

- **Divergence reduction factor:** 1000.0×
- **Baseline divergence rate:** 100.00%
- **Fixed-root divergence rate:** 0.10%

## Interpretation

⚠️ **Baseline validation:** New base code still shows 100.00% divergences (target: <1%), indicating possible remaining geometry issues or need for further tuning.

**Variant comparison:** Fixed-root variant shows 1000.0× divergence reduction, suggesting the root dynamics (even non-centered) still contribute some geometry complexity.

⚠️ **Convergence warning:** Baseline max R-hat = 14462920.057 > 1.1, indicating potential convergence issues. Consider increasing draws/tune further.

---

**Reference:** plans/v0.2.1_divergence_plan_2.md, Issue #1, Test Group 9 (post-fix validation)
