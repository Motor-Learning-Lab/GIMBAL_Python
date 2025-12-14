# Test Group 11.2: Parameter Redundancy Diagnostic (Strong Priors)

**Generated:** 2025-12-14T08:46:37.623803

## Purpose

Test whether imposing strong, data-driven priors on directions and lengths reduces divergences without completely fixing parameters.

## Configuration

- T = 100
- C = 3
- Strong prior settings:
  - Length relative SD: 2.0%
  - Raw direction SD: 0.05
- draws = 500, tune = 500
- seed = 42

## Results

### Baseline: Weak Priors

- Divergences: 1000/1000 (100.00%)
- Runtime: 270.1s

### Variant: Strong GT-Based Priors

- Divergences: 0/1000 (0.00%)
- Runtime: 868.1s

## Comparison

- **Divergence reduction factor:** 1000.0×
- **Baseline divergence rate:** 100.00%
- **Strong-priors divergence rate:** 0.00%

## Interpretation

### Cross-Test Pattern (Test 11.1 + 11.2)

Consider results from both Test 11.1 (fixed) and Test 11.2 (strong priors):

- **Monotonic improvement** (weak priors → strong priors → fixed) suggests parameter redundancy amplifies geometric pathologies.
- **No clear pattern** suggests redundancy is secondary to Issues #1 (root RW) and #2 (camera conditioning).

### This Test (Strong Priors vs Baseline)

**Moderate-to-strong evidence:** Divergence reduction ≥3× from strong priors (without full fixing) suggests that **constraining parameter redundancy** helps sampling, even when uncertainty is preserved.

---

**Reference:** plans/v0.2.1_divergence_plan_2.md, Issue #3, Test Group 11.2
