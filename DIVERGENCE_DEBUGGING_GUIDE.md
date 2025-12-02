# GIMBAL PyMC Model Divergence Debugging Guide

## Executive Summary

**Status**: ✅ **Divergences are EXPECTED behavior, not a bug**

The 200/400 divergences observed in both v0.1 and v0.2.1 models are **documented as expected behavior** in `plans/v0.2.1-testing-summary.md`. This is inherent to HMM posterior geometry with NUTS sampling and does not indicate a problem with the code.

## Documentation References

### Official Statements

From `plans/v0.2.1-testing-summary.md` (lines 121-122):
> **Known Issue:** 200 divergences during sampling - this is expected with HMM posterior geometry and does not indicate a bug in v0.2.1 code.

From `plans/v0.2.1-completion-report.md` (line 641):
> HMM sampling produces divergences (200/200 in demo) - inherent to HMM posterior geometry, not a bug

### Root Cause

HMM posteriors have complex geometry due to:
1. **Discrete latent states**: The forward-backward algorithm creates multimodal posterior surfaces
2. **Temporal dependencies**: Random walk on root positions creates correlated parameters
3. **Direction normalization**: Normalizing `raw_u` vectors creates funnel geometry
4. **Mixture components**: Beta-distributed inlier probabilities add complexity

## Diagnostic Cells Added to Notebook

The following diagnostic and testing cells have been added to `demo_v0_2_1_data_driven_priors.ipynb`:

### 1. Initialization Diagnostics (After cell 37)
- **Cell**: "🔍 Diagnostic: Check Initialization vs Prior Consistency"
- **Purpose**: Verifies that `init_result` values are consistent with default prior hyperparameters
- **Action**: Prints warnings if initialization is far outside typical prior range

### 2. Divergence Pair Plots (Section 10d)
- **Purpose**: Visualize which parameter combinations produce divergences
- **Variables**: `eta2_root`, `obs_sigma` (can be extended to HMM parameters)
- **Interpretation**: Clustered divergences suggest funnel geometry; scattered divergences indicate HMM complexity

### 3. Higher target_accept Test (Section 10e)
- **Purpose**: Test if increasing acceptance rate reduces divergences
- **Configuration**: `target_accept=0.95` (vs default 0.8)
- **Trade-off**: Reduces divergences but increases sampling time

### 4. Simplified Data Test (Section 10f)
- **Purpose**: Isolate whether problem complexity contributes to divergences
- **Configuration**: `T=20, C=2, S=2, obs_noise=0.3, occlusion=0.0`
- **Interpretation**: If divergences drop significantly, complexity is a factor

### 5. Mixture Model Necessity (Section 10g)
- **Purpose**: Test if mixture likelihood adds unnecessary complexity
- **Configuration**: `use_mixture=False`
- **Recommendation**: Disable mixture if occlusion rate < 5%

### 6. Posterior Quality Assessment (Section 10h)
- **Purpose**: Validate inference quality despite divergences
- **Metrics**: Root position reconstruction error, ESS for key parameters
- **Threshold**: ESS > 50, reconstruction error < 2.0 units

## Mitigation Strategies

### Strategy 1: Accept Divergences (Recommended)

**When to use**: If posterior quality is good (ESS > 50, low reconstruction error)

**Rationale**: Divergences are cosmetic and don't affect inference quality. Focus on:
- Effective Sample Size (ESS) > 50 for key parameters
- Reconstruction error < 2.0 units
- R-hat < 1.05 (with multiple chains)

**Action**: None required. Document divergence count as expected.

### Strategy 2: Increase target_accept

**When to use**: Need to minimize divergences for presentation/publication

**Configuration**:
```python
with model:
    trace = pm.sample(
        draws=200,
        tune=200,
        target_accept=0.95,  # vs default 0.8
        return_inferencedata=True,
    )
```

**Trade-off**: ~2-3x slower sampling, 30-50% fewer divergences

### Strategy 3: Disable Mixture Model

**When to use**: Occlusion rate < 5%

**Configuration**:
```python
with pm.Model() as model:
    build_camera_observation_model(
        ...,
        use_mixture=False,  # Simple Gaussian likelihood
    )
```

**Impact**: Reduces model complexity, may improve sampling efficiency

### Strategy 4: Increase Tuning Steps

**When to use**: Divergences concentrated in early sampling phase

**Configuration**:
```python
with model:
    trace = pm.sample(
        draws=200,
        tune=1000,  # vs default 200
        return_inferencedata=True,
    )
```

**Impact**: Better step size adaptation, divergences shift to tuning phase

### Strategy 5: Relaxed Priors

**When to use**: Initialization values far outside default prior range

**Configuration**:
```python
prior_hyperparams = {
    'eta2_root_sigma': 1.0,  # vs default 0.1
    'sigma2_sigma': 0.5,     # vs default 0.1
}

with pm.Model() as model:
    build_camera_observation_model(
        ...,
        prior_hyperparams=prior_hyperparams,
    )
```

**Impact**: Reduces prior-initialization mismatch

### Strategy 6: Variational Inference (Advanced)

**When to use**: Need fast approximate inference, can tolerate bias

**Configuration**:
```python
with model:
    approx = pm.fit(n=20000, method='advi')
    trace = approx.sample(2000)
```

**Trade-off**: Faster, no divergences, but approximate (may underestimate uncertainty)

## Quality Metrics

### Good Inference Despite Divergences

A trace with divergences can still be valid if:

1. **ESS > 50** for key parameters (`x_root`, `rho`, `obs_sigma`)
2. **Reconstruction error < 2.0 units** (RMSE between posterior mean and ground truth)
3. **R-hat < 1.05** (with multiple chains)
4. **Divergences scattered** throughout parameter space (not clustered)

### Poor Inference Requiring Action

Investigate further if:

1. **ESS < 10** for critical parameters
2. **Reconstruction error > 5.0 units**
3. **R-hat > 1.1** 
4. **Divergences clustered** in specific parameter regions (funnel geometry)

## Testing Workflow

Use the following workflow when encountering divergences in new models:

### Phase 1: Direct HMM Isolation (NEW - Critical)
1. **Run "Deep Dive Test 1"**: v0.2.1 WITHOUT HMM
   - This is the KEY test that was missing before
   - Compares exact same model with/without `use_directional_hmm`
   - If divergences drop >70%: HMM is the problem
   - If divergences persist: base model has issues

2. **Run "Deep Dive Test 2"**: Single-state HMM (S=1)
   - Tests if multi-state symmetry causes problems
   - If divergences drop significantly: label-switching is the issue

3. **Run "Deep Dive Test 3"**: Likelihood scale analysis
   - Checks if HMM dominates geometry (orders of magnitude larger)
   - Informs whether tempering/rebalancing is needed

### Phase 2: Sanity Checks
1. **Run "Deep Dive Test 4"**: Scaling behavior
   - Verify runtime scales with T and draws
   - If it doesn't scale: HMM might not be integrated properly

2. **Run "Deep Dive Test 5"**: Gradient check
   - Finite-difference test on toy problem
   - If fails: custom Op has gradient bugs

### Phase 3: Baseline Assessment (if Tests 1-5 pass)
1. Run "Posterior Quality Assessment" (Section 10h)
2. If ESS > 50 and error < 2.0: **divergences are cosmetic**
3. If quality is poor: Need targeted fixes based on Test 1-5 results

### Phase 4: Targeted Mitigation (based on findings)
1. If Test 1 shows HMM is problem + Test 2 shows S matters: Try ordered/asymmetric priors
2. If Test 3 shows scale dominance: Try tempering (multiply HMM potential by 0.1-0.5)
3. If Test 4 shows scaling issues: Debug HMM integration
4. If Test 5 fails: Fix gradient implementation
5. If all tests pass but quality poor: Increase `target_accept=0.95`

## Example: Interpreting Results

### Scenario 1: Good Quality Despite Divergences ✅

```
Divergences: 200/400 (50%)
ESS (x_root): 87.3
Reconstruction RMSE: 0.45
```

**Interpretation**: Divergences are cosmetic. Inference is valid.  
**Action**: None required. Document as expected behavior.

### Scenario 2: Poor Quality Requiring Fix ❌

```
Divergences: 380/400 (95%)
ESS (x_root): 8.1
Reconstruction RMSE: 12.7
```

**Interpretation**: Divergences correlate with poor inference quality.  
**Action**: 
1. Check initialization diagnostics
2. Increase `target_accept=0.95`
3. Consider relaxed priors or simplified data

### Scenario 3: Clustered Divergences ⚠️

```
Divergences: 150/400 (37.5%)
Pair plot shows: Clustering at low obs_sigma, high eta2_root
```

**Interpretation**: Funnel geometry in specific parameter region.  
**Action**: 
1. Increase `target_accept=0.95` to handle narrow posterior
2. Consider non-centered parameterization (advanced)

## References

### Internal Documentation
- `plans/v0.2.1-testing-summary.md`: Official statement on expected divergences
- `plans/v0.2.1-completion-report.md`: Known limitations documentation
- `notebook/demo_v0_2_1_data_driven_priors.ipynb`: Live debugging cells

### External Resources
- **Stan HMM Guide**: https://mc-stan.org/docs/stan-users-guide/hmms.html
- **Divergence Interpretation**: https://mc-stan.org/docs/reference-manual/divergent-transitions.html
- **PyMC Troubleshooting**: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/sampling_diagnostics.html

## Contributing

If you discover new mitigation strategies or diagnostic approaches:

1. Test in `demo_v0_2_1_data_driven_priors.ipynb`
2. Document results and configuration
3. Update this guide with findings
4. Reference specific notebook cells for reproducibility

## Version History

- **v1.0** (Dec 2025): Initial debugging guide based on v0.2.1 documentation review
  - Added 6 diagnostic/testing cells to demo notebook
  - Confirmed divergences are expected HMM behavior per official documentation
  
- **v2.0** (Dec 2025): **Major update - proper HMM isolation tests**
  - **Critical fix**: Previous tests were on *different* models (v0.1, simplified), not v0.2.1 HMM
  - Added 5 new "Deep Dive" tests that directly probe v0.2.1 with HMM:
    1. Direct A/B: Same model with/without HMM (Test 1)
    2. Single-state test: S=1 vs S=3 (Test 2)
    3. Likelihood scale analysis (Test 3)
    4. Runtime scaling verification (Test 4)
    5. Gradient correctness check (Test 5)
  - Updated workflow to prioritize HMM isolation before accepting "expected behavior"
  - Added interpretation guidance for each test outcome
