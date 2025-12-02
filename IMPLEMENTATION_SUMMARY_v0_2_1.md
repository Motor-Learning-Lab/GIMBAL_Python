# GIMBAL v0.2.1 Prior Fix Implementation Summary

## Completed Changes (December 2, 2025)

### A. Overview
Successfully implemented data-driven priors with Gamma prior on obs_sigma and streamlined notebook for v0.2.1 demo.

---

## B. Code Changes

### B1-B3: Gamma Prior on obs_sigma (gimbal/pymc_model.py)

**Added:**
- `gamma_from_mode_sd()` helper function to convert mode/SD to Gamma(alpha, beta) parameters
- New hyperparameters: `obs_sigma_mode` and `obs_sigma_sd` (in pixel units)
- Gamma prior replaces HalfNormal for obs_sigma
- Backward compatibility: still accepts `obs_sigma_sigma` for old code

**Location:** Lines 28-88 in pymc_model.py

**Key Benefits:**
- Mode/SD parameterization is more interpretable (directly in pixel units)
- Avoids pathological funnels of HalfNormal
- Data-driven scaling based on config.obs_noise_std

---

### C1-C2: Initialization with obs_noise_std (gimbal/fit_params.py)

**Modified:** `initialize_from_groundtruth()`
- Added `obs_noise_std` parameter (optional)
- Sets `obs_sigma` initialization to `max(0.1, obs_noise_std * 1.5)`
- Fallback to 2.0 if not provided

**Location:** Lines 575-648 in fit_params.py

**Key Benefits:**
- Initialization matches data scale
- Prevents under/over-estimation of observation noise
- Smoother sampling startup

---

### D1-D3: Data-Driven Hyperparameters (notebook cells)

**Added to notebook:**
- New cell after Section 7: "Configure Data-Driven Hyperparameters"
- Computes `prior_hyperparams` dict from init_result:
  ```python
  prior_hyperparams = {
      "eta2_root_sigma": max(init_result.eta2[0] / 2.0, 0.5),
      "sigma2_sigma": max(init_result.sigma2.mean() * 2.0, 0.5),
      "obs_sigma_mode": max(0.1, data.config.obs_noise_std * 1.0),
      "obs_sigma_sd": max(0.1, data.config.obs_noise_std * 0.5),
  }
  ```
- Passes `prior_hyperparams` to model builders

**Key Benefits:**
- Priors scale with actual data statistics
- No hard-coded magic numbers
- Flexible per-dataset tuning

---

### E: Test Script (tests/test_sampling_camera_model.py)

**Created:** New minimal debug script for sampling diagnostics

**Tests:**
- M1: No mixture, no HMM, Gamma prior
- M2: Mixture on, no HMM, Gamma prior
- (M3: Mixture + HMM - ready for future testing)

**Metrics:**
- Divergence count and percentage
- ESS (mean, min) for x_root, obs_sigma, eta2_root, rho
- Root RMSE vs ground truth
- obs_sigma posterior mean vs true value

**Usage:** `python tests/test_sampling_camera_model.py`

**Key Benefits:**
- Isolated testing without notebook clutter
- Reproducible diagnostics
- Easy comparison across configurations

---

### F: Notebook Simplification

**Changes:**
1. Updated Section 8: Split into initialization + hyperparameter config
2. Simplified Section 9 (model building):
   - v0.1: mixture=True, hmm=False, Gamma prior
   - v0.2.1: mixture=True, hmm=True, Gamma prior + data-driven directional priors
3. Updated all section numbers (8→9, 9→10, 10→11, etc.)
4. Added note pointing to test script for detailed diagnostics
5. Removed redundant old API calls

**Key Benefits:**
- Cleaner narrative flow
- Focus on 2D→3D→statistics→priors pipeline
- Heavy diagnostics delegated to test script
- Less brittle to API changes

---

## C. Updated API

### Old Way (deprecated):
```python
init_result = initialize_from_groundtruth(
    x_gt=data.x_true,
    parents=DEMO_V0_1_SKELETON.parents,
)
# obs_sigma always 2.0

with pm.Model() as model:
    # Manual model building with hard-coded priors
    ...
```

### New Way (v0.2.1):
```python
init_result = initialize_from_groundtruth(
    x_gt=data.x_true,
    parents=DEMO_V0_1_SKELETON.parents,
    obs_noise_std=data.config.obs_noise_std,  # NEW
)
# obs_sigma = obs_noise_std * 1.5

prior_hyperparams = {
    "obs_sigma_mode": data.config.obs_noise_std * 1.0,  # NEW
    "obs_sigma_sd": data.config.obs_noise_std * 0.5,     # NEW
    # ... other hyperparams
}

with pm.Model() as model:
    build_camera_observation_model(
        y_observed=data.y_observed,
        camera_proj=data.camera_proj,
        parents=DEMO_V0_1_SKELETON.parents,
        init_result=init_result,
        use_mixture=True,
        use_directional_hmm=False,  # or True for v0.2.1
        prior_hyperparams=prior_hyperparams,  # NEW
    )
```

---

## D. Execution Order for Debugging

1. **First**: Run test script to verify camera + kinematic model
   ```bash
   python tests/test_sampling_camera_model.py
   ```
   - Check M1 (no mixture) and M2 (mixture) for divergences and ESS
   - Verify obs_sigma posterior matches true value

2. **Second**: Run notebook cells 1-10 to verify pipeline
   - Generate data → triangulate → clean → compute statistics
   - Build priors and hyperparameters
   - Sample v0.1 and v0.2.1 models

3. **Third**: Compare ESS and convergence
   - v0.2.1 should show higher ESS than v0.1
   - Divergences should be <5% with Gamma prior

4. **Future**: If HMM causes divergences, focus on HMM parameterization
   - Use test script M3 to isolate HMM vs camera issues
   - Consider non-centered parameterization for HMM

---

## E. Files Modified

1. `/home/opher/Documents/GitHub/GIMBAL_Python/gimbal/pymc_model.py`
   - Added gamma_from_mode_sd()
   - Updated KNOWN_HYPERPARAMS
   - Added Gamma prior for obs_sigma

2. `/home/opher/Documents/GitHub/GIMBAL_Python/gimbal/fit_params.py`
   - Added obs_noise_std parameter
   - Updated obs_sigma initialization logic

3. `/home/opher/Documents/GitHub/GIMBAL_Python/notebook/demo_v0_2_1_data_driven_priors.ipynb`
   - Added hyperparameter configuration cell
   - Updated model building cells
   - Simplified and reorganized sections

4. `/home/opher/Documents/GitHub/GIMBAL_Python/tests/test_sampling_camera_model.py` (**NEW**)
   - Standalone sampling diagnostic script

---

## F. Backward Compatibility

✅ Old code using `obs_sigma_sigma` will still work (with deprecation warning)
✅ Old code not passing `obs_noise_std` will use fallback (2.0)
✅ Old code not passing `prior_hyperparams` will use defaults

---

## G. Next Steps

1. Run test script and verify M1/M2 work cleanly
2. Run notebook and verify pipeline end-to-end
3. If sampling looks good, re-enable M3 (HMM test)
4. Consider non-centered parameterization for other variance parameters if needed
5. Document empirical results (divergences, ESS, RMSE) for paper/report

---

## H. Testing Checklist

- [ ] Run `python tests/test_sampling_camera_model.py`
  - [ ] M1 completes without errors
  - [ ] M2 completes without errors
  - [ ] Divergences < 10%
  - [ ] ESS > 50 for all variables
  - [ ] obs_sigma posterior ~ true value

- [ ] Run notebook cells 1-11
  - [ ] All imports work
  - [ ] Data generation works
  - [ ] Triangulation and cleaning work
  - [ ] Statistics computation works
  - [ ] Prior config builds
  - [ ] Hyperparameters configure
  - [ ] Both models sample successfully
  - [ ] ESS comparison shows v0.2.1 improvement

- [ ] Verify API compatibility
  - [ ] Old notebooks still work (with warnings)
  - [ ] New API is cleaner and more explicit

---

## I. Known Limitations

1. Single chain (n_chains=1) means no R-hat diagnostics
   - For production, use n_chains=4
   
2. Short run (200 tune, 200 draws) for quick iteration
   - For final results, use 1000 tune, 1000 draws

3. HMM test (M3) not yet enabled in script
   - Waiting for camera model to stabilize first

4. No automatic hyperparameter tuning yet
   - Manual scaling factors (1.5x, 2.0x, etc.)
   - Future: cross-validation or empirical Bayes

---

## J. Summary

**Problem**: HalfNormal prior on obs_sigma caused funnels, hard to interpret, not data-driven

**Solution**: Gamma prior with mode/SD in pixel units, derived from config.obs_noise_std

**Result**: 
- More stable sampling (fewer divergences expected)
- Better initialization (scaled to data)
- Cleaner notebook (focused on pipeline)
- Isolated diagnostics (test script)

**Status**: ✅ Implementation complete, ready for testing
