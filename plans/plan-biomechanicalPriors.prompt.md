# Plan: Biomechanical Priors for PyMC GIMBAL

Add three biomechanical constraints to `gimbal/pymc_model.py` enabled by default: global temporal smoothness for direction continuity, learned parent-relative canonical directions with per-joint concentration, and inference without explicit angle limits. All features integrate with nutpie sampling.

## Steps

1. **Extend `KNOWN_HYPERPARAMS` in [`gimbal/pymc_model.py`](gimbal/pymc_model.py:281-288)** — Add `temporal_smoothness_sigma` (default 0.1) and `canonical_kappa_sigma` (default 10.0)

2. **Add temporal smoothness prior after line 348** — Insert loop over `u_all` creating `pm.Potential(f"temporal_smooth_{k+1}", -0.5 * sum((u_k[1:] - u_k[:-1])**2) / temporal_smoothness_sigma²)` with global smoothness parameter, gated by `use_direction_smoothing` kwarg (default True)

3. **Add parent-relative canonical directions with learned parameters** — For each joint k, sample `canonical_direction_{k} ~ VonMisesFisher(mu=[0,0,1], kappa=large)` as prior mean direction relative to parent, then sample `kappa_canonical_{k} ~ HalfNormal(sigma=canonical_kappa_sigma)` for concentration, compute `dot_products = sum(u_k * canonical_direction_k)` over time, add `pm.Potential(f"canonical_dir_{k+1}", kappa_k * dot_products)`, gated by `use_canonical_directions` kwarg (default True)

4. **Update function signature in [`gimbal/pymc_model.py`](gimbal/pymc_model.py)** — Add kwargs `use_direction_smoothing=True` and `use_canonical_directions=True` to `build_camera_observation_model`, update docstring Parameters section with detailed descriptions, add Examples demonstrating how to disable priors if needed

5. **Create visualization module `gimbal/viz_biomech.py`** — Implement `plot_direction_smoothness(trace, parents)` showing per-joint temporal ∆u magnitude time series and histograms, `plot_canonical_alignment(trace, parents)` showing per-joint learned canonical directions as 3D arrows and kappa posteriors as distributions, `plot_biomech_summary(trace, parents)` combining both panels with skeleton overlay

6. **Add new demo notebook `notebook/demo_pymc_biomechanical.ipynb`** — Create standalone demonstration of biomechanical priors: (1) generate synthetic data with known canonical directions and smoothness, (2) fit model with priors enabled (default), (3) fit model with priors disabled for comparison, (4) visualize learned canonical directions and kappa values using `gimbal.viz_biomech`, (5) show posterior predictive checks for smoothness

7. **Update existing demo notebooks** — In [`demo_pymc_camera_simple.ipynb`](notebook/demo_pymc_camera_simple.ipynb) and `demo_pymc_camera_full.ipynb`, add `use_direction_smoothing=False, use_canonical_directions=False` to `build_camera_observation_model()` calls to preserve existing behavior without biomechanical priors

8. **Create `test_pymc_biomech.py` at repo root** — Add unit tests: (1) `test_temporal_smoothness_reduces_jitter` validates smoothness prior effect, (2) `test_canonical_directions_learned` checks that learned canonical directions concentrate around true values in synthetic data, (3) `test_kappa_posteriors_reasonable` validates learned concentration parameters, (4) `test_biomech_nutpie_compatibility` confirms priors work with nutpie compilation, (5) `test_priors_can_be_disabled` ensures backward compatibility flags work

## Further Considerations

1. **Parent-relative coordinate transform?** — To make canonical directions parent-relative, need to transform `u_k` into parent bone's coordinate frame before computing dot product with canonical direction. Should we implement full rotation matrix transform, or use simpler approach assuming parent direction defines local z-axis? Recommend rotation matrix for generality.

2. **Canonical direction prior strength?** — Currently using `VonMisesFisher(mu=[0,0,1], kappa=large)` as weak prior on canonical directions. Should kappa be configurable hyperparameter, or hard-coded? Recommend `canonical_direction_prior_kappa` hyperparameter (default 10.0) for user control over how much canonical directions can deviate from default.
