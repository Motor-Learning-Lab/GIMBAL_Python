# GIMBAL: AI Agent Development Guide

## Project Overview

GIMBAL (Geometric Manifolds for Body Articulation and Localization) is a Bayesian framework for inferring 3D skeletal motion from multi-camera 2D keypoint observations using Hidden Markov Models.

**Current Status:** v0.2.1 (Data-driven priors with real motion capture data)

## Three-Stage PyMC Architecture (v0.1+)

The core pipeline consists of three modular stages that must be built in sequence:

1. **Stage 1: Collapsed HMM Engine** (`gimbal/hmm_pytensor.py`)
   - Generic forward algorithm for marginalizing discrete states in log-space
   - Entry point: `collapsed_hmm_loglik(logp_emit, logp_init, logp_trans)`
   - Independent of cameras and kinematics—just HMM math

2. **Stage 2: Camera Observation Model** (`gimbal/pymc_model.py`)
   - Combines skeletal kinematics with multi-camera 2D projections
   - Entry point: `build_camera_observation_model(y_obs, proj_param, parents, bone_lengths)`
   - Returns: `(model, U, x_all, y_pred, log_obs_t)` where `U` is joint directions and `log_obs_t` is per-timestep likelihood
   - **Critical:** This stage exposes `U` (directions) and `log_obs_t` (observation likelihood) for Stage 3

3. **Stage 3: Directional HMM Prior** (`gimbal/hmm_directional.py`)
   - Adds directional prior over joint orientations with state-dependent canonical poses
   - Entry point: `add_directional_hmm_prior(U, log_obs_t, S, **kwargs)`
   - Uses dot-product energy for vMF-like directional emissions
   - **v0.2.1 feature:** Supports data-driven priors via `prior_config` parameter

**Typical usage pattern:**
```python
with pm.Model() as model:
    _, U, x_all, y_pred, log_obs_t = gimbal.build_camera_observation_model(...)
    gimbal.add_directional_hmm_prior(U, log_obs_t, S=3)
    # Now ready for sampling
```

## Package Management: Pixi (Not Conda/Pip Directly)

**Critical:** This project uses [Pixi](https://pixi.sh) for environment management, NOT standard conda/pip.

- Environment file: `pixi.toml` (replaces `environment.yml` or `requirements.txt`)
- Install dependencies: `pixi install`
- Run commands: `pixi run <task>` or `pixi shell` to enter environment
- Available tasks: `pixi run run-demo`, `pixi run notebook`
- When suggesting dependency changes, update `pixi.toml` under `[dependencies]`

## File Naming Conventions (MANDATORY)

**Read `IMPORTANT_FILE_NAMING_CONVENTIONS.md` before creating diagnostic tests.**

For diagnostic work, follow the three-file pattern:
- `test_group_N_<description>.py` - Executable test
- `results_group_N_<description>.json` - Raw metrics (with timestamp, config)
- `report_group_N_<description>.md` - Human-readable analysis

Example: `tests/diagnostics/v0_2_1_divergence/test_group_1_baseline_no_hmm.py`

**Key rules:**
- Each test plan in `plans/` gets a matching directory in `tests/diagnostics/`
- Test outputs go in `results/`, NOT in `tests/`
- Use lowercase with underscores, not camelCase or kebab-case

## Code Organization Patterns

### Skeleton Configuration
- Skeletons defined via `SkeletonConfig(joint_names, parents, bone_lengths)`
- Demo skeleton: `DEMO_V0_1_SKELETON` (pre-configured in `skeleton_config.py`)
- Parents array defines tree structure: `parents[k]` is parent of joint `k`, root is -1

### Synthetic Data Generation
- Use `generate_demo_sequence(skeleton, SyntheticDataConfig(...))` for test data
- Returns `SyntheticMotionData` with ground truth and observations
- Critical fields: `.y_observed` (2D keypoints), `.camera_proj` (projection matrices), `.true_states` (HMM states)

### PyTensor vs NumPy
- PyMC model building uses **PyTensor tensors** (`pt.tensor`), not NumPy arrays
- Shape broadcasting: Use `.dimshuffle()` for adding dimensions (equivalent to NumPy `None` indexing)
- Numerical stability: Always use `pt.logsumexp()` for log-space operations
- Scan operations: Use `scan()` for temporal recursions (see `hmm_pytensor.py`)

## Testing Strategy

### Test Organization
- `tests/unit/` - Fast, isolated module tests
- `tests/integration/` - Multi-module interaction tests
- `tests/smoke/` - Quick validation tests
- `tests/diagnostics/` - Comprehensive diagnostic suites (e.g., divergence analysis)

### Running Tests
- Diagnostic suites: `pixi run python tests/diagnostics/<suite>/test_runner.py`
- Unit tests: `pixi run pytest tests/unit/`
- **Do NOT use pytest fixtures** in diagnostic tests—use explicit configuration classes

## v0.2.1 Data-Driven Priors Pipeline

New feature for building priors from real motion capture data:

1. `triangulate_multi_view()` - Reconstruct 3D from 2D observations
2. `clean_keypoints_2d()` / `clean_keypoints_3d()` - Data cleaning with outlier detection
3. `compute_direction_statistics()` - Analyze directional data
4. `build_priors_from_statistics()` - Convert statistics to PyMC priors

Example: See `examples/demo_v0_2_1_data_driven_priors.py`

## Common Pitfalls

1. **Windows OpenMP conflict:** Set `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at script start
2. **Nutpie sampling:** Requires proper initialization—use `fit_params.initialize_from_observations()` first
3. **Stage order matters:** Must build Stage 2 before Stage 3 (need `U` and `log_obs_t`)
4. **Don't modify legacy:** `gimbal/torch_legacy/` is deprecated—use PyMC pipeline instead
5. **Kappa sharing:** `add_directional_hmm_prior()` has 4 sharing modes—check docstring for current default

## Key Documentation References

- Architecture: `plans/v0.1-overview.md`, `plans/v0.2-overview.md`
- Stage completion reports: `plans/v0.1.{1,2,3}-completion-report.md`
- Algorithm spec: `GIMBAL spec.md` (mathematical formulation)
- Module APIs: Check docstrings in `gimbal/__init__.py` for public exports

## When Making Changes

1. If adding diagnostic tests: Follow `IMPORTANT_FILE_NAMING_CONVENTIONS.md`
2. If modifying core stages: Run corresponding `tests/unit/test_hmm_v0_1_*` tests
3. If changing API: Update `gimbal/__init__.py` exports and docstrings
4. Version planning: Document in `plans/` before implementing
