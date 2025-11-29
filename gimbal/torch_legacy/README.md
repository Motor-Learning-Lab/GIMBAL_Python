# Legacy Torch GIMBAL Implementation

This directory contains the original PyTorch-based GIMBAL implementation.

## Status

**⚠️ Legacy / Maintenance Mode**

This code is kept for:
- Historical reference
- Compatibility with existing experiments
- Performance comparisons

**Primary development now uses the PyMC pipeline** (see parent `gimbal/` directory).

## Contents

- `model.py` — Probabilistic model definitions and log-densities
- `inference.py` — Gibbs sampler and HMC inference routines
- `camera.py` — Camera projection utilities

## Usage

```python
from gimbal.torch_legacy.camera import project_points
from gimbal.torch_legacy.fit_params import build_gimbal_parameters
from gimbal.torch_legacy.inference import run_gibbs_sampler

# ... (see examples/run_gimbal_demo.py for full example)
```

## Migration to PyMC

For new projects, use the PyMC pipeline:

```python
import gimbal

# Stage 1-3 PyMC pipeline
model, U, x_all, y_pred, log_obs_t = gimbal.build_camera_observation_model(...)
gimbal.add_directional_hmm_prior(U, log_obs_t, S=3)
```

See `examples/demo_pymc_pipeline.py` for a complete example.
