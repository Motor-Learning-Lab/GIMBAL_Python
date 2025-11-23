# GIMBAL Python (Prototype)

This repository contains a prototype implementation of the **GIMBAL**
algorithm (Geometric Manifolds for Body Articulation and Localization)
for 3D pose estimation from multi-view 2D keypoints.

The implementation follows the high-level specification in `GIMBAL spec.md`.

## Package structure

- `gimbal/`
  - `__init__.py` – package namespace
  - `camera.py` – camera projection utilities
  - `model.py` – probabilistic model and log-densities
  - `inference.py` – MCMC (HMC + Gibbs) inference algorithms
  - `fit_params.py` – parameter initialization from ground-truth data
- `examples/run_gimbal_demo.py` – small synthetic demo

## Installation

Create and activate a virtual environment (recommended), then install
PyTorch and scikit-learn, for example:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install torch scikit-learn
```

## Running the demo

From the repository root:

```powershell
. .venv\Scripts\Activate.ps1
python -m examples.run_gimbal_demo
```

This will simulate a tiny dataset, run a few iterations of the GIMBAL
sampler, and print the posterior mean 3D trajectories for the first
joint over the first few frames.
