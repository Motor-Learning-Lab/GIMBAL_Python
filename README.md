# GIMBAL Python

PyMC implementation of **GIMBAL** (General Inference for Multimodal Biological Articulated Locomotion) - a Bayesian framework for inferring 3D skeletal motion from multi-camera 2D keypoint observations.

## Current Status: v0.1 Complete ✅

The project has completed its three-phase v0.1 implementation:

- **v0.1.1:** Collapsed HMM engine with forward algorithm in log-space
- **v0.1.2:** Camera observation model with kinematics and 2D projections  
- **v0.1.3:** Directional HMM prior over joint directions

See `plans/v0.1-overview.md` for architecture details and `plans/v0.1.{1,2,3}-completion-report.md` for implementation documentation.

## Repository Structure

```
gimbal/                    # Core library modules
├── hmm_pytensor.py       # v0.1.1: Collapsed HMM engine
├── pymc_model.py         # v0.1.2: Camera observation model
├── hmm_directional.py    # v0.1.3: Directional HMM prior
├── camera.py             # Camera projection utilities
├── fit_params.py         # Initialization from observations (DLT, Anipose)
├── pymc_utils.py         # PyMC helper functions
└── pymc_distributions.py # Custom distributions (VonMisesFisher)

tests/                     # All test files
├── test_hmm_v0_1_*.py    # Phase-specific tests
├── test_v0_1_3_directional_hmm.py  # Comprehensive v0.1.3 test suite
└── run_v0_1_3_tests.py   # Test runner

notebook/                  # Interactive demonstrations
├── demo_v0_1_complete.ipynb  # Full v0.1.1-1.3 integration demo
└── demo_pymc_*.ipynb     # Camera model demos

plans/                     # Design docs and completion reports
├── v0.1.{1,2,3}-detailed-spec.md
├── v0.1.{1,2,3}-completion-report.md
└── v0.2-overview.md      # Next phase roadmap

examples/                  # Standalone examples
└── run_gimbal_demo.py    # Original demo
```

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
