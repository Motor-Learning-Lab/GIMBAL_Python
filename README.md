# GIMBAL Python

**GIMBAL** (Geometric Manifolds for Body Articulation and Localization) is a Bayesian framework for inferring 3D skeletal motion from multi-camera 2D keypoint observations using Hidden Markov Models.

## Current Status: v0.2.1 (Data-Driven Priors) ✅

**v0.1 Complete ✅** - The PyMC HMM pipeline (Stage 1-3) is fully implemented and tested.

**v0.2.1 Complete ✅** - Data-driven priors from real motion capture data, comprehensive diagnostic framework.

---

## PyMC HMM Pipeline (v0.1+) — Primary Implementation

The main GIMBAL pipeline consists of three stages:

### Stage 1: Collapsed HMM Engine (`hmm_pytensor.py`)
Forward algorithm for marginalizing discrete states in log-space. Provides numerically stable HMM inference via `collapsed_hmm_loglik()`.

### Stage 2: Camera Observation Model (`pymc_model.py`)
Combines skeletal kinematics with multi-camera 2D projections. Builds joint positions, directions, and observation likelihoods via `build_camera_observation_model()`.

### Stage 3: Directional HMM Prior (`hmm_directional.py`)
Adds directional prior over joint orientations with state-dependent canonical poses. Uses dot-product energy for computational efficiency via `add_directional_hmm_prior()`.

### Quick Start

```python
import gimbal
from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig
import pymc as pm

# Generate synthetic data
config = SyntheticDataConfig(T=20, C=2, S=2)
data = gimbal.generate_demo_sequence(DEMO_V0_1_SKELETON, config)

# Build complete model
with pm.Model() as model:
    _, U, x_all, y_pred, log_obs_t = gimbal.build_camera_observation_model(
        y_obs=data.y_observed,
        proj_param=data.camera_proj,
        parents=DEMO_V0_1_SKELETON.parents,
        bone_lengths=DEMO_V0_1_SKELETON.bone_lengths,
    )
    gimbal.add_directional_hmm_prior(U, log_obs_t, S=2)
    
    # Sample with nutpie or PyMC samplers
    # idata = pm.sample(...)
```

See `examples/demo_v0_1_complete.ipynb` or `notebook/demo_stage3_complete.ipynb` for complete runnable examples.

---

## v0.2.1: Data-Driven Priors from Motion Capture

**New capabilities:**
- **3D Triangulation** — Reconstruct 3D joint positions from 2D multi-view observations
- **Directional Statistics** — Analyze joint orientation distributions across motion sequences
- **Data-Driven Priors** — Build priors from real motion capture data instead of synthetic distributions

**Key functions:**
- `triangulate_multi_view()` — Multi-camera 3D reconstruction via Direct Linear Transform (DLT)
- `clean_keypoints_2d()` / `clean_keypoints_3d()` — Robust data cleaning with outlier detection
- `compute_direction_statistics()` — Compute mean directions and concentration parameters
- `build_priors_from_statistics()` — Convert statistics to PyMC-compatible priors

**Example:**
```python
import gimbal

# Load real motion capture data (2D keypoints)
y_2d = load_observations()  # Shape: (T, C, N, 2)

# Triangulate to 3D
x_3d, valid_triangulations = gimbal.triangulate_multi_view(y_2d, proj_matrices)

# Clean outliers
x_clean = gimbal.clean_keypoints_3d(x_3d, threshold=0.05)

# Compute directional statistics
statistics = gimbal.compute_direction_statistics(x_clean, skeleton)

# Build priors
priors = gimbal.build_priors_from_statistics(statistics, skeleton)

# Use in HMM
with pm.Model() as model:
    _, U, x_all, y_pred, log_obs_t = gimbal.build_camera_observation_model(...)
    gimbal.add_directional_hmm_prior(U, log_obs_t, S=3, prior_config=priors)
```

See `examples/demo_v0_2_1_data_driven_priors.py` for a complete walkthrough.

---

## Legacy Torch Implementation

The original PyTorch-based GIMBAL implementation (Gibbs sampler + HMC) is available in `gimbal.torch_legacy`. This code is maintained for reference and compatibility but is not the primary development path.

See `gimbal/torch_legacy/README.md` for details and `examples/run_gimbal_demo.py` for usage.

---## Repository Structure

```
gimbal/                        # Core library modules
├── __init__.py               # Public API (imports Stage 1-3 functions)
├── skeleton_config.py        # Skeleton definitions (DEMO_V0_1_SKELETON)
├── synthetic_data.py         # Synthetic data generation utilities
│
├── hmm_pytensor.py           # Stage 1: Collapsed HMM engine
├── pymc_model.py             # Stage 2: Camera observation model
├── hmm_directional.py        # Stage 3: Directional HMM prior
│
├── fit_params.py             # Initialization from observations (DLT)
├── pymc_utils.py             # PyMC helper functions
├── pymc_distributions.py     # Custom distributions (experimental vMF)
│
└── torch_legacy/             # Legacy PyTorch implementation
    ├── model.py              # Torch probabilistic model
    ├── inference.py          # Gibbs sampler + HMC
    └── camera.py             # Torch camera utilities

tests/                         # Test suite
├── test_hmm_v0_1_*.py        # Stage-specific tests
├── test_v0_1_3_directional_hmm.py  # Comprehensive Stage 3 tests
├── test_demo_v0_1_smoke.py   # Full pipeline smoke test
└── run_v0_1_3_tests.py       # Test runner

notebook/                      # Interactive demonstrations
├── demo_v0_1_complete.ipynb  # Full v0.1 integration demo
└── demo_pymc_*.ipynb         # Component demos

examples/                      # Standalone examples
├── demo_pymc_pipeline.py     # PyMC "hello world" demo (NEW)
└── run_gimbal_demo.py        # Legacy Torch demo

plans/                         # Design documents and roadmaps
├── v0.1-overview.md          # v0.1 architecture
├── v0.1.{1,2,3}-completion-report.md
├── v0.2-overview.md          # v0.2 roadmap (priors, AIST++, PCA)
└── v0.2.0-detailed-spec.md   # Current phase (restructuring)
```

## Installation

GIMBAL requires Python 3.10+ and PyMC. We recommend using a virtual environment:

```powershell
# Create and activate virtual environment
python -m venv .venv
. .venv\Scripts\Activate.ps1

# Install PyMC and dependencies
pip install pymc nutpie

# For legacy Torch demo (optional)
pip install torch scikit-learn
```

## Running the Demos

### PyMC Pipeline Demo (Recommended)

```powershell
python examples/demo_pymc_pipeline.py
```

This demonstrates the complete v0.1 PyMC HMM pipeline in ~50 lines of code.

### Legacy Torch Demo

```powershell
python -m examples.run_gimbal_demo
```

Runs the original PyTorch Gibbs sampler implementation.

### Interactive Notebooks

```powershell
jupyter notebook notebook/demo_v0_1_complete.ipynb
```

Full walkthrough with visualizations of the three-stage PyMC pipeline.

---

## Development Roadmap

**v0.1 (Complete)** — Core PyMC HMM pipeline  
**v0.2.1 (Complete)** — Data-driven priors from motion capture, comprehensive diagnostics  
**v0.3.0** — Anatomical priors, AIST++ dataset integration, PCA-based priors  

See `plans/v0.2-overview.md` for detailed roadmap.

---

## Documentation

- **Architecture:** `plans/v0.1-overview.md`
- **Implementation Reports:** `plans/v0.1.{1,2,3}-completion-report.md`
- **API Reference:** Module docstrings in `gimbal/`
- **Examples:** `examples/` and `notebook/`

---

## Citation

If you use GIMBAL in your research, please cite:

```bibtex
@software{gimbal2024,
  title = {GIMBAL: General Inference for Multimodal Biological Articulated Locomotion},
  author = {Motor Learning Lab},
  year = {2024},
  url = {https://github.com/Motor-Learning-Lab/GIMBAL_Python}
}
```
