# GIMBAL Python

**GIMBAL** (Geometric Manifolds for Body Articulation and Localization) is a Bayesian framework for inferring 3D skeletal motion from multi-camera 2D keypoint observations using Hidden Markov Models.

## Current Status: v0.2.2 (Module Reorganization) ✅

**v0.1 Complete ✅** - The PyMC HMM pipeline (Stage 1-3) is fully implemented and tested.

**v0.2.1 Complete ✅** - Data-driven priors from real motion capture data, comprehensive diagnostic framework.

**v0.2.2 P0 Complete ✅** - Package restructured into 6 domain-specific subpackages with cleaner naming. All imports use explicit subpackage paths.

---

## PyMC HMM Pipeline (v0.1+) — Primary Implementation

The main GIMBAL pipeline consists of three stages:

### Stage 1: Collapsed HMM Engine (`gimbal_pymc/hmm/engine.py`)
Forward algorithm for marginalizing discrete states in log-space. Provides numerically stable HMM inference via `collapsed_hmm_loglik()`.

### Stage 2: Camera Observation Model (`gimbal_pymc/hmm/observation_model.py`)
Combines skeletal kinematics with multi-camera 2D projections. Builds joint positions, directions, and observation likelihoods via `build_camera_observation_model()`.

### Stage 3: Directional HMM Prior (`gimbal_pymc/hmm/directional_prior.py`)
Adds directional prior over joint orientations with state-dependent canonical poses. Uses dot-product energy for computational efficiency via `add_directional_hmm_prior()`.

### Quick Start

```python
import pymc as pm
from gimbal_pymc.skeleton.synthetic_data import generate_demo_sequence, SyntheticDataConfig
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.priors.initialization import initialize_from_observations_dlt
from gimbal_pymc.hmm.observation_model import build_camera_observation_model
from gimbal_pymc.hmm.directional_prior import add_directional_hmm_prior

# Generate synthetic data
config = SyntheticDataConfig(T=20, C=2, S=2)
data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

# Initialize from observations
init_result = initialize_from_observations_dlt(
    data.y_observed, data.camera_proj, 
    DEMO_V0_1_SKELETON.parents
)

# Build complete model
with pm.Model() as model:
    build_camera_observation_model(
        y_observed=data.y_observed,
        camera_proj=data.camera_proj,
        parents=DEMO_V0_1_SKELETON.parents,
        init_result=init_result
    )
    # Access U and log_obs_t from model
    U = model["U"]
    log_obs_t = model["log_obs_t"]
    
    add_directional_hmm_prior(U, log_obs_t, S=2)
    
    # Sample with nutpie or PyMC samplers
    # idata = pm.sample(...)
```

See `examples/demo_v0_2_0_pymc_pipeline.py` and `examples/demo_v0_2_1_data_driven_priors.py` for complete runnable examples.

---

## v0.2.1: Data-Driven Priors from Motion Capture

**New capabilities:**
- **3D Triangulation** — Reconstruct 3D joint positions from 2D multi-view observations
- **Directional Statistics** — Analyze joint orientation distributions across motion sequences
- **Data-Driven Priors** — Build priors from real motion capture data instead of synthetic distributions

**Key functions:**
- `gimbal_pymc.cameras.triangulation.triangulate_multi_view()` — Multi-camera 3D reconstruction via Direct Linear Transform (DLT)
- `gimbal_pymc.data_cleaning.cleaning.clean_keypoints_2d()` / `clean_keypoints_3d()` — Robust data cleaning with outlier detection
- `gimbal_pymc.joints.statistics.compute_direction_statistics()` — Compute mean directions and concentration parameters
- `gimbal_pymc.priors.building.build_priors_from_statistics()` — Convert statistics to PyMC-compatible priors

**Example:**
```python
from gimbal_pymc.cameras.triangulation import triangulate_multi_view
from gimbal_pymc.data_cleaning.cleaning import CleaningConfig, clean_keypoints_3d
from gimbal_pymc.joints.statistics import compute_direction_statistics
from gimbal_pymc.priors.building import build_priors_from_statistics
from gimbal_pymc.hmm.observation_model import build_camera_observation_model
from gimbal_pymc.hmm.directional_prior import add_directional_hmm_prior
import pymc as pm

# Triangulate to 3D
x_3d = triangulate_multi_view(y_2d, proj_matrices)

# Clean outliers
config = CleaningConfig()
x_clean, valid, use_stats, summary = clean_keypoints_3d(x_3d, parents, config)

# Compute directional statistics
statistics = compute_direction_statistics(x_clean, parents, use_stats, joint_names)

# Build priors
priors = build_priors_from_statistics(statistics, joint_names)

# Use in HMM
with pm.Model() as model:
    build_camera_observation_model(...)
    U = model["U"]
    log_obs_t = model["log_obs_t"]
    add_directional_hmm_prior(U, log_obs_t, S=3, prior_config=priors)
```

See `examples/demo_v0_2_1_data_driven_priors.py` for a complete walkthrough.

---

## Repository Structure

```
gimbal_pymc/                   # Core library (v0.2.2: subpackage organization)
├── __init__.py               # Subpackage exports only (no convenience exports)
├── hmm/                      # Hidden Markov Model components
│   ├── engine.py             # Stage 1: Collapsed HMM forward algorithm
│   ├── observation_model.py  # Stage 2: Camera observation model
│   ├── directional_prior.py  # Stage 3: Directional HMM prior
│   └── gaussian_example.py   # Gaussian HMM example
├── cameras/                  # Multi-view geometry
│   ├── projection.py         # Camera projection utilities
│   ├── triangulation.py      # DLT 3D reconstruction
│   └── identifiability.py    # Camera configuration validation
├── skeleton/                 # Articulated body kinematics
│   ├── config.py             # Skeleton definitions (DEMO_V0_1_SKELETON)
│   ├── synthetic_data.py     # Synthetic motion generation
│   ├── metrics.py            # Quality metrics
│   └── visualization.py      # Plotting utilities
├── joints/                   # Directional statistics
│   ├── statistics.py         # vMF parameter estimation
│   └── distributions.py      # vMF distribution for PyMC
├── priors/                   # Prior construction & initialization
│   ├── building.py           # Prior construction from data (v0.2.1)
│   ├── initialization.py     # Initialization from observations (DLT)
│   └── utils.py              # Distribution helpers
└── data_cleaning/            # Data preprocessing (v0.2.1)
    └── cleaning.py           # 2D/3D keypoint cleaning

tests/                         # Test suite
├── test_stage1_collapsed_hmm.py    # Stage 1: Collapsed HMM tests
├── test_stage2_camera_model.py     # Stage 2: Camera observation model tests
├── integration/
│   ├── test_stage3_directional_hmm.py  # Stage 3: Directional HMM tests
│   └── test_data_driven_priors.py      # Data-driven priors pipeline tests
├── unit/                               # Unit tests (DLT, initialization, utilities)
├── smoke/                              # Quick validation tests
├── pipeline/                           # End-to-end pipeline tests
│   ├── test_synthetic_data_generator.py
│   └── configs/v0.2.1/                 # Dataset configurations
└── diagnostics/                        # Comprehensive diagnostic suites

notebook/                      # Interactive demonstrations
├── demo_v0_1_complete.ipynb  # Full v0.1 integration demo
└── demo_v0_2_1_data_driven_priors.ipynb

examples/                      # Standalone examples
├── demo_v0_2_0_pymc_pipeline.py     # PyMC pipeline demo
└── demo_v0_2_1_data_driven_priors.py # Data-driven priors demo

plans/                         # Design documents and roadmaps
├── v0.1-overview.md          # v0.1 architecture
├── v0.1.{1,2,3}-completion-report.md
├── v0.2-overview.md          # v0.2 roadmap
└── v0.2.{0,1}-completion-report.md
```

## Installation

GIMBAL uses [Pixi](https://pixi.sh) for environment management. Pixi automatically handles all dependencies including Python 3.11, PyMC, PyTorch, and development tools.

### Install Pixi

**Windows (PowerShell):**
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

**Linux/macOS:**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Install Dependencies and Package

From the repository root:

```powershell
# 1. Install all dependencies
pixi install

# 2. Install gimbal_pymc in editable mode
pixi run install-dev
```

**Note:** `install-dev` installs the `gimbal_pymc` package in editable mode so tests and scripts can import it.

### Verify Installation

```powershell
pixi run check-setup
# Should output: ✓ Environment OK
```

---

## Sanity Checks

Quick validation after installation:

```powershell
# Quick smoke tests (~30 seconds)
pixi run test-smoke

# Integration tests (~2 minutes)
pixi run test-integration

# Pipeline tests (~10 seconds)
pixi run test-pipeline
```

---

## Testing

### Test Taxonomy

Tests are organized by purpose (defined in `pytest.ini`):

- **Smoke tests** (`tests/smoke/`) — Quick API validation (~30s)
- **Unit tests** (`tests/unit/`) — Individual module tests
- **Integration tests** (`tests/integration/`) — Multi-module features (Stage 3, priors)
- **Pipeline tests** (`tests/pipeline/`) — Synthetic data generation and validation
- **Diagnostics** (`tests/diagnostics/`) — Comprehensive analysis suites (opt-in)

### Running Tests

```powershell
# Run normal tests (excludes diagnostics by default)
pixi run test-all

# Run specific test suites
pixi run test-smoke        # Quick validation
pixi run test-unit         # Unit tests only
pixi run test-integration  # Integration features
pixi run test-pipeline     # Synthetic data pipeline

# Run diagnostics (opt-in, comprehensive analysis)
pixi run test-diagnostics
```

**Note:** By default, `test-all` excludes `diagnostics/` via `pytest.ini` configuration.

See `tests/README.md` for detailed test organization and documentation.

## Running the Code

### Available Pixi Tasks

Run any task with `pixi run <task-name>`:

**Setup & Demos:**
- `check-setup` — Verify environment configuration
- `demo-pymc` — v0.2.0 PyMC pipeline demo
- `demo-priors` — v0.2.1 data-driven priors demo
- `notebook` — Launch JupyterLab

**Testing:**
- `test-p0` — P0-scoped tests (33 tests: core reorganization)
- `test-smoke` — Quick validation (~30s)
- `test-unit` — Unit tests
- `test-integration` — Integration tests (Stage 3, priors)
- `test-pipeline` — Synthetic data pipeline tests
- `test-all` — All tests (60 tests: includes pipeline)
- `test-diagnostics` — Comprehensive diagnostic suites (opt-in)

**Data Generation:**
- `generate-datasets` — Generate synthetic datasets from configs
- `pipeline-clean` — Stage B: Clean 2D keypoints (L00 dataset)
- `pipeline-triangulate` — Stage C: Triangulate to 3D (L00 dataset)
- `pipeline-full` — Complete L00 pipeline (stages A-J)

### PyMC Pipeline Demos

**Complete v0.1 PyMC pipeline:**
```powershell
pixi run demo-pymc
```

**Data-driven priors (v0.2.1):**
```powershell
pixi run demo-priors
```

### Interactive Notebooks

```powershell
pixi run notebook
# Then open notebook/demo_v0_1_complete.ipynb
```

Full walkthrough with visualizations of the three-stage PyMC pipeline.

### Running Tests

```powershell
pixi run test-unit
pixi run test-pipeline
pixi run test-all
```

### Running the Pipeline

To run the complete data processing pipeline on the L00_minimal dataset:

```powershell
pixi run pipeline-full
```

Or run individual stages:
```powershell
pixi run pipeline-clean         # Stage B: Clean 2D keypoints
pixi run pipeline-triangulate   # Stage C: 3D triangulation
```

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
