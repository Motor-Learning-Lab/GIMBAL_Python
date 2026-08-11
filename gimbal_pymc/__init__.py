"""GIMBAL: Geometric Manifolds for Body Articulation and Localization.

This package provides a Bayesian framework for inferring 3D skeletal motion
from multi-camera 2D keypoint observations using Hidden Markov Models.

v0.2.2 Module Organization
===========================

The package is organized into focused subpackages:

**hmm/** - Hidden Markov Model components
    - engine: Collapsed HMM forward algorithm (Stage 1)
    - directional_prior: Directional HMM prior (Stage 3)
    - observation_model: Camera observation model builder (Stage 2)
    - gaussian_example: Example Gaussian HMM implementation

**cameras/** - Camera projection and triangulation
    - projection: Camera projection utilities
    - triangulation: DLT and Anipose triangulation
    - identifiability: Camera configuration validation

**skeleton/** - Skeletal structure and motion
    - config: Skeleton configuration and validation
    - synthetic_data: Synthetic data generation
    - metrics: Quality metrics for skeletal motion
    - visualization: Visualization utilities

**joints/** - Joint-level operations
    - distributions: VonMisesFisher and other joint distributions
    - statistics: Direction distribution analysis

**priors/** - Prior specification and initialization
    - building: Prior distribution builders
    - initialization: Parameter initialization strategies
    - utils: Nutpie integration utilities

**data_cleaning/** - Data preprocessing (v0.2.1)
    - cleaning: 2D/3D keypoint cleaning

Quick Start
-----------
>>> import gimbal_pymc as gp
>>> import pymc as pm
>>>
>>> # Generate synthetic data
>>> data = gp.skeleton.synthetic_data.generate_demo_sequence(
>>>     gp.skeleton.config.DEMO_V0_1_SKELETON
>>> )
>>>
>>> # Initialize from observations
>>> init_result = gp.priors.initialization.initialize_from_observations_dlt(
>>>     y_observed=data.y_observed,
>>>     camera_proj=data.camera_proj,
>>>     parents=gp.skeleton.config.DEMO_V0_1_SKELETON.parents,
>>> )
>>>
>>> # Build Stage 2 model
>>> with pm.Model() as model:
>>>     gp.hmm.observation_model.build_camera_observation_model(
>>>         y_observed=data.y_observed,
>>>         camera_proj=data.camera_proj,
>>>         parents=gp.skeleton.config.DEMO_V0_1_SKELETON.parents,
>>>         init_result=init_result,
>>>         use_mixture=False,
>>>     )
>>>     # Access model variables via dictionary
>>>     U = model["U"]
>>>     log_obs_t = model["log_obs_t"]
>>>
>>>     # Add Stage 3 directional HMM prior
>>>     gp.hmm.directional_prior.add_directional_hmm_prior(U, log_obs_t, S=3)
>>>
>>>     # Sample with PyMC or nutpie samplers
>>>     # trace = pm.sample()

See Also
--------
examples/demo_v0_2_1_data_driven_priors.py : Complete PyMC pipeline example
notebook/demo_v0_2_1_data_driven_priors.ipynb : Detailed walkthrough
plans/v0.2.2/v0.2.2_overview.md : v0.2.2 architecture documentation
"""

# Subpackage imports - use explicit subpackage paths for all imports
from . import hmm
from . import cameras
from . import skeleton
from . import joints
from . import priors
from . import data_cleaning

__all__ = [
    "hmm",
    "cameras",
    "skeleton",
    "joints",
    "priors",
    "data_cleaning",
]
