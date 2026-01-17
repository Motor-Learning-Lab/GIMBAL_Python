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
    - utils: Camera projection utilities
    - triangulation: DLT and Anipose triangulation

**skeleton/** - Skeletal structure and motion
    - config: Skeleton configuration and validation
    - synthetic_data: Synthetic data generation
    - metrics: Quality metrics for skeletal motion
    - visualization: Visualization utilities

**joints/** - Joint-level operations
    - forward_kinematics: Joint position computation
    - directions: Direction vector utilities

**priors/** - Prior specification and initialization
    - building: Prior distribution builders
    - initialization: Parameter initialization strategies
    - utils: Nutpie integration utilities

**data_cleaning/** - Data preprocessing (v0.2.1)
    - keypoint_cleaning: 2D/3D keypoint cleaning
    - direction_statistics: Direction distribution analysis

Quick Start
-----------
>>> import gimbal_pymc as gp
>>> from gimbal_pymc import DEMO_V0_1_SKELETON
>>> from gimbal_pymc import generate_demo_sequence
>>>
>>> # Generate synthetic data
>>> data = generate_demo_sequence(DEMO_V0_1_SKELETON)
>>>
>>> # Build Stage 1-3 model
>>> import pymc as pm
>>> with pm.Model() as model:
>>>     model, U, x_all, y_pred, log_obs_t = gp.build_camera_observation_model(
>>>         y_obs=data.y_observed,
>>>         proj_param=data.camera_proj,
>>>         parents=DEMO_V0_1_SKELETON.parents,
>>>         bone_lengths=DEMO_V0_1_SKELETON.bone_lengths,
>>>     )
>>>     gp.add_directional_hmm_prior(U, log_obs_t, S=3)
>>>     # Sample with nutpie or PyMC samplers...

See Also
--------
examples/demo_v0_2_1_data_driven_priors.py : Complete PyMC pipeline example
notebook/demo_v0_2_1_data_driven_priors.ipynb : Detailed walkthrough
plans/v0.2.2_overview.md : v0.2.2 architecture documentation
"""

# Stage 1-3: Core PyMC HMM Pipeline
from .hmm.engine import collapsed_hmm_loglik
from .hmm.observation_model import (
    _build_camera_observation_model_full as build_camera_observation_model,
)
from .hmm.directional_prior import add_directional_hmm_prior

# v0.2.1: Data-driven priors pipeline
from .cameras.triangulation import triangulate_multi_view
from .data_cleaning.cleaning import (
    CleaningConfig,
    clean_keypoints_2d,
    clean_keypoints_3d,
)
from .joints.statistics import compute_direction_statistics
from .priors.building import (
    build_priors_from_statistics,
    get_gamma_shape_rate,
)

# Skeleton configuration and synthetic data
from .skeleton.config import (
    DEMO_V0_1_SKELETON,
    L00_SKELETON,
    SkeletonConfig,
    validate_skeleton,
)
from .skeleton.synthetic_data import (
    generate_demo_sequence,
    SyntheticDataConfig,
    SyntheticMotionData,
)

# Subpackage exports for direct access
from . import hmm
from . import cameras
from . import skeleton
from . import joints
from . import priors
from . import data_cleaning

# Backward compatibility: Keep old module names as aliases
from .priors import initialization as fit_params
from .skeleton import metrics as skeleton_metrics
from .skeleton import visualization as skeleton_visualization  
from . import identifiability

__all__ = [
    # Stage 1-3 PyMC pipeline
    "collapsed_hmm_loglik",
    "build_camera_observation_model",
    "add_directional_hmm_prior",
    # v0.2.1: Data-driven priors
    "triangulate_multi_view",
    "CleaningConfig",
    "clean_keypoints_2d",
    "clean_keypoints_3d",
    "compute_direction_statistics",
    "build_priors_from_statistics",
    "get_gamma_shape_rate",
    # Skeleton configuration
    "DEMO_V0_1_SKELETON",
    "L00_SKELETON",
    "SkeletonConfig",
    "validate_skeleton",
    # Synthetic data generation
    "generate_demo_sequence",
    "SyntheticDataConfig",
    "SyntheticMotionData",
    # Subpackages
    "hmm",
    "cameras",
    "skeleton",
    "joints",
    "priors",
    "data_cleaning",
    # Backward compatibility
    "fit_params",
    "skeleton_metrics",
    "skeleton_visualization",
    "identifiability",
]
