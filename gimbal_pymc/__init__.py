"""GIMBAL: Geometric Manifolds for Body Articulation and Localization.
...
"""

# Stage 1-3 PyMC pipeline
from gimbal_pymc.hmm.pytensor import collapsed_hmm_loglik
from gimbal_pymc.hmm.directional import add_directional_hmm_prior
from gimbal_pymc.cameras.observation_model import (
    _build_camera_observation_model_full as build_camera_observation_model,
)

# v0.2.1: Data-driven priors pipeline
from gimbal_pymc.cameras.triangulation import triangulate_multi_view
from gimbal_pymc.data_cleaning.cleaning import (
    CleaningConfig,
    clean_keypoints_2d,
    clean_keypoints_3d,
)
from gimbal_pymc.joints.direction_statistics import compute_direction_statistics
from gimbal_pymc.priors.building import (
    build_priors_from_statistics,
    get_gamma_shape_rate,
)

# Skeleton and synthetic data utilities
from gimbal_pymc.skeleton.config import (
    DEMO_V0_1_SKELETON,
    L00_SKELETON,
    SkeletonConfig,
    validate_skeleton,
)
from gimbal_pymc.skeleton.synthetic_data import (
    generate_demo_sequence,
    SyntheticDataConfig,
    SyntheticMotionData,
)

# Initialization utilities (kept as submodule for gp.fit_params.* usage)
from gimbal_pymc.priors import fit_params
from gimbal_pymc.priors.fit_params import initialize_from_observations_dlt

# Submodules exposed at top level (moved — import from new locations)
from gimbal_pymc.skeleton import metrics as skeleton_metrics
from gimbal_pymc.skeleton import visualization as skeleton_visualization
from gimbal_pymc.cameras import identifiability

# Legacy Torch implementation
from gimbal_pymc import torch_legacy

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
    # Initialization
    "initialize_from_observations_dlt",
    # Submodules
    "fit_params",
    "skeleton_metrics",
    "skeleton_visualization",
    "identifiability",
    "torch_legacy",
]