"""GIMBAL: Geometric Manifolds for Body Articulation and Localization.

This package provides a Bayesian framework for inferring 3D skeletal motion
from multi-camera 2D keypoint observations using Hidden Markov Models.

Current Pipeline: PyMC HMM (v0.1+)
===================================

The main GIMBAL pipeline consists of three stages:

**Stage 1: Collapsed HMM Engine** (`hmm_pytensor`)
    Forward algorithm for marginalizing discrete states in log-space.
    Provides numerically stable HMM inference via collapsed_hmm_loglik().

**Stage 2: Camera Observation Model** (`pymc_model`)
    Combines skeletal kinematics with multi-camera 2D projections.
    Builds joint positions, directions, and observation likelihoods.

**Stage 3: Directional HMM Prior** (`hmm_directional`)
    Adds directional prior over joint orientations with state-dependent
    canonical poses. Uses dot-product energy for computational efficiency.

Quick Start
-----------
>>> import gimbal
>>> from gimbal import DEMO_V0_1_SKELETON
>>> from gimbal.synthetic_data import generate_demo_sequence
>>>
>>> # Generate synthetic data
>>> data = generate_demo_sequence(DEMO_V0_1_SKELETON)
>>>
>>> # Build Stage 1-3 model
>>> import pymc as pm
>>> with pm.Model() as model:
>>>     model, U, x_all, y_pred, log_obs_t = gimbal.build_camera_observation_model(
>>>         y_obs=data.y_observed,
>>>         proj_param=data.camera_proj,
>>>         parents=DEMO_V0_1_SKELETON.parents,
>>>         bone_lengths=DEMO_V0_1_SKELETON.bone_lengths,
>>>     )
>>>     gimbal.add_directional_hmm_prior(U, log_obs_t, S=3)
>>>     # Sample with nutpie or PyMC samplers...

Legacy Torch Implementation
============================

The original PyTorch-based GIMBAL implementation is available in the
`torch_legacy` subpackage. This code is maintained for reference but is
not the primary development path.

See `gimbal.torch_legacy` for the Gibbs sampler and HMC inference code.

See Also
--------
examples/demo_pymc_pipeline.py : Complete PyMC pipeline example
notebook/demo_v0_1_complete.ipynb : Detailed walkthrough with visualizations
plans/v0.1-overview.md : Architecture documentation
"""

# PyMC HMM Pipeline (Stage 1-3)
from .hmm_pytensor import collapsed_hmm_loglik
from .pymc_model import (
    _build_camera_observation_model_full as build_camera_observation_model,
)
from .hmm_directional import add_directional_hmm_prior

# Skeleton and synthetic data utilities
from .skeleton_config import DEMO_V0_1_SKELETON, SkeletonConfig, validate_skeleton
from .synthetic_data import (
    generate_demo_sequence,
    SyntheticDataConfig,
    SyntheticMotionData,
)

# Initialization utilities
from . import fit_params

# Legacy Torch implementation
from . import torch_legacy

__all__ = [
    # Stage 1-3 PyMC pipeline
    "collapsed_hmm_loglik",
    "build_camera_observation_model",
    "add_directional_hmm_prior",
    # Skeleton configuration
    "DEMO_V0_1_SKELETON",
    "SkeletonConfig",
    "validate_skeleton",
    # Synthetic data generation
    "generate_demo_sequence",
    "SyntheticDataConfig",
    "SyntheticMotionData",
    # Submodules
    "fit_params",
    "torch_legacy",
]
