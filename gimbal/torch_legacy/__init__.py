"""Legacy Torch-based GIMBAL implementation.

This subpackage contains the original PyTorch implementation of GIMBAL,
including the Gibbs sampler and HMC inference routines.

**Status:** This code is maintained for reference and compatibility, but is
not the primary development path. The main GIMBAL pipeline now uses PyMC
(see parent gimbal module for Stage 1-3 PyMC implementation).

Historical Context
------------------
The Torch-based implementation was the original GIMBAL research code that
demonstrated the feasibility of the approach. It includes:

- Full probabilistic model with vMF directional priors
- Gibbs sampler for discrete latent variables (pose states, outlier flags)
- HMC for continuous variables (3D positions, heading)
- Multi-camera projection and observation model

Current Use Cases
-----------------
- Reference implementation for algorithmic comparisons
- Baseline for performance benchmarking
- Historical experiments and research code

Future Development
------------------
New features and improvements are being implemented in the PyMC pipeline.
This Torch code will not receive major updates but will be kept functional.

See Also
--------
gimbal.hmm_pytensor : Stage 1 - Collapsed HMM engine (PyMC)
gimbal.pymc_model : Stage 2 - Camera observation model (PyMC)
gimbal.hmm_directional : Stage 3 - Directional HMM prior (PyMC)
"""

from . import model, inference, camera  # noqa: F401

__all__ = ["model", "inference", "camera"]
