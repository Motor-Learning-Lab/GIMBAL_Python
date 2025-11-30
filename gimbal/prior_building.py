"""Build prior configurations from empirical directional statistics.

This module converts empirical statistics (mean direction, concentration) into
prior parameter configurations for PyMC model construction.

v0.2.1 addition for data-driven priors pipeline.
"""

from typing import Dict, List

import numpy as np


def _gamma_from_mode_sd(mode: float, sd: float) -> tuple[float, float]:
    """
    Convert Gamma(mode, sd) to Gamma(shape, rate) parameterization.

    For a Gamma distribution:
        mode = (shape - 1) / rate  (for shape > 1)
        variance = shape / rate^2

    Given mode and sd (standard deviation), solve for shape and rate.

    Parameters
    ----------
    mode : float
        Mode of the Gamma distribution (must be > 0)
    sd : float
        Standard deviation (must be > 0)

    Returns
    -------
    shape : float
        Shape parameter (alpha)
    rate : float
        Rate parameter (beta)

    Notes
    -----
    This uses the approximation:
        shape = (mode / sd)^2 + 2
        rate = (shape - 1) / mode
    which ensures shape > 1 and matches mode exactly.
    """
    # Ensure valid inputs
    if mode <= 0 or sd <= 0:
        raise ValueError("mode and sd must be positive")

    # Approximation that ensures shape > 1
    shape = (mode / sd) ** 2 + 2
    rate = (shape - 1) / mode

    return shape, rate


def build_priors_from_statistics(
    empirical_stats: Dict[str, Dict[str, np.ndarray]],
    joint_names: List[str],
    kappa_min: float = 0.1,
    kappa_scale: float = 5.0,
) -> Dict[str, Dict]:
    """
    Build prior configuration from empirical directional statistics.

    Converts empirical mean directions and concentrations into prior parameters
    suitable for PyMC model construction using projected Normal distribution.

    Parameters
    ----------
    empirical_stats : dict
        Dictionary from compute_direction_statistics() with joint names as keys.
        Each value has 'mu', 'kappa', and 'n_samples'.
    joint_names : list of str
        Complete list of joint names (including root)
    kappa_min : float, optional
        Minimum kappa value (prevents over-confidence). Default: 0.1
    kappa_scale : float, optional
        Scaling factor for empirical kappa to convert to prior kappa.
        kappa_prior = kappa_empirical / kappa_scale
        Lower values = weaker priors. Default: 5.0

    Returns
    -------
    prior_config : dict
        Configuration for hmm_directional.py with structure:
        {
            'joint_name': {
                'mu_mean': ndarray, shape (3,) - mean direction
                'mu_sd': float - sigma for projected Normal (1/sqrt(kappa_prior))
                'kappa_mode': float - mode of Gamma prior
                'kappa_sd': float - sd of Gamma prior
            },
            ...
        }
        Joints with invalid statistics or root are omitted from config.

    Notes
    -----
    - Only joints with valid (non-NaN) empirical statistics are included
    - Root joint is automatically excluded (no parent, no direction)
    - For projected Normal: sigma = 1 / sqrt(kappa_prior)
    - For Gamma(kappa): uses mode and sd parameterization with conversion
    - The kappa_scale parameter controls prior strength:
        * Higher values (e.g., 10) = weaker priors (more uncertainty)
        * Lower values (e.g., 2) = stronger priors (closer to empirical)
    """
    prior_config = {}

    for joint_name in joint_names:
        # Skip if not in empirical stats (e.g., root)
        if joint_name not in empirical_stats:
            continue

        stats = empirical_stats[joint_name]
        mu_emp = stats["mu"]
        kappa_emp = stats["kappa"]

        # Skip if invalid statistics
        if np.any(np.isnan(mu_emp)) or np.isnan(kappa_emp):
            continue

        # Compute prior kappa (weakened by kappa_scale)
        kappa_prior = kappa_emp / kappa_scale
        kappa_prior = max(kappa_prior, kappa_min)

        # For projected Normal: sigma = 1 / sqrt(kappa)
        mu_sd = 1.0 / np.sqrt(kappa_prior)

        # For Gamma(kappa) prior: use mode = kappa_prior, sd = mode/2
        # This gives a fairly broad prior centered at the empirical value
        kappa_mode = kappa_prior
        kappa_sd = kappa_prior / 2.0

        # Store in config
        prior_config[joint_name] = {
            "mu_mean": mu_emp,
            "mu_sd": mu_sd,
            "kappa_mode": kappa_mode,
            "kappa_sd": kappa_sd,
        }

    return prior_config


def get_gamma_shape_rate(mode: float, sd: float) -> tuple[float, float]:
    """
    Public utility to convert Gamma(mode, sd) to (shape, rate).

    This is exposed for testing and advanced usage. The build_priors_from_statistics
    function uses this internally but stores mode/sd in the config for transparency.

    Parameters
    ----------
    mode : float
        Mode of Gamma distribution
    sd : float
        Standard deviation of Gamma distribution

    Returns
    -------
    shape : float
        Shape parameter (alpha)
    rate : float
        Rate parameter (beta)

    Examples
    --------
    >>> shape, rate = get_gamma_shape_rate(mode=2.0, sd=1.0)
    >>> # Use shape and rate in PyMC: pm.Gamma('kappa', alpha=shape, beta=rate)
    """
    return _gamma_from_mode_sd(mode, sd)
