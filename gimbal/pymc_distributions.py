"""
Custom PyMC distributions for GIMBAL.

This module implements directional distributions needed for GIMBAL that are not
available in the core PyMC library:
- VonMisesFisher: Distribution on the unit sphere S^2 for directional vectors
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def vmf_log_norm_const(kappa):
    """
    Compute log normalization constant for von Mises-Fisher on S^2.

    For S^2, the normalization constant is:
        C(kappa) = kappa / (4*pi*sinh(kappa))

    So log C(kappa) = log(kappa) - log(4*pi) - log(sinh(kappa))
                    = log(kappa) - log(4*pi) - log((exp(kappa) - exp(-kappa))/2)
                    = log(kappa) - log(4*pi) - (log(exp(kappa) - exp(-kappa)) - log(2))

    For numerical stability when kappa is large:
        sinh(kappa) ≈ exp(kappa)/2, so log(sinh(kappa)) ≈ kappa - log(2)

    Parameters
    ----------
    kappa : tensor
        Concentration parameter (kappa >= 0)

    Returns
    -------
    log_c : tensor
        Log normalization constant
    """
    log_4pi = pt.log(4.0 * np.pi)

    # For numerical stability, use different formulas for large/small kappa
    # When kappa is large: log(sinh(kappa)) ≈ kappa - log(2)
    # When kappa is small: use exact formula

    # Exact formula (stable for small kappa)
    log_sinh_exact = pt.log(pt.sinh(kappa))

    # Asymptotic formula (stable for large kappa)
    log_sinh_asymp = kappa - pt.log(2.0)

    # Switch between them at kappa = 10
    log_sinh = pt.switch(kappa > 10.0, log_sinh_asymp, log_sinh_exact)

    return pt.log(kappa) - log_4pi - log_sinh


def vmf_logp(value, mu, kappa):
    """
    Log-probability for von Mises-Fisher distribution on S^2.

    Parameters
    ----------
    value : tensor, shape (..., 3)
        Unit vectors on S^2 (should have norm 1)
    mu : tensor, shape (..., 3) or (3,)
        Mean direction (unit vector)
    kappa : tensor, scalar or shape (...)
        Concentration parameter (kappa >= 0)

    Returns
    -------
    logp : tensor, shape (...)
        Log-probability density
    """
    # Compute dot product between value and mu
    # Broadcasting handles different shapes
    dot = pt.sum(value * mu, axis=-1)

    # Log normalization constant
    log_c = vmf_log_norm_const(kappa)

    # Log-probability: log C(kappa) + kappa * mu^T * x
    return log_c + kappa * dot


def vmf_random(mu, kappa, size=None, rng=None):
    """
    Sample from von Mises-Fisher distribution on S^2 using Wood's algorithm.

    This implements the rejection sampling algorithm from:
    Wood, A. T. A. (1994). Simulation of the von Mises Fisher distribution.
    Communications in Statistics - Simulation and Computation, 23(1), 157-164.

    Parameters
    ----------
    mu : array, shape (..., 3)
        Mean direction (unit vector)
    kappa : array, scalar or shape (...)
        Concentration parameter
    size : tuple, optional
        Output shape (not including the final dimension of 3)
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    samples : array, shape size + (3,) or mu.shape
        Samples from vMF(mu, kappa)
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu)
    kappa = np.asarray(kappa)

    # Ensure mu is a unit vector
    mu_norm = np.linalg.norm(mu, axis=-1, keepdims=True)
    mu = mu / mu_norm

    # Handle scalar kappa
    kappa_scalar = kappa.ndim == 0
    if kappa_scalar:
        kappa = np.array([kappa])

    # Determine output shape
    if size is None:
        output_shape = np.broadcast_shapes(mu.shape[:-1], kappa.shape)
    else:
        output_shape = size

    n_samples = int(np.prod(output_shape))

    # Flatten for sampling
    mu_flat = np.broadcast_to(mu, output_shape + (3,)).reshape(n_samples, 3)
    kappa_flat = np.broadcast_to(kappa, output_shape).reshape(n_samples)

    samples = np.zeros((n_samples, 3))

    for i in range(n_samples):
        mu_i = mu_flat[i]
        kappa_i = kappa_flat[i]

        # Wood's algorithm for sampling the angle w = cos(theta)
        # where theta is the angle from mu

        # For kappa = 0, uniform on sphere
        if kappa_i < 1e-10:
            v = rng.standard_normal(3)
            v = v / np.linalg.norm(v)
            samples[i] = v
            continue

        # Rejection sampling for w ~ f(w) ∝ exp(kappa * w)
        b = (-2.0 * kappa_i + np.sqrt(4.0 * kappa_i**2 + 1.0)) / 1.0
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa_i * x0 + 2.0 * np.log(1.0 - x0**2)

        accepted = False
        while not accepted:
            z = rng.beta(1.0, 1.0)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = rng.uniform(0, 1)

            if kappa_i * w + 2.0 * np.log(1.0 - x0 * w) - c >= np.log(u):
                accepted = True

        # Sample uniformly from the tangent plane at mu
        # Create orthonormal basis perpendicular to mu
        if abs(mu_i[0]) < 0.9:
            v1 = np.array([1.0, 0.0, 0.0])
        else:
            v1 = np.array([0.0, 1.0, 0.0])

        v1 = v1 - np.dot(v1, mu_i) * mu_i
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(mu_i, v1)

        # Sample angle uniformly
        phi = rng.uniform(0, 2 * np.pi)

        # Construct sample: x = w*mu + sqrt(1-w^2)*(cos(phi)*v1 + sin(phi)*v2)
        perp = np.sqrt(1.0 - w**2) * (np.cos(phi) * v1 + np.sin(phi) * v2)
        samples[i] = w * mu_i + perp

    # Reshape to output shape
    samples = samples.reshape(output_shape + (3,))

    return samples


def VonMisesFisher(name, mu, kappa, **kwargs):
    """
    Von Mises-Fisher distribution on the unit sphere S^2.

    The von Mises-Fisher distribution is the spherical analog of the Normal
    distribution for directional data on the unit sphere. It is parameterized by:
    - mu: mean direction (unit vector on S^2)
    - kappa: concentration parameter (kappa >= 0)

    When kappa = 0, the distribution is uniform on the sphere.
    As kappa increases, the distribution concentrates around mu.

    Parameters
    ----------
    name : str
        Name of the random variable
    mu : tensor, shape (..., 3)
        Mean direction (unit vector)
    kappa : tensor, scalar or shape (...)
        Concentration parameter (kappa >= 0)
    **kwargs
        Additional arguments passed to pm.CustomDist (dims, shape, etc.)

    Returns
    -------
    dist : pm.CustomDist
        PyMC distribution on S^2

    Examples
    --------
    >>> import pymc as pm
    >>> import numpy as np
    >>>
    >>> with pm.Model() as model:
    >>>     # Mean direction pointing in z-direction
    >>>     mu = np.array([0., 0., 1.])
    >>>
    >>>     # Concentration parameter
    >>>     kappa = pm.Exponential("kappa", 1.0)
    >>>
    >>>     # vMF distribution for K directional vectors over T time steps
    >>>     u = VonMisesFisher("u", mu=mu, kappa=kappa, shape=(T, K, 3))
    """
    return pm.CustomDist(
        name,
        mu,
        kappa,
        logp=vmf_logp,
        random=vmf_random,
        signature="(d),()->(d)",  # (mu, kappa) -> output with same d as mu
        **kwargs
    )
