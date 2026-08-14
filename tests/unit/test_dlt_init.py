"""
Tests for initialize_from_observations_dlt and its eta2 (temporal variance) estimator.

Previously this file was a print-only script with no `def test_*` functions and no
assertions -- pytest collected and executed it (so a hard crash would be caught) but
nothing checked whether the printed values were actually correct. That gap is why a
systematic ~10-60x bias in the estimated `eta2_root` (see
plans/v0.2.2/v0.2.2_C1_findings.md, "initialize_from_observations_dlt's eta2_root
initval: test coverage audit and failure characterization") went uncaught. This file
now asserts on the values, including two tests specifically for that bug.
"""

import numpy as np
import pytest

from gimbal_pymc.cameras.projection import build_projection_matrix
from gimbal_pymc.priors.initialization import (
    _estimate_temporal_variances_numpy,
    initialize_from_observations_dlt,
)
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.skeleton.synthetic_data import (
    SyntheticDataConfig,
    generate_demo_sequence,
)


def _make_simple_scene(seed=42, T=20, K=3, C=3):
    """A small hand-built multi-camera scene (not the full skeleton generator)."""
    rng = np.random.default_rng(seed)
    parents = np.array([-1, 0, 1])

    cameras = []
    for i in range(C):
        rvec = np.array([0.0, np.pi / 6 * i, 0.0])
        tvec = np.array([200.0 * np.sin(np.pi / 3 * i), 100.0, 500.0 + 100 * i])
        cameras.append(build_projection_matrix(rvec, tvec, focal_length=800.0))
    camera_proj = np.stack(cameras)

    x_true = np.zeros((T, K, 3))
    for t in range(T):
        x_true[t, 0] = [np.sin(t / 5) * 50, np.cos(t / 5) * 50, 100]
        x_true[t, 1] = x_true[t, 0] + [50, 0, 0]
        x_true[t, 2] = x_true[t, 1] + [0, 50, 0]

    y_obs = np.zeros((C, T, K, 2))
    for c in range(C):
        for t in range(T):
            for k in range(K):
                x_h = np.append(x_true[t, k], 1)
                p = camera_proj[c] @ x_h
                y_obs[c, t, k] = (p[:2] / p[2]) + rng.normal(0, 1.0, 2)

    return y_obs, camera_proj, parents


def test_dlt_init_shapes_and_positivity():
    """Basic sanity: shapes, finiteness, and strictly-positive scale parameters."""
    y_obs, camera_proj, parents = _make_simple_scene()
    result = initialize_from_observations_dlt(y_obs, camera_proj, parents, min_cameras=2)

    T, K = 20, 3
    assert result.x_init.shape == (T, K, 3)
    assert result.u_init.shape == (T, K, 3)
    assert not np.isnan(result.x_init).any()
    assert not np.isnan(result.u_init).any()
    assert (result.eta2 > 0).all()
    assert (result.sigma2 > 0).all()
    assert result.obs_sigma > 0

    # Root (joint 0) has no direction vector by convention -- _estimate_direction_
    # vectors_numpy leaves it at its zero-initialized default. Only non-root joints
    # should be unit vectors.
    u_norms = np.linalg.norm(result.u_init[:, 1:, :], axis=-1)
    assert np.allclose(u_norms, 1.0, atol=0.01)


def test_eta2_recovers_known_variance_with_noiseless_positions():
    """
    Isolates the estimator formula from triangulation noise: feed
    _estimate_temporal_variances_numpy positions from a random walk with a known
    ground-truth eta2 directly (as if triangulation were perfect), and check it
    recovers that value closely.

    This is the direct regression test for the formula bug: the old code applied the
    Gaussian MAD-to-SD conversion factor to a chi-squared(3)-distributed quantity (the
    sum of 3 squared per-coordinate deltas), which biases the estimate high by ~2x even
    with zero triangulation noise. Fails at the old code's ~2x bias; passes at the
    fixed per-coordinate formula's <10% bias (verified empirically before fixing).
    """
    rng = np.random.default_rng(123)
    T = 500
    true_eta2 = 2.25  # sd = 1.5

    x = np.zeros((T, 1, 3))
    for t in range(1, T):
        x[t, 0] = x[t - 1, 0] + rng.normal(0, np.sqrt(true_eta2), 3)

    eta2_est = _estimate_temporal_variances_numpy(x, parents=np.array([-1]))

    assert eta2_est[0] == pytest.approx(true_eta2, rel=0.3)


def test_eta2_root_not_wildly_biased_with_realistic_triangulation():
    """
    Regression test for the actual reported failure: with realistic DLT triangulation
    noise (not the noiseless case above), the root joint's eta2 initval must not land
    tens of prior-standard-deviations into a HalfNormal(0.5)-scale prior's tail the way
    it did before this fix (observed 23-57x true value across 100 seeds, 100% failure
    rate). This uses the full production skeleton/camera synthetic-data generator, the
    same one the TG14 diagnostic that first surfaced this bug uses, at its default
    root_noise_std=1.0 (true eta2_root = 1.0).
    """
    config = SyntheticDataConfig(
        T=100, C=3, S=3, kappa=10.0, obs_noise_std=0.5, occlusion_rate=0.0,
        random_seed=42,
    )
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

    result = initialize_from_observations_dlt(
        data.y_observed, data.camera_proj, DEMO_V0_1_SKELETON.parents
    )

    true_eta2_root = config.root_noise_std**2
    eta2_root = result.eta2[0]

    assert eta2_root >= 0.0
    assert np.isfinite(eta2_root)
    # The old formula gave 23-57x true value on this exact scenario across 100 seeds;
    # the fix should never again be that far off. A generous factor-of-10 bound
    # catches a regression without being a tight, seed-fragile assertion.
    assert eta2_root < 10 * true_eta2_root, (
        f"eta2_root={eta2_root:.3f} is more than 10x the true value "
        f"({true_eta2_root}) -- looks like a regression of the triangulation-noise "
        f"and/or MAD-formula bias bug"
    )
