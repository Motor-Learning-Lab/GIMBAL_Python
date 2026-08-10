"""
Smoke test for v0.1 complete pipeline.

This test validates that the full PyMC HMM pipeline (Stage 1-3) can be
built and that the model graph is valid. It does not test convergence,
only that:
- Imports work correctly
- Shapes match expectations
- The model can be constructed without errors
- Prior predictive sampling runs

This provides a fast regression test to catch obvious breakage.
"""

import pytest
import numpy as np
import pymc as pm

from gimbal_pymc.hmm.observation_model import build_camera_observation_model
from gimbal_pymc.hmm.directional_prior import add_directional_hmm_prior
from gimbal_pymc.skeleton.synthetic_data import (
    generate_demo_sequence,
    SyntheticDataConfig,
)
from gimbal_pymc.skeleton.config import DEMO_V0_1_SKELETON
from gimbal_pymc.priors.initialization import initialize_from_observations_dlt


def test_v0_1_pipeline_builds():
    """Test that the full v0.1 pipeline can be built."""
    # Generate minimal synthetic data
    config = SyntheticDataConfig(
        T=20,  # Sufficient frames for robust parameter estimation (min 10 per bone)
        C=2,  # Minimal cameras
        S=2,  # Minimal states
        random_seed=42,
    )
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

    # Validate synthetic data shapes
    assert data.x_true.shape == (20, 6, 3)
    assert data.u_true.shape == (20, 6, 3)
    assert data.true_states.shape == (20,)
    assert data.y_observed.shape == (2, 20, 6, 2)
    assert data.camera_proj.shape == (2, 3, 4)

    # Initialize from observations
    init_result = initialize_from_observations_dlt(
        y_observed=data.y_observed,
        camera_proj=data.camera_proj,
        parents=DEMO_V0_1_SKELETON.parents,
    )

    # Build Stage 2 model
    with pm.Model() as model:
        model_result = build_camera_observation_model(
            y_observed=data.y_observed,
            camera_proj=data.camera_proj,
            parents=DEMO_V0_1_SKELETON.parents,
            init_result=init_result,
            use_mixture=False,  # Simple Gaussian for smoke test
            validate_init_points=False,
        )

        # Extract key variables
        U = model["U"]
        x_all = model["x_all"]
        y_pred = model["y_pred"]
        log_obs_t = model["log_obs_t"]

        # Validate Stage 2 shapes (robust to PyTensor symbolic inference)
        # Check rank (ndim)
        assert U.ndim == 3  # (T, K, 3)
        assert x_all.ndim == 3  # (T, K, 3)
        assert y_pred.ndim == 4  # (C, T, K, 2)
        assert log_obs_t.ndim == 1  # (T,)

        # Check known dimensions (trailing dims are stable, leading dims may be symbolic)
        assert U.type.shape[-1] == 3  # Direction vectors
        assert U.type.shape[-2] == 6  # K joints
        assert x_all.type.shape[-1] == 3  # 3D positions
        assert x_all.type.shape[-2] == 6  # K joints
        assert y_pred.type.shape[-1] == 2  # 2D image points
        assert y_pred.type.shape[-2] == 6  # K joints

        # Allow symbolic dimensions for C and T (PyTensor may infer None)
        assert y_pred.type.shape[0] in (None, 2)  # C cameras
        assert y_pred.type.shape[1] in (None, 20)  # T frames
        assert U.type.shape[0] in (None, 20)  # T frames
        assert x_all.type.shape[0] in (None, 20)  # T frames
        assert log_obs_t.type.shape[0] in (None, 20)  # T frames

        # Add Stage 3 directional HMM prior
        hmm_vars = add_directional_hmm_prior(
            U=U,
            log_obs_t=log_obs_t,
            S=2,  # Minimal states
        )

        # Validate Stage 3 variables exist
        assert "mu" in hmm_vars
        assert "kappa" in hmm_vars
        assert "hmm_loglik" in hmm_vars
        assert "logp_emit" in hmm_vars

        # Validate Stage 3 shapes (robust to symbolic inference)
        assert hmm_vars["mu"].ndim == 3  # (S, K, 3)
        assert hmm_vars["kappa"].ndim == 2  # (S, K)

        # Check known dimensions
        assert hmm_vars["mu"].type.shape[-1] == 3  # Direction vectors
        assert hmm_vars["mu"].type.shape[-2] == 6  # K joints
        assert hmm_vars["mu"].type.shape[0] in (None, 2)  # S states
        assert hmm_vars["kappa"].type.shape[-1] == 6  # K joints
        assert hmm_vars["kappa"].type.shape[0] in (None, 2)  # S states

        # Check that model has the expected number of free variables
        # This is a rough check - exact number depends on DLT initialization
        assert len(model.free_RVs) > 0

        # Verify that model can generate initial values successfully
        # This tests that the full computation graph is valid
        test_point = model.initial_point()
        assert test_point is not None
        assert len(test_point) > 0, "Model should have initialized parameters"


def test_v0_1_prior_predictive_sampling():
    """Test that prior predictive sampling works."""
    config = SyntheticDataConfig(T=20, C=2, S=2, random_seed=42)
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

    init_result = initialize_from_observations_dlt(
        y_observed=data.y_observed,
        camera_proj=data.camera_proj,
        parents=DEMO_V0_1_SKELETON.parents,
    )

    with pm.Model() as model:
        build_camera_observation_model(
            y_observed=data.y_observed,
            camera_proj=data.camera_proj,
            parents=DEMO_V0_1_SKELETON.parents,
            init_result=init_result,
            use_mixture=False,
            validate_init_points=False,
        )

        U = model["U"]
        log_obs_t = model["log_obs_t"]

        add_directional_hmm_prior(U=U, log_obs_t=log_obs_t, S=2)

        # Run minimal prior predictive sampling
        prior_pred = pm.sample_prior_predictive(
            samples=10,
            random_seed=42,
        )

        # Check that we got samples
        assert prior_pred.prior is not None
        # Check that key variables are present
        assert "dir_hmm_mu" in prior_pred.prior
        assert "dir_hmm_kappa_full" in prior_pred.prior


def test_v0_1_behavioral_invariance():
    """Test that v0.1 default behavior is preserved.

    This tests that explicitly passing prior_config=None produces the same
    model structure as using the default (HalfNormal priors, not data-driven).
    This is behavioral invariance, not module-path backward compatibility.
    """
    config = SyntheticDataConfig(T=20, C=2, S=2, random_seed=42)
    data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

    init_result = initialize_from_observations_dlt(
        y_observed=data.y_observed,
        camera_proj=data.camera_proj,
        parents=DEMO_V0_1_SKELETON.parents,
    )

    # Build model with default parameters (v0.1 behavior)
    with pm.Model() as model_v1:
        build_camera_observation_model(
            y_observed=data.y_observed,
            camera_proj=data.camera_proj,
            parents=DEMO_V0_1_SKELETON.parents,
            init_result=init_result,
            use_mixture=False,
            validate_init_points=False,
        )
        U1 = model_v1["U"]
        log_obs_t1 = model_v1["log_obs_t"]
        add_directional_hmm_prior(U=U1, log_obs_t=log_obs_t1, S=2)

    # Build model with explicit prior_config=None (should be identical)
    with pm.Model() as model_v2:
        build_camera_observation_model(
            y_observed=data.y_observed,
            camera_proj=data.camera_proj,
            parents=DEMO_V0_1_SKELETON.parents,
            init_result=init_result,
            use_mixture=False,
            validate_init_points=False,
        )
        U2 = model_v2["U"]
        log_obs_t2 = model_v2["log_obs_t"]
        add_directional_hmm_prior(
            U=U2,
            log_obs_t=log_obs_t2,
            S=2,
            prior_config=None,  # Explicit None should be same as default
        )

    # Both models should have same structure
    assert len(model_v1.free_RVs) == len(model_v2.free_RVs)
    assert set(rv.name for rv in model_v1.free_RVs) == set(
        rv.name for rv in model_v2.free_RVs
    )


def test_synthetic_data_deterministic():
    """Test that synthetic data generation is deterministic with seed."""
    config = SyntheticDataConfig(T=10, C=2, S=2, random_seed=42)

    data1 = generate_demo_sequence(DEMO_V0_1_SKELETON, config)
    data2 = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

    # Should produce identical results with same seed
    np.testing.assert_array_equal(data1.x_true, data2.x_true)
    np.testing.assert_array_equal(data1.u_true, data2.u_true)
    np.testing.assert_array_equal(data1.true_states, data2.true_states)

    # Observations may differ slightly due to occlusion randomness,
    # but non-NaN values should be close
    mask1 = ~np.isnan(data1.y_observed)
    mask2 = ~np.isnan(data2.y_observed)
    np.testing.assert_array_equal(mask1, mask2)
    np.testing.assert_allclose(
        data1.y_observed[mask1],
        data2.y_observed[mask2],
        rtol=1e-10,
    )


if __name__ == "__main__":
    # Run tests directly
    test_v0_1_pipeline_builds()
    print("✓ Pipeline builds successfully")

    test_v0_1_prior_predictive_sampling()
    print("✓ Prior predictive sampling works")

    test_v0_1_backward_compatibility()
    print("✓ Backward compatibility maintained")

    test_synthetic_data_deterministic()
    print("✓ Synthetic data generation is deterministic")

    print("\nAll smoke tests passed!")
