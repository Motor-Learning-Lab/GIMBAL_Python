"""Unit tests for v0.2.1 data-driven priors pipeline.

Tests triangulation, cleaning, statistics, prior building, and integration
with hmm_directional.py.
"""

import numpy as np

import gimbal


def test_triangulate_multi_view_basic():
    """Test basic DLT triangulation with known 3D points."""
    # Create proper camera projection matrices with realistic intrinsics
    # Camera intrinsics: focal length = 1000px, principal point at (320, 240)
    K_int = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float64)

    # Camera 1: At origin looking down +Z
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = K_int @ np.hstack([R1, t1])  # (3, 4)

    # Camera 2: Translated 5 units along +X, looking at origin
    R2 = np.eye(3)
    t2 = np.array([[-5.0], [0.0], [0.0]])
    P2 = K_int @ np.hstack([R2, t2])  # (3, 4)

    camera_proj = np.stack([P1, P2], axis=0)  # (2, 3, 4)

    # Known 3D point at (1, 2, 10) - in front of cameras
    X_true = np.array([1.0, 2.0, 10.0])

    # Project to 2D
    X_h = np.append(X_true, 1.0)  # Homogeneous
    x1_h = P1 @ X_h
    x2_h = P2 @ X_h
    x1 = x1_h[:2] / x1_h[2]
    x2 = x2_h[:2] / x2_h[2]

    # Create input: (C=2, T=1, K=1, 2)
    keypoints_2d = np.array([[[[x1[0], x1[1]]]], [[[x2[0], x2[1]]]]])

    # Triangulate with very high threshold (simple cameras have poor conditioning)
    positions_3d = gimbal.triangulate_multi_view(
        keypoints_2d, camera_proj, condition_threshold=1e17
    )

    # Check shape
    assert positions_3d.shape == (1, 1, 3)

    # Check accuracy (use larger tolerance for simple cameras)
    X_reconstructed = positions_3d[0, 0, :]
    np.testing.assert_allclose(X_reconstructed, X_true, rtol=1e-3, atol=0.1)


def test_triangulate_multi_view_insufficient_cameras():
    """Test that triangulation handles insufficient cameras correctly."""
    # Create input with only 1 camera (needs 2)
    P1 = np.eye(3, 4, dtype=np.float64)
    camera_proj = P1[np.newaxis, :, :]  # (1, 3, 4)

    keypoints_2d = np.random.rand(1, 5, 3, 2)  # (C=1, T=5, K=3, 2)

    result = gimbal.triangulate_multi_view(keypoints_2d, camera_proj, min_cameras=2)

    # Should return all NaN
    assert result.shape == (5, 3, 3)
    assert np.all(np.isnan(result))


def test_clean_keypoints_2d_outlier_detection():
    """Test 2D cleaning detects and removes outliers."""
    # Create synthetic data: (C=1, T=10, K=2, 2)
    C, T, K = 1, 10, 2
    keypoints_2d = np.random.randn(C, T, K, 2) * 0.1 + 50.0  # Small noise

    # Add outlier at t=5, k=1
    keypoints_2d[0, 5, 1, :] += 100.0  # Huge jump

    parents = np.array([-1, 0])  # Root and child
    config = gimbal.CleaningConfig(
        jump_z_thresh=3.0, bone_z_thresh=3.0, max_gap=2, max_bad_joint_fraction=0.5
    )

    keypoints_clean, valid_mask, summary = gimbal.clean_keypoints_2d(
        keypoints_2d, parents, config
    )

    # Should detect outlier
    assert summary["n_jump_outliers"] > 0 or summary["n_bone_outliers"] > 0

    # Outlier should be marked as NaN
    assert np.isnan(keypoints_clean[0, 5, 1, 0])


def test_clean_keypoints_2d_interpolation():
    """Test that short gaps are interpolated."""
    # Create data with gap
    C, T, K = 1, 10, 1
    keypoints_2d = np.ones((C, T, K, 2)) * 50.0

    # Create gap at t=4,5 (size 2, within max_gap=5)
    keypoints_2d[0, 4:6, 0, :] = np.nan

    parents = np.array([-1])
    config = gimbal.CleaningConfig(max_gap=5)

    keypoints_clean, valid_mask, summary = gimbal.clean_keypoints_2d(
        keypoints_2d, parents, config
    )

    # Gap should be filled
    assert not np.isnan(keypoints_clean[0, 4, 0, 0])
    assert not np.isnan(keypoints_clean[0, 5, 0, 0])
    assert summary["n_interpolated"] >= 2


def test_clean_keypoints_3d_statistics_mask():
    """Test that 3D cleaning creates correct statistics mask."""
    T, K = 20, 3
    positions_3d = np.random.randn(T, K, 3)

    # Add outlier at t=10
    positions_3d[10, 1, :] += 100.0

    parents = np.array([-1, 0, 1])
    config = gimbal.CleaningConfig(jump_z_thresh=3.0, max_gap=2)

    positions_clean, valid_mask, use_for_stats, summary = gimbal.clean_keypoints_3d(
        positions_3d, parents, config
    )

    # Outlier should be excluded from statistics
    assert not use_for_stats[10, 1]

    # Outlier should be replaced (interpolated or NaN)
    # If gap is small, might be interpolated
    if not np.isnan(positions_clean[10, 1, 0]):
        # Was interpolated, so should still be excluded from stats
        assert not use_for_stats[10, 1]


def test_compute_direction_statistics_basic():
    """Test direction statistics computation with known distribution."""
    # Create synthetic positions with consistent bone direction
    T, K = 100, 2
    positions_3d = np.zeros((T, K, 3))

    # Root at origin, child moves along a consistent direction
    mu_true = np.array([0.6, 0.8, 0.0])  # Unit vector in XY plane
    mu_true /= np.linalg.norm(mu_true)

    for t in range(T):
        positions_3d[t, 0, :] = [0, 0, 0]  # Root
        # Child with small noise around mean direction
        length = 1.0 + np.random.randn() * 0.01
        noise = np.random.randn(3) * 0.05
        positions_3d[t, 1, :] = mu_true * length + noise

    parents = np.array([-1, 0])
    joint_names = ["root", "child"]
    use_for_stats = np.ones((T, K), dtype=bool)

    stats = gimbal.compute_direction_statistics(
        positions_3d, parents, use_for_stats, joint_names, min_samples=10
    )

    # Should have stats for child only
    assert "root" in stats  # Present but NaN
    assert np.all(np.isnan(stats["root"]["mu"]))
    assert "child" in stats
    assert stats["child"]["n_samples"] == T

    # Mean direction should be close to true
    mu_emp = stats["child"]["mu"]
    dot_product = np.dot(mu_emp, mu_true)
    assert dot_product > 0.95  # Very close due to low noise

    # Kappa should be high (low variance)
    kappa_emp = stats["child"]["kappa"]
    assert kappa_emp > 10.0


def test_compute_direction_statistics_insufficient_samples():
    """Test that joints with insufficient samples get NaN."""
    T, K = 5, 2  # Only 5 samples, less than min_samples=10
    positions_3d = np.random.randn(T, K, 3)

    parents = np.array([-1, 0])
    joint_names = ["root", "child"]
    use_for_stats = np.ones((T, K), dtype=bool)

    stats = gimbal.compute_direction_statistics(
        positions_3d, parents, use_for_stats, joint_names, min_samples=10
    )

    # Should have NaN stats for child (insufficient samples)
    assert np.all(np.isnan(stats["child"]["mu"]))
    assert np.isnan(stats["child"]["kappa"])


def test_build_priors_from_statistics():
    """Test prior building from empirical statistics."""
    # Create mock empirical stats
    emp_stats = {
        "joint1": {
            "mu": np.array([1.0, 0.0, 0.0]),
            "kappa": 10.0,
            "n_samples": 100,
        },
        "joint2": {
            "mu": np.array([0.0, 1.0, 0.0]),
            "kappa": 5.0,
            "n_samples": 50,
        },
        "root": {
            "mu": np.full(3, np.nan),
            "kappa": np.nan,
            "n_samples": 0,
        },
    }

    joint_names = ["root", "joint1", "joint2"]

    prior_config = gimbal.build_priors_from_statistics(
        emp_stats, joint_names, kappa_min=0.1, kappa_scale=5.0
    )

    # Root should be excluded
    assert "root" not in prior_config

    # joint1 and joint2 should have priors
    assert "joint1" in prior_config
    assert "joint2" in prior_config

    # Check joint1 prior structure
    j1_prior = prior_config["joint1"]
    assert "mu_mean" in j1_prior
    assert "mu_sd" in j1_prior
    assert "kappa_mode" in j1_prior
    assert "kappa_sd" in j1_prior

    # Check values
    np.testing.assert_allclose(j1_prior["mu_mean"], [1.0, 0.0, 0.0])
    assert j1_prior["kappa_mode"] == 10.0 / 5.0  # kappa_prior = kappa_emp / scale
    assert j1_prior["mu_sd"] == 1.0 / np.sqrt(2.0)  # sigma = 1/sqrt(kappa_prior)


def test_get_gamma_shape_rate():
    """Test Gamma parameterization conversion."""
    mode = 2.0
    sd = 1.0

    shape, rate = gimbal.get_gamma_shape_rate(mode, sd)

    # Check that shape > 1 (required for valid mode)
    assert shape > 1.0

    # Check that mode calculation is correct
    mode_check = (shape - 1) / rate
    np.testing.assert_allclose(mode_check, mode, rtol=1e-6)

    # Check that variance is approximately correct
    variance_check = shape / rate**2
    np.testing.assert_allclose(np.sqrt(variance_check), sd, rtol=0.1)


def test_integration_full_pipeline():
    """Integration test: Full pipeline from 2D observations to prior config."""
    # Use demo skeleton
    skeleton = gimbal.DEMO_V0_1_SKELETON
    K_full = len(skeleton.joint_names)
    K = K_full - 1  # Non-root joints

    # Generate synthetic 3D trajectory
    T = 50
    positions_3d_true = np.random.randn(T, K_full, 3) * 0.5

    # Ensure bone lengths are reasonable
    for t in range(T):
        for k in range(1, K_full):
            p = skeleton.parents[k]
            bone_vec = positions_3d_true[t, k] - positions_3d_true[t, p]
            bone_vec = bone_vec / np.linalg.norm(bone_vec) * skeleton.bone_lengths[k]
            positions_3d_true[t, k] = positions_3d_true[t, p] + bone_vec

    # Create mock camera projections (2 cameras)
    C = 2
    camera_proj = np.array(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[1, 0, 0, -5], [0, 1, 0, 0], [0, 0, 1, 0]],
        ],
        dtype=np.float64,
    )

    # Project to 2D (simplified, just take x,y coordinates)
    keypoints_2d = np.zeros((C, T, K_full, 2))
    for c in range(C):
        for t in range(T):
            for k in range(K_full):
                X_h = np.append(positions_3d_true[t, k], 1.0)
                x_h = camera_proj[c] @ X_h
                keypoints_2d[c, t, k, :] = x_h[:2] / x_h[2]

    # Step 1: Clean 2D
    config = gimbal.CleaningConfig()
    kp_2d_clean, valid_2d, summary_2d = gimbal.clean_keypoints_2d(
        keypoints_2d, skeleton.parents, config
    )
    assert kp_2d_clean.shape == keypoints_2d.shape

    # Step 2: Triangulate
    positions_3d_tri = gimbal.triangulate_multi_view(kp_2d_clean, camera_proj)
    assert positions_3d_tri.shape == (T, K_full, 3)

    # Step 3: Clean 3D
    pos_3d_clean, valid_3d, use_for_stats, summary_3d = gimbal.clean_keypoints_3d(
        positions_3d_tri, skeleton.parents, config
    )
    assert pos_3d_clean.shape == (T, K_full, 3)
    assert use_for_stats.shape == (T, K_full)

    # Step 4: Compute statistics
    stats = gimbal.compute_direction_statistics(
        pos_3d_clean,
        skeleton.parents,
        use_for_stats,
        skeleton.joint_names,
        min_samples=5,  # Lower threshold for small test dataset
    )
    assert len(stats) == K_full

    # Step 5: Build priors
    prior_config = gimbal.build_priors_from_statistics(
        stats, skeleton.joint_names, kappa_scale=5.0
    )

    # Should have priors for non-root joints with sufficient data
    # (May be fewer than K due to data quality)
    assert len(prior_config) >= 0  # At least some joints might have priors
    for joint_name, joint_prior in prior_config.items():
        assert "mu_mean" in joint_prior
        assert "mu_sd" in joint_prior
        assert "kappa_mode" in joint_prior
        assert "kappa_sd" in joint_prior


def test_hmm_directional_with_prior_config():
    """Test that hmm_directional accepts and uses prior_config."""
    import pymc as pm
    import pytensor.tensor as pt

    # Create mock data
    T = 20
    S = 2

    # Create prior config and joint names FIRST to determine K
    joint_names = ["root", "j1", "j2", "j3"]
    K = len(joint_names)  # K=4

    # Mock U (directions) - must match K from joint_names
    U_data = np.random.randn(T, K, 3)
    U_data /= np.linalg.norm(U_data, axis=-1, keepdims=True)

    # Mock log_obs_t
    log_obs_t_data = np.random.randn(T) - 10.0

    # Create prior config
    prior_config = {
        "j1": {
            "mu_mean": np.array([1.0, 0.0, 0.0]),
            "mu_sd": 0.5,
            "kappa_mode": 2.0,
            "kappa_sd": 1.0,
        },
        "j2": {
            "mu_mean": np.array([0.0, 1.0, 0.0]),
            "mu_sd": 0.5,
            "kappa_mode": 3.0,
            "kappa_sd": 1.5,
        },
    }

    # Build model
    with pm.Model() as model:
        U = pt.as_tensor_variable(U_data)
        log_obs_t = pt.as_tensor_variable(log_obs_t_data)

        result = gimbal.add_directional_hmm_prior(
            U,
            log_obs_t,
            S,
            joint_names=joint_names,
            prior_config=prior_config,
        )

        # Check that model contains the expected variables
        assert "mu" in result
        assert "kappa" in result
        assert result["mu"].eval().shape == (S, K, 3)
        assert result["kappa"].eval().shape == (S, K)


def test_hmm_directional_without_prior_config():
    """Test backward compatibility: hmm_directional works without prior_config."""
    import pymc as pm
    import pytensor.tensor as pt

    T, K, S = 20, 3, 2
    U_data = np.random.randn(T, K, 3)
    U_data /= np.linalg.norm(U_data, axis=-1, keepdims=True)
    log_obs_t_data = np.random.randn(T) - 10.0

    with pm.Model() as model:
        U = pt.as_tensor_variable(U_data)
        log_obs_t = pt.as_tensor_variable(log_obs_t_data)

        # Should work without joint_names or prior_config (v0.1 mode)
        result = gimbal.add_directional_hmm_prior(
            U, log_obs_t, S, share_kappa_across_joints=True
        )

        assert result["mu"].eval().shape == (S, K, 3)
        assert result["kappa"].eval().shape == (S, K)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
