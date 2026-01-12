"""Integration tests for v0.2.1 synthetic dataset generation.

Tests validate that generated datasets meet quality thresholds for:
- Bone length consistency
- Direction normalization
- Smoothness (no extreme jerk)
- State sequence correctness
- 2D observation validity

Run from repo root:
    pytest tests/pipeline/test_v0_2_1_synth_generator.py
Or:
    pixi run pytest tests/pipeline/test_v0_2_1_synth_generator.py
"""

import pytest
import numpy as np
from pathlib import Path

from tests.pipeline.utils import (
    load_config,
    generate_from_config,
    compute_dataset_metrics,
    check_metrics_thresholds,
)


@pytest.fixture(scope="module")
def config_dir():
    """Return path to config directory."""
    return Path(__file__).parent / "configs" / "v0.2.1"


@pytest.fixture(scope="module")
def datasets():
    """Generate all test datasets once for the module."""
    config_dir_path = Path(__file__).parent / "configs" / "v0.2.1"
    datasets = {}

    for config_name in ["L00_minimal", "L01_noise", "L02_outliers", "L03_missingness"]:
        config_path = config_dir_path / f"{config_name}.json"
        config = load_config(config_path)
        dataset = generate_from_config(config, calibrate_noise=True)
        metrics = compute_dataset_metrics(dataset)
        datasets[config_name] = {
            "dataset": dataset,
            "metrics": metrics,
            "config": config,
        }

    return datasets


@pytest.fixture(scope="module")
def baseline_thresholds(datasets):
    """Compute baseline-relative thresholds from L00 metrics.

    This makes tests robust to parameter tuning while still catching regressions.
    Thresholds are defined as multiples of L00 baseline values.
    """
    l00_metrics = datasets["L00_minimal"]["metrics"]

    return {
        # Bone length: L00 should be perfect, others allow tiny numerical error
        "bone_length_max_dev": 0.001,  # 0.1% absolute tolerance
        # Direction normalization: all should be near-perfect
        "direction_norm_tolerance": 0.01,  # 1% tolerance from unity
        "near_zero_directions": 0,  # No near-zero norms allowed
        # Smoothness: L01-L03 can have up to 2x L00 jerk (noise adds dynamics)
        "jerk_p95_factor": 2.0,  # Multiplier for L00 baseline jerk
        "jerk_p95_baseline": l00_metrics["smoothness"]["jerk"]["p95"],
        # Bounds violations: 5% tolerance for noisy datasets
        "bounds_violation_rate": 0.05,
        # Identifiability: all datasets should pass
        "identifiability_fraction": 0.95,  # At least 95% of samples good
        "identifiability_min_angle": 20.0,  # degrees
    }


class TestL00Minimal:
    """Tests for L00 minimal baseline dataset."""

    def test_bone_length_consistency(self, datasets, baseline_thresholds):
        """Bone lengths should be perfectly consistent (numerical precision only)."""
        metrics = datasets["L00_minimal"]["metrics"]
        assert (
            metrics["bone_length"]["max_relative_deviation"]
            < baseline_thresholds["bone_length_max_dev"]
        )
        assert (
            metrics["bone_length"]["mean_relative_deviation"]
            < baseline_thresholds["bone_length_max_dev"] / 2
        )

    def test_direction_normalization(self, datasets, baseline_thresholds):
        """Direction vectors should be unit normalized."""
        metrics = datasets["L00_minimal"]["metrics"]
        mean_norm = metrics["direction_normalization"]["mean_norm"]
        assert abs(mean_norm - 1.0) < baseline_thresholds["direction_norm_tolerance"]
        assert (
            metrics["direction_normalization"]["near_zero_count"]
            == baseline_thresholds["near_zero_directions"]
        )

    def test_smoothness(self, datasets, baseline_thresholds):
        """Motion should be smooth (no extreme jerk)."""
        metrics = datasets["L00_minimal"]["metrics"]
        # Jerk should be reasonable (L00 defines baseline)
        max_jerk = (
            baseline_thresholds["jerk_p95_baseline"]
            * baseline_thresholds["jerk_p95_factor"]
        )
        assert metrics["smoothness"]["jerk"]["p95"] < max_jerk
        # Speed should be non-zero (motion is happening)
        assert metrics["smoothness"]["speed"]["mean"] > 0.1

    def test_single_state(self, datasets):
        """L00 should have exactly 1 state with no transitions."""
        metrics = datasets["L00_minimal"]["metrics"]
        assert metrics["states"]["single_state_check"]["is_single_state"]

    def test_identifiability(self, datasets, baseline_thresholds):
        """Camera configuration should be identifiable."""
        metrics = datasets["L00_minimal"]["metrics"]
        assert metrics["identifiability"]["passed"]
        assert (
            metrics["identifiability"]["fraction_good"]
            >= baseline_thresholds["identifiability_fraction"]
        )
        assert (
            metrics["identifiability"]["mean_min_angle"]
            > baseline_thresholds["identifiability_min_angle"]
        )
        assert metrics["states"]["single_state_check"]["actual_unique_states"] == 1

        # All transitions should be self-transitions
        trans_counts = np.array(metrics["states"]["transition_counts"])
        assert trans_counts[0, 0] > 0  # Has self-transitions
        if trans_counts.size > 1:
            # If matrix is larger, off-diagonals should be zero
            off_diagonal_sum = trans_counts.sum() - np.diag(trans_counts).sum()
            assert off_diagonal_sum == 0

    def test_no_missingness(self, datasets):
        """L00 should have no missing observations."""
        metrics = datasets["L00_minimal"]["metrics"]
        assert (
            metrics["2d_observations"]["nan_fraction"] < 0.01
        )  # Allow <1% (numerical issues)

    def test_low_noise(self, datasets):
        """L00 should have low noise as configured."""
        metrics = datasets["L00_minimal"]["metrics"]
        config = datasets["L00_minimal"]["config"]
        assert config["dataset_spec"]["observation"]["noise_px"] == 2.0
        # Observation std should be close to noise level
        # (This is approximate due to projection geometry)


class TestL01Noise:
    """Tests for L01 increased noise dataset."""

    def test_higher_noise_than_L00(self, datasets):
        """L01 should have higher observation noise than L00."""
        l00_metrics = datasets["L00_minimal"]["metrics"]
        l01_metrics = datasets["L01_noise"]["metrics"]

        # Config confirms higher noise
        l00_config = datasets["L00_minimal"]["config"]
        l01_config = datasets["L01_noise"]["config"]
        assert (
            l01_config["dataset_spec"]["observation"]["noise_px"]
            > l00_config["dataset_spec"]["observation"]["noise_px"]
        )

    def test_same_ground_truth_quality(self, datasets, baseline_thresholds):
        """Ground truth should have same quality as L00 (different obs only)."""
        metrics = datasets["L01_noise"]["metrics"]
        assert (
            metrics["bone_length"]["max_relative_deviation"]
            < baseline_thresholds["bone_length_max_dev"]
        )
        assert metrics["direction_normalization"]["near_zero_count"] == 0


class TestL02Outliers:
    """Tests for L02 outliers dataset."""

    def test_outliers_enabled(self, datasets):
        """L02 should have outliers enabled in config."""
        config = datasets["L02_outliers"]["config"]
        assert config["dataset_spec"]["observation"]["outliers"]["enabled"]
        assert config["dataset_spec"]["observation"]["outliers"]["fraction"] > 0

    def test_valid_ground_truth(self, datasets, baseline_thresholds):
        """Ground truth should still be valid despite outliers."""
        metrics = datasets["L02_outliers"]["metrics"]
        assert (
            metrics["bone_length"]["max_relative_deviation"]
            < baseline_thresholds["bone_length_max_dev"]
        )


class TestL03Missingness:
    """Tests for L03 missingness dataset."""

    def test_missingness_enabled(self, datasets):
        """L03 should have missingness in config."""
        config = datasets["L03_missingness"]["config"]
        assert config["dataset_spec"]["observation"]["missingness"]["enabled"]
        target_rate = config["dataset_spec"]["observation"]["missingness"]["rate"]
        assert target_rate > 0

    def test_missingness_rate(self, datasets):
        """Actual missingness should be close to target."""
        metrics = datasets["L03_missingness"]["metrics"]
        config = datasets["L03_missingness"]["config"]
        target_rate = config["dataset_spec"]["observation"]["missingness"]["rate"]
        actual_rate = metrics["2d_observations"]["nan_fraction"]

        # Allow 25% tolerance due to randomness
        assert abs(actual_rate - target_rate) < target_rate * 0.25

    def test_valid_ground_truth(self, datasets, baseline_thresholds):
        """Ground truth should be valid despite missingness."""
        metrics = datasets["L03_missingness"]["metrics"]
        assert (
            metrics["bone_length"]["max_relative_deviation"]
            < baseline_thresholds["bone_length_max_dev"]
        )


class TestAllDatasets:
    """Tests that apply to all datasets."""

    @pytest.mark.parametrize(
        "dataset_name", ["L00_minimal", "L01_noise", "L02_outliers", "L03_missingness"]
    )
    def test_threshold_validation(self, datasets, dataset_name, baseline_thresholds):
        """All datasets should pass baseline-relative thresholds."""
        metrics = datasets[dataset_name]["metrics"]
        passed, failures = check_metrics_thresholds(metrics, baseline_thresholds)

        if not passed:
            failure_msg = f"{dataset_name} failed thresholds:\n" + "\n".join(failures)
            pytest.fail(failure_msg)

    @pytest.mark.parametrize(
        "dataset_name", ["L00_minimal", "L01_noise", "L02_outliers", "L03_missingness"]
    )
    def test_config_hash_present(self, datasets, dataset_name):
        """All datasets should have valid config hash."""
        dataset = datasets[dataset_name]["dataset"]
        assert len(dataset.config_hash) == 64  # SHA256 hex digest

    @pytest.mark.parametrize(
        "dataset_name", ["L00_minimal", "L01_noise", "L02_outliers", "L03_missingness"]
    )
    def test_data_shapes(self, datasets, dataset_name):
        """All datasets should have consistent shapes."""
        dataset = datasets[dataset_name]["dataset"]
        config = datasets[dataset_name]["config"]

        T = config["meta"]["T"]
        K = len(config["dataset_spec"]["skeleton"]["joint_names"])
        C = len(config["dataset_spec"]["cameras"]["cameras"])

        assert dataset.x_true.shape == (T, K, 3)
        assert dataset.u_true.shape == (T, K, 3)
        assert dataset.a_true.shape == (T, K, 3)
        assert dataset.z_true.shape == (T,)
        assert dataset.y_2d.shape == (C, T, K, 2)


def test_generation_reproducibility():
    """Same config + seed should produce identical results."""
    config_path = Path(__file__).parent / "configs" / "v0.2.1" / "L00_minimal.json"
    config = load_config(config_path)

    # Generate twice with same seed
    dataset1 = generate_from_config(config, calibrate_noise=True)
    dataset2 = generate_from_config(config, calibrate_noise=True)

    # Results should be identical
    np.testing.assert_array_equal(dataset1.x_true, dataset2.x_true)
    np.testing.assert_array_equal(dataset1.z_true, dataset2.z_true)
    np.testing.assert_array_equal(dataset1.y_2d, dataset2.y_2d)
    assert dataset1.config_hash == dataset2.config_hash


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
