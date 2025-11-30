"""Data cleaning for 2D and 3D keypoint trajectories.

This module provides outlier detection, interpolation, and quality assessment
for both 2D multi-camera keypoints and 3D triangulated positions.

v0.2.1 addition for data-driven priors pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class CleaningConfig:
    """Configuration for keypoint cleaning.

    Attributes
    ----------
    jump_z_thresh : float
        Z-score threshold for jump-based outlier detection.
        Typical values: 2.5-4.0. Higher = more permissive.
    bone_z_thresh : float
        Z-score threshold for bone-length-based outlier detection.
        Typical values: 2.5-4.0. Higher = more permissive.
    max_gap : int
        Maximum gap size (in frames) to interpolate.
        Gaps larger than this remain NaN.
    max_bad_joint_fraction : float
        Maximum fraction of joints that can be NaN before marking frame invalid.
        Range: [0, 1]. Typical: 0.2-0.4.
    """

    jump_z_thresh: float = 3.0
    bone_z_thresh: float = 3.0
    max_gap: int = 5
    max_bad_joint_fraction: float = 0.3


def _detect_jump_outliers(data: np.ndarray, z_thresh: float) -> np.ndarray:
    """
    Detect outliers based on frame-to-frame jumps using MAD-based Z-scores.

    Parameters
    ----------
    data : ndarray, shape (T,) or (T, D)
        Time series data with time on first axis
    z_thresh : float
        Z-score threshold for outlier detection

    Returns
    -------
    outliers : ndarray, shape (T,)
        Boolean array where True indicates outlier
    """
    # Compute frame-to-frame differences along time axis
    diffs = np.diff(data, axis=0)

    # Compute magnitude of jumps
    if data.ndim > 1:
        # For multi-dimensional data (T, D), compute Euclidean distance
        magnitudes = np.sqrt(np.sum(diffs**2, axis=1))
    else:
        # For 1D data (T,)
        magnitudes = np.abs(diffs)

    # Compute median and MAD for robust statistics
    median = np.nanmedian(magnitudes)
    mad = np.nanmedian(np.abs(magnitudes - median))

    # Avoid division by zero (constant signal)
    if mad < 1e-6:
        mad = 1.0

    # Compute Z-scores
    z_scores = (magnitudes - median) / (
        1.4826 * mad
    )  # 1.4826 makes MAD consistent with std

    # Identify outliers
    is_outlier_diff = z_scores > z_thresh

    # Convert from differences to frames (mark both endpoints)
    T = data.shape[0]
    outliers = np.zeros(T, dtype=bool)
    outliers[:-1] |= is_outlier_diff
    outliers[1:] |= is_outlier_diff

    return outliers


def _detect_bone_length_outliers(
    positions: np.ndarray, parents: np.ndarray, z_thresh: float
) -> np.ndarray:
    """
    Detect outliers based on bone length inconsistency.

    Parameters
    ----------
    positions : ndarray, shape (T, K, D) or (C, T, K, D)
        Joint positions (D=2 for 2D, D=3 for 3D)
    parents : ndarray, shape (K,)
        Parent indices for each joint
    z_thresh : float
        Z-score threshold

    Returns
    -------
    outliers : ndarray, shape matching positions[..., :, :-1]
        Boolean array where True indicates bone length outlier
    """
    K = positions.shape[-2]

    # Determine if multi-camera (C, T, K, D) or single (T, K, D)
    if positions.ndim == 4:
        C, T = positions.shape[:2]
        outliers = np.zeros((C, T, K), dtype=bool)
        is_multi_camera = True
    else:
        T = positions.shape[0]
        outliers = np.zeros((T, K), dtype=bool)
        is_multi_camera = False
        C = 0  # Not used, but keeps linter happy

    for k in range(1, K):  # Skip root (no parent)
        p = parents[k]
        if p < 0:
            continue

        # Compute bone lengths over time
        if is_multi_camera:
            # Multi-camera: (C, T)
            bone_vecs = positions[:, :, k, :] - positions[:, :, p, :]
            bone_lengths = np.sqrt(np.sum(bone_vecs**2, axis=-1))

            for c in range(C):
                lengths_c = bone_lengths[c, :]
                valid = ~np.isnan(lengths_c)
                if valid.sum() < 3:
                    continue

                median = np.nanmedian(lengths_c)
                mad = np.nanmedian(np.abs(lengths_c - median))

                if mad < 1e-6:
                    continue

                z_scores = np.abs(lengths_c - median) / (1.4826 * mad)
                is_outlier = z_scores > z_thresh

                # Mark both parent and child as outliers
                outliers[c, is_outlier, k] = True
                outliers[c, is_outlier, p] = True
        else:
            # Single view: (T,)
            bone_vecs = positions[:, k, :] - positions[:, p, :]
            bone_lengths = np.sqrt(np.sum(bone_vecs**2, axis=-1))

            valid = ~np.isnan(bone_lengths)
            if valid.sum() < 3:
                continue

            median = np.nanmedian(bone_lengths)
            mad = np.nanmedian(np.abs(bone_lengths - median))

            if mad < 1e-6:
                continue

            z_scores = np.abs(bone_lengths - median) / (1.4826 * mad)
            is_outlier = z_scores > z_thresh

            # Mark both parent and child as outliers
            outliers[is_outlier, k] = True
            outliers[is_outlier, p] = True

    return outliers


def _interpolate_gaps(data: np.ndarray, max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate NaN gaps up to max_gap size.

    Parameters
    ----------
    data : ndarray, shape (T, ...) with time as first axis
        Data with NaN values to interpolate
    max_gap : int
        Maximum gap size to interpolate

    Returns
    -------
    data_filled : ndarray, same shape as data
        Data with interpolated values
    was_interpolated : ndarray, shape (T, ...), bool
        Mask indicating which values were interpolated
    """
    data_filled = data.copy()
    was_interpolated = np.zeros(data.shape, dtype=bool)

    T = data.shape[0]

    # Flatten to (T, N) for easier processing
    data_flat = data.reshape(T, -1)
    filled_flat = data_filled.reshape(T, -1)
    interp_flat = was_interpolated.reshape(T, -1)

    N = data_flat.shape[1]

    for n in range(N):
        series = data_flat[:, n]
        valid = ~np.isnan(series)

        if valid.sum() < 2:
            continue

        # Find gaps
        gap_start = None
        for t in range(T):
            if not valid[t]:
                if gap_start is None:
                    gap_start = t
            else:
                if gap_start is not None:
                    gap_end = t - 1
                    gap_size = gap_end - gap_start + 1

                    if gap_size <= max_gap:
                        # Find valid endpoints
                        t_before = gap_start - 1
                        t_after = gap_end + 1

                        # Check boundaries
                        if t_before >= 0 and t_after < T:
                            # Linear interpolation
                            t_indices = np.arange(gap_start, gap_end + 1)
                            filled_flat[t_indices, n] = np.interp(
                                t_indices,
                                [t_before, t_after],
                                [series[t_before], series[t_after]],
                            )
                            interp_flat[gap_start : gap_end + 1, n] = True

                    gap_start = None

    # Reshape back
    data_filled = filled_flat.reshape(data.shape)
    was_interpolated = interp_flat.reshape(data.shape)

    return data_filled, was_interpolated


def clean_keypoints_2d(
    keypoints_2d: np.ndarray, parents: np.ndarray, config: CleaningConfig
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Clean 2D keypoint observations per camera.

    Detects and removes outliers based on frame-to-frame jumps and bone length
    consistency, then interpolates short gaps. This is applied per-camera
    independently to remove obvious garbage before triangulation.

    Parameters
    ----------
    keypoints_2d : ndarray, shape (C, T, K, 2)
        2D keypoint observations from C cameras
    parents : ndarray, shape (K,)
        Parent indices for skeleton tree
    config : CleaningConfig
        Cleaning parameters

    Returns
    -------
    keypoints_2d_clean : ndarray, shape (C, T, K, 2)
        Cleaned 2D keypoints with interpolation
    valid_frame_mask_2d : ndarray, shape (C, T), bool
        Per-camera frame validity mask (False if too many bad joints)
    summary : dict
        Diagnostic information with keys:
        - n_jump_outliers : int
        - n_bone_outliers : int
        - n_interpolated : int
        - n_invalid_frames : int
    """
    C, T, K, _ = keypoints_2d.shape
    keypoints_clean = keypoints_2d.copy()
    valid_frame_mask = np.ones((C, T), dtype=bool)

    n_jump_outliers = 0
    n_bone_outliers = 0
    n_interpolated = 0

    # Process each camera independently
    for c in range(C):
        kp_c = keypoints_clean[c]  # (T, K, 2)

        # 1. Detect jump outliers per joint
        for k in range(K):
            for dim in range(2):  # u and v
                series = kp_c[:, k, dim]
                if np.all(np.isnan(series)):
                    continue

                # Detect jumps
                jump_outliers = _detect_jump_outliers(
                    series[..., np.newaxis], config.jump_z_thresh
                )
                jump_outliers = jump_outliers.squeeze()

                # Mark as NaN
                kp_c[jump_outliers, k, dim] = np.nan
                n_jump_outliers += jump_outliers.sum()

        # 2. Detect bone length outliers
        bone_outliers = _detect_bone_length_outliers(
            kp_c, parents, config.bone_z_thresh
        )
        n_bone_outliers += bone_outliers.sum()

        # Mark bone outliers as NaN
        for t in range(T):
            for k in range(K):
                if bone_outliers[t, k]:
                    kp_c[t, k, :] = np.nan

        # 3. Interpolate short gaps per joint and dimension
        for k in range(K):
            for dim in range(2):
                series = kp_c[:, k, dim]
                filled, was_interp = _interpolate_gaps(series, config.max_gap)
                kp_c[:, k, dim] = filled
                n_interpolated += was_interp.sum()

        # 4. Mark invalid frames
        for t in range(T):
            n_nan_joints = np.sum(np.any(np.isnan(kp_c[t]), axis=1))
            if n_nan_joints / K > config.max_bad_joint_fraction:
                valid_frame_mask[c, t] = False

        keypoints_clean[c] = kp_c

    n_invalid_frames = (~valid_frame_mask).sum()

    summary = {
        "n_jump_outliers": n_jump_outliers,
        "n_bone_outliers": n_bone_outliers,
        "n_interpolated": n_interpolated,
        "n_invalid_frames": n_invalid_frames,
    }

    return keypoints_clean, valid_frame_mask, summary


def clean_keypoints_3d(
    positions_3d: np.ndarray, parents: np.ndarray, config: CleaningConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Clean 3D joint position trajectories.

    Detects and removes outliers, interpolates short gaps, and creates masks
    for frame validity and statistical analysis.

    Parameters
    ----------
    positions_3d : ndarray, shape (T, K, 3)
        3D joint positions over time
    parents : ndarray, shape (K,)
        Parent indices for skeleton tree
    config : CleaningConfig
        Cleaning parameters

    Returns
    -------
    positions_clean : ndarray, shape (T, K, 3)
        Cleaned positions with interpolation
    valid_frame_mask : ndarray, shape (T,), bool
        Frame validity mask (False if too many bad joints)
    use_for_stats_mask : ndarray, shape (T, K), bool
        Statistical validity mask (True only for original non-outlier data)
    summary : dict
        Diagnostic information with keys:
        - n_jump_outliers : int
        - n_bone_outliers : int
        - n_interpolated : int
        - n_invalid_frames : int
    """
    T, K, _ = positions_3d.shape
    positions_clean = positions_3d.copy()
    use_for_stats_mask = np.ones((T, K), dtype=bool)

    # Track what's original data (not NaN, not outlier, not interpolated)
    was_originally_nan = np.any(np.isnan(positions_3d), axis=-1)  # (T, K)

    n_jump_outliers = 0
    n_bone_outliers = 0
    n_interpolated = 0

    # 1. Detect jump outliers per joint
    for k in range(K):
        pos_k = positions_clean[:, k, :]  # (T, 3)

        jump_outliers = _detect_jump_outliers(pos_k, config.jump_z_thresh)
        n_jump_outliers += jump_outliers.sum()

        # Mark as NaN
        positions_clean[jump_outliers, k, :] = np.nan
        use_for_stats_mask[jump_outliers, k] = False

    # 2. Detect bone length outliers
    bone_outliers = _detect_bone_length_outliers(
        positions_clean, parents, config.bone_z_thresh
    )
    n_bone_outliers += bone_outliers.sum()

    # Mark bone outliers as NaN
    for t in range(T):
        for k in range(K):
            if bone_outliers[t, k]:
                positions_clean[t, k, :] = np.nan
                use_for_stats_mask[t, k] = False

    # 3. Interpolate short gaps per joint and dimension
    for k in range(K):
        for dim in range(3):  # x, y, z
            series = positions_clean[:, k, dim]
            filled, was_interp = _interpolate_gaps(series, config.max_gap)
            positions_clean[:, k, dim] = filled

            # Mark interpolated points
            use_for_stats_mask[was_interp, k] = False
            n_interpolated += was_interp.sum()

    # 4. Also exclude originally NaN values from statistics
    use_for_stats_mask[was_originally_nan] = False

    # 5. Mark invalid frames
    valid_frame_mask = np.ones(T, dtype=bool)
    for t in range(T):
        n_nan_joints = np.sum(np.any(np.isnan(positions_clean[t]), axis=-1))
        if n_nan_joints / K > config.max_bad_joint_fraction:
            valid_frame_mask[t] = False

    n_invalid_frames = (~valid_frame_mask).sum()

    summary = {
        "n_jump_outliers": n_jump_outliers,
        "n_bone_outliers": n_bone_outliers,
        "n_interpolated": n_interpolated,
        "n_invalid_frames": n_invalid_frames,
    }

    return positions_clean, valid_frame_mask, use_for_stats_mask, summary
