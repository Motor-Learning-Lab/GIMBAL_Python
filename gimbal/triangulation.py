"""Multi-view triangulation for GIMBAL.

This module provides triangulation functionality to convert multi-camera 2D
keypoint observations into 3D positions using Direct Linear Transform (DLT).

v0.2.1 addition for data-driven priors pipeline.
"""

import numpy as np


def triangulate_multi_view(
    keypoints_2d: np.ndarray,
    camera_proj: np.ndarray,
    min_cameras: int = 2,
    condition_threshold: float = 1e6,
) -> np.ndarray:
    """
    Triangulate 3D positions from multi-view 2D keypoint observations.

    Uses Direct Linear Transform (DLT) with SVD. NaN observations are excluded
    from triangulation. If fewer than min_cameras have valid observations for
    a given joint at a given timestep, the 3D position is set to NaN.

    Parameters
    ----------
    keypoints_2d : ndarray, shape (C, T, K, 2)
        2D keypoint observations from C cameras over T timesteps for K joints.
        NaN values indicate missing/occluded observations.
    camera_proj : ndarray, shape (C, 3, 4)
        Camera projection matrices [A | b] for each camera.
        Each matrix maps homogeneous 3D coordinates to 2D: y = P @ [x; 1]
    min_cameras : int, optional
        Minimum number of cameras required for triangulation. Default: 2.
    condition_threshold : float, optional
        Maximum condition number for SVD (largest/smallest singular value).
        Triangulation with higher condition number is rejected. Default: 1e6.

    Returns
    -------
    positions_3d : ndarray, shape (T, K, 3)
        Triangulated 3D joint positions.
        NaN where triangulation failed (insufficient cameras or poor conditioning).

    Notes
    -----
    The DLT method constructs a linear system A @ X = 0 where:
    - Each camera contributes 2 equations (u and v components)
    - The solution is the nullspace of A (last column of V in SVD)
    - Homogeneous coordinates are converted to 3D via division by w

    References
    ----------
    Hartley, R., & Zisserman, A. (2003). Multiple view geometry in computer vision.
    Cambridge university press. (Chapter 12: Structure Computation)
    """
    C, T, K, _ = keypoints_2d.shape
    positions_3d = np.zeros((T, K, 3))

    for k in range(K):
        for t in range(T):
            # Get observations for this joint at this time across all cameras
            y_tk = keypoints_2d[:, t, k, :]  # (C, 2)

            # Find cameras with valid (non-NaN) observations
            valid_mask = ~np.isnan(y_tk[:, 0]) & ~np.isnan(y_tk[:, 1])
            n_valid = valid_mask.sum()

            if n_valid < min_cameras:
                positions_3d[t, k, :] = np.nan
                continue

            # Construct DLT matrix A
            # Each camera contributes 2 rows:
            #   u * P[2,:] - P[0,:]  (x-equation)
            #   v * P[2,:] - P[1,:]  (y-equation)
            A = []
            for c in np.where(valid_mask)[0]:
                u, v = y_tk[c]
                P = camera_proj[c]
                A.append(u * P[2, :] - P[0, :])
                A.append(v * P[2, :] - P[1, :])

            A = np.array(A)  # Shape: (2*n_valid, 4)

            # Solve via SVD: A = U S V^T
            # Solution is last column of V (nullspace of A)
            try:
                _, S, Vt = np.linalg.svd(A)

                # Check conditioning
                cond = S[0] / (S[-1] + 1e-10)
                if cond > condition_threshold:
                    positions_3d[t, k, :] = np.nan
                    continue

                # Extract homogeneous coordinates [x, y, z, w]
                X_homog = Vt[-1, :]

                # Convert to 3D via division by w
                if np.abs(X_homog[3]) < 1e-8:
                    positions_3d[t, k, :] = np.nan
                else:
                    positions_3d[t, k, :] = X_homog[:3] / X_homog[3]

            except np.linalg.LinAlgError:
                # SVD failed (rare, but handle gracefully)
                positions_3d[t, k, :] = np.nan

    return positions_3d
