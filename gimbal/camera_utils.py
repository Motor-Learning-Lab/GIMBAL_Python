"""
Camera projection utilities.

Provides NumPy implementations of camera operations that mirror the PyTensor
projector used in the PyMC model, plus helper functions for camera geometry.
"""

import numpy as np
from typing import Tuple


def project_points_numpy(x: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """
    Project 3D points to 2D using perspective projection.

    This mirrors the PyTensor implementation in project_points_pytensor(),
    ensuring consistency between synthetic data generation and inference.

    Args:
        x: 3D positions, shape (T, K, 3)
        proj: Camera projection matrices P = K[R|t], shape (C, 3, 4)

    Returns:
        y: 2D projected points with perspective division, shape (C, T, K, 2)

    Algorithm:
        1. Convert to homogeneous: [x, y, z, 1]
        2. Apply projection: [u', v', w'] = P @ [x, y, z, 1]
        3. Perspective division: u = u'/w', v = v'/w'
    """
    T, K, _ = x.shape

    # Homogeneous coordinates
    ones = np.ones((*x.shape[:-1], 1))  # (T, K, 1)
    x_h = np.concatenate([x, ones], axis=-1)  # (T, K, 4)

    # Project: x_cam = proj @ x_h^T for each camera
    # Using einsum: "cij,tkj->ctki" means:
    #   c: camera index
    #   i,j: matrix multiply proj[c] @ x_h[t,k]
    #   Result indexed by (c, t, k, i)
    x_cam = np.einsum("cij,tkj->ctki", proj, x_h)  # (C, T, K, 3)

    # Extract components
    u = x_cam[:, :, :, 0]  # (C, T, K)
    v = x_cam[:, :, :, 1]  # (C, T, K)
    w = x_cam[:, :, :, 2]  # (C, T, K)

    # Perspective division (avoid division by zero)
    w = np.maximum(np.abs(w), 1e-6) * np.sign(w + 1e-10)

    # Stack into (C, T, K, 2)
    y = np.stack([u / w, v / w], axis=-1)

    return y


def camera_center_from_proj(P: np.ndarray) -> np.ndarray:
    """
    Extract camera center from projection matrix P.

    For P = [A | b], the camera center C satisfies P @ [C; 1] = 0,
    which gives C = -A^{-1} @ b.

    Args:
        P: Projection matrix, shape (3, 4) or (C, 3, 4)

    Returns:
        Camera center(s), shape (3,) or (C, 3)
    """
    if P.ndim == 2:
        # Single camera
        A = P[:, :3]
        b = P[:, 3]
        return -np.linalg.inv(A) @ b
    else:
        # Multiple cameras
        C = P.shape[0]
        centers = np.zeros((C, 3))
        for c in range(C):
            A = P[c, :, :3]
            b = P[c, :, 3]
            centers[c] = -np.linalg.inv(A) @ b
        return centers


def build_look_at_matrix(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    up_world: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """
    Build a camera rotation matrix using "look-at" convention.

    The camera's local coordinate system is defined as:
    - z_cam: points FROM camera TO target (forward/optical axis)
    - x_cam: points to the right (perpendicular to up_world and z_cam)
    - y_cam: points up in the camera frame (completes right-handed system)

    Args:
        camera_pos: Camera position in world coords, shape (3,)
        target_pos: Point the camera looks at, shape (3,)
        up_world: World "up" direction (used to resolve roll), shape (3,)

    Returns:
        R: Rotation matrix from world to camera coords, shape (3, 3)
           R @ world_point gives camera-frame coordinates

    Convention:
        For extrinsic matrix [R | t], we use t = -R @ camera_pos
        so that P = K @ [R | t] transforms world points to camera space.
    """
    # Forward direction: camera looks toward target
    z_cam = target_pos - camera_pos
    z_cam = z_cam / np.linalg.norm(z_cam)

    # Right direction: perpendicular to both up_world and forward
    x_cam = np.cross(up_world, z_cam)
    x_cam_norm = np.linalg.norm(x_cam)

    # Handle degenerate case: camera looking straight up/down
    if x_cam_norm < 1e-6:
        # Choose arbitrary right vector perpendicular to z_cam
        if np.abs(z_cam[0]) < 0.9:
            x_cam = np.cross([1, 0, 0], z_cam)
        else:
            x_cam = np.cross([0, 1, 0], z_cam)
        x_cam = x_cam / np.linalg.norm(x_cam)
    else:
        x_cam = x_cam / x_cam_norm

    # Up direction: completes right-handed system
    y_cam = np.cross(z_cam, x_cam)

    # Rotation matrix: rows are camera axes expressed in world coords
    R = np.vstack([x_cam, y_cam, z_cam])

    return R


def build_intrinsic_matrix(
    focal_length: float, principal_point: Tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """
    Build camera intrinsic matrix K.

    Args:
        focal_length: Focal length in pixels (same for x and y)
        principal_point: (cx, cy) principal point offset

    Returns:
        K: Intrinsic matrix, shape (3, 3)
           [[f, 0, cx],
            [0, f, cy],
            [0, 0,  1]]
    """
    cx, cy = principal_point
    K = np.array([[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]])
    return K


def build_projection_matrix(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    focal_length: float,
    up_world: np.ndarray = np.array([0.0, 0.0, 1.0]),
    principal_point: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Build full projection matrix P = K[R|t].

    Args:
        camera_pos: Camera position, shape (3,)
        target_pos: Point camera looks at, shape (3,)
        focal_length: Focal length in pixels
        up_world: World up direction
        principal_point: (cx, cy) principal point

    Returns:
        P: Projection matrix, shape (3, 4)
    """
    # Build components
    R = build_look_at_matrix(camera_pos, target_pos, up_world)
    K = build_intrinsic_matrix(focal_length, principal_point)

    # Translation: t = -R @ camera_pos
    t = -R @ camera_pos

    # Combine: P = K @ [R | t]
    Rt = np.column_stack([R, t])  # (3, 4)
    P = K @ Rt

    return P


def camera_from_placement(
    position: np.ndarray,
    target: np.ndarray,
    fov_deg: float,
    image_size: Tuple[int, int],
    up_world: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build camera parameters from placement specification.

    This is a convenience function for generating camera configurations
    from intuitive placement parameters (position, target, FOV) rather
    than requiring explicit K, R, t specification.

    Args:
        position: Camera position in world coords, shape (3,)
        target: Point the camera looks at, shape (3,)
        fov_deg: Horizontal field of view in degrees
        image_size: (width, height) in pixels
        up_world: World "up" direction, shape (3,)

    Returns:
        K: Intrinsic matrix (3, 3) - focal length and principal point
        R: Rotation matrix (3, 3) - world to camera frame
        t: Translation vector (3,) - for extrinsic matrix [R|t]

    Notes:
        - Focal length computed as: f = image_width / (2 * tan(fov_deg/2))
        - Principal point placed at image center: (width/2, height/2)
        - Translation vector t = -R @ position for standard extrinsics
    """
    width, height = image_size

    # Convert FOV to focal length
    # fov_deg is horizontal FOV, so we use image width
    fov_rad = np.deg2rad(fov_deg)
    focal_length = width / (2.0 * np.tan(fov_rad / 2.0))

    # Principal point at image center
    cx = width / 2.0
    cy = height / 2.0

    # Build intrinsic matrix
    K = build_intrinsic_matrix(focal_length, (cx, cy))

    # Build rotation matrix using look-at
    R = build_look_at_matrix(position, target, up_world)

    # Translation vector: t = -R @ position
    t = -R @ position

    return K, R, t
