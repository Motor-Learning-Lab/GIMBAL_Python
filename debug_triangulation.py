"""Debug triangulation issue - detailed"""

import numpy as np
import sys

sys.path.insert(0, ".")

# Test manually first
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
P2 = np.array([[1, 0, 0, -5], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)

X_true = np.array([2.0, 3.0, 10.0])
X_h = np.append(X_true, 1.0)

x1_h = P1 @ X_h
x2_h = P2 @ X_h
x1 = x1_h[:2] / x1_h[2]
x2 = x2_h[:2] / x2_h[2]

print(f"x1 = {x1}, x2 = {x2}")

# Manual DLT
u1, v1 = x1
u2, v2 = x2

A = []
A.append(u1 * P1[2, :] - P1[0, :])
A.append(v1 * P1[2, :] - P1[1, :])
A.append(u2 * P2[2, :] - P2[0, :])
A.append(v2 * P2[2, :] - P2[1, :])

A = np.array(A)
print(f"\nA matrix:\n{A}")
print(f"A shape: {A.shape}")

U_svd, S, Vt = np.linalg.svd(A)
print(f"\nSingular values: {S}")
print(f"Condition number: {S[0] / S[-1]}")

X_homog = Vt[-1, :]
print(f"\nX_homog: {X_homog}")
print(f"w = {X_homog[3]}")

X_reconstructed = X_homog[:3] / X_homog[3]
print(f"X_reconstructed: {X_reconstructed}")
print(f"X_true: {X_true}")
print(f"Error: {np.linalg.norm(X_reconstructed - X_true)}")

# Now test with gimbal
print("\n" + "=" * 60)
print("Testing with gimbal.triangulate_multi_view:")
import gimbal

# Create simple 3x3 projection matrices for 2 cameras
# Camera 1: Identity view
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)

# Camera 2: Translated along X
P2 = np.array([[1, 0, 0, -5], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)

camera_proj = np.stack([P1, P2], axis=0)  # (2, 3, 4)

# Known 3D point at (2, 3, 10)
X_true = np.array([2.0, 3.0, 10.0])

# Project to 2D
X_h = np.append(X_true, 1.0)  # Homogeneous
print(f"X_h: {X_h}")

x1_h = P1 @ X_h
x2_h = P2 @ X_h
print(f"x1_h: {x1_h}")
print(f"x2_h: {x2_h}")

x1 = x1_h[:2] / x1_h[2]
x2 = x2_h[:2] / x2_h[2]
print(f"x1: {x1}")
print(f"x2: {x2}")

# Create input: (C=2, T=1, K=1, 2)
keypoints_2d = np.array([[[[x1[0], x1[1]]]], [[[x2[0], x2[1]]]]])
print(f"keypoints_2d shape: {keypoints_2d.shape}")
print(f"keypoints_2d: {keypoints_2d}")

# Triangulate
positions_3d = gimbal.triangulate_multi_view(keypoints_2d, camera_proj)
print(f"positions_3d shape: {positions_3d.shape}")
print(f"positions_3d: {positions_3d}")

X_reconstructed = positions_3d[0, 0, :]
print(f"X_reconstructed: {X_reconstructed}")
print(f"X_true: {X_true}")
print(f"Error: {X_reconstructed - X_true}")
