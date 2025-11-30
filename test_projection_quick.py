"""Quick test of camera projection after fix."""

import numpy as np
from gimbal.synthetic_data import generate_camera_matrices

# Generate camera matrices
scene_center = np.array([0.0, 0.0, 100.0])
cam_proj = generate_camera_matrices(3, np.random.default_rng(42))

print("Camera matrices generated with focal_length=1.0\n")

# Test projection of a point near skeleton root
skeleton_point = np.array([0, 0, 110, 1])  # Point 10 units above root

print("Testing projection of point [0, 0, 110]:")
for c in range(3):
    pixel = cam_proj[c] @ skeleton_point
    print(f"  Camera {c}: pixel = [{pixel[0]:.1f}, {pixel[1]:.1f}]")

# Test a point 10 units to the side
side_point = np.array([10, 0, 110, 1])
print("\nTesting projection of point [10, 0, 110]:")
for c in range(3):
    pixel = cam_proj[c] @ side_point
    print(f"  Camera {c}: pixel = [{pixel[0]:.1f}, {pixel[1]:.1f}]")

print("\nExpected: Pixel coordinates should be in range [-150, 150]")
print("          (based on distance ~150 and focal_length=1)")
