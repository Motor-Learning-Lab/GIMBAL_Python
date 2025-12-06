"""Debug camera orientations"""

from gimbal.synthetic_data import generate_camera_matrices
from gimbal.camera_utils import camera_center_from_proj
import numpy as np

rng = np.random.default_rng(42)
camera_proj = generate_camera_matrices(C=3, rng=rng)

print("Camera matrices:")
for c in range(3):
    print(f"\nCamera {c}:")
    print(camera_proj[c])

    # Extract components
    P = camera_proj[c]

    # Try to extract K, R, t
    # P = K[R|t] where P is (3, 4)
    M = P[:, :3]  # First 3 columns
    print(f"  M (should be K@R):")
    print(f"    {M}")

    # Camera center
    center = camera_center_from_proj(P)
    print(f"  Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

# Test projection manually
print("\n\nManual projection test:")
scene_center = np.array([0.0, 0.0, 100.0])
point = scene_center + np.array([0, 0, 10])  # 10 units above center

for c in range(3):
    point_h = np.append(point, 1)
    proj = camera_proj[c] @ point_h
    print(
        f"Camera {c}: point={point}, proj={proj}, u/w={proj[0]/proj[2]:.3f}, v/w={proj[1]/proj[2]:.3f}"
    )
