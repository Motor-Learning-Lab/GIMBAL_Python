"""Analyze 3D skeleton positions and camera views"""
import numpy as np
from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence
from gimbal.camera_utils import camera_center_from_proj

config = SyntheticDataConfig(T=100, C=3, S=3, kappa=5.0, random_seed=42)
data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

# Analyze 3D extents
x_true = data.x_true  # (T, K, 3)

print("3D Skeleton Analysis")
print("=" * 60)

for t in [0, 25, 50, 75]:
    x_t = x_true[t]
    state = data.true_states[t]
    
    x_extent = x_t[:, 0].max() - x_t[:, 0].min()
    y_extent = x_t[:, 1].max() - x_t[:, 1].min()
    z_extent = x_t[:, 2].max() - x_t[:, 2].min()
    
    print(f"\nt={t}, state={state}:")
    print(f"  Root position: [{x_t[0, 0]:.2f}, {x_t[0, 1]:.2f}, {x_t[0, 2]:.2f}]")
    print(f"  3D extents: X={x_extent:.2f}, Y={y_extent:.2f}, Z={z_extent:.2f} units")
    
    # Compute 2D extents for each camera
    y_2d = data.y_observed[:, t, :, :]  # (C, K, 2)
    for c in range(3):
        y_cam = y_2d[c]
        u_extent = np.nanmax(y_cam[:, 0]) - np.nanmin(y_cam[:, 0])
        v_extent = np.nanmax(y_cam[:, 1]) - np.nanmin(y_cam[:, 1])
        print(f"    Camera {c}: 2D extent u={u_extent:.2f}, v={v_extent:.2f} pixels")

# Camera positions
print(f"\n" + "=" * 60)
print("Camera Positions:")
centers = camera_center_from_proj(data.camera_proj)
scene_center = np.array([0.0, 0.0, 100.0])
for c in range(3):
    dist = np.linalg.norm(centers[c] - scene_center)
    print(f"  Camera {c}: [{centers[c, 0]:.1f}, {centers[c, 1]:.1f}, {centers[c, 2]:.1f}], dist={dist:.1f}")

# Expected projection scale
focal_length = 10.0
distance = 80.0
print(f"\n" + "=" * 60)
print(f"Expected Projection Scale:")
print(f"  Focal length: {focal_length}")
print(f"  Distance: ~{distance} units")
print(f"  Scale factor: f/d = {focal_length/distance:.3f}")
print(f"  A 40-unit 3D extent should project to ~{40 * focal_length / distance:.1f} pixels")
print(f"  A 20-unit 3D extent should project to ~{20 * focal_length / distance:.1f} pixels")
