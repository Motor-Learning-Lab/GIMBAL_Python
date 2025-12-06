"""Debug why skeleton appears degenerate in camera views."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve()))

from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence

# Generate data
config = SyntheticDataConfig(T=5, C=3, S=3, random_seed=42)
data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

print("=" * 80)
print("SKELETON PROJECTION DEBUG")
print("=" * 80)

# Look at first timestep
t = 0
print(f"\nTimestep {t}:")
print(f"  State: {data.true_states[t]}")

# Print 3D skeleton positions
print("\n3D Skeleton Positions:")
for k, name in enumerate(DEMO_V0_1_SKELETON.joint_names):
    pos = data.x_true[t, k]
    print(f"  {name:10s}: [{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}]")

# Print bone vectors
print("\nBone Vectors (child - parent):")
for k in range(1, len(DEMO_V0_1_SKELETON.joint_names)):
    parent_idx = DEMO_V0_1_SKELETON.parents[k]
    bone_vec = data.x_true[t, k] - data.x_true[t, parent_idx]
    bone_len = np.linalg.norm(bone_vec)
    bone_dir = bone_vec / bone_len
    print(
        f"  {DEMO_V0_1_SKELETON.joint_names[k]:10s}: vec=[{bone_vec[0]:6.2f}, {bone_vec[1]:6.2f}, {bone_vec[2]:6.2f}], "
        f"len={bone_len:5.2f}, dir=[{bone_dir[0]:5.2f}, {bone_dir[1]:5.2f}, {bone_dir[2]:5.2f}]"
    )

# Compute skeleton extent
x_extent = data.x_true[t, :, 0].max() - data.x_true[t, :, 0].min()
y_extent = data.x_true[t, :, 1].max() - data.x_true[t, :, 1].min()
z_extent = data.x_true[t, :, 2].max() - data.x_true[t, :, 2].min()
print(f"\nSkeleton Extent:")
print(f"  X: {x_extent:.2f} units")
print(f"  Y: {y_extent:.2f} units")
print(f"  Z: {z_extent:.2f} units")

# Print camera info
print("\n" + "=" * 80)
print("CAMERA INFORMATION")
print("=" * 80)
for c in range(data.config.C):
    P = data.camera_proj[c]
    A = P[:, :3]
    b = P[:, 3]
    focal_length = A[0, 0]
    cam_pos = -b / focal_length

    print(f"\nCamera {c}:")
    print(f"  Position: [{cam_pos[0]:7.2f}, {cam_pos[1]:7.2f}, {cam_pos[2]:7.2f}]")
    print(
        f"  Distance to skeleton root: {np.linalg.norm(cam_pos - data.x_true[t, 0]):.2f}"
    )
    print(f"  Focal length: {focal_length}")

    # Project all skeleton points
    print(f"  Projected skeleton:")
    for k, name in enumerate(DEMO_V0_1_SKELETON.joint_names):
        pt_3d = data.x_true[t, k]
        pt_hom = np.append(pt_3d, 1)
        px = P @ pt_hom
        print(
            f"    {name:10s}: 3D [{pt_3d[0]:7.2f}, {pt_3d[1]:7.2f}, {pt_3d[2]:7.2f}] -> "
            f"2D [{px[0]:8.1f}, {px[1]:8.1f}]"
        )

    # Get actual observations (with noise)
    print(f"  Observed 2D (with noise/occlusions):")
    for k, name in enumerate(DEMO_V0_1_SKELETON.joint_names):
        obs = data.y_observed[c, t, k]
        if np.any(np.isnan(obs)):
            print(f"    {name:10s}: OCCLUDED")
        else:
            print(f"    {name:10s}: [{obs[0]:8.1f}, {obs[1]:8.1f}]")

    # Compute 2D extent
    projected_points = []
    for k in range(len(DEMO_V0_1_SKELETON.joint_names)):
        pt_hom = np.append(data.x_true[t, k], 1)
        px = P @ pt_hom
        projected_points.append(px[:2])
    projected_points = np.array(projected_points)

    u_extent = projected_points[:, 0].max() - projected_points[:, 0].min()
    v_extent = projected_points[:, 1].max() - projected_points[:, 1].min()

    print(f"  2D Extent: u={u_extent:.1f}, v={v_extent:.1f} pixels")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# Check if skeleton is actually vertical or horizontal
if z_extent > max(x_extent, y_extent):
    print("\n✓ Skeleton is primarily VERTICAL (extends in Z)")
    print(f"  Main extent: {z_extent:.1f} units in Z")
    print(f"  Side extents: {x_extent:.1f} in X, {y_extent:.1f} in Y")
else:
    print("\n✗ Skeleton is NOT primarily vertical!")
    print(f"  X extent: {x_extent:.1f}")
    print(f"  Y extent: {y_extent:.1f}")
    print(f"  Z extent: {z_extent:.1f}")

# Check camera positions relative to skeleton
skeleton_center_z = data.x_true[t, :, 2].mean()
print(f"\nSkeleton vertical center: Z = {skeleton_center_z:.1f}")
print("Camera Z positions relative to skeleton:")
for c in range(data.config.C):
    P = data.camera_proj[c]
    A = P[:, :3]
    b = P[:, 3]
    focal_length = A[0, 0]
    cam_pos = -b / focal_length
    print(
        f"  Camera {c}: Z = {cam_pos[2]:7.1f} ({'BELOW' if cam_pos[2] < skeleton_center_z else 'ABOVE'} skeleton center)"
    )
