"""Test script to diagnose and fix camera positioning for synthetic data."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Simulate a simple skeleton at the scene
def create_test_skeleton():
    """Create a simple test skeleton centered at [0, 0, 100]."""
    # Root at [0, 0, 100]
    # 5 joints extending upward/outward
    skeleton_3d = np.array(
        [
            [0, 0, 100],  # Root
            [0, 0, 110],  # Joint 1 (10 units up)
            [0, 0, 120],  # Joint 2 (20 units up)
            [5, 0, 115],  # Joint 3 (side)
            [-5, 0, 115],  # Joint 4 (other side)
            [0, 5, 115],  # Joint 5 (front)
        ]
    )
    return skeleton_3d


def project_point(point_3d, camera_proj):
    """Project a 3D point using camera matrix."""
    # Convert to homogeneous coordinates
    point_h = np.append(point_3d, 1)
    # Project: y = P @ x
    y = camera_proj @ point_h
    # Return pixel coordinates (u, v)
    return y[:2]


def test_camera_setup(camera_pos, scene_center, focal_length):
    """Test a camera configuration."""
    print(f"\n{'='*60}")
    print(f"Testing camera at: {camera_pos}")
    print(f"Scene center: {scene_center}")
    print(f"Focal length: {focal_length}")

    # Create projection matrix
    A = np.eye(3) * focal_length
    b = -A @ camera_pos
    camera_proj = np.column_stack([A, b])

    print(f"\nCamera matrix P:")
    print(camera_proj)

    # Create test skeleton
    skeleton_3d = create_test_skeleton()

    # Project all points
    skeleton_2d = []
    for point in skeleton_3d:
        pixel = project_point(point, camera_proj)
        skeleton_2d.append(pixel)
    skeleton_2d = np.array(skeleton_2d)

    print(f"\n3D skeleton bounds:")
    print(f"  X: [{skeleton_3d[:, 0].min():.1f}, {skeleton_3d[:, 0].max():.1f}]")
    print(f"  Y: [{skeleton_3d[:, 1].min():.1f}, {skeleton_3d[:, 1].max():.1f}]")
    print(f"  Z: [{skeleton_3d[:, 2].min():.1f}, {skeleton_3d[:, 2].max():.1f}]")

    print(f"\n2D projections (pixels):")
    print(
        f"  u: [{skeleton_2d[:, 0].min():.1f}, {skeleton_2d[:, 0].max():.1f}] (range: {skeleton_2d[:, 0].max() - skeleton_2d[:, 0].min():.1f})"
    )
    print(
        f"  v: [{skeleton_2d[:, 1].min():.1f}, {skeleton_2d[:, 1].max():.1f}] (range: {skeleton_2d[:, 1].max() - skeleton_2d[:, 1].min():.1f})"
    )

    # Check if projection is reasonable (e.g., fits in a typical image)
    u_range = skeleton_2d[:, 0].max() - skeleton_2d[:, 0].min()
    v_range = skeleton_2d[:, 1].max() - skeleton_2d[:, 1].min()

    # Distance from camera to scene center
    distance = np.linalg.norm(camera_pos - scene_center)
    print(f"\nCamera-to-scene distance: {distance:.1f}")

    # Angular size estimate
    skeleton_size = 25  # Rough estimate of skeleton extent
    angular_size_rad = skeleton_size / distance
    pixel_size = angular_size_rad * focal_length
    print(f"Expected pixel extent: ~{pixel_size:.1f} pixels")

    return skeleton_3d, skeleton_2d, camera_proj


def visualize_setup(camera_positions, scene_center, focal_length):
    """Visualize multiple camera setups."""
    fig = plt.figure(figsize=(18, 6))

    skeleton_3d = create_test_skeleton()

    # 3D view
    ax_3d = fig.add_subplot(1, 4, 1, projection="3d")

    # Plot skeleton
    ax_3d.plot(
        skeleton_3d[:, 0],
        skeleton_3d[:, 1],
        skeleton_3d[:, 2],
        "o-",
        markersize=10,
        linewidth=2,
        label="Skeleton",
    )
    ax_3d.scatter(
        [skeleton_3d[0, 0]],
        [skeleton_3d[0, 1]],
        [skeleton_3d[0, 2]],
        c="red",
        s=200,
        marker="*",
        label="Root",
        zorder=100,
    )

    # Plot cameras
    for i, cam_pos in enumerate(camera_positions):
        ax_3d.scatter(
            [cam_pos[0]],
            [cam_pos[1]],
            [cam_pos[2]],
            c="orange",
            s=200,
            marker="^",
            edgecolors="black",
            linewidth=2,
            label=f"Cam {i}",
            zorder=20,
        )
        # Draw line to scene center
        ax_3d.plot(
            [cam_pos[0], scene_center[0]],
            [cam_pos[1], scene_center[1]],
            [cam_pos[2], scene_center[2]],
            "orange",
            linewidth=1,
            alpha=0.3,
            linestyle="--",
        )

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title("3D Scene")
    ax_3d.legend(fontsize=8)

    # Set equal aspect
    ax_3d.set_xlim([scene_center[0] - 100, scene_center[0] + 100])
    ax_3d.set_ylim([scene_center[1] - 100, scene_center[1] + 100])
    ax_3d.set_zlim([scene_center[2] - 50, scene_center[2] + 50])

    # 2D projections for each camera
    for i, cam_pos in enumerate(camera_positions[:3]):
        A = np.eye(3) * focal_length
        b = -A @ cam_pos
        camera_proj = np.column_stack([A, b])

        skeleton_2d = []
        for point in skeleton_3d:
            pixel = project_point(point, camera_proj)
            skeleton_2d.append(pixel)
        skeleton_2d = np.array(skeleton_2d)

        ax_2d = fig.add_subplot(1, 4, i + 2)
        ax_2d.plot(
            skeleton_2d[:, 0], skeleton_2d[:, 1], "o-", markersize=8, linewidth=2
        )
        ax_2d.scatter(
            [skeleton_2d[0, 0]],
            [skeleton_2d[0, 1]],
            c="red",
            s=100,
            marker="*",
            zorder=100,
        )
        ax_2d.set_xlabel("u (pixels)")
        ax_2d.set_ylabel("v (pixels)")
        ax_2d.set_title(f"Camera {i} View")
        ax_2d.invert_yaxis()
        ax_2d.grid(True, alpha=0.3)
        ax_2d.set_aspect("equal")

        # Add range info
        u_range = skeleton_2d[:, 0].max() - skeleton_2d[:, 0].min()
        v_range = skeleton_2d[:, 1].max() - skeleton_2d[:, 1].min()
        ax_2d.text(
            0.05,
            0.95,
            f"Range: {u_range:.0f}×{v_range:.0f}",
            transform=ax_2d.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Testing Camera Positioning for GIMBAL Synthetic Data")
    print("=" * 60)

    scene_center = np.array([0.0, 0.0, 100.0])

    # Test current setup
    print("\n\n### CURRENT SETUP (from code) ###")
    camera_positions_current = [
        scene_center + np.array([150, 0, 0]),
        scene_center + np.array([0, 150, 0]),
        scene_center + np.array([0, 0, 100]),
    ]
    focal_length_current = 500

    for i, cam_pos in enumerate(camera_positions_current):
        test_camera_setup(cam_pos, scene_center, focal_length_current)

    visualize_setup(camera_positions_current, scene_center, focal_length_current)

    # Test alternative setup with smaller focal length
    print("\n\n### PROPOSED SETUP (adjusted focal length) ###")
    focal_length_new = 50  # Much smaller focal length

    for i, cam_pos in enumerate(camera_positions_current):
        test_camera_setup(cam_pos, scene_center, focal_length_new)

    visualize_setup(camera_positions_current, scene_center, focal_length_new)

    # Test alternative setup with closer cameras
    print("\n\n### ALTERNATIVE SETUP (closer cameras) ###")
    camera_positions_close = [
        scene_center + np.array([50, 0, 0]),  # Much closer
        scene_center + np.array([0, 50, 0]),
        scene_center + np.array([0, 0, 30]),
    ]
    focal_length_alt = 100

    for i, cam_pos in enumerate(camera_positions_close):
        test_camera_setup(cam_pos, scene_center, focal_length_alt)

    visualize_setup(camera_positions_close, scene_center, focal_length_alt)

    print("\n\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    print("For skeleton centered at [0, 0, 100] with extent ~25 units:")
    print("  - Option 1: Keep distance=150, reduce focal_length to 50-100")
    print("  - Option 2: Reduce distance to 50-80, use focal_length=100-200")
    print("  - Target: Skeleton should span 100-300 pixels in 2D view")
