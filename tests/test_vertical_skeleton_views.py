"""
Test: Vertical skeleton sanity check.

Verifies that cameras with proper orientations produce geometrically
correct views of a purely vertical skeleton:

- Front camera (from +X): Should see vertical line with large extent
- Side camera (from +Y): Should see vertical line with large extent
- Overhead camera (from +Z): Should see near-point with small extent

This was the key issue in the original orthographic, axis-aligned model.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gimbal.synthetic_data import generate_camera_matrices
from gimbal.camera_utils import project_points_numpy, camera_center_from_proj


def test_vertical_skeleton_views():
    """
    Test that a vertical skeleton produces correct views from different cameras.

    With proper perspective + orientation:
    - Cameras 0, 1 (horizontal): should see ~42 unit vertical extent
    - Camera 2 (overhead): should see ~0 unit extent (looking down at point)
    """

    print("=" * 70)
    print("TEST: Vertical Skeleton - Camera Views")
    print("=" * 70)

    # Create perfectly vertical skeleton (root at origin, extending up +Z)
    skeleton_vertical = np.array(
        [
            [0, 0, 100],  # Root
            [0, 0, 110],  # Joint 1 (+10 in Z)
            [0, 0, 120],  # Joint 2 (+10 in Z)
            [0, 0, 128],  # Joint 3 (+8 in Z)
            [0, 0, 136],  # Joint 4 (+8 in Z)
            [0, 0, 142],  # Joint 5 (+6 in Z)
        ]
    )

    # Total vertical extent
    z_min = skeleton_vertical[:, 2].min()
    z_max = skeleton_vertical[:, 2].max()
    vertical_extent = z_max - z_min

    print(f"\nVertical skeleton:")
    print(f"  Z extent: {z_min:.1f} to {z_max:.1f} ({vertical_extent:.1f} units)")
    print(f"  X extent: ~0 units (vertical line)")
    print(f"  Y extent: ~0 units (vertical line)")

    # Reshape for projection: (T=1, K=6, 3)
    x_true = skeleton_vertical[np.newaxis, :, :]

    # Generate cameras with new perspective + orientation model
    rng = np.random.default_rng(42)
    camera_proj = generate_camera_matrices(C=3, rng=rng)

    # Extract camera positions
    camera_centers = camera_center_from_proj(camera_proj)

    print(f"\nCamera positions:")
    scene_center = np.array([0.0, 0.0, 100.0])
    for c in range(3):
        print(
            f"  Camera {c}: [{camera_centers[c, 0]:7.2f}, {camera_centers[c, 1]:7.2f}, {camera_centers[c, 2]:7.2f}]"
        )

    # Project skeleton
    y_proj = project_points_numpy(x_true, camera_proj)  # (C=3, T=1, K=6, 2)

    # Compute 2D extents for each camera
    print(f"\n2D Projected extents:")
    print(
        f"  (With focal_length=10, distance~80, expect ~{vertical_extent * 10 / 80:.1f} pixels for horizontal cameras)"
    )

    extents = []
    for c in range(3):
        y_cam = y_proj[c, 0, :, :]  # (K=6, 2)

        u_min, u_max = y_cam[:, 0].min(), y_cam[:, 0].max()
        v_min, v_max = y_cam[:, 1].min(), y_cam[:, 1].max()

        u_extent = u_max - u_min
        v_extent = v_max - v_min

        extents.append((u_extent, v_extent))

        print(
            f"  Camera {c}: u_extent = {u_extent:6.2f} pixels, v_extent = {v_extent:6.2f} pixels"
        )

    # Check expectations
    print(f"\nValidation:")

    # Cameras 0 and 1 are horizontal - should see large extent
    success = True

    # Camera 0 (from +X): skeleton extends in Z, projects to v-axis
    # Camera 1 (from +Y): skeleton extends in Z, projects to u-axis
    # Expected extent: vertical_extent * focal_length / distance ≈ 42 * 10 / 80 ≈ 5.25 pixels
    horizontal_threshold = 3.0  # At least 3 pixels

    for c in [0, 1]:
        max_extent = max(extents[c])
        if max_extent >= horizontal_threshold:
            print(
                f"  ✅ Camera {c} (horizontal): max extent = {max_extent:.2f} pixels (>= {horizontal_threshold})"
            )
        else:
            print(
                f"  ❌ Camera {c} (horizontal): max extent = {max_extent:.2f} pixels (< {horizontal_threshold})"
            )
            success = False

    # Camera 2 (from above): should see near-point
    overhead_threshold = 2.0  # Less than 2 pixels
    max_extent_overhead = max(extents[2])
    if max_extent_overhead <= overhead_threshold:
        print(
            f"  ✅ Camera 2 (overhead): max extent = {max_extent_overhead:.2f} pixels (<= {overhead_threshold})"
        )
    else:
        print(
            f"  ❌ Camera 2 (overhead): max extent = {max_extent_overhead:.2f} pixels (> {overhead_threshold})"
        )
        success = False

    # Horizontal cameras should see MUCH larger extent than overhead
    ratio_threshold = 2.0
    min_horizontal = min(max(extents[0]), max(extents[1]))
    if min_horizontal / max_extent_overhead >= ratio_threshold:
        print(
            f"  ✅ Horizontal/overhead ratio = {min_horizontal / max_extent_overhead:.2f} (>= {ratio_threshold})"
        )
    else:
        print(
            f"  ❌ Horizontal/overhead ratio = {min_horizontal / max_extent_overhead:.2f} (< {ratio_threshold})"
        )
        success = False

    # Visualize
    visualize_projections(y_proj, extents, camera_centers, scene_center)

    print(f"\n{'=' * 70}")
    if success:
        print("✅ SUCCESS: Cameras produce geometrically correct views!")
        print("  - Horizontal cameras see vertical line (large extent)")
        print("  - Overhead camera sees point (small extent)")
        print("=" * 70)
        return True
    else:
        print("❌ FAILURE: Camera views don't match geometric expectations!")
        print("  This suggests orientation or projection issues.")
        print("=" * 70)
        return False


def visualize_projections(y_proj, extents, camera_centers, scene_center):
    """Create visualization of projected skeleton views."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    camera_names = ["Camera 0 (Front)", "Camera 1 (Side)", "Camera 2 (Overhead)"]

    for c in range(3):
        ax = axes[c]
        y_cam = y_proj[c, 0, :, :]  # (K=6, 2)

        # Plot skeleton points
        ax.scatter(y_cam[:, 0], y_cam[:, 1], s=100, c="blue", zorder=10)

        # Connect joints
        for k in range(len(y_cam) - 1):
            ax.plot(
                [y_cam[k, 0], y_cam[k + 1, 0]],
                [y_cam[k, 1], y_cam[k + 1, 1]],
                "k-",
                linewidth=2,
            )

        # Mark root
        ax.scatter(
            [y_cam[0, 0]],
            [y_cam[0, 1]],
            s=200,
            c="red",
            marker="o",
            zorder=11,
            label="Root",
        )

        ax.set_xlabel("u (pixels)")
        ax.set_ylabel("v (pixels)")
        ax.set_title(f"{camera_names[c]}\nExtent: {max(extents[c]):.2f} px")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()

    plt.suptitle("Vertical Skeleton: Projected Views", fontsize=14)
    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent.parent / "test_vertical_skeleton_views.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Visualization saved: {output_path}")

    plt.close()


if __name__ == "__main__":
    success = test_vertical_skeleton_views()

    if success:
        exit(0)
    else:
        exit(1)
