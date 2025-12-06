"""Quick visualization of new camera projection system"""

import numpy as np
import matplotlib.pyplot as plt
from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence

# Generate data with new camera system
config = SyntheticDataConfig(
    T=100,
    C=3,
    S=3,
    kappa=5.0,
    obs_noise_std=3.0,
    occlusion_rate=0.1,
    random_seed=42,
)

print("Generating data with NEW perspective camera system...")
data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

# Plot 2D projections
timesteps_to_plot = [0, 25, 50, 75]
colors_by_state = ["red", "green", "blue"]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))

# Get axis limits
y_observed_2d = data.y_observed
u_min, u_max = np.nanmin(y_observed_2d[:, :, :, 0]), np.nanmax(
    y_observed_2d[:, :, :, 0]
)
v_min, v_max = np.nanmin(y_observed_2d[:, :, :, 1]), np.nanmax(
    y_observed_2d[:, :, :, 1]
)

u_range = u_max - u_min
v_range = v_max - v_min
padding = 0.2
u_lim = [u_min - padding * u_range, u_max + padding * u_range]
v_lim = [v_min - padding * v_range, v_max + padding * v_range]

print(f"\n2D projection ranges:")
print(f"  u: [{u_min:.1f}, {u_max:.1f}], extent = {u_range:.1f} pixels")
print(f"  v: [{v_min:.1f}, {v_max:.1f}], extent = {v_range:.1f} pixels")

for c in range(3):
    for t_idx, t in enumerate(timesteps_to_plot):
        ax = axes[c, t_idx]

        y_cam = y_observed_2d[c, t]

        # Plot bones
        for k, parent_idx in enumerate(DEMO_V0_1_SKELETON.parents):
            if parent_idx >= 0:
                parent_2d = y_cam[parent_idx]
                child_2d = y_cam[k]

                if not np.any(np.isnan(parent_2d)) and not np.any(np.isnan(child_2d)):
                    ax.plot(
                        [parent_2d[0], child_2d[0]],
                        [parent_2d[1], child_2d[1]],
                        "k-",
                        linewidth=2,
                        alpha=0.6,
                    )

        # Plot joints
        for k in range(len(DEMO_V0_1_SKELETON.joint_names)):
            if not np.any(np.isnan(y_cam[k])):
                ax.scatter(
                    y_cam[k, 0],
                    y_cam[k, 1],
                    c=colors_by_state[data.true_states[t]],
                    s=100,
                    edgecolors="black",
                    linewidth=1.5,
                    zorder=10,
                )

        ax.set_xlim(u_lim)
        ax.set_ylim(v_lim)

        if c == 2:
            ax.set_xlabel("u (pixels)")
        if t_idx == 0:
            ax.set_ylabel(f"Camera {c}\nv (pixels)")

        if c == 0:
            ax.set_title(f"t={t}, state={data.true_states[t]}")

        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

plt.suptitle(
    "NEW Perspective Cameras with Proper Orientations", fontsize=16, fontweight="bold"
)
plt.tight_layout()

output_path = "new_camera_projection_demo.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n✅ Saved visualization: {output_path}")

# Print per-camera statistics
print(f"\nPer-camera 2D extents:")
for c in range(3):
    y_cam_all = y_observed_2d[c, :, :, :]  # All timesteps
    u_extent = np.nanmax(y_cam_all[:, :, 0]) - np.nanmin(y_cam_all[:, :, 0])
    v_extent = np.nanmax(y_cam_all[:, :, 1]) - np.nanmin(y_cam_all[:, :, 1])
    print(f"  Camera {c}: u_extent={u_extent:.1f}px, v_extent={v_extent:.1f}px")

plt.show()
