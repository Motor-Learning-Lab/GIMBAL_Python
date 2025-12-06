import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve()))

from gimbal import DEMO_V0_1_SKELETON, SyntheticDataConfig, generate_demo_sequence

config = SyntheticDataConfig(T=100, C=3, S=3, random_seed=42)
data = generate_demo_sequence(DEMO_V0_1_SKELETON, config)

print("Skeleton extents and camera views at visualized timesteps:\n")
for t in [0, 25, 50, 75]:
    extent_x = data.x_true[t, :, 0].max() - data.x_true[t, :, 0].min()
    extent_y = data.x_true[t, :, 1].max() - data.x_true[t, :, 1].min()
    extent_z = data.x_true[t, :, 2].max() - data.x_true[t, :, 2].min()

    print(f"t={t:3d}, state={data.true_states[t]}:")
    print(f"  3D extent: X={extent_x:5.2f}, Y={extent_y:5.2f}, Z={extent_z:5.2f}")

    # Check 2D extents for each camera
    for c in range(3):
        y_cam = data.y_observed[c, t]
        if not np.all(np.isnan(y_cam)):
            u_extent = np.nanmax(y_cam[:, 0]) - np.nanmin(y_cam[:, 0])
            v_extent = np.nanmax(y_cam[:, 1]) - np.nanmin(y_cam[:, 1])
            print(f"  Camera {c}: 2D extent u={u_extent:6.1f}, v={v_extent:6.1f}")
    print()
