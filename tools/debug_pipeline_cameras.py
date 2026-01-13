"""Quick diagnostic to see what's wrong with pipeline camera configuration."""

import numpy as np
from pathlib import Path
from tests.pipeline.utils import (
    load_config,
    generate_from_config,
    compute_dataset_metrics,
)

config_path = Path("tests/pipeline/configs/v0.2.1/L00_minimal.json")
config = load_config(config_path)
dataset = generate_from_config(config, calibrate_noise=True)
metrics = compute_dataset_metrics(dataset)

print("\n=== Camera Configuration ===")
for i, cam in enumerate(dataset.camera_metadata):
    print(f"\nCamera {i}: {cam['name']}")
    print(f"  Position: {cam['position']}")
    print(f"  Target: {cam['target']}")
    print(f"  Image size: {cam['image_size']}")

print("\n=== Skeleton bounds ===")
x_min = dataset.x_true.min(axis=(0, 1))
x_max = dataset.x_true.max(axis=(0, 1))
x_mean = dataset.x_true.mean(axis=(0, 1))
print(f"X range: [{x_min[0]:.1f}, {x_max[0]:.1f}], mean: {x_mean[0]:.1f}")
print(f"Y range: [{x_min[1]:.1f}, {x_max[1]:.1f}], mean: {x_mean[1]:.1f}")
print(f"Z range: [{x_min[2]:.1f}, {x_max[2]:.1f}], mean: {x_mean[2]:.1f}")

print("\n=== Observation metrics ===")
obs_metrics = metrics["2d_observations"]
print(f"Total observations: {obs_metrics['total_observations']}")
print(f"Valid observations: {obs_metrics['valid_observations']}")
print(f"NaN fraction: {obs_metrics['nan_fraction']:.4f}")
print(f"Bounds violations: {obs_metrics['bounds_violations']}")
print(f"Bounds violation rate: {obs_metrics['bounds_violation_rate']:.4f}")

print("\n=== Identifiability metrics ===")
ident_metrics = metrics["identifiability"]
print(f"Passed: {ident_metrics['passed']}")
print(f"Fraction good: {ident_metrics['fraction_good']:.4f}")
print(f"Mean min angle: {ident_metrics['mean_min_angle']:.2f}°")
print(f"Min min angle: {ident_metrics['min_min_angle']:.2f}°")

# Sample a few 2D observations to check values
print("\n=== Sample 2D projections (cam 0, time 0) ===")
for k in range(min(3, dataset.y_2d.shape[2])):
    u, v = dataset.y_2d[0, 0, k]
    print(f"Joint {k}: ({u:.1f}, {v:.1f})")
