"""Runner script to generate datasets from configs.

Usage:
    python generate_dataset.py L00_minimal
    python generate_dataset.py L01_noise L02_outliers L03_missingness
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.pipeline.utils.config_generator import (
    load_config,
    generate_from_config,
    save_dataset,
)
from tests.pipeline.utils.metrics import compute_dataset_metrics, save_metrics
from tests.pipeline.utils.visualization import generate_all_figures


def main():
    """Generate datasets from command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python generate_dataset.py <config_name> [<config_name> ...]")
        print("Example: python generate_dataset.py L00_minimal L01_noise")
        sys.exit(1)

    config_names = sys.argv[1:]

    # Base paths
    config_dir = Path(__file__).parent / "configs" / "v0.2.1"
    output_base = Path(__file__).parent / "datasets" / "v0.2.1"

    for config_name in config_names:
        print(f"\n{'='*60}")
        print(f"Generating: {config_name}")
        print(f"{'='*60}")

        # Load config
        config_path = config_dir / f"{config_name}.json"
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            continue

        config = load_config(config_path)
        print(f"Loaded config from {config_path}")

        # Generate dataset
        print("Generating dataset...")
        dataset = generate_from_config(config, calibrate_noise=True)
        print(
            f"Generated {dataset.x_true.shape[0]} timesteps with {dataset.x_true.shape[1]} joints"
        )

        # Save dataset
        output_dir = output_base / config_name
        save_dataset(dataset, output_dir)

        # Compute and save metrics
        print("Computing metrics...")
        metrics = compute_dataset_metrics(dataset)
        metrics_path = output_dir / "metrics.json"
        save_metrics(metrics, metrics_path)

        # Generate figures
        generate_all_figures(dataset, output_dir)

        # Print summary
        print(f"\nSummary for {config_name}:")
        print(
            f"  Bone length max deviation: {metrics['bone_length']['max_relative_deviation']:.6f}"
        )
        print(
            f"  Direction norm mean: {metrics['direction_normalization']['mean_norm']:.6f}"
        )
        print(f"  Speed 95th pct: {metrics['smoothness']['speed']['p95']:.2f}")
        print(f"  Jerk 95th pct: {metrics['smoothness']['jerk']['p95']:.2f}")
        print(f"  NaN fraction: {metrics['2d_observations']['nan_fraction']:.4f}")
        print(f"  Config hash: {dataset.config_hash[:16]}...")

    print(f"\n{'='*60}")
    print(f"Generation complete. Datasets saved to {output_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
