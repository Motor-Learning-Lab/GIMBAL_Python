"""
Stage E: Direction Statistics for L00_minimal dataset

Compute empirical directional statistics (mean direction, concentration) for each joint.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gimbal


def plot_direction_distributions(
    stats: Dict, joint_names: list[str], output_path: Path
) -> None:
    """Plot mean directions and concentration parameters."""
    # Filter out root (has NaN stats)
    non_root_joints = [
        name for name in joint_names if not np.any(np.isnan(stats[name]["mu"]))
    ]
    K_non_root = len(non_root_joints)

    if K_non_root == 0:
        print("  WARNING: No valid directional statistics to plot")
        return

    fig = plt.figure(figsize=(16, 8))

    # 3D plot of mean directions
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_title("Mean Bone Directions")

    for joint_name in non_root_joints:
        mu = stats[joint_name]["mu"]
        # Plot as arrow from origin
        ax3d.quiver(
            0,
            0,
            0,
            mu[0],
            mu[1],
            mu[2],
            arrow_length_ratio=0.1,
            linewidth=2,
            label=joint_name,
        )

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_xlim([-1, 1])
    ax3d.set_ylim([-1, 1])
    ax3d.set_zlim([-1, 1])
    ax3d.legend()

    # Bar plot of kappa values
    ax_bar = fig.add_subplot(1, 2, 2)
    ax_bar.set_title("Concentration Parameters (κ)")

    kappas = [stats[name]["kappa"] for name in non_root_joints]
    x_pos = np.arange(len(non_root_joints))

    ax_bar.bar(x_pos, kappas, color="steelblue", alpha=0.7)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(non_root_joints, rotation=45, ha="right")
    ax_bar.set_ylabel("κ")
    ax_bar.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def format_stats_for_json(stats: Dict) -> Dict:
    """Convert stats to JSON-serializable format."""
    json_stats = {}
    for joint_name, joint_stats in stats.items():
        json_stats[joint_name] = {
            "mu": (
                joint_stats["mu"].tolist()
                if isinstance(joint_stats["mu"], np.ndarray)
                else joint_stats["mu"]
            ),
            "kappa": (
                float(joint_stats["kappa"])
                if not np.isnan(joint_stats["kappa"])
                else None
            ),
            "n_samples": int(joint_stats["n_samples"]),
        }
    return json_stats


def run_stage_e(
    dataset_dir: Path, stage_d_output_dir: Path, output_dir: Path
) -> Dict[str, Any]:
    """
    Run Stage E: Direction Statistics

    Parameters
    ----------
    dataset_dir : Path
        Directory containing original dataset (for skeleton config)
    stage_d_output_dir : Path
        Directory containing Stage D outputs (x_3d_clean.npz)
    output_dir : Path
        Directory for outputs

    Returns
    -------
    metrics : dict
        Direction statistics and metadata
    """
    print("=" * 80)
    print("STAGE E: Direction Statistics")
    print("=" * 80)

    # Load config
    with open(dataset_dir / "config.json") as f:
        config = json.load(f)

    joint_names = config["dataset_spec"]["skeleton"]["joint_names"]
    parents = np.array(config["dataset_spec"]["skeleton"]["parents"])

    # Load cleaned 3D from Stage D
    stage_d_data = np.load(stage_d_output_dir / "x_3d_clean.npz")
    x_3d_clean = stage_d_data["x_3d_clean"]
    use_for_stats_mask = stage_d_data["use_for_stats_mask"]

    T, K, _ = x_3d_clean.shape
    print(f"Cleaned 3D shape: {x_3d_clean.shape}")
    print(
        f"Stats-usable points: {np.sum(use_for_stats_mask)} / {T * K} ({100 * np.sum(use_for_stats_mask) / (T * K):.1f}%)"
    )

    # Compute direction statistics
    print("\n[1/3] Computing directional statistics...")
    stats = gimbal.compute_direction_statistics(
        x_3d_clean, parents, use_for_stats_mask, joint_names, min_samples=10
    )

    print("\nStatistics per joint:")
    for joint_name in joint_names:
        s = stats[joint_name]
        if s["n_samples"] == 0:
            print(f"  {joint_name}: [ROOT - no parent]")
        else:
            mu_str = f"[{s['mu'][0]:.3f}, {s['mu'][1]:.3f}, {s['mu'][2]:.3f}]"
            print(f"  {joint_name}: μ={mu_str}, κ={s['kappa']:.2f}, n={s['n_samples']}")

    # Validate statistics
    print("\n[2/3] Validating statistics...")
    validation = {
        "all_joints_have_data": True,
        "sufficient_samples": True,
        "reasonable_kappa": True,
        "issues": [],
    }

    for joint_name in joint_names:
        s = stats[joint_name]
        if parents[joint_names.index(joint_name)] >= 0:  # Non-root
            if s["n_samples"] == 0:
                validation["all_joints_have_data"] = False
                validation["issues"].append(f"{joint_name} has no valid samples")
            elif s["n_samples"] < 100:
                validation["sufficient_samples"] = False
                validation["issues"].append(
                    f"{joint_name} has only {s['n_samples']} samples (< 100)"
                )

            if not np.isnan(s["kappa"]) and (s["kappa"] < 0 or s["kappa"] > 200):
                validation["reasonable_kappa"] = False
                validation["issues"].append(
                    f"{joint_name} has unusual κ={s['kappa']:.2f}"
                )

    if validation["issues"]:
        print("  ⚠️  Validation issues:")
        for issue in validation["issues"]:
            print(f"    - {issue}")
    else:
        print("  ✓ All validations passed")

    # Plot statistics
    print("\n[3/3] Plotting directional statistics...")
    figures_dir = output_dir / "figures"
    plot_direction_distributions(
        stats, joint_names, figures_dir / "directions_statistics.png"
    )
    print(f"  Saved direction plots")

    # Compile metrics
    full_metrics = {
        "stage": "E_direction_statistics",
        "statistics": format_stats_for_json(stats),
        "validation": validation,
        "summary": {
            "total_joints": len(joint_names),
            "non_root_joints": sum(
                1 for name in joint_names if stats[name]["n_samples"] > 0
            ),
            "mean_samples_per_joint": float(
                np.mean([s["n_samples"] for s in stats.values() if s["n_samples"] > 0])
            ),
            "mean_kappa": float(
                np.nanmean(
                    [s["kappa"] for s in stats.values() if not np.isnan(s["kappa"])]
                )
            ),
        },
    }

    # Save metrics
    output_path = output_dir / "direction_stats.json"
    with open(output_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\n✓ Stage E complete. Statistics saved to {output_path}")

    return full_metrics


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    repo_root = Path(__file__).parent.parent.parent
    dataset_dir = repo_root / "tests" / "pipeline" / "datasets" / "v0.2.1_L00_minimal"
    stage_d_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"
    output_dir = repo_root / "tests" / "pipeline" / "fits" / "v0.2.1_L00_minimal"

    metrics = run_stage_e(dataset_dir, stage_d_dir, output_dir)

    passed = (
        metrics["validation"]["all_joints_have_data"]
        and metrics["validation"]["sufficient_samples"]
        and metrics["validation"]["reasonable_kappa"]
    )

    if passed:
        print("\n" + "=" * 80)
        print("STAGE E: PASSED ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("STAGE E: WARNING - Validation issues detected")
        print("=" * 80)
