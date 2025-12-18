"""Skeleton configuration definitions for GIMBAL.

This module defines skeleton structures used throughout GIMBAL, including
joint hierarchies, bone lengths, and coordinate system conventions.

Note: This module currently defines only the v0.1 demo skeleton (6 joints).
It will be extended in v0.2.5 to support AIST++ and other real-world datasets.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SkeletonConfig:
    """Configuration for a skeletal structure.

    Attributes
    ----------
    joint_names : list[str]
        Names of joints in the skeleton (length K)
    parents : np.ndarray
        Parent indices for each joint, shape (K,)
        Root joint has parent index -1
    bone_lengths : np.ndarray | None
        Length of each bone segment, shape (K,)
        Root typically has length 0
        Can be None if bone lengths are unknown/variable
    up_axis : np.ndarray
        Unit vector indicating the "up" direction in the skeleton's
        coordinate system, shape (3,)
        Default is +Z axis: [0, 0, 1]
    """

    joint_names: list[str]
    parents: np.ndarray
    bone_lengths: np.ndarray | None
    up_axis: np.ndarray


# v0.1 Demo Skeleton: Simple 6-joint chain
# This is the skeleton used in demo_v0_1_complete and related notebooks
DEMO_V0_1_SKELETON = SkeletonConfig(
    joint_names=[
        "root",  # 0: Root/pelvis
        "joint1",  # 1: First segment
        "joint2",  # 2: Second segment
        "joint3",  # 3: Third segment
        "joint4",  # 4: Fourth segment
        "joint5",  # 5: Fifth segment (end effector)
    ],
    parents=np.array([-1, 0, 1, 2, 3, 4]),
    bone_lengths=np.array([0.0, 10.0, 10.0, 8.0, 8.0, 6.0]),
    up_axis=np.array([0.0, 0.0, 1.0]),
)


# v0.2.1 L00-L03 Skeleton: Minimal 3-joint Y-shaped structure
# Root with proximal joint, splitting into two distal joints
# Total: 3 joints, 3 bone segments
# Structure:  distal_left (joint 2)
#              /
#    root - proximal (joint 1)
#              \
#              distal_right (joint 3) [INCORRECT - see note]
#
# NOTE: This is ACTUALLY a 4-joint structure in the tree representation:
#   - joint 0: root (parent=-1)
#   - joint 1: proximal (parent=0, bone from root to proximal)
#   - joint 2: distal_left (parent=1, bone from proximal to distal_left)
#   - joint 3: distal_right (parent=1, bone from proximal to distal_right)
#
# The "3 joints, 3 segments" description from user maps to:
#   - 3 segments (bones): root→proximal, proximal→distal_left, proximal→distal_right
#   - 4 joints (articulation points including root)
L00_SKELETON = SkeletonConfig(
    joint_names=[
        "root",  # 0: Root/base
        "proximal",  # 1: Proximal joint
        "distal_left",  # 2: Left distal end
        "distal_right",  # 3: Right distal end
    ],
    parents=np.array([-1, 0, 1, 1]),  # root, proximal←root, both distals←proximal
    bone_lengths=np.array([0.0, 10.0, 8.0, 8.0]),  # root=0, proximal=10, distals=8 each
    up_axis=np.array([0.0, 0.0, 1.0]),
)


def validate_skeleton(skeleton: SkeletonConfig) -> None:
    """Validate that a skeleton configuration is well-formed.

    Parameters
    ----------
    skeleton : SkeletonConfig
        Skeleton configuration to validate

    Raises
    ------
    ValueError
        If skeleton configuration is invalid
    """
    K = len(skeleton.joint_names)

    # Check parents array
    if skeleton.parents.shape != (K,):
        raise ValueError(
            f"parents shape {skeleton.parents.shape} doesn't match "
            f"number of joints {K}"
        )

    # Check that root has parent -1
    root_indices = np.where(skeleton.parents == -1)[0]
    if len(root_indices) != 1:
        raise ValueError(
            f"Expected exactly one root (parent=-1), found {len(root_indices)}"
        )

    # Check that parent indices are valid
    for k, parent in enumerate(skeleton.parents):
        if parent != -1 and (parent < 0 or parent >= K):
            raise ValueError(
                f"Joint {k} has invalid parent index {parent} "
                f"(must be -1 or in [0, {K-1}])"
            )
        if parent >= k:
            raise ValueError(f"Joint {k} has parent {parent} >= k (non-tree structure)")

    # Check bone lengths if provided
    if skeleton.bone_lengths is not None:
        if skeleton.bone_lengths.shape != (K,):
            raise ValueError(
                f"bone_lengths shape {skeleton.bone_lengths.shape} "
                f"doesn't match number of joints {K}"
            )
        if np.any(skeleton.bone_lengths < 0):
            raise ValueError("bone_lengths must be non-negative")

    # Check up_axis
    if skeleton.up_axis.shape != (3,):
        raise ValueError(f"up_axis shape {skeleton.up_axis.shape} must be (3,)")
    norm = np.linalg.norm(skeleton.up_axis)
    if not np.isclose(norm, 1.0):
        raise ValueError(f"up_axis must be a unit vector (norm={norm:.4f})")


# Validate the demo skeletons on import
validate_skeleton(DEMO_V0_1_SKELETON)
validate_skeleton(L00_SKELETON)
