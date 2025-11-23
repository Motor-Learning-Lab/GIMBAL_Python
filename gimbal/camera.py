"""Camera projection utilities for GIMBAL (Section 1, Camera model).

We assume a calibrated projective mapping f_c: R^3 -> R^2 of the form

    (u, v, w)^T = A_c x + b_c,
    f_c(x) = (u/w, v/w)^T.

This module primarily provides a batched projection function used by
`gimbal.model.log_observation_likelihood` and the example script.
"""

from __future__ import annotations

import torch
from torch import Tensor


def project_points(x: Tensor, proj: Tensor) -> Tensor:
    """Project 3D points into multiple cameras.

    Parameters
    ----------
    x : Tensor
        3D points, shape (..., 3).
    proj : Tensor
        Camera parameters, shape (C, 3, 4) where each row is [A_c | b_c].

    Returns
    -------
    Tensor
        2D projections, shape (..., C, 2).
    """

    C = proj.shape[0]
    x_h = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)  # (..., 4)
    x_cam = torch.einsum("cij,...j->...ci", proj, x_h)  # (..., C, 3)
    u = x_cam[..., 0]
    v = x_cam[..., 1]
    w = x_cam[..., 2].clamp_min(1e-6)
    return torch.stack([u / w, v / w], dim=-1)  # (..., C, 2)
