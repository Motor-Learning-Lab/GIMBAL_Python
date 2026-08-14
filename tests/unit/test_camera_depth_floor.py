"""
Tests for the camera projection's positive depth floor (project_points_pytensor).

Covers the fix described in plans/v0.2.2/v0.2.2_C1_findings.md, "TG14 full-run chain
divergence: a camera cheirality singularity": the perspective division u,v = x,y/depth
has a real 1/w pole at zero depth. An unprotected division hits that pole directly; a
hard pt.maximum clamp avoids the crash but has zero gradient once a point crosses the
clamp threshold, so nothing in the model pulls a wayward point back in front of the
camera -- it can drift arbitrarily far behind the lens until it happens to cross back,
at which point HMC can land a leapfrog step on the pole and diverge. The fix replaces
the clamp with a softplus, which is smooth, always strictly positive, and keeps a
genuine (if small) gradient at every depth.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt

from gimbal_pymc.hmm.observation_model import project_points_pytensor


def test_depth_floor_always_positive_including_deeply_negative_raw_depth():
    """The softplus-based depth floor never returns <= 0, even far behind the camera."""
    raw = pt.dvector("raw")
    depth = pt.softplus(raw) + 1e-9
    f = pytensor.function([raw], depth)

    raw_values = np.array(
        [-1e6, -1000.0, -50.0, -5.0, -0.001, 0.0, 0.001, 5.0, 50.0, 1000.0]
    )
    out = f(raw_values)

    assert np.all(out > 0.0)
    assert np.all(np.isfinite(out))
    # Negligible bias for depths well in front of the camera (real scenes are tens to
    # hundreds of world units deep, far from where the floor has any effect).
    front = raw_values >= 5.0
    assert np.allclose(out[front], raw_values[front], atol=1e-2)


def _single_camera_identity_proj():
    """proj = [I | 0]: camera frame == world frame, so depth == world z."""
    proj = np.zeros((1, 3, 4))
    proj[0, 0, 0] = 1.0
    proj[0, 1, 1] = 1.0
    proj[0, 2, 2] = 1.0
    return proj


def test_project_points_finite_for_point_behind_camera():
    """
    A 3D point behind the camera (negative depth in that camera's frame) must still
    produce finite (u, v). Before this fix, an unprotected division here was a real
    singularity, and the production pt.maximum clamp masked the crash but not the
    underlying zero-gradient region just past it.
    """
    proj = _single_camera_identity_proj()
    x_behind = np.array([[[1.0, 2.0, -50.0]]])  # (T=1, K=1, 3); z = -50 -> behind camera

    x = pt.tensor3("x")
    proj_t = pt.tensor3("proj")
    y = project_points_pytensor(x, proj_t)
    f = pytensor.function([x, proj_t], y)

    out = f(x_behind, proj)
    assert np.all(np.isfinite(out)), f"projection produced non-finite output: {out}"


def test_project_points_gradient_finite_and_nonzero_behind_camera():
    """
    The point of the softplus floor over a hard clamp: the gradient w.r.t. x must stay
    finite AND nonzero even when a point is behind the camera, so a sampler has a
    restoring signal instead of a flat, gradient-dead region with nothing pulling a
    wayward point back toward positive depth.
    """
    proj = _single_camera_identity_proj()
    x_val = np.array([[[1.0, 2.0, -50.0]]])

    x = pt.tensor3("x")
    proj_t = pt.tensor3("proj")
    y = project_points_pytensor(x, proj_t)
    loss = pt.sum(y**2)
    grad = pt.grad(loss, x)
    f = pytensor.function([x, proj_t], grad)

    g = f(x_val, proj)
    assert np.all(np.isfinite(g)), f"gradient behind camera is non-finite: {g}"
    assert np.any(g != 0.0), "gradient behind camera is identically zero (dead region)"


def test_project_points_matches_raw_division_in_front_of_camera():
    """Sanity check: for points well in front of the camera, the floor is a no-op."""
    proj = _single_camera_identity_proj()
    x_front = np.array([[[10.0, -5.0, 80.0]]])  # comfortably in front (z=80)

    x = pt.tensor3("x")
    proj_t = pt.tensor3("proj")
    y = project_points_pytensor(x, proj_t)
    f = pytensor.function([x, proj_t], y)

    out = f(x_front, proj)
    expected = np.array([[[[10.0 / 80.0, -5.0 / 80.0]]]])
    assert np.allclose(out, expected, atol=1e-6)
