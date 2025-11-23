"""Minimal demo script for the GIMBAL sampler.

This script simulates a tiny synthetic dataset with two cameras and a
simple 2-joint skeleton, constructs rough model parameters, and runs a
few MCMC iterations, printing the posterior mean 3D trajectories.

The goal is to provide an end-to-end runnable example consistent with
`GIMBAL spec.md`, not to be a faithful reproduction of the original
experiments.
"""

from __future__ import annotations

import torch

from gimbal.camera import project_points
from gimbal.fit_params import build_gimbal_parameters
from gimbal.inference import HMCConfig, run_gibbs_sampler


def simulate_synthetic_data(T: int = 20):
    device = torch.device("cpu")

    # Simple 2-joint skeleton: root (0) and child (1)
    K = 2
    parent = torch.tensor([-1, 0], dtype=torch.long, device=device)

    # Ground-truth root motion: straight line
    t = torch.linspace(0, 1, T, device=device)
    root_traj = torch.stack(
        [t * 100.0, torch.zeros_like(t), torch.ones_like(t) * 1000.0], dim=-1
    )

    # Child offset in world coordinates
    offset = torch.tensor([50.0, 0.0, 0.0], device=device)
    x_gt = torch.zeros(T, K, 3, device=device)
    x_gt[:, 0] = root_traj
    x_gt[:, 1] = root_traj + offset

    # Two simple pinhole cameras
    C = 2
    proj = torch.zeros(C, 3, 4, device=device)
    # Camera 0: front-facing
    proj[0, :, :3] = torch.eye(3, device=device)
    proj[0, :, 3] = torch.tensor([0.0, 0.0, 0.0], device=device)
    # Camera 1: shifted to the side
    proj[1, :, :3] = torch.eye(3, device=device)
    proj[1, :, 3] = torch.tensor([100.0, 0.0, 0.0], device=device)

    y_clean = project_points(x_gt.view(T * K, 3), proj).view(T, K, C, 2)
    noise = torch.randn_like(y_clean) * 1.0
    y_obs = y_clean + noise

    return x_gt, y_obs, proj, parent


def main():
    torch.manual_seed(0)

    T = 20
    x_gt, y_obs, proj, parent = simulate_synthetic_data(T=T)

    # Build parameters using ground truth
    num_states = 3
    params = build_gimbal_parameters(x_gt, parent, y_obs, proj, num_states)

    # Initial latent variables
    x_init = x_gt.clone()
    u_init = torch.zeros_like(x_init)
    s_init = torch.zeros(T, dtype=torch.long)
    h_init = torch.zeros(T, dtype=torch.float32)
    z_init = torch.zeros(T, x_gt.shape[1], y_obs.shape[2])

    num_iters = 5
    hmc_cfg = HMCConfig(step_size=0.001, num_steps=5)

    xs, us, ss, hs, zs = run_gibbs_sampler(
        x_init=x_init,
        u_init=u_init,
        s_init=s_init,
        h_init=h_init,
        z_init=z_init,
        y=y_obs,
        proj=proj,
        params=params,
        num_iters=num_iters,
        hmc_config=hmc_cfg,
    )

    # Posterior mean over iterations
    x_mean = xs.mean(dim=0)

    print("Posterior mean 3D trajectories (first few frames, first joint):")
    print(x_mean[:5, 0])


if __name__ == "__main__":
    main()
