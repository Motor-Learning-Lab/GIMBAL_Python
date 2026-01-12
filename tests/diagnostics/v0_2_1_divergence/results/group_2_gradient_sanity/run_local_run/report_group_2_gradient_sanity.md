# Group 2: Gradient Sanity Check

**Run ID:** local_run
**Timestamp:** 2025-12-20T00:32:52.735105
**Status:** ✓ SUCCESS

## Configuration

- **Dataset:** v0.2.1_L00_minimal
- **Max T:** None
- **Max K:** None
- **Seed:** 42

## Data Shape

- **Cameras:** 3
- **Frames:** 1800
- **Joints:** 4

## Gradient Statistics at Initial Point

- **L2 Norm:** nan
- **L-infinity Norm:** nan
- **NaN Count:** 1802 / 27025
- **Inf Count:** 0 / 27025

## Top 20 Gradient Components (by Absolute Value)

| Rank | Variable | Index | Gradient Value |
|------|----------|-------|----------------|
| 1 | `eta2_root` |  | 5.471772e+05 |
| 2 | `rho` | 0 | NaN |
| 3 | `sigma2` | 0 | NaN |
| 4 | `sigma2` | 2 | -8.990000e+02 |
| 5 | `sigma2` | 1 | -8.990000e+02 |
| 6 | `eps_root` | 636 | 5.062464e+01 |
| 7 | `eps_root` | 3304 | -4.933059e+01 |
| 8 | `eps_root` | 3888 | -4.781114e+01 |
| 9 | `eps_root` | 3177 | 4.644142e+01 |
| 10 | `eps_root` | 4874 | 4.611100e+01 |
| 11 | `eps_root` | 4042 | -4.509714e+01 |
| 12 | `eps_root` | 3660 | 4.443102e+01 |
| 13 | `eps_root` | 3909 | 4.440658e+01 |
| 14 | `eps_root` | 1866 | 4.356882e+01 |
| 15 | `eps_root` | 1858 | -4.349160e+01 |
| 16 | `eps_root` | 628 | -4.307717e+01 |
| 17 | `eps_root` | 2520 | 4.306832e+01 |
| 18 | `eps_root` | 4947 | -4.275745e+01 |
| 19 | `eps_root` | 1654 | 4.246161e+01 |
| 20 | `eps_root` | 1967 | -4.240489e+01 |

## Perturbation Analysis

Testing gradient behavior under small perturbations (scale=1e-3):

| Perturbation | Logp | Logp Finite | Grad L2 Norm | Grad Finite |
|--------------|------|-------------|--------------|-------------|
| 1 | -inf | ✗ | nan | ✗ |
| 2 | -inf | ✗ | nan | ✗ |
| 3 | -inf | ✗ | nan | ✗ |

## Environment

- **Python:** 3.11.14
- **PyMC:** 5.26.1
- **PyTensor:** 2.35.1

---

**Results JSON:** `C:\Repositories\GIMBAL_Python\tests\diagnostics\v0_2_1_divergence\results\group_2_gradient_sanity\run_local_run\results_group_2_gradient_sanity.json`