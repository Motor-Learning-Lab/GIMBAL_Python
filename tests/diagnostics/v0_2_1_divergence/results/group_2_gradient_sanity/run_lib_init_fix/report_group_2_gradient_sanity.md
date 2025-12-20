# Group 2: Gradient Sanity Check

**Run ID:** lib_init_fix
**Timestamp:** 2025-12-20T13:14:51.723268
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

- **L2 Norm:** 1.544672e+08
- **L-infinity Norm:** 8.948522e+07
- **NaN Count:** 0 / 27025
- **Inf Count:** 0 / 27025

## Top 20 Gradient Components (by Absolute Value)

| Rank | Variable | Index | Gradient Value |
|------|----------|-------|----------------|
| 1 | `dir_hmm_mu_raw` | 5 | 8.948522e+07 |
| 2 | `dir_hmm_mu_raw` | 11 | 6.580669e+07 |
| 3 | `dir_hmm_mu_raw` | 8 | 6.363523e+07 |
| 4 | `dir_hmm_mu_raw` | 6 | 6.221624e+07 |
| 5 | `dir_hmm_mu_raw` | 9 | -5.995736e+07 |
| 6 | `dir_hmm_mu_raw` | 4 | -1.945811e+06 |
| 7 | `dir_hmm_mu_raw` | 10 | -7.399513e+05 |
| 8 | `dir_hmm_mu_raw` | 7 | -6.682483e+05 |
| 9 | `dir_hmm_mu_raw` | 3 | -3.540306e+05 |
| 10 | `obs_sigma` |  | 1.991627e+05 |
| 11 | `eps_root` | 311 | 5.441447e+04 |
| 12 | `eps_root` | 305 | 5.436920e+04 |
| 13 | `eps_root` | 299 | 5.390018e+04 |
| 14 | `eps_root` | 314 | 5.354059e+04 |
| 15 | `eps_root` | 293 | 5.344916e+04 |
| 16 | `eps_root` | 308 | 5.338475e+04 |
| 17 | `eps_root` | 302 | 5.337473e+04 |
| 18 | `eps_root` | 296 | 5.325348e+04 |
| 19 | `eps_root` | 290 | 5.262107e+04 |
| 20 | `eps_root` | 320 | 5.202027e+04 |

## Perturbation Analysis

Testing gradient behavior under small perturbations (scale=1e-3):

| Perturbation | Logp | Logp Finite | Grad L2 Norm | Grad Finite |
|--------------|------|-------------|--------------|-------------|
| 1 | -306258.01 | ✓ | 3.23e+07 | ✓ |
| 2 | -290843.25 | ✓ | 3.49e+07 | ✓ |
| 3 | -294275.39 | ✓ | 2.93e+07 | ✓ |

## Environment

- **Python:** 3.11.14
- **PyMC:** 5.26.1
- **PyTensor:** 2.35.1

---

**Results JSON:** `C:\Repositories\GIMBAL_Python\tests\diagnostics\v0_2_1_divergence\results\group_2_gradient_sanity\run_lib_init_fix\results_group_2_gradient_sanity.json`