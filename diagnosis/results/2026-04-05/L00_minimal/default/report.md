# GIMBAL C1 Ladder ¡ª L00_minimal  (default)

**Date:** 2026-04-05 12:50

**Description:** Minimal baseline: single state, 4 joints, 3 cameras, low noise, no outliers, no missingness.

## Complexity profile

| Parameter | Value |
|---|---|
| T | 200 |
| Joints | 4 |
| States | 1 |
| Cameras | 3 |
| Noise (px) | 2.0 |
| Outliers | 0% (off) |
| Missingness | 0% (off) |

## Stage results

**Overall: SOME FAILED**

| Stage | Name | Result |
|---|---|---|
| A | Load Validation | PASSED |
| B | 2D Cleaning | PASSED |
| C | Triangulation | PASSED |
| D | 3D Cleaning | PASSED |
| E | Direction Statistics | PASSED |
| F | Prior Building | PASSED |
| G | Model Building | ERROR: 'compilation_success' |
| H | Fitting/Sampling | NOT RUN |
| I | Posterior Diagnostics | NOT RUN |
| J | Ground Truth Comparison | NOT RUN |

---
Output directory: `C:\Users\86153\Desktop\GIMBAL_Python\diagnosis\results\2026-04-05\L00_minimal\default`
