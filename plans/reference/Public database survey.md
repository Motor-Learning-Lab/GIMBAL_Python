# Public Datasets for Validating 3D Human Keypoint Models

### *A Revised and Fully-Sourced Survey for GIMBAL-Style Algorithm Validation*

*(Replaces NotebookLM file) — Opher Donchin, 2025*

---

# 1. Overview

For validating **3D human keypoint estimation**—particularly **model-based, generative, or Bayesian algorithms** inspired by GIMBAL—the critical requirement is **ground-truth provenance**: *how accurate, physical, and unbiased the “truth” is*. The datasets fall into three major classes:

1. **Marker-based optical motion capture** (physical ground truth; highest precision)
2. **Multi-view video with algorithmically inferred 3D labels** (semi-physical but less precise)
3. **In-the-wild datasets with pseudo-ground-truth SMPL fits** (parametric GT; best for generalization)

This document catalogs the most important modern, *truly public* human datasets, identifies their strengths and weaknesses for **GIMBAL-style inference**, and offers guidance for constructing a robust validation pipeline.

---

# 2. Ground-Truth Fidelity Levels

## Level A — Physical GT (marker-based mocap)

Best for validating *absolute geometry* (MPJPE, bone lengths, joint offsets, kinematic chains).

* **Pros:** Millimetric accuracy, calibrated environment, gold standard
* **Cons:** Limited diversity (clothing, lighting), indoor only

## Level B — Semi-physical GT (markerless multi-view triangulation)

* **Pros:** More diversity of backgrounds, clothing, environment
* **Cons:** GT contains reconstruction bias

## Level C — Parametric pseudo-GT (SMPL fits)

* **Pros:** Outdoor and “in-the-wild” diversity
* **Cons:** GT depends heavily on SMPL assumptions

---

# 3. High-Fidelity Marker-Based Datasets (Level A)

## 3.1 Human3.6M (H3.6M)

* **GT:** Vicon optical mocap (sub-millimeter)
* **Scale:** ~3.6M frames, 11 actors
* **Cameras:** 4 synchronized RGB
* **Strengths:** Extremely stable GT; industrial benchmark
* **Link:** [http://vision.imar.ro/human3.6m](http://vision.imar.ro/human3.6m)

### Recommended for GIMBAL

Absolutely. Strongest baseline for geometric accuracy.

---

## 3.2 CMU Mocap

* **GT:** Optical mocap
* **Scale:** 2605 trials, broad action range
* **Limitations:** No RGB; distal joints noisy
* **Link:** [http://mocap.cs.cmu.edu/](http://mocap.cs.cmu.edu/)

### Recommended for GIMBAL

Excellent for kinematic modeling, less useful for multi-view validation.

---

## 3.3 TotalCapture

* **GT:** Vicon mocap
* **Data:** RGB (8 cams) + synchronized IMUs
* **Strengths:** Multi-modal validation
* **Link:** [https://cvssp.org/data/TotalCapture/](https://cvssp.org/data/TotalCapture/)

### Recommended for GIMBAL

Yes—ideal for validating **multi-view + dynamics + IMU alignment**.

---

## 3.4 HumanEva

* **GT:** Mocap
* **Scale:** small, older
* **Link:** [http://humaneva.is.tue.mpg.de/](http://humaneva.is.tue.mpg.de/)

### Recommended for GIMBAL

Mostly historical interest.

---

# 4. Multi-View Markerless Datasets (Level B)

## 4.1 MPI-INF-3DHP

* **GT:** Multi-view triangulation
* **Scale:** ~1.3M frames
* **Environment:** Indoor & outdoor
* **Link:** [https://vcai.mpi-inf.mpg.de/3dhp/](https://vcai.mpi-inf.mpg.de/3dhp/)

### Recommended for GIMBAL

Excellent for testing robustness to varied environments.

---

## 4.2 Panoptic Studio

* **GT:** Ultra-dense camera array (>500)
* **Strengths:** Multi-person interactions
* **Limitations:** Indoor-only, markerless GT
* **Link:** [http://domedb.perception.cs.cmu.edu/](http://domedb.perception.cs.cmu.edu/)

### Recommended for GIMBAL

Useful if multi-person modeling becomes relevant.

---

# 5. In-the-Wild (SMPL-Based) Datasets (Level C)

## 5.1 3DPW

* **GT:** IMU-driven SMPL fits
* **Environment:** Outdoor, real-world videos
* **Link:** [https://virtualhumans.mpi-inf.mpg.de/3DPW/](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

### Recommended for GIMBAL

Yes—best for testing in-the-wild generalization, but GT is model-based.

---

## 5.2 H3WB (WholeBody)

* **GT:** Derived from H3.6M via multi-view labeling
* **Coverage:** 133 keypoints
* **Link:** [https://github.com/facebookresearch/VideoPose3D/tree/master/data/H3WB](https://github.com/facebookresearch/VideoPose3D/tree/master/data/H3WB)

### Recommended for GIMBAL

Good extension beyond the standard 17-joint skeleton.

---

# 6. IMU-Based Datasets (for Temporal Validation)

## 6.1 DIP-IMU

* **GT:** SMPL
* **Strengths:** High-frequency dynamics
* **Link:** [https://dip.is.tue.mpg.de/](https://dip.is.tue.mpg.de/)

## 6.2 GaitMap & Gait Datasets

* **Use:** Benchmarking segmentation and temporal modeling
* **Link:** [https://gaitmap.readthedocs.io/](https://gaitmap.readthedocs.io/)

### Recommended for GIMBAL

Strong for validating HMM or segmentation components.

---

# 7. Evaluation Metrics (Choose According to GT Fidelity)

## Level A (Mocap)

* **MPJPE**
* **Bone-length error**
* **Acceleration/velocity error** (for dynamic priors)

## Level B (Triangulated)

* **P-MPJPE** (alignment required)
* **mPJVE** (velocity error)

## Level C (SMPL-Based)

* **Mesh error (per-vertex)**
* **PA-MPJPE** (scale + rotation alignment)

---

# 8. Recommended Validation Pipeline

## Step 1 — Core geometric validation

* Human3.6M
* TotalCapture

## Step 2 — Robustness & generalization

* MPI-INF-3DHP
* Panoptic (optional)

## Step 3 — In-the-wild performance

* 3DPW
* H3WB

## Step 4 — Temporal modeling validation

* TotalCapture
* DIP-IMU

---

# 9. Dataset Summary Table

| Dataset      | GT Type      | Cameras     | Environment    | Fidelity | Best Use           |
| ------------ | ------------ | ----------- | -------------- | -------- | ------------------ |
| Human3.6M    | Marker       | 4 RGB       | Indoor         | ★★★★★    | Geometry, accuracy |
| CMU          | Marker       | None        | Indoor         | ★★★★☆    | Kinematics         |
| TotalCapture | Marker       | 8 RGB + IMU | Indoor         | ★★★★☆    | Multi-modal        |
| MPI-INF-3DHP | Triangulated | 14          | Indoor/Outdoor | ★★★☆☆    | Robustness         |
| Panoptic     | Triangulated | 500+        | Indoor         | ★★★☆☆    | Multi-person       |
| 3DPW         | SMPL         | Handheld    | Outdoor        | ★★☆☆☆    | In-the-wild        |
| H3WB         | Derived      | 4           | Indoor         | ★★★★☆    | Whole-body         |
| DIP-IMU      | SMPL         | IMU         | Indoor         | ★☆☆☆☆    | Dynamics           |

---

# 10. Key Observations for Human-Oriented GIMBAL Development

* Humans have more constrained pose manifolds → **directional HMMs are powerful**.
* Occlusion is common → tests the algorithm’s camera + kinematic priors.
* Bone lengths differ across subjects → consider **hierarchical priors**.
* Multi-view coverage is lower than rodent datasets → projection errors increase.

---

# 11. Minimal Dataset Set (If Time is Limited)

* **Human3.6M** — core accuracy
* **MPI-INF-3DHP** — robustness
* **3DPW** — in-the-wild
* **TotalCapture** — multi-modal/dynamics

---

# 12. References (Canonical Sources)

* Ionescu et al., *Human3.6M*, TPAMI 2014
* Mehta et al., *MPI-INF-3DHP*, ECCV 2017
* von Marcard et al., *3DPW*, ECCV 2018
* Joo et al., *Panoptic Studio*, TPAMI 2018
* Trumble et al., *TotalCapture*, FG 2017
* Huang et al., *DIP-IMU*, SIGGRAPH 2018
