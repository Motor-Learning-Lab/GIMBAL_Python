# GIMBAL Plans Directory

This directory contains planning documents, specifications, reviews, and completion reports for GIMBAL development.

## File Naming Convention

Files follow the pattern: `v{major}.{minor}.{patch}-{type}.md`

**Types:**
- `overview.md` - High-level roadmap for a version
- `detailed-spec.md` - Detailed implementation specification for a phase
- `completion-report.md` - Post-implementation documentation and results
- `review*.md` - Review comments and iteration notes
- `*-notes.md` - Supporting notes and background research

## Version 0.1 — Three-Stage HMM Integration

**Status:** ✅ Complete

### v0.1 Overview
- **`v0.1-overview.md`** - Three-stage architecture overview
- **`v0.1-review-iteration2.md`** - Second iteration review comments

### v0.1.1 — Collapsed HMM Engine (Stage 1)
- **`v0.1.1-detailed-spec.md`** - Forward algorithm specification
- **`v0.1.1-review-improvements.md`** - Review and improvement suggestions
- **`v0.1.1-review-revision3.md`** - Third revision comments
- **`v0.1.1-completion-report.md`** - ✅ Implementation results and validation

**Key Achievement:** Numerically stable collapsed HMM with PyTensor scan integration

### v0.1.2 — Camera Observation Model (Stage 2)
- **`v0.1.2-detailed-spec.md`** - Camera projection and kinematics specification
- **`v0.1.2-spec-prompt.md`** - Specification development prompt
- **`v0.1.2-review1.md`** - First review comments
- **`v0.1.2-completion-report.md`** - ✅ Implementation results and interface contracts

**Key Achievement:** Per-timestep likelihood computation, exposed U and x_all for Stage 3

### v0.1.3 — Directional HMM Prior (Stage 3)
- **`v0.1.3-detailed-spec.md`** - Directional prior over joint directions specification
- **`v0.1.3-completion-report.md`** - ✅ Implementation results and label switching mitigation

**Key Achievement:** State-dependent canonical directions with flexible concentration sharing

## Version 0.2 — Priors, Real Data, and Robustness

**Status:** 📋 Planning

### v0.2 Overview
- **`v0.2-overview.md`** - Eight-phase roadmap for v0.2 development

### Phases (0.2.1 - 0.2.8)
1. **Coarse Anatomical Priors & Basic Cleaning**
2. **k-Means / Clustering Empirical-Bayes Priors**
3. **Sampler Decision Spike**
4. **Minimal Synthetic Diagnostics**
5. **First Public Dataset Loader + Baseline Real-Data Fit**
6. **Real-Data Diagnostics & Data-Driven Anatomical Priors**
7. **Coarse PCA + Low-D HMM + Transition Upsampling**
8. **PCA-Informed Priors for Full Model & State-Number Selection**

### Supporting Documents
- **`v0.2-biomechanical-priors-notes.md`** - Notes on biomechanical prior development

## Reference Documents

These documents provide background and are not tied to specific versions:

- **`blended_local_frame_design.md`** - Coordinate frame design considerations
- **`coordinates.md`** - Coordinate system documentation
- **`Public database survey.md`** - Survey of available public motion capture datasets

## Development Workflow

For each phase:
1. Create `vX.Y.Z-detailed-spec.md` with implementation details
2. Implement and test
3. Write `vX.Y.Z-completion-report.md` documenting results
4. Add review files (`vX.Y.Z-review*.md`) as needed during development

## Quick Reference

| Version | Status | Key Files |
|---------|--------|-----------|
| v0.1 (3-stage HMM) | ✅ Complete | v0.1-overview.md, v0.1.{1,2,3}-completion-report.md |
| v0.2 (Priors & Real Data) | 📋 Planning | v0.2-overview.md |

---

**Last Updated:** November 28, 2025
