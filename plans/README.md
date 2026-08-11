# GIMBAL Plans Directory

This directory contains planning documents, specifications, reviews, and completion reports for GIMBAL development.

## Directory Structure

```
plans/
├── README.md              # This file
├── reference/             # General design references (not version-specific)
├── v0.1/                  # v0.1 complete documentation
├── v0.2.0/                # v0.2.0 complete documentation
├── v0.2.1/                # v0.2.1 active documentation
├── v0.2.2/                # v0.2.2 planning (create as needed)
└── archive/               # Historical and intermediate files
    ├── meta/              # Repository-level historical docs
    └── v0.2.1/            # v0.2.1 intermediate debug/iteration files
```

## File Naming Convention

Files follow the pattern: `v{major}.{minor}.{patch}_{type}.md`

**Types:**
- `overview` - High-level roadmap for a version
- `detailed_spec` / `spec_*` - Detailed implementation specifications
- `completion_report` - Post-implementation documentation and results
- `review*` - Review comments and iteration notes
- `step{N}_*` - Phase-specific documentation
- `debug_*` / `divergence_*` - Debugging plans (archive when resolved)

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

**Status:** 🔄 In Progress (v0.2.1 Complete, v0.2.2 Planning)

### v0.2 Overview
- **[v0.2-overview.md](v0.2.0/v0.2-overview.md)** - Eight-phase roadmap for v0.2 development
- **[v0.2-biomechanical-priors-notes.md](v0.2.0/v0.2-biomechanical-priors-notes.md)** - Notes on biomechanical prior development

### v0.2.0 — Project Restructuring
**Status:** ✅ Complete

See `v0.2.0/` directory for completion reports and planning documents.

### v0.2.1 — Data-Driven Priors & Synthetic Data Pipeline

**Status:** ✅ Core features complete

**Key Documents:**
- **[v0.2.1_step3_completion_summary.md](v0.2.1/v0.2.1_step3_completion_summary.md)** ← **Step 3 canonical reference**
- **[v0.2.1_completion_report.md](v0.2.1/v0.2.1_completion_report.md)** - Data-driven priors implementation
- **[v0.2.1_spec_data_driven_priors.md](v0.2.1/v0.2.1_spec_data_driven_priors.md)** - Technical specification
- **[v0.2.1_Quick_reference_after_step_3_for_step_4.md](v0.2.1/v0.2.1_Quick_reference_after_step_3_for_step_4.md)** - Step 4 planning reference
- **[v0.2.1_step4_implementation_summary.md](v0.2.1/v0.2.1_step4_implementation_summary.md)** - Step 4 sampler work
- **[v0.2.1_divergence_plan_2.md](v0.2.1/v0.2.1_divergence_plan_2.md)** - Current divergence debugging plan

**Key Achievements:**
- Data-driven priors pipeline (triangulation → cleaning → statistics → priors)
- Synthetic data generation with second-order dynamics
- Camera visualization and 3-tier identifiability checking
- Consolidated skeleton metrics and visualization modules
- Sampler comparison and initialization improvements

**Additional Files:**
- [v0.2.1_step3_synthetic_data_generation.md](v0.2.1/v0.2.1_step3_synthetic_data_generation.md) - Step 3 original plan
- [v0.2.1_step3_clarifications_and_questions.md](v0.2.1/v0.2.1_step3_clarifications_and_questions.md) - Design decisions
- [v0.2.1_step4_fit_L00.md](v0.2.1/v0.2.1_step4_fit_L00.md) - Fit configuration documentation
- [v0.2.1_toolchain_switches_inventory.md](v0.2.1/v0.2.1_toolchain_switches_inventory.md) - Configuration options

> **Archived:** Intermediate debug plans and iteration documents are in `archive/v0.2.1/`

### v0.2.2 — Module Reorganization + Anipose-Derived Priors + Capability Ladder

- [v0.2.2_overview.md](v0.2.2/v0.2.2_overview.md) - Architecture and sub-project roadmap (P0, P1, A1-A3, C1)
- [v0.2.2_P0_plan.md](v0.2.2/v0.2.2_P0_plan.md) - P0 (module reorganization) plan
- [v0.2.2_P0_plan_original.md](v0.2.2/v0.2.2_P0_plan_original.md) - P0 plan, original draft
- [v0.2.2_P0_completion_report.md](v0.2.2/v0.2.2_P0_completion_report.md) - P0 completion report

**Sub-project documents:** each sub-project (P1, A1, A2, A3, C1) gets its own plan and
findings documents here as work on it starts, following the pattern established by P0:

- `v0.2.2_<ID>_plan.md` - scope and design for that sub-project
- `v0.2.2_<ID>_findings.md` - accumulated evidence and results (a living document, not a
  one-time report — see the Living Documents convention below)
- `v0.2.2_<ID>_completion_report.md` - written once the sub-project is done

Don't create a sub-project's documents speculatively before there is real content — an
empty or stub document is worse than none, since it looks current and isn't. The
corresponding GitHub issue is authoritative for scope in the meantime.

### Living Documents

Some planning documents are not one-time specs but accumulate findings over the life of
a sub-project (e.g. a divergence investigation, or the C1 capability ladder). These
follow the pattern set by `v0.2.1/v0.2.1_divergence_plan_2.md`:

- An editor's note at the top: preserve existing content, append or revise sections in
  place rather than rewriting, and describe what results *show* rather than asserting
  that a problem is solved.
- A "Current Status" section carrying a revision date, updated in place as findings
  accumulate — not superseded by a new document each time.
- Stable IDs for named findings/issues (e.g. I1, I2, I3), assigned once and never
  reused. Other documents and GitHub issues should link to a finding by ID rather than
  restating it, so there is exactly one place it's defined.

See `AGENTS.md` for how these documents relate to GitHub issues.

### Phases (0.2.1 - 0.2.8) - Overall Roadmap
1. **Coarse Anatomical Priors & Basic Cleaning**
2. **k-Means / Clustering Empirical-Bayes Priors**
3. **Sampler Decision Spike** ✅
4. **Minimal Synthetic Diagnostics**
5. **First Public Dataset Loader + Baseline Real-Data Fit**
6. **Real-Data Diagnostics & Data-Driven Anatomical Priors**
7. **Coarse PCA + Low-D HMM + Transition Upsampling**
8. **PCA-Informed Priors for Full Model & State-Number Selection**

## Reference Documents (Non-Version-Specific)

Located in `reference/` directory - design documents that apply across versions:

- **[blended_local_frame_design.md](reference/blended_local_frame_design.md)** - Local anatomical coordinate frames design
- **[coordinates.md](reference/coordinates.md)** - Coordinate system construction formulas
- **[Public database survey.md](reference/Public%20database%20survey.md)** - Survey of available public motion capture datasets

## Development Workflow

For each phase:
1. Create `vX.Y.Z_detailed_spec.md` or `vX.Y.Z_stepN_*.md` with implementation details
2. Implement and test
3. Write `vX.Y.Z_completion_report.md` documenting results
4. Add review files (`vX.Y.Z_review*.md`) as needed during development
5. Move intermediate/debug files to `archive/vX.Y.Z/` when work is complete

## Archive Directory

The `archive/` directory contains:
- **`meta/`** - Repository-level historical documents (e.g., machine transfer notes, cleanup plans)
- **`v0.2.1/`** - v0.2.1 intermediate debug, divergence plans, and iteration documents

Files are archived when:
- They represent intermediate iterations superseded by final documents
- They are debugging plans that have been resolved
- They are historical context no longer needed for active development

## Quick Reference

| Version | Status | Key Files |
|---------|--------|-----------|
| v0.1 (3-stage HMM) | ✅ Complete | [v0.1/](v0.1/) - All Stage 1-3 specs and completion reports |
| v0.2.0 (Restructuring) | ✅ Complete | [v0.2.0/](v0.2.0/) |
| v0.2.1 (Data-Driven Priors) | ✅ Complete | [v0.2.1/](v0.2.1/) |
| v0.2.2 (Reorg + Priors + Ladder) | 🔄 In Progress (P0 ✅ complete) | [v0.2.2/](v0.2.2/) |

---

**Last Updated:** August 11, 2026
