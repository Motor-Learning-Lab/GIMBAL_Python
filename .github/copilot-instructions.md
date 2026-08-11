# GitHub Copilot Instructions

Agent guidance for this repository lives in **[AGENTS.md](../AGENTS.md)** at the repo root.

Read that file before making changes. It is the single source; this file exists only so
that tools looking for `.github/copilot-instructions.md` are pointed at it. Do not
duplicate content here — update `AGENTS.md` instead.

> **Note:** the previous contents of this file were written for v0.2.1 and had gone stale.
> They referenced pre-reorganization module paths (`gimbal/pymc_model.py`) and a
> top-level import style (`gimbal.build_camera_observation_model(...)`) that no longer
> works, which contributed to a class of import bugs. Anything still useful from that
> version has been carried into `AGENTS.md`.
