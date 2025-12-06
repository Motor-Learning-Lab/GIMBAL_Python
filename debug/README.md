# Debug Scripts

Exploration and debugging scripts for GIMBAL development.

## Structure

- **camera/**: Camera-related debugging
  - `debug_camera_orient.py`: Debug camera orientation calculations
  - `debug_skeleton_projection.py`: Debug skeleton projection
  - `visualize_camera_fix.py`: Visualize camera fixes
  - `show_camera_differences.py`: Compare camera configurations

- **triangulation/**: Triangulation debugging
  - `debug_triangulate.py`: Debug triangulation
  - `debug_triangulation.py`: Debug triangulation (alternate)
  - `analyze_skeleton_3d.py`: Analyze 3D skeleton structure

- **misc/**: Other debugging scripts
  - `debug_zero_noise.py`: Test zero-noise scenarios
  - `check_skeleton_extents.py`: Check skeleton coordinate ranges

## Usage

These are ad-hoc exploration scripts, not formal tests. Run directly:

```bash
pixi run python debug/camera/debug_camera_orient.py
```

## Outputs

Debug outputs typically print to console. Save any important findings to `docs/`.
