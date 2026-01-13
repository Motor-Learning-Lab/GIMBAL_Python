# Tools Directory

Ad-hoc scripts and utilities used during development and migration.

## Migration Tools (Historical)

These scripts were used for one-time migrations during package rename:

- `clean_sys_path.py` - Remove sys.path manipulations
- `remove_sys_path.py` - Alternative sys.path cleanup
- `rename_imports.py` - Rename import statements
- `standardize_imports.py` - Standardize to `import gimbal_pymc as gp`

## Configuration Testing (Historical)

One-off scripts for testing configurations:

- `test_config_refactor.py` - Test configuration refactoring
- `test_new_config.py` - Test new configuration system
- `debug_pipeline_cameras.py` - Debug camera geometry issues
- `compute_correct_camera_configs.py` - Compute canonical camera configs

## Usage Note

These tools are kept for reference but are **not part of the main package**. They were used during solo development and should not be relied upon for production use.

For current configuration and testing, use:
- `pixi run test-*` commands (see root README.md)
- `tests/pipeline/generate_dataset.py` for dataset generation
