"""Test loading new L00 config."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from tests.pipeline.utils.config_generator import load_config

config_path = Path("tests/pipeline/datasets/v0.2.1_L00_minimal/config.json")

print(f"Loading config from: {config_path}")
print(f"File exists: {config_path.exists()}")

if config_path.exists():
    config = load_config(config_path)
    print("\n✓ Config loaded successfully!")
    print(f"\nMeta:")
    print(f"  name: {config['meta']['name']}")
    print(f"  T: {config['meta']['T']}")
    print(f"  dt: {config['meta']['dt']}")
    print(f"\nSkeleton:")
    print(f"  joints: {config['dataset_spec']['skeleton']['joint_names']}")
    print(f"  parents: {config['dataset_spec']['skeleton']['parents']}")
    print(f"\nStates:")
    print(f"  num_states: {config['dataset_spec']['states']['num_states']}")
    print(f"\nCameras:")
    print(f"  image_size: {config['dataset_spec']['cameras']['image_size']}")
    print(f"  num_cameras: {len(config['dataset_spec']['cameras']['cameras'])}")
    print(f"\nObservation:")
    print(f"  noise_px: {config['dataset_spec']['observation']['noise_px']}")
else:
    print("ERROR: Config file not found!")
