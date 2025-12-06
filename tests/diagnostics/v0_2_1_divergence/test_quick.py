"""Quick test to verify HMM ON model works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.diagnostics.v0_2_1_divergence.test_utils import (
    get_standard_synth_data,
    build_test_model,
    sample_model,
)

print("Testing HMM ON model...")
synth_data = get_standard_synth_data(T=50, C=3, S=3, seed=42)

print("Building model...")
model = build_test_model(synth_data, use_directional_hmm=True, S=3)

print(f"Model built successfully with {len(model.free_RVs)} free variables")
print("\nAttempting to sample (2 draws, 2 tune)...")

try:
    trace = sample_model(model, draws=2, tune=2, chains=1)
    print(f"[OK] Sampling successful!")
    print(f"  Divergences: {trace.sample_stats.diverging.sum().values}")
except Exception as e:
    print(f"[FAIL] Sampling failed: {e}")
    import traceback

    traceback.print_exc()
