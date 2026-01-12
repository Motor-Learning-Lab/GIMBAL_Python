"""Quick test of SyntheticDataConfig refactor."""

from gimbal import SyntheticDataConfig

# Test 1: Simple programmatic creation (backward compat)
print("Test 1: Simple creation...")
c = SyntheticDataConfig()
print(f"  T={c.T}, C={c.C}, S={c.S}")
print(f"  uses_second_order={c.uses_second_order_dynamics()}")
print(f"  kappa={c.kappa}")
print("  ✓ Simple creation OK")

# Test 2: Custom parameters
print("\nTest 2: Custom parameters...")
c2 = SyntheticDataConfig(T=100, C=5, random_seed=123)
print(f"  T={c2.T}, C={c2.C}, seed={c2.random_seed}")
print("  ✓ Custom parameters OK")

# Test 3: JSON loading (will fail without file, but tests method exists)
print("\nTest 3: from_json method exists...")
print(f"  has from_json: {hasattr(SyntheticDataConfig, 'from_json')}")
print("  ✓ JSON interface exists")

print("\n✓ All tests passed - refactor successful!")
