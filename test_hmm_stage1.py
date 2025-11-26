"""
Validation tests for Stage 1 HMM implementation.

Includes:
- Tiny HMM brute-force verification
- T=1 edge case test
- Gradient validation with finite differences
- Normalization checks
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
from scipy.special import logsumexp
from gimbal.hmm_pytensor import forward_log_prob_single, collapsed_hmm_loglik


def test_tiny_hmm_brute_force():
    """
    Test forward algorithm against brute-force enumeration.

    Uses S=2, T=3 to enumerate all 8 possible state sequences and
    compare with the forward algorithm result.

    Returns
    -------
    passed : bool
        True if test passes (agreement within 1e-6)
    """
    print("Running tiny HMM brute-force test (S=2, T=3)...")

    S = 2
    T = 3

    # Create simple test parameters
    logp_init = np.log(np.array([0.6, 0.4]))
    logp_trans = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))
    logp_emit = np.array(
        [
            [-1.0, -2.0],  # t=0
            [-0.5, -1.5],  # t=1
            [-2.0, -0.5],  # t=2
        ]
    )

    # Brute force: enumerate all 8 state sequences
    log_probs = []
    for z0 in range(S):
        for z1 in range(S):
            for z2 in range(S):
                z = [z0, z1, z2]
                # log p(z_0)
                logp = logp_init[z[0]]
                # log p(y_0 | z_0)
                logp += logp_emit[0, z[0]]
                # log p(z_1 | z_0)
                logp += logp_trans[z[0], z[1]]
                # log p(y_1 | z_1)
                logp += logp_emit[1, z[1]]
                # log p(z_2 | z_1)
                logp += logp_trans[z[1], z[2]]
                # log p(y_2 | z_2)
                logp += logp_emit[2, z[2]]
                log_probs.append(logp)

    brute_force_logp = logsumexp(log_probs)

    # Forward algorithm via PyTensor
    logp_init_pt = pt.as_tensor_variable(logp_init)
    logp_trans_pt = pt.as_tensor_variable(logp_trans)
    logp_emit_pt = pt.as_tensor_variable(logp_emit)

    forward_logp_pt = forward_log_prob_single(logp_emit_pt, logp_init_pt, logp_trans_pt)
    forward_fn = pytensor.function([], forward_logp_pt)
    forward_logp = forward_fn()

    # Compare
    diff = abs(brute_force_logp - forward_logp)
    passed = diff < 1e-6

    print(f"  Brute force log p: {brute_force_logp:.10f}")
    print(f"  Forward algorithm:  {forward_logp:.10f}")
    print(f"  Difference:         {diff:.2e}")
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


def test_t1_edge_case():
    """
    Test T=1 edge case handling.

    For T=1, the result should be logsumexp(logp_init + logp_emit[0]).

    Returns
    -------
    passed : bool
        True if test passes
    """
    print("\nRunning T=1 edge case test...")

    S = 3
    T = 1

    logp_init = np.log(np.array([0.5, 0.3, 0.2]))
    logp_trans = np.log(np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]]))
    logp_emit = np.array([[-1.5, -0.5, -2.0]])  # shape (1, 3)

    # Expected: logsumexp(logp_init + logp_emit[0])
    expected_logp = logsumexp(logp_init + logp_emit[0])

    # Forward algorithm
    logp_init_pt = pt.as_tensor_variable(logp_init)
    logp_trans_pt = pt.as_tensor_variable(logp_trans)
    logp_emit_pt = pt.as_tensor_variable(logp_emit)

    forward_logp_pt = forward_log_prob_single(logp_emit_pt, logp_init_pt, logp_trans_pt)
    forward_fn = pytensor.function([], forward_logp_pt)
    forward_logp = forward_fn()

    diff = abs(expected_logp - forward_logp)
    passed = diff < 1e-10

    print(f"  Expected log p:    {expected_logp:.10f}")
    print(f"  Forward algorithm: {forward_logp:.10f}")
    print(f"  Difference:        {diff:.2e}")
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


def test_gradient_correctness(epsilon=1e-5, tolerance=1e-4):
    """
    Test gradient correctness via finite differences.

    Compares PyTensor automatic differentiation gradients with
    numerical finite-difference approximations.

    Parameters
    ----------
    epsilon : float
        Finite difference step size
    tolerance : float
        Maximum allowed relative error

    Returns
    -------
    passed : bool
        True if all gradients match within tolerance
    """
    print("\nRunning gradient correctness test...")

    S = 2
    T = 4

    # Create test parameters as shared variables for gradient computation
    init_logits = np.array([0.2, -0.3])
    trans_logits = np.array([[0.5, -0.5], [-0.2, 0.3]])
    logp_emit_vals = np.random.randn(T, S)

    # PyTensor variables
    init_logits_pt = pt.dvector("init_logits")
    trans_logits_pt = pt.dmatrix("trans_logits")
    logp_emit_pt = pt.dmatrix("logp_emit")

    # Normalize
    logp_init_pt = init_logits_pt - pt.logsumexp(init_logits_pt)
    logp_trans_pt = trans_logits_pt - pt.logsumexp(
        trans_logits_pt, axis=1, keepdims=True
    )

    # Forward pass
    forward_logp_pt = forward_log_prob_single(logp_emit_pt, logp_init_pt, logp_trans_pt)

    # Compute gradients
    grad_init = pytensor.grad(forward_logp_pt, init_logits_pt)
    grad_trans = pytensor.grad(forward_logp_pt, trans_logits_pt)

    # Compile functions
    forward_fn = pytensor.function(
        [init_logits_pt, trans_logits_pt, logp_emit_pt], forward_logp_pt
    )
    grad_init_fn = pytensor.function(
        [init_logits_pt, trans_logits_pt, logp_emit_pt], grad_init
    )
    grad_trans_fn = pytensor.function(
        [init_logits_pt, trans_logits_pt, logp_emit_pt], grad_trans
    )

    # Get analytical gradients
    f0 = forward_fn(init_logits, trans_logits, logp_emit_vals)
    grad_init_auto = grad_init_fn(init_logits, trans_logits, logp_emit_vals)
    grad_trans_auto = grad_trans_fn(init_logits, trans_logits, logp_emit_vals)

    print(f"  Forward log p: {f0:.6f}")
    print(f"  Testing init_logits gradient...")

    # Finite differences for init_logits
    grad_init_fd = np.zeros_like(init_logits)
    for i in range(len(init_logits)):
        init_plus = init_logits.copy()
        init_plus[i] += epsilon
        f_plus = forward_fn(init_plus, trans_logits, logp_emit_vals)
        grad_init_fd[i] = (f_plus - f0) / epsilon

    init_diff = np.abs(grad_init_auto - grad_init_fd)
    init_rel_error = init_diff / (np.abs(grad_init_fd) + 1e-10)

    print(f"    Auto grad: {grad_init_auto}")
    print(f"    FD grad:   {grad_init_fd}")
    print(f"    Max abs diff: {init_diff.max():.2e}")
    print(f"    Max rel error: {init_rel_error.max():.2e}")

    init_passed = init_rel_error.max() < tolerance

    print(f"  Testing trans_logits gradient...")

    # Finite differences for trans_logits
    grad_trans_fd = np.zeros_like(trans_logits)
    for i in range(trans_logits.shape[0]):
        for j in range(trans_logits.shape[1]):
            trans_plus = trans_logits.copy()
            trans_plus[i, j] += epsilon
            f_plus = forward_fn(init_logits, trans_plus, logp_emit_vals)
            grad_trans_fd[i, j] = (f_plus - f0) / epsilon

    trans_diff = np.abs(grad_trans_auto - grad_trans_fd)
    trans_rel_error = trans_diff / (np.abs(grad_trans_fd) + 1e-10)

    print(f"    Max abs diff: {trans_diff.max():.2e}")
    print(f"    Max rel error: {trans_rel_error.max():.2e}")

    trans_passed = trans_rel_error.max() < tolerance

    passed = init_passed and trans_passed
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


def test_normalization_checks():
    """
    Verify that probability distributions are properly normalized.

    Checks that:
    - exp(logp_init) sums to 1
    - Each row of exp(logp_trans) sums to 1

    Returns
    -------
    passed : bool
        True if normalization is correct within 1e-6
    """
    print("\nRunning normalization checks...")

    S = 3

    # Create unnormalized logits
    init_logits = np.array([0.5, -0.2, 0.8])
    trans_logits = np.array([[0.3, -0.5, 0.2], [0.1, 0.6, -0.3], [-0.4, 0.2, 0.5]])

    # Normalize
    logp_init = init_logits - logsumexp(init_logits)
    logp_trans = trans_logits - logsumexp(trans_logits, axis=1, keepdims=True)

    # Check normalization
    pi = np.exp(logp_init)
    A = np.exp(logp_trans)

    init_sum = pi.sum()
    trans_sums = A.sum(axis=1)

    init_error = abs(init_sum - 1.0)
    trans_error = np.abs(trans_sums - 1.0).max()

    print(f"  Initial distribution sum: {init_sum:.10f} (error: {init_error:.2e})")
    print(f"  Transition row sums: {trans_sums}")
    print(f"  Max transition error: {trans_error:.2e}")

    passed = init_error < 1e-6 and trans_error < 1e-6
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


def run_all_tests():
    """
    Run all Stage 1 validation tests.

    Returns
    -------
    all_passed : bool
        True if all tests passed
    """
    print("=" * 60)
    print("Stage 1 HMM Validation Test Suite")
    print("=" * 60)

    results = {}

    results["brute_force"] = test_tiny_hmm_brute_force()
    results["t1_edge"] = test_t1_edge_case()
    results["gradient"] = test_gradient_correctness()
    results["normalization"] = test_normalization_checks()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed.'}")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
