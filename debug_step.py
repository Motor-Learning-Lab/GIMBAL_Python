"""Debug the step function"""

import numpy as np
import pytensor.tensor as pt
import pytensor
from scipy.special import logsumexp

S = 2

logp_trans = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))
alpha_0 = np.array([-1.51082562, -2.91629073])

print("Step function test:")
print(f"alpha_0: {alpha_0}")
print(f"logp_trans:\n{logp_trans}")
print()

# Manual calculation for alpha[1,j]
logp_emit_1 = np.array([-0.5, -1.5])

for j in range(S):
    contributions = alpha_0 + logp_trans[:, j]
    print(f"For state j={j}:")
    print(f"  alpha_0 + logp_trans[:, {j}] = {contributions}")
    pred_j = logsumexp(contributions)
    alpha_1_j = logp_emit_1[j] + pred_j
    print(f"  logsumexp = {pred_j}")
    print(f"  alpha[1, {j}] = {logp_emit_1[j]} + {pred_j} = {alpha_1_j}")
print()

# PyTensor version
alpha_prev_pt = pt.dvector("alpha_prev")
logp_trans_pt = pt.dmatrix("logp_trans")
logp_emit_t_pt = pt.dvector("logp_emit_t")

# What the spec says: alpha_prev.dimshuffle(0, "x") creates (S, 1)
# Then add logp_trans (S, S) to get (S, S)
# Then logsumexp over axis=0 (sum over rows) to get (S,)
alpha_pred_pt = pt.logsumexp(
    alpha_prev_pt.dimshuffle(0, "x") + logp_trans_pt,
    axis=0,
)

step_fn = pytensor.function(
    [alpha_prev_pt, logp_trans_pt, logp_emit_t_pt],
    [alpha_pred_pt, logp_emit_t_pt + alpha_pred_pt],
)

alpha_pred, alpha_next = step_fn(alpha_0, logp_trans, logp_emit_1)
print(f"PyTensor alpha_pred: {alpha_pred}")
print(f"PyTensor alpha_next: {alpha_next}")
print()

# The issue: axis=0 sums over dimension 0, which are the ROWS (previous states i)
# Let's check what we get with dimshuffle(0, "x")
alpha_expanded = alpha_0[:, np.newaxis]  # Shape (S, 1) = (2, 1)
print(f"alpha expanded shape: {alpha_expanded.shape}")
print(f"alpha expanded:\n{alpha_expanded}")
print()

combined = alpha_expanded + logp_trans
print(f"alpha_expanded + logp_trans:\n{combined}")
print(f"  This is: combined[i,j] = alpha_0[i] + logp_trans[i,j]")
print()

result_axis0 = logsumexp(combined, axis=0)
print(f"logsumexp(combined, axis=0): {result_axis0}")
print(f"  This sums over i (rows), giving result[j]")
