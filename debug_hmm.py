"""Debug script for brute force test"""

import numpy as np
import pytensor
import pytensor.tensor as pt
from scipy.special import logsumexp
from gimbal.hmm_pytensor import forward_log_prob_single

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

print("Parameters:")
print(f"logp_init: {logp_init}")
print(f"logp_trans:\n{logp_trans}")
print(f"logp_emit:\n{logp_emit}")
print()

# Manual forward algorithm in numpy
alpha = np.zeros((T, S))

# t=0: alpha[0,s] = logp_init[s] + logp_emit[0,s]
alpha[0] = logp_init + logp_emit[0]
print(f"alpha[0]: {alpha[0]}")

# t=1: alpha[1,j] = logp_emit[1,j] + logsumexp_i(alpha[0,i] + logp_trans[i,j])
for j in range(S):
    alpha[1, j] = logp_emit[1, j] + logsumexp(alpha[0] + logp_trans[:, j])
print(f"alpha[1]: {alpha[1]}")

# t=2: alpha[2,j] = logp_emit[2,j] + logsumexp_i(alpha[1,i] + logp_trans[i,j])
for j in range(S):
    alpha[2, j] = logp_emit[2, j] + logsumexp(alpha[1] + logp_trans[:, j])
print(f"alpha[2]: {alpha[2]}")

# Final logp
manual_logp = logsumexp(alpha[2])
print(f"\nManual forward logp: {manual_logp}")

# PyTensor version
logp_init_pt = pt.as_tensor_variable(logp_init)
logp_trans_pt = pt.as_tensor_variable(logp_trans)
logp_emit_pt = pt.as_tensor_variable(logp_emit)

forward_logp_pt = forward_log_prob_single(logp_emit_pt, logp_init_pt, logp_trans_pt)
forward_fn = pytensor.function([], forward_logp_pt)
pytensor_logp = forward_fn()
print(f"PyTensor forward logp: {pytensor_logp}")

print(f"\nDifference: {abs(manual_logp - pytensor_logp)}")
