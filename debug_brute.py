"""Check brute force calculation"""

import numpy as np
from scipy.special import logsumexp

S = 2
T = 3

logp_init = np.log(np.array([0.6, 0.4]))
logp_trans = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))
logp_emit = np.array(
    [
        [-1.0, -2.0],  # t=0
        [-0.5, -1.5],  # t=1
        [-2.0, -0.5],  # t=2
    ]
)

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

            print(f"Path {z}: {logp:.4f}")
            log_probs.append(logp)

brute_force_logp = logsumexp(log_probs)
print(f"\nBrute force logsumexp: {brute_force_logp}")
