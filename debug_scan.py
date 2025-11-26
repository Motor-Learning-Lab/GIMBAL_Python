"""Test scan behavior"""

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor import scan
from scipy.special import logsumexp

logp_trans = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))
logp_emit = np.array(
    [
        [-1.0, -2.0],  # t=0
        [-0.5, -1.5],  # t=1
        [-2.0, -0.5],  # t=2
    ]
)
alpha_0 = np.array([-1.51082562, -2.91629073])


# Define step using closure
def make_step(logp_trans):
    def step(alpha_prev, logp_emit_t):
        alpha_pred = pt.logsumexp(
            alpha_prev.dimshuffle(0, "x") + logp_trans,
            axis=0,
        )
        alpha_t = logp_emit_t + alpha_pred
        return alpha_t

    return step


logp_trans_pt = pt.as_tensor_variable(logp_trans)
alpha_prev_pt = pt.dvector("alpha_prev")
logp_emit_rest_pt = pt.dmatrix("logp_emit_rest")

step_fn = make_step(logp_trans_pt)

alpha_all, _ = scan(
    fn=step_fn,
    sequences=logp_emit_rest_pt,
    outputs_info=alpha_prev_pt,
)

scan_fn = pytensor.function([alpha_prev_pt, logp_emit_rest_pt], alpha_all)

result = scan_fn(alpha_0, logp_emit[1:])
print("Scan result:")
print(result)
print()

# Manual calculation
alpha_1 = np.array([-2.23634551, -3.81568282])
alpha_2 = logp_emit[2] + logsumexp(alpha_1[:, np.newaxis] + logp_trans, axis=0)
print(f"Manual alpha_2: {alpha_2}")
print()

final_logp_scan = logsumexp(result[-1])
final_logp_manual = logsumexp(alpha_2)
print(f"Final logp from scan: {final_logp_scan}")
print(f"Final logp manual: {final_logp_manual}")
