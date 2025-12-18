"""State sequence metrics for HMM validation."""

import numpy as np
from typing import Dict, Any


def compute_state_metrics(z_true: np.ndarray, num_states: int) -> Dict[str, Any]:
    """Compute state sequence metrics: dwell times, transitions.

    Parameters
    ----------
    z_true : np.ndarray, shape (T,)
        Hidden state sequence
    num_states : int
        Number of states in the HMM

    Returns
    -------
    metrics : dict
        Dictionary with dwell times, transition counts, and sanity checks
    """
    # Dwell times per state
    dwell_times = {s: [] for s in range(num_states)}
    current_state = z_true[0]
    dwell_start = 0

    for t in range(1, len(z_true)):
        if z_true[t] != current_state:
            dwell_times[current_state].append(t - dwell_start)
            current_state = z_true[t]
            dwell_start = t
    dwell_times[current_state].append(len(z_true) - dwell_start)

    # Transition counts
    transition_counts = np.zeros((num_states, num_states), dtype=int)
    for t in range(1, len(z_true)):
        transition_counts[z_true[t - 1], z_true[t]] += 1

    return {
        "num_states": num_states,
        "dwell_times": {
            s: {"mean": float(np.mean(times)) if times else 0.0, "count": len(times)}
            for s, times in dwell_times.items()
        },
        "transition_counts": transition_counts.tolist(),
        "single_state_check": {
            "is_single_state": (num_states == 1),
            "actual_unique_states": int(len(np.unique(z_true))),
        },
    }
