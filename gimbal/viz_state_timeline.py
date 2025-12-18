"""State sequence timeline visualization."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_state_timeline(
    z_true: np.ndarray,
    transition_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Plot state sequence over time and transition matrix.

    Parameters
    ----------
    z_true : np.ndarray, shape (T,)
        Hidden state sequence
    transition_matrix : np.ndarray, shape (S, S)
        State transition probability matrix
    output_path : Path
        Output file path
    """
    T = len(z_true)
    S = transition_matrix.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # State timeline
    ax1.plot(z_true, drawstyle="steps-post", linewidth=2, color="blue")
    ax1.set_xlabel("Timestep", fontsize=12)
    ax1.set_ylabel("State", fontsize=12)
    ax1.set_title("Hidden State Sequence", fontsize=14)
    ax1.set_yticks(range(S))
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T - 1)

    # Add state duration annotations (optional, for clarity)
    current_state = z_true[0]
    duration_start = 0
    for t in range(1, T):
        if z_true[t] != current_state:
            duration = t - duration_start
            ax1.axvspan(
                duration_start,
                t,
                alpha=0.2,
                color=f"C{current_state}",
                label=f"State {current_state}" if duration_start == 0 else "",
            )
            current_state = z_true[t]
            duration_start = t
    # Last segment
    ax1.axvspan(duration_start, T, alpha=0.2, color=f"C{current_state}")

    # Transition matrix heatmap
    im = ax2.imshow(transition_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax2.set_xlabel("To State", fontsize=12)
    ax2.set_ylabel("From State", fontsize=12)
    ax2.set_title("State Transition Matrix", fontsize=14)
    ax2.set_xticks(range(S))
    ax2.set_yticks(range(S))

    # Add probability annotations
    for i in range(S):
        for j in range(S):
            text = ax2.text(
                j,
                i,
                f"{transition_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if transition_matrix[i, j] > 0.5 else "black",
                fontsize=10,
            )

    plt.colorbar(im, ax=ax2, label="Transition Probability")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
