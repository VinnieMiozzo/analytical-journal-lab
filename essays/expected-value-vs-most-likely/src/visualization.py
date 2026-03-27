import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": True,
})

PAIRINGS = [
    ("Option A", "Option B", "Example 1 · Higher EV despite worse mode"),
    ("Strategy 1", "Strategy 2", "Example 2 · Safer-looking option destroys value"),
]

PALETTE = {
    "Option A": "#4C78A8",
    "Option B": "#F58518",
    "Strategy 1": "#54A24B",
    "Strategy 2": "#E45756",
}

def _empirical_distribution(trials: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values, counts = np.unique(trials, return_counts=True)
    probs = counts / counts.sum()
    return values.astype(float), probs.astype(float)

def _mode_from_distribution(values: np.ndarray, probs: np.ndarray) -> float:
    return float(values[np.argmax(probs)])

def _prob_label(p: float) -> str:
    return f"{p:.0%}" if p >= 0.095 else f"{p:.1%}"

def plot_outcome_distributions(
    results: Dict[str, Dict],
    save_path: str = "results/ev_decision_map.png",
    pairings: Optional[List[Tuple[str, str, str]]] = None
):
    """
    Cleaner discrete-outcome view:
    - possible outcomes as dots
    - dot size proportional to probability
    - diamond at the modal outcome
    - dashed line at expected value
    """
    pairings = pairings or PAIRINGS
    fig, axes = plt.subplots(len(pairings), 1, figsize=(14, 8.5))
    if len(pairings) == 1:
        axes = [axes]

    for ax, (left_name, right_name, panel_title) in zip(axes, pairings):
        row_names = [left_name, right_name]
        y_positions = [1, 0]

        all_x = []
        evs = []
        for name in row_names:
            values, probs = _empirical_distribution(np.asarray(results[name]["trials"]))
            all_x.extend(values.tolist())
            evs.append(float(results[name]["ev_analytic"]))

        xmin = min(all_x + evs)
        xmax = max(all_x + evs)
        span = (xmax - xmin) if xmax != xmin else 1
        xpad_left = max(2.5, 0.08 * span)
        xpad_right = max(5.5, 0.16 * span)
        ax.set_xlim(xmin - xpad_left, xmax + xpad_right)
        ax.set_ylim(-0.25, 1.25)

        winner = max(row_names, key=lambda n: results[n]["ev_analytic"])

        for y, name in zip(y_positions, row_names):
            color = PALETTE.get(name, None)
            trials = np.asarray(results[name]["trials"])
            values, probs = _empirical_distribution(trials)
            ev = float(results[name]["ev_analytic"])
            mode = float(results[name].get("mode", _mode_from_distribution(values, probs)))
            std = float(results[name]["std"])

            ax.hlines(y, xmin - xpad_left, xmax + xpad_right, color="#d9d9d9", linewidth=2, zorder=1)

            sizes = 2300 * probs + 180
            ax.scatter(values, [y] * len(values), s=sizes, color=color, alpha=0.86,
                       edgecolor="white", linewidth=1.8, zorder=3)

            for x, p in zip(values, probs):
                ax.annotate(
                    _prob_label(float(p)),
                    (x, y),
                    xytext=(0, 13),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    color="#333333"
                )

            ax.axvline(ev, color=color, linestyle=(0, (6, 3)), linewidth=2.3, alpha=0.95, zorder=2)
            ax.scatter([mode], [y], marker="D", s=85, color="white", edgecolor=color, linewidth=2.3, zorder=4)

            label = name + ("  ★ higher EV" if name == winner else "")
            ax.text(
                0.015, (y + 0.02 - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
                label,
                transform=ax.transAxes,
                fontsize=12.5,
                fontweight="bold",
                ha="left",
                va="center",
                color="#202020"
            )

            summary = f"EV {ev:.2f}   |   Mode {mode:.0f}   |   Std {std:.2f}"
            ax.text(
                xmax + xpad_right * 0.52, y, summary,
                fontsize=10.5,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#fafafa", edgecolor="#dddddd")
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(["", ""])
        ax.set_xlabel("Outcome")
        ax.set_title(panel_title, loc="left", pad=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.22)
        ax.grid(axis="y", visible=False)
        ax.text(
            0.995, 1.035,
            "Dot size = probability   ·   Diamond = mode   ·   Dashed line = expected value",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color="#555555"
        )

    fig.suptitle("Expected Value vs. Most Likely Outcome", fontsize=24, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Decision map saved to {save_path}")

def plot_ev_convergence(
    results: Dict[str, Dict],
    save_path: str = "results/ev_convergence_improved.png",
    pairings: Optional[List[Tuple[str, str, str]]] = None
):
    """
    Pairwise cumulative-average convergence with:
    - one panel per comparison
    - log-scaled x-axis to make early volatility visible
    - analytic EV reference lines
    """
    pairings = pairings or PAIRINGS
    fig, axes = plt.subplots(1, len(pairings), figsize=(14, 5.2))
    if len(pairings) == 1:
        axes = [axes]

    for ax, (left_name, right_name, panel_title) in zip(axes, pairings):
        for name in [left_name, right_name]:
            color = PALETTE.get(name, None)
            trials = np.asarray(results[name]["trials"], dtype=float)
            cumulative_avg = np.cumsum(trials) / np.arange(1, len(trials) + 1)
            ev = float(results[name]["ev_analytic"])

            x = np.arange(1, len(trials) + 1)
            ax.plot(x, cumulative_avg, color=color, linewidth=2.4, label=f"{name} path")
            ax.axhline(ev, color=color, linestyle=(0, (5, 3)), linewidth=1.8, alpha=0.85,
                       label=f"{name} EV = {ev:.2f}")

        ax.set_xscale("log")
        ax.set_title(panel_title, loc="left", fontweight="bold")
        ax.set_xlabel("Number of trials (log scale)")
        ax.set_ylabel("Cumulative average return")
        ax.grid(True, which="major", alpha=0.22)
        ax.grid(True, which="minor", alpha=0.08)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4)
    fig.suptitle("Law of Large Numbers: Convergence Toward Theoretical EV", fontsize=19, fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Improved convergence plot saved to {save_path}")

