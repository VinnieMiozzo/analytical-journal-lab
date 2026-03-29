import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_risk_posterior(
    revenue_samples,
    ev_revenue,
    cvar_5,
    var_5,
    p_ruin,
    save_path="results/risk_posterior.png"
):
    """Posterior distribution with EV, VaR, CVaR, and ruin probability highlighted."""

    revenue_samples = np.asarray(revenue_samples, dtype=float)
    sample_min = float(revenue_samples.min())

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(
        revenue_samples,
        bins=80,
        density=True,
        alpha=0.75,
        edgecolor="white",
        color="#2E86AB"
    )

    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Break-even")
    ax.axvline(ev_revenue, color="black", linestyle="-", linewidth=2.5, label=f"EV = ${ev_revenue:,.0f}")
    ax.axvline(var_5, color="#F18F01", linestyle="-.", linewidth=2, label=f"5% VaR = ${var_5:,.0f}")
    ax.axvline(cvar_5, color="#7B2CBF", linestyle=":", linewidth=2, label=f"5% CVaR = ${cvar_5:,.0f}")

    # Shade the worst 5% tail
    ax.axvspan(sample_min, var_5, alpha=0.20, color="#C73E1D", label="Worst 5% tail")

    ax.set_title(
        "Posterior Distribution of Incremental Revenue\n(with VaR, CVaR, and Probability of Loss)",
        fontweight="bold"
    )
    ax.set_xlabel("Incremental Revenue (total)")
    ax.set_ylabel("Density")

    summary = (
        f"P(loss) = {p_ruin:.1%}\n"
        f"EV = ${ev_revenue:,.0f}\n"
        f"5% VaR = ${var_5:,.0f}\n"
        f"5% CVaR = ${cvar_5:,.0f}"
    )
    ax.text(
        0.02,
        0.97,
        summary,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9)
    )

    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Risk posterior plot saved -> {save_path}")
