import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from scipy.stats import norm


def _pct(x, pos):
    return f"{x:.0%}"


def _money(x, pos):
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def plot_ab_test_summary(
    agg,
    posterior,
    save_path="results/ab_test_summary.png",
):
    """
    3-panel summary:
    1) conversion rate by variant
    2) revenue per user by variant
    3) posterior distribution of relative lift
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("A/B Test Summary: Treatment vs. Control", fontsize=16, fontweight="bold")

    colors = {"Control": "#4C78A8", "Treatment": "#F58518"}
    variants = agg["variant"].tolist()

    # Panel 1: Conversion rate
    ax = axes[0]
    cr = agg["conversion_rate"].values
    bars = ax.bar(variants, cr, color=[colors[v] for v in variants], width=0.6)
    ax.set_title("Observed Conversion Rate", fontweight="bold")
    ax.set_ylabel("Conversion Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, max(cr) * 1.25)

    for b, v in zip(bars, cr):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(cr) * 0.03,
            f"{v:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    uplift = (cr[1] - cr[0]) / cr[0] if cr[0] > 0 else np.nan
    ax.text(
        0.5,
        0.95,
        f"Observed lift: {uplift:.1%}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
    )

    # Panel 2: Revenue per user
    ax = axes[1]
    rpu = agg["revenue_per_user"].values
    bars = ax.bar(variants, rpu, color=[colors[v] for v in variants], width=0.6)
    ax.set_title("Observed Revenue per User", fontweight="bold")
    ax.set_ylabel("Revenue per User")
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_ylim(0, max(rpu) * 1.25 if max(rpu) > 0 else 1)

    for b, v in zip(bars, rpu):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(rpu) * 0.03 if max(rpu) > 0 else 0.02,
            f"${v:,.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    incr_rpu = rpu[1] - rpu[0]
    ax.text(
        0.5,
        0.95,
        f"Incremental RPU: ${incr_rpu:,.2f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
    )

    # Panel 3: Posterior lift distribution
    ax = axes[2]
    lift = posterior["relative_lift_samples"]
    ev_lift = posterior["ev_lift"]
    ci_low, ci_high = posterior["lift_ci_95"]
    p_beat = posterior["p_beat_control"]

    ax.hist(lift, bins=60, density=True, alpha=0.80, color="#72B7B2", edgecolor="white")
    ax.axvline(0, color="#C44E52", linestyle="--", linewidth=2, label="No lift")
    ax.axvline(ev_lift, color="#222222", linestyle="-", linewidth=2, label=f"Posterior mean = {ev_lift:.1%}")
    ax.axvspan(ci_low, ci_high, color="#BDBDBD", alpha=0.20, label=f"95% CI [{ci_low:.1%}, {ci_high:.1%}]")

    ax.set_title("Posterior Distribution of Relative Lift", fontweight="bold")
    ax.set_xlabel("Relative Lift")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=True)

    ax.text(
        0.98,
        0.95,
        f"P(Treatment > Control) = {p_beat:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def estimate_power_curve(
    baseline_conversion: float = 0.05,
    true_lift: float = 0.15,
    alpha: float = 0.05,
    users_grid=None,
    n_sims: int = 2000,
    seed: int = 42,
):
    """
    Estimate power via repeated simulation of a two-proportion z-test.
    """
    if users_grid is None:
        users_grid = np.array([2000, 4000, 6000, 8000, 10000, 15000, 20000, 30000, 40000, 50000])

    rng = np.random.default_rng(seed)

    p_c = baseline_conversion
    p_t = baseline_conversion * (1 + true_lift)

    power = []

    for n in users_grid:
        x_c = rng.binomial(n, p_c, size=n_sims)
        x_t = rng.binomial(n, p_t, size=n_sims)

        p_pool = (x_c + x_t) / (2 * n)
        se = np.sqrt(p_pool * (1 - p_pool) * (2 / n))
        z = np.divide(
            (x_t / n) - (x_c / n),
            se,
            out=np.zeros_like(se, dtype=float),
            where=se > 0,
        )
        pvals = 2 * (1 - norm.cdf(np.abs(z)))
        power.append(np.mean(pvals < alpha))

    return np.asarray(users_grid), np.asarray(power)


def plot_power_curve(
    baseline_conversion: float = 0.05,
    true_lift: float = 0.15,
    alpha: float = 0.05,
    users_grid=None,
    n_sims: int = 2000,
    seed: int = 42,
    save_path="results/power_curve.png",
):
    users, power = estimate_power_curve(
        baseline_conversion=baseline_conversion,
        true_lift=true_lift,
        alpha=alpha,
        users_grid=users_grid,
        n_sims=n_sims,
        seed=seed,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(users, power, marker="o", linewidth=2)
    ax.axhline(0.80, linestyle="--", linewidth=1.5, label="80% power target")

    idx = np.where(power >= 0.80)[0]
    if len(idx) > 0:
        n_required = users[idx[0]]
        ax.axvline(n_required, linestyle="--", linewidth=1.5, label=f"~{n_required:,} users / variant")
        ax.annotate(
            f"{n_required:,} users/variant",
            xy=(n_required, power[idx[0]]),
            xytext=(10, -25),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_title("Power Curve for Detecting Treatment Lift", fontweight="bold")
    ax.set_xlabel("Users per Variant")
    ax.set_ylabel("Estimated Statistical Power")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
