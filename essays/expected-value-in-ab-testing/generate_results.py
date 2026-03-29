import os
import numpy as np
from src.simulation import simulate_ab_test
from src.visualization import plot_ab_test_summary, plot_power_curve

os.makedirs("results", exist_ok=True)

print("🚀 Generating full senior A/B simulation (Part 2)...")

result = simulate_ab_test(
    n_users_per_variant=15_000,
    baseline_conversion=0.05,
    true_lift=0.15,
    revenue_per_conversion=25.0,
    revenue_std=15.0,
    prior_alpha=1.0,
    prior_beta=1.0,
    posterior_draws=20_000,
    seed=42
)

# Save figures for LaTeX
plot_ab_test_summary(
    agg=result["agg"],
    posterior=result["posterior"],
    save_path="results/ab_test_summary.png"
)

plot_power_curve(
    baseline_conversion=0.05,
    true_lift=0.15,
    save_path="results/power_curve.png"
)

print("✅ All files generated!")
print(f"EV Lift (posterior): {result['posterior']['ev_lift']:.1%}")
print(f"P(Beat Control): {result['posterior']['p_beat_control']:.1%}")
print(f"EV Revenue Gain: ${result['posterior']['ev_revenue_gain']:,.0f}")
print("\nFigures saved in results/ folder → ready for LaTeX")
