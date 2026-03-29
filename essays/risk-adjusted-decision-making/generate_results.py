import os
import numpy as np
from src.simulation import simulate_risk_adjusted_ab_test
from src.visualization import plot_risk_posterior

os.makedirs("results", exist_ok=True)

print("🚀 Generating Part 3 — Risk-Adjusted Decision Making...")

# Two scenarios: repeatable small tests vs one-shot high-stakes launch
repeatable = simulate_risk_adjusted_ab_test(is_one_shot=False, seed=42)
one_shot   = simulate_risk_adjusted_ab_test(is_one_shot=True,  seed=42)

# Generate the key risk visualization (using the high-stakes scenario)
plot_risk_posterior(
    revenue_samples=one_shot["posterior"]["revenue_samples"],
    ev_revenue=one_shot["posterior"]["ev_revenue_gain"],
    cvar_5=one_shot["posterior"]["cvar_5"],
    var_5=one_shot["posterior"]["var_5"],
    p_ruin=one_shot["posterior"]["p_ruin"],
    save_path="results/risk_posterior.png"
)

print("\n✅ Part 3 results generated successfully!")
print(f"Repeatable EV revenue gain : ${repeatable['posterior']['ev_revenue_gain']:,.0f}")
print(f"One-shot CVaR 5%           : ${one_shot['posterior']['cvar_5']:,.0f}")
print(f"One-shot P(ruin)           : {one_shot['posterior']['p_ruin']:.1%}")
