from src.simulation import run_monte_carlo, STRATEGIES
from src.visualization import plot_outcome_distributions, plot_ev_convergence
import os

os.makedirs("results", exist_ok=True)

print("🚀 Running Monte Carlo simulations...")

results = {}
for name, strategy in STRATEGIES.items():
    results[name] = run_monte_carlo(strategy, n_trials=10000)

# Generate the two charts used in the LaTeX study note
plot_outcome_distributions(results)
plot_ev_convergence(results)

print("\n✅ All done! Results folder now contains the two PNGs for your Overleaf PDF.")
