import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

sns.set_style("whitegrid")

def plot_outcome_distributions(results: Dict[str, Dict], save_path: str = "results/ev_histogram.png"):
    """Create side-by-side histograms with dashed EV lines."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    
    for i, (name, data) in enumerate(results.items()):
        trials = data["trials"]
        ev = data["ev_analytic"]
        
        sns.histplot(trials, bins=50, kde=False, ax=axes[i], color="#1f77b4", alpha=0.7)
        axes[i].axvline(ev, color="red", linestyle="--", linewidth=2, label=f"EV = {ev:.2f}")
        axes[i].set_title(f"{name}\nMode = {data['mode']:.0f} | Std = {data['std']:.2f}")
        axes[i].set_xlabel("Outcome")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Histogram saved to {save_path}")


def plot_ev_convergence(results: Dict[str, Dict], save_path: str = "results/ev_convergence.png"):
    """Show Law of Large Numbers convergence for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, data in results.items():
        trials = data["trials"]
        cumulative_avg = np.cumsum(trials) / np.arange(1, len(trials) + 1)
        ax.plot(cumulative_avg, label=f"{name} (final EV = {data['ev_analytic']:.2f})", linewidth=2)
    
    ax.set_xlabel("Number of Trials")
    ax.set_ylabel("Cumulative Average Return")
    ax.set_title("Law of Large Numbers: EV Convergence Over Trials")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Convergence plot saved to {save_path}")
