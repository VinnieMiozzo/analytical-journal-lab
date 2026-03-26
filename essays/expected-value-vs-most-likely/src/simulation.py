import numpy as np
from typing import List, Tuple, Dict, Any

def run_monte_carlo(
    strategy: List[Tuple[float, float]],
    n_trials: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation for any (probability, outcome) strategy.
    
    Args:
        strategy: List of (probability, outcome) tuples
        n_trials: Number of simulation trials
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with analytic EV, simulated EV, mode, std, and raw trials
    """
    np.random.seed(seed)
    probs, outcomes = zip(*strategy)
    
    # Simulate trials
    trials = np.random.choice(outcomes, size=n_trials, p=probs)
    
    # Compute metrics
    ev_analytic = sum(p * o for p, o in strategy)
    ev_simulated = float(np.mean(trials))
    std = float(np.std(trials))
    
    # Mode (most frequent outcome)
    unique, counts = np.unique(trials, return_counts=True)
    mode = float(unique[np.argmax(counts)])
    
    return {
        "ev_analytic": ev_analytic,
        "ev_simulated": ev_simulated,
        "mode": mode,
        "std": std,
        "trials": trials,
        "n_trials": n_trials
    }


# Pre-defined strategies from the study note
STRATEGIES = {
    "Option A": [(0.9, 10), (0.1, -5)],
    "Option B": [(0.4, 30), (0.6, 0)],
    "Strategy 1": [(0.8, 5), (0.2, -20)],
    "Strategy 2": [(0.45, 15), (0.55, 0)]
}
