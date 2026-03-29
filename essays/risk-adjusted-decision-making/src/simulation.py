import numpy as np
import pandas as pd
from typing import Dict, Any


def simulate_risk_adjusted_ab_test(
    n_users_per_variant: int = 15000,
    baseline_conversion: float = 0.05,
    true_lift: float = 0.15,
    revenue_per_conversion: float = 25.0,
    revenue_std: float = 15.0,
    catastrophic_prob: float = 0.05,
    catastrophic_loss_multiplier: float = -8.0,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    posterior_draws: int = 20000,
    bootstrap_batch_size: int = 500,
    seed: int = 42,
    is_one_shot: bool = False
) -> Dict[str, Any]:
    """
    Risk-adjusted A/B simulation.

    Key idea:
    - Conversion uncertainty is modeled with a Beta posterior.
    - Revenue risk is modeled from the user-level revenue distribution via bootstrap.
    - If is_one_shot=True, catastrophe is treated as treatment-specific tail risk.
    """

    rng = np.random.default_rng(seed)

    # True conversion rates
    control_conv = baseline_conversion
    treatment_conv = baseline_conversion * (1 + true_lift)

    # Simulate conversions
    control_converted = rng.binomial(1, control_conv, n_users_per_variant).astype(bool)
    treatment_converted = rng.binomial(1, treatment_conv, n_users_per_variant).astype(bool)

    # Lognormal revenue parameters
    cv = revenue_std / revenue_per_conversion
    sigma_log = np.sqrt(np.log1p(cv**2))
    mu_log = np.log(revenue_per_conversion) - 0.5 * sigma_log**2

    # User-level revenue arrays
    control_revenue = np.zeros(n_users_per_variant, dtype=float)
    treatment_revenue = np.zeros(n_users_per_variant, dtype=float)

    control_revenue[control_converted] = rng.lognormal(
        mean=mu_log,
        sigma=sigma_log,
        size=control_converted.sum()
    )
    treatment_revenue[treatment_converted] = rng.lognormal(
        mean=mu_log,
        sigma=sigma_log,
        size=treatment_converted.sum()
    )

    # One-shot catastrophic tail risk:
    # if this risk belongs only to the launch/treatment, apply it only there.
    if is_one_shot:
        cat_mask_treatment = rng.random(n_users_per_variant) < catastrophic_prob
        treatment_revenue[cat_mask_treatment] *= catastrophic_loss_multiplier

    # User-level dataset
    df = pd.DataFrame({
        "variant": np.repeat(["Control", "Treatment"], n_users_per_variant),
        "converted": np.concatenate([control_converted.astype(int), treatment_converted.astype(int)]),
        "revenue": np.concatenate([control_revenue, treatment_revenue])
    })

    # Aggregates
    agg = (
        df.groupby("variant", sort=False)
        .agg(
            users=("variant", "size"),
            conversions=("converted", "sum"),
            revenue=("revenue", "sum")
        )
        .assign(
            conversion_rate=lambda x: x["conversions"] / x["users"],
            revenue_per_user=lambda x: x["revenue"] / x["users"]
        )
    )

    # Explicit lookups
    x_c = int(agg.loc["Control", "conversions"])
    n_c = int(agg.loc["Control", "users"])
    x_t = int(agg.loc["Treatment", "conversions"])
    n_t = int(agg.loc["Treatment", "users"])

    # Beta posterior for conversion
    alpha_c = prior_alpha + x_c
    beta_c = prior_beta + (n_c - x_c)
    alpha_t = prior_alpha + x_t
    beta_t = prior_beta + (n_t - x_t)

    control_post = rng.beta(alpha_c, beta_c, posterior_draws)
    treatment_post = rng.beta(alpha_t, beta_t, posterior_draws)

    eps = 1e-12
    lift_post = (treatment_post - control_post) / np.clip(control_post, eps, None)

    # Revenue posterior / predictive via bootstrap on user-level revenue
    revenue_samples = np.empty(posterior_draws, dtype=float)

    for start in range(0, posterior_draws, bootstrap_batch_size):
        batch = min(bootstrap_batch_size, posterior_draws - start)

        c_idx = rng.integers(0, n_users_per_variant, size=(batch, n_users_per_variant))
        t_idx = rng.integers(0, n_users_per_variant, size=(batch, n_users_per_variant))

        control_mean_rpu = control_revenue[c_idx].mean(axis=1)
        treatment_mean_rpu = treatment_revenue[t_idx].mean(axis=1)

        revenue_samples[start:start + batch] = (
            treatment_mean_rpu - control_mean_rpu
        ) * n_users_per_variant

    # Risk metrics from revenue distribution
    ev_lift = float(lift_post.mean())
    p_beat_control = float((treatment_post > control_post).mean())

    ev_revenue = float(revenue_samples.mean())
    median_revenue = float(np.median(revenue_samples))
    var_5 = float(np.percentile(revenue_samples, 5))
    tail = revenue_samples[revenue_samples <= var_5]
    cvar_5 = float(tail.mean()) if len(tail) > 0 else var_5
    p_ruin = float((revenue_samples < 0).mean())
    p_positive_revenue = float((revenue_samples > 0).mean())

    observed_incremental_revenue = float(
        (agg.loc["Treatment", "revenue_per_user"] - agg.loc["Control", "revenue_per_user"])
        * n_users_per_variant
    )

    return {
        "data": df,
        "agg": agg.reset_index(),
        "posterior": {
            "ev_lift": ev_lift,
            "p_beat_control": p_beat_control,
            "ev_revenue_gain": ev_revenue,
            "median_revenue_gain": median_revenue,
            "observed_incremental_revenue": observed_incremental_revenue,
            "var_5": var_5,
            "cvar_5": cvar_5,
            "p_ruin": p_ruin,
            "p_positive_revenue": p_positive_revenue,
            "lift_samples": lift_post,
            "revenue_samples": revenue_samples
        },
        "params": {
            "is_one_shot": is_one_shot,
            "catastrophic_prob": catastrophic_prob,
            "catastrophic_loss_multiplier": catastrophic_loss_multiplier
        }
    }
