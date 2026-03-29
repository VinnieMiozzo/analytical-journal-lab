import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy.stats import norm


def _validate_inputs(
    n_users_per_variant: int,
    baseline_conversion: float,
    true_lift: float,
    revenue_per_conversion: float,
    revenue_std: float,
    prior_alpha: float,
    prior_beta: float,
) -> None:
    if n_users_per_variant <= 0:
        raise ValueError("n_users_per_variant must be > 0.")
    if not (0 < baseline_conversion < 1):
        raise ValueError("baseline_conversion must be between 0 and 1.")
    if revenue_per_conversion <= 0:
        raise ValueError("revenue_per_conversion must be > 0.")
    if revenue_std < 0:
        raise ValueError("revenue_std must be >= 0.")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be > 0.")

    treatment_conversion = baseline_conversion * (1 + true_lift)
    if not (0 < treatment_conversion < 1):
        raise ValueError(
            "baseline_conversion * (1 + true_lift) must stay between 0 and 1."
        )


def _lognormal_params(mean: float, std: float) -> Tuple[float, float]:
    """
    Convert target mean/std of a lognormal variable into underlying normal params.
    """
    if std == 0:
        return np.log(mean), 0.0

    variance = std**2
    sigma2 = np.log(1 + variance / (mean**2))
    mu = np.log(mean) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    return mu, sigma


def _two_proportion_pvalue(x_c: int, n_c: int, x_t: int, n_t: int) -> float:
    p_pool = (x_c + x_t) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    if se == 0:
        return 1.0
    z = ((x_t / n_t) - (x_c / n_c)) / se
    return float(2 * (1 - norm.cdf(abs(z))))


def _credible_interval(x: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    alpha = 1 - level
    low, high = np.quantile(x, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def simulate_ab_test(
    n_users_per_variant: int = 10_000,
    baseline_conversion: float = 0.05,
    true_lift: float = 0.15,              # relative lift
    revenue_per_conversion: float = 25.0,
    revenue_std: float = 15.0,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    posterior_draws: int = 20_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulate a user-level A/B test with:
    - Bernoulli conversion process
    - positive revenue generated only on conversion
    - Beta-Binomial posterior for conversion rates
    - posterior distribution for delta conversion rate / lift / incremental revenue

    Revenue per conversion is simulated as lognormal so values stay non-negative.
    """
    _validate_inputs(
        n_users_per_variant=n_users_per_variant,
        baseline_conversion=baseline_conversion,
        true_lift=true_lift,
        revenue_per_conversion=revenue_per_conversion,
        revenue_std=revenue_std,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )

    rng = np.random.default_rng(seed)

    control_conv_true = baseline_conversion
    treatment_conv_true = baseline_conversion * (1 + true_lift)

    # Simulate conversions
    control_converted = rng.binomial(1, control_conv_true, size=n_users_per_variant)
    treatment_converted = rng.binomial(1, treatment_conv_true, size=n_users_per_variant)

    # Simulate positive order values only for converters
    mu_log, sigma_log = _lognormal_params(revenue_per_conversion, revenue_std)

    control_revenue = np.zeros(n_users_per_variant)
    treatment_revenue = np.zeros(n_users_per_variant)

    n_control_conv = int(control_converted.sum())
    n_treatment_conv = int(treatment_converted.sum())

    if sigma_log == 0:
        control_revenue[control_converted == 1] = revenue_per_conversion
        treatment_revenue[treatment_converted == 1] = revenue_per_conversion
    else:
        control_revenue[control_converted == 1] = rng.lognormal(
            mean=mu_log, sigma=sigma_log, size=n_control_conv
        )
        treatment_revenue[treatment_converted == 1] = rng.lognormal(
            mean=mu_log, sigma=sigma_log, size=n_treatment_conv
        )

    df = pd.DataFrame(
        {
            "variant": np.repeat(["Control", "Treatment"], n_users_per_variant),
            "converted": np.concatenate([control_converted, treatment_converted]),
            "revenue": np.concatenate([control_revenue, treatment_revenue]),
        }
    )

    # Aggregate
    agg = (
        df.groupby("variant", sort=False)
        .agg(
            users=("variant", "size"),
            conversions=("converted", "sum"),
            revenue=("revenue", "sum"),
        )
        .reset_index()
    )

    agg["conversion_rate"] = agg["conversions"] / agg["users"]
    agg["revenue_per_user"] = agg["revenue"] / agg["users"]
    agg["avg_order_value"] = np.where(
        agg["conversions"] > 0,
        agg["revenue"] / agg["conversions"],
        0.0,
    )

    control_row = agg.loc[agg["variant"] == "Control"].iloc[0]
    treatment_row = agg.loc[agg["variant"] == "Treatment"].iloc[0]

    x_c = int(control_row["conversions"])
    n_c = int(control_row["users"])
    x_t = int(treatment_row["conversions"])
    n_t = int(treatment_row["users"])

    cr_c = float(control_row["conversion_rate"])
    cr_t = float(treatment_row["conversion_rate"])
    rpu_c = float(control_row["revenue_per_user"])
    rpu_t = float(treatment_row["revenue_per_user"])

    # Beta-Binomial posterior for conversion rate
    alpha_c = prior_alpha + x_c
    beta_c = prior_beta + (n_c - x_c)

    alpha_t = prior_alpha + x_t
    beta_t = prior_beta + (n_t - x_t)

    control_rate_post = rng.beta(alpha_c, beta_c, size=posterior_draws)
    treatment_rate_post = rng.beta(alpha_t, beta_t, size=posterior_draws)

    delta_rate_post = treatment_rate_post - control_rate_post
    relative_lift_post = np.divide(
        delta_rate_post,
        control_rate_post,
        out=np.zeros_like(delta_rate_post),
        where=control_rate_post > 0,
    )

    # Use pooled observed average order value for incremental revenue
    converted_revenue = df.loc[df["converted"] == 1, "revenue"]
    pooled_aov = (
        float(converted_revenue.mean())
        if len(converted_revenue) > 0
        else revenue_per_conversion
    )

    incremental_revenue_per_user_post = delta_rate_post * pooled_aov
    incremental_revenue_total_post = incremental_revenue_per_user_post * n_users_per_variant

    p_beat_control = float(np.mean(treatment_rate_post > control_rate_post))
    p_positive_revenue = float(np.mean(incremental_revenue_total_post > 0))

    lift_ci_low, lift_ci_high = _credible_interval(relative_lift_post, level=0.95)
    delta_ci_low, delta_ci_high = _credible_interval(delta_rate_post, level=0.95)
    rev_ci_low, rev_ci_high = _credible_interval(incremental_revenue_total_post, level=0.95)

    var_5 = float(np.percentile(incremental_revenue_total_post, 5))
    cvar_5 = float(incremental_revenue_total_post[incremental_revenue_total_post <= var_5].mean())

    p_value = _two_proportion_pvalue(x_c, n_c, x_t, n_t)

    observed_relative_lift = (cr_t - cr_c) / cr_c if cr_c > 0 else np.nan
    observed_incremental_rpu = rpu_t - rpu_c
    observed_incremental_revenue_total = observed_incremental_rpu * n_users_per_variant

    return {
        "data": df,
        "agg": agg,
        "observed": {
            "control_conversion_rate": cr_c,
            "treatment_conversion_rate": cr_t,
            "delta_conversion_rate": cr_t - cr_c,
            "relative_lift": float(observed_relative_lift),
            "control_rpu": rpu_c,
            "treatment_rpu": rpu_t,
            "incremental_rpu": observed_incremental_rpu,
            "incremental_revenue_total": observed_incremental_revenue_total,
            "p_value": p_value,
        },
        "posterior": {
            "samples": posterior_draws,
            "control_rate_samples": control_rate_post,
            "treatment_rate_samples": treatment_rate_post,
            "delta_rate_samples": delta_rate_post,
            "relative_lift_samples": relative_lift_post,
            "incremental_revenue_total_samples": incremental_revenue_total_post,
            "ev_lift": float(relative_lift_post.mean()),
            "ev_delta_rate": float(delta_rate_post.mean()),
            "ev_revenue_gain": float(incremental_revenue_total_post.mean()),
            "p_beat_control": p_beat_control,
            "p_positive_revenue": p_positive_revenue,
            "lift_ci_95": (lift_ci_low, lift_ci_high),
            "delta_rate_ci_95": (delta_ci_low, delta_ci_high),
            "revenue_gain_ci_95": (rev_ci_low, rev_ci_high),
            "var_5": var_5,
            "cvar_5": cvar_5,
        },
        "params": {
            "n_users_per_variant": n_users_per_variant,
            "baseline_conversion": baseline_conversion,
            "treatment_conversion_true": treatment_conv_true,
            "true_lift": true_lift,
            "revenue_per_conversion": revenue_per_conversion,
            "revenue_std": revenue_std,
            "prior_alpha": prior_alpha,
            "prior_beta": prior_beta,
            "seed": seed,
        },
    }
