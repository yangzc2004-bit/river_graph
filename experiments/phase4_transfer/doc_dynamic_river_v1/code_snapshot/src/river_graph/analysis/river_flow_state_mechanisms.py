"""Observed signal and partial-water explanations of confluence DOC SD ratios."""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

GROUPS = ("correlation", "amplitude_balance", "mean_flow_share")


def mixing_potential(parameters: dict) -> float:
    """Fixed-weight variance reduction relative to weighted branch variances, in %."""
    w, sa, sb, rho = (float(parameters[k]) for k in ("w", "sd_a", "sd_b", "rho"))
    if not 0 <= w <= 1 or min(sa, sb) < 0 or not -1.00000001 <= rho <= 1.00000001:
        raise ValueError("Invalid mixing summaries")
    rho = np.clip(rho, -1, 1)  # floating-point correlation tolerance only
    reference = w*sa**2 + (1-w)*sb**2
    if reference <= 1e-12:
        return np.nan
    mixed = w*w*sa**2 + (1-w)**2*sb**2 + 2*w*(1-w)*rho*sa*sb
    return float(100*(1-mixed/reference))


def shapley_mixing_change(low: dict, high: dict) -> tuple[dict, list[dict]]:
    """All-order attribution of an analytic summary change, not a causal effect."""
    keys = {"correlation": ("rho",), "amplitude_balance": ("sd_a", "sd_b"),
            "mean_flow_share": ("w",)}
    values, rows = {}, []
    for size in range(4):
        for group in combinations(GROUPS, size):
            selected = frozenset(group)
            parameters = {key: low[key] for key in ("w", "sd_a", "sd_b", "rho")}
            for name in group:
                for key in keys[name]:
                    parameters[key] = high[key]
            value = mixing_potential(parameters)
            values[selected] = value
            rows.append({"substituted_high_state_groups": "+".join(group) or "none",
                         "mixing_potential_pct": value, **parameters})
    contributions = {}
    for name in GROUPS:
        effect = 0.
        others = [g for g in GROUPS if g != name]
        for size in range(3):
            weight = factorial(size)*factorial(2-size)/factorial(3)
            for group in combinations(others, size):
                selected = frozenset(group)
                effect += weight*(values[selected | {name}]-values[selected])
        contributions[name+"_pp"] = float(effect)
    total = values[frozenset(GROUPS)]-values[frozenset()]
    contributions["mixing_potential_change_pp"] = float(total)
    if np.isfinite(total) and not np.isclose(total, sum(contributions[g+"_pp"] for g in GROUPS), atol=1e-9):
        raise ValueError("Shapley contributions do not sum to the observed summary change")
    return contributions, rows


def departure_identity(mixture: np.ndarray, outlet: np.ndarray) -> dict:
    """The departure includes unmonitored inputs and flow uncertainty, not only a channel."""
    mixture, outlet = np.asarray(mixture, dtype=float), np.asarray(outlet, dtype=float)
    if len(mixture) < 5 or mixture.shape != outlet.shape:
        raise ValueError("Need at least five paired observed concentrations")
    if not np.isfinite(mixture).all() or not np.isfinite(outlet).all():
        raise ValueError("Nonfinite observed signal")
    departure = outlet-mixture
    vm, vo, vd = (float(np.var(x, ddof=1)) for x in (mixture, outlet, departure))
    covariance = float(np.cov(mixture, departure, ddof=1)[0, 1])
    if not np.isclose(vo, vm+vd+2*covariance, atol=1e-9, rtol=1e-9):
        raise ValueError("Outlet-departure variance identity failed")
    if min(vm, vo) <= 1e-12:
        raise ValueError("No identifiable outlet/mixture fluctuation ratio")
    return {"mixture_sd": np.sqrt(vm), "outlet_sd": np.sqrt(vo),
            "departure_sd": np.sqrt(vd), "departure_variance": vd,
            "mixture_departure_covariance": covariance,
            "normalized_departure_variance": vd/vm,
            "normalized_departure_covariance_term": 2*covariance/vm,
            "outlet_mixture_sd_ratio": np.sqrt(vo/vm)}


def no_processing_feasibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain invalid water budgets and negative required unmonitored concentrations."""
    result = frame.copy()
    f = result.known_upstream_flow_share
    usable = np.isfinite(f) & f.gt(0) & f.le(.95)
    result["feasibility_budget_usable"] = usable
    result["budget_exclusion"] = np.select(
        [~np.isfinite(f) | f.le(0), f.gt(1), f.gt(.95)],
        ["invalid_or_nonpositive_share", "measured_share_exceeds_one", "near_complete_share"],
        default="included")
    implied = (result.doc_receiver-f*result.doc_dynamic_mix)/(1-f).where(usable)
    result["implied_unmonitored_doc_no_processing"] = implied
    result["nonnegative_unmonitored_solution"] = usable & implied.ge(0)
    return result


def state_summary(frame: pd.DataFrame) -> dict:
    if len(frame) < 5:
        raise ValueError("Need five observed dates in a flow state")
    a, b = frame.adjusted_doc_a.to_numpy(), frame.adjusted_doc_b.to_numpy()
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    if min(sa, sb) <= 1e-12:
        raise ValueError("No identifiable branch coordination")
    rho = float(np.corrcoef(a, b)[0, 1])
    w = float(frame.weight_a.mean())
    parameters = {"w": w, "sd_a": sa, "sd_b": sb, "rho": rho}
    population_w = float(frame.fixed_weight_a.iloc[0])
    fixed = w*a+(1-w)*b
    fixed_sd = float(np.std(fixed, ddof=1))
    if fixed_sd <= 1e-12:
        raise ValueError("Fixed-weight mixture has no fluctuation")
    terms = departure_identity(frame.adjusted_doc_dynamic_mix.to_numpy(),
                               frame.adjusted_doc_receiver.to_numpy())
    budget = no_processing_feasibility(frame)
    valid = budget.loc[budget.feasibility_budget_usable]
    result = {"n_campaigns": len(frame), "n_years": frame.date_local.dt.year.nunique(),
        **parameters, **terms, "mixing_potential_pct": mixing_potential(parameters),
        "population_weight_mixing_potential_pct": mixing_potential(parameters | {"w": population_w}),
        "fixed_state_weight_mixture_sd": fixed_sd,
        "dynamic_fixed_mixture_sd_ratio": terms["mixture_sd"]/fixed_sd,
        "mean_branch_a_flow_fraction": w, "sd_branch_a_flow_fraction": float(frame.weight_a.std(ddof=1)),
        "median_upstream_flow_share": float(frame.known_upstream_flow_share.median()),
        "n_usable_water_budgets": len(valid),
        "n_nonnegative_unmonitored_solutions": int(valid.nonnegative_unmonitored_solution.sum()),
        "fraction_nonnegative_unmonitored_solutions": float(valid.nonnegative_unmonitored_solution.mean()) if len(valid) else np.nan,
        "median_implied_unmonitored_doc": float(valid.implied_unmonitored_doc_no_processing.median()) if len(valid) else np.nan}
    return result


def state_contrast(low: dict, high: dict) -> dict:
    outlet_change = float(np.log(high["outlet_sd"]/low["outlet_sd"]))
    mixture_contribution = float(-np.log(high["mixture_sd"]/low["mixture_sd"]))
    total = float(np.log(high["outlet_mixture_sd_ratio"]/low["outlet_mixture_sd_ratio"]))
    if not np.isclose(total, outlet_change+mixture_contribution, atol=1e-10):
        raise ValueError("Ratio-change identity failed")
    contributions, _ = shapley_mixing_change(low, high)
    return {"log_ratio_change": total, "log_outlet_sd_contribution": outlet_change,
        "log_mixture_sd_contribution": mixture_contribution,
        "outlet_sd_change_pct": 100*(high["outlet_sd"]/low["outlet_sd"]-1),
        "mixture_sd_change_pct": 100*(high["mixture_sd"]/low["mixture_sd"]-1),
        "branch_correlation_change": high["rho"]-low["rho"],
        "mean_flow_fraction_change": high["w"]-low["w"],
        "dynamic_fixed_ratio_log_change": float(np.log(high["dynamic_fixed_mixture_sd_ratio"]/low["dynamic_fixed_mixture_sd_ratio"])),
        "departure_covariance_term_change": high["normalized_departure_covariance_term"]-low["normalized_departure_covariance_term"],
        "departure_variance_term_change": high["normalized_departure_variance"]-low["normalized_departure_variance"],
        **contributions}
