import numpy as np
from typing import Dict, Any

def estimate_training_cost_tool(dataset_id: str, y: np.ndarray, model_class: str, seasonality_period: int = 1) -> Dict[str, Any]:
    n = len(y)
    
    # Base compute assumptions (e.g., standard Fargate sizing or standard compute instance)
    cost_per_minute = 0.0005
    minutes = 0.01 # Base container / import overhead
    
    # 🧠 Seasonality Penalty Factor
    # Seasonal models have more parameters to estimate. ARIMA expands combinatorially.
    sp_penalty = 1.0
    
    # Naive and Polynomial don't use seasonality parameters in their base implementations
    if seasonality_period > 1 and not ("Naive" in model_class or "Polynomial" in model_class):
        if "ARIMA" in model_class:
            # SARIMA permutations multiply the grid search space drastically
            sp_penalty = 4.0 + (seasonality_period * 0.05)
        elif "ETS" in model_class:
            # ETS state space expands non-linearly with seasonal period
            sp_penalty = 3.0 + (seasonality_period * 0.04)
        elif "Exponential" in model_class:
            # Adds strict seasonal states equivalent to the period length
            sp_penalty = 2.0 + (seasonality_period * 0.02)
        elif "Prophet" in model_class or "BATS" in model_class or "Theta" in model_class:
            # Fourier terms / trigonometric / Theta seasonality adds linear-ish overhead
            sp_penalty = 1.5 + (seasonality_period * 0.01)
        else:
            sp_penalty = 1.2
            
    # Heuristic mapping for standard Sktime scaling properties
    if "Naive" in model_class or "Polynomial" in model_class:
        minutes += (n * 0.00001)  # sp_penalty intentionally ignored
    elif "Theta" in model_class:
        minutes += (n * 0.00005) * sp_penalty
    elif "Exponential" in model_class:
        minutes += (n * 0.00005) * sp_penalty
    elif "Prophet" in model_class or "BATS" in model_class:
        # Roughly O(N log N)
        minutes += (n * np.log(n + 1) * 0.0001) * sp_penalty
    elif "ARIMA" in model_class or "ETS" in model_class:
        # Non-linear O(N^3) scaling typical of exact statespace solvers / likelihood maximization
        minutes += ((n / 100) ** 3) * 0.0015 * sp_penalty
        if minutes < 0.1:
            minutes = 0.1
    else:
        # Default fallback
        minutes += (n * 0.001) * sp_penalty

    cost_usd = minutes * cost_per_minute
    
    # SLA / Threshold Recommendation
    if minutes > 60:
        rec = "reject_too_expensive"
        hint = "get_model_complexity_budget" # Go back and pick a smaller model
    elif minutes > 10:
        rec = "high_cost_warning"
        hint = "fit_model"
    else:
        rec = "proceed"
        hint = "fit_model"
        
    return {
        "minutes": round(float(minutes), 3),
        "cost_usd": round(float(cost_usd), 5),
        "recommendation": rec,
        "next_action_hint": hint
    }
