import warnings
import numpy as np
from typing import Dict, Any
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

def run_stationarity_test_tool(dataset_id: str, y: np.ndarray) -> Dict[str, Any]:
    '''
    Statistical tool: Runs Augmented Dickey-Fuller (ADF) and KPSS tests to determine stationarity.
    Returns: p-values, conclusion, and next_action_hint.
    '''
    # Need at least a few points
    if len(y) < 10:
        return {
            "adf_pvalue": 1.0,
            "kpss_pvalue": 0.0,
            "is_stationary": False,
            "conclusion": "insufficient_data",
            "next_action_hint": "get_model_complexity_budget"
        }
        
    # Run ADF test (Null Hypothesis: Non-Stationary / has unit root)
    # Low p-value (< 0.05) -> Reject Null -> Stationary
    try:
        adf_result = adfuller(y, autolag='AIC')
        adf_pval = float(adf_result[1])
    except Exception:
        adf_pval = 1.0 # default to non-stationary on failure
        
    # Run KPSS test (Null Hypothesis: Stationary)
    # Low p-value (< 0.05) -> Reject Null -> Non-Stationary
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_result = kpss(y, regression='c', nlags="auto")
            kpss_pval = float(kpss_result[1])
    except Exception:
        kpss_pval = 0.0 # default to non-stationary on failure

    # Evaluate joint condition
    adf_stationary = bool(adf_pval < 0.05)
    kpss_stationary = bool(kpss_pval >= 0.05)
    
    if adf_stationary and kpss_stationary:
        conclusion = "strictly_stationary"
        is_stat = True
        next_action = "detect_seasonality"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "strictly_non_stationary"
        is_stat = False
        next_action = "check_structural_break" # structural breaks often mimic unit roots
    elif not adf_stationary and kpss_stationary:
        # Stationary around a deterministic trend
        conclusion = "trend_stationary"
        is_stat = False
        next_action = "detect_seasonality" 
    else: # adf_stationary and not kpss_stationary
        # Difference stationary (needs differencing but no specific trend)
        conclusion = "difference_stationary"
        is_stat = False
        next_action = "check_structural_break"
        
    return {
        "adf_pvalue": round(adf_pval, 4),
        "kpss_pvalue": round(kpss_pval, 4),
        "is_stationary": is_stat,
        "conclusion": conclusion,
        "next_action_hint": next_action
    }
