import numpy as np
from typing import Dict, Any

def get_model_complexity_budget_tool(dataset_id: str, y: np.ndarray) -> Dict[str, Any]:
    n = len(y)
    
    # Categorized by algorithmic complexity and data hunger
    O_1_models = ["NaiveForecaster"]
    O_N_models = ["ThetaForecaster", "ExponentialSmoothing", "PolynomialTrendForecaster"]
    O_N2_models = ["Prophet", "TBATS", "BATS"]
    O_N3_models = ["AutoARIMA", "AutoETS"]
    DL_models = ["LSTMForecaster", "Transformers"] # Deep learning requires large N
    
    permitted = O_1_models + O_N_models
    forbidden = []
    reason = ""
    
    if n < 30:
        forbidden.extend(O_N2_models + O_N3_models + DL_models)
        reason = f"Dataset size {n} is highly constrained. Complex parameter spaces will overfit violently. Restricted to Naive and simple Linear models."
    elif n < 200:
        permitted.extend(O_N2_models + O_N3_models)
        forbidden.extend(DL_models)
        reason = f"Dataset size {n} is sufficient for classical statistical models, but deep learning models will likely overfit or fail to learn."
    elif n < 5000:
        permitted.extend(O_N2_models + O_N3_models + DL_models)
        reason = f"Dataset size {n} is in the optimal band for all model types."
    else:
        # Huge datasets - block O(N^3) standard algorithms
        permitted.extend(O_N2_models + DL_models)
        forbidden.extend(O_N3_models)
        reason = f"Dataset size {n} is too large for O(N^3) models like AutoARIMA. Inference and training will exceed timeout thresholds. Use O(N) or O(N log N) models, or deep learning."
        
    return {
        "permitted": permitted,
        "forbidden": forbidden,
        "reason": reason,
        "next_action_hint": "estimate_training_cost"
    }
