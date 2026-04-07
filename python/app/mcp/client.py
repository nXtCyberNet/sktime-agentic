from matplotlib.pylab import Any
import numpy as np
from typing import Dict, Any, List, Optional

from .check_structural_break import check_structural_break_tool
from .detect_seasonality import detect_seasonality_tool
from .get_model_complexity_budget import get_model_complexity_budget_tool
from .estimate_training_cost import estimate_training_cost_tool

class MCPClient:

    def __init__(self, data_loader=None):
        # data_loader is a callable that returns a numpy array for a dataset_id
        self.data_loader = data_loader

    def _get_data(self, dataset_id: str) -> np.ndarray:
        if self.data_loader:
            return self.data_loader(dataset_id)
        # Fallback safe mock data for testing if no loader provided
        return np.random.randn(100)

    def profile_dataset(self, dataset_id: str) -> Dict[str, Any]:
        pass
        
    def run_stationarity_test(self, dataset_id: str) -> Dict[str, Any]:
        pass

    def check_structural_break(self, dataset_id: str) -> Dict[str, Any]:
        y = self._get_data(dataset_id)
        return check_structural_break_tool(dataset_id, y)
    
    def detect_seasonality(self, dataset_id: str) -> Dict[str, Any]:
        y = self._get_data(dataset_id)
        return detect_seasonality_tool(dataset_id, y)
    
    def get_model_complexity_budget(self, dataset_id: str) -> Dict[str, Any]:
        y = self._get_data(dataset_id)
        return get_model_complexity_budget_tool(dataset_id, y)

    def estimate_training_cost(self, dataset_id: str, model_class: str, seasonality_period: int = 1) -> Dict[str, Any]:
        y = self._get_data(dataset_id)
        return estimate_training_cost_tool(dataset_id, y, model_class, seasonality_period)
