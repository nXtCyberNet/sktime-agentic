from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.schemas import DataProfile

CANDIDATE_ESTIMATORS: list[str] = [
    # O(1)
    "NaiveForecaster",
    # O(N)
    "ThetaForecaster",
    "ExponentialSmoothing",
    "PolynomialTrendForecaster",
    # O(N log N)
    "Prophet",
    "TBATS",
    "BATS",
    # O(N^3)
    "AutoARIMA",
    "AutoETS",
    # Deep learning (large N only)
    "LSTMForecaster",
    "Transformers",
]

# Kept as a module-level name for import compatibility with existing tests that
# patch app.agents.model_selector.ALLOWED_ESTIMATORS.  It is now populated
# from CANDIDATE_ESTIMATORS rather than left empty.
ALLOWED_ESTIMATORS: list[str] = list(CANDIDATE_ESTIMATORS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_for_profile(profile: "DataProfile") -> list[str]:
    """
    Return the subset of CANDIDATE_ESTIMATORS that are permitted for the
    given DataProfile's complexity budget.

    This is the canonical filter used by ModelSelectorAgent before calling
    the LLM, and by tests that need a deterministic allowed list without
    an LLM call.
    """
    permitted: list[str] = profile.complexity_budget.get("permitted_models", [])
    forbidden: list[str] = profile.complexity_budget.get("forbidden_models", [])
    forbidden_set = set(forbidden)
    return [e for e in CANDIDATE_ESTIMATORS if e in permitted and e not in forbidden_set]


def validate_pipeline_spec(spec: dict[str, Any], registry: list[str] | None = None) -> bool:
    if registry is None:
        registry = ALLOWED_ESTIMATORS

    estimators = spec.get("estimators")
    if not isinstance(estimators, list):
        return False
    if not estimators:
        return False

    allowed_set = set(registry)
    for name in estimators:
        if name not in allowed_set:
            return False
    return True