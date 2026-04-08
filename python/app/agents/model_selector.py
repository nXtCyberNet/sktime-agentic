"""
ModelSelectorAgent
==================
Chooses the ranked list of candidate estimators for a given dataset.

Responsibilities
----------------
1. Pull the full DataProfile from Valkey (written by PipelineArchitectAgent).
2. Consult MLflow for any previously registered model versions on this dataset.
3. Call the LLM (via _llm_select) to produce a ranked list of sktime-compatible
   estimator class names, respecting complexity-budget hard constraints and
   production history (failed estimators, drift events).
4. Write the ranked list back to Valkey so TrainingAgent can consume it.

Design constraints
------------------
- LLM selection is advisory; forbidden_models from the complexity budget are
  always stripped before the list is returned to the caller.
- The agent must be idempotent: re-running with the same dataset_id always
  produces the same Valkey key (with a fresh TTL).
- No business logic lives in the prompt string; all thresholds and rules are
  injected as structured JSON so the LLM reasons from data, not from magic
  numbers baked into prose.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import AsyncAnthropic

from app.schemas import DataProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valkey key helpers
# ---------------------------------------------------------------------------
_PROFILE_KEY   = "profile:{dataset_id}"
_CANDIDATE_KEY = "candidates:{dataset_id}"
_CANDIDATE_TTL = 3600  # seconds


class ModelSelectorAgent:
    """
    Parameters
    ----------
    valkey      : async Valkey/Redis client
    mlflow_client : synchronous MLflow tracking client
    mcp_client  : MCPClient (already constructed with data/memory loaders)
    settings    : app.config.Settings
    """

    def __init__(self, valkey, mlflow_client, mcp_client, settings):
        self.valkey   = valkey
        self.mlflow   = mlflow_client
        self.mcp      = mcp_client
        self.settings = settings
        self._llm     = AsyncAnthropic()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def select(self, job) -> list[str]:
        """
        Main entry point called by the orchestrator.

        Parameters
        ----------
        job : any object with a .dataset_id attribute
              (typically a ForecastRequest or a retrain job dict)

        Returns
        -------
        List of estimator class names in preference order, e.g.
        ["AutoARIMA", "ExponentialSmoothing", "NaiveForecaster"]
        """
        dataset_id: str = job.dataset_id if hasattr(job, "dataset_id") else job["dataset_id"]
        logger.info("ModelSelectorAgent.select: starting for dataset_id=%s", dataset_id)

        # ---- 1. Load DataProfile from Valkey ----
        profile: DataProfile = await self._load_profile(dataset_id)

        # ---- 2. Enrich with MLflow history ----
        mlflow_context = self._fetch_mlflow_context(dataset_id)

        # ---- 3. Ask the LLM ----
        raw_candidates: list[str] = await self._llm_select(profile, mlflow_context)

        # ---- 4. Strip forbidden models (hard constraint – never delegated to LLM) ----
        forbidden: set[str] = set(profile.complexity_budget.get("forbidden_models", []))
        candidates = [m for m in raw_candidates if m not in forbidden]

        if not candidates:
            # Absolute fallback: cheapest permitted model
            permitted = profile.complexity_budget.get("permitted_models", ["NaiveForecaster"])
            candidates = permitted[:1]
            logger.warning(
                "ModelSelectorAgent: LLM returned only forbidden models for %s; "
                "falling back to %s",
                dataset_id, candidates,
            )

        # ---- 5. Persist to Valkey ----
        key = _CANDIDATE_KEY.format(dataset_id=dataset_id)
        await self.valkey.setex(key, _CANDIDATE_TTL, json.dumps(candidates))
        logger.info("ModelSelectorAgent: wrote %d candidates for %s → %s", len(candidates), dataset_id, candidates)

        return candidates

    # ------------------------------------------------------------------
    # LLM selection
    # ------------------------------------------------------------------

    async def _llm_select(
        self,
        profile: DataProfile,
        mlflow_context: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Send the full data profile to Claude and get back a JSON-encoded
        ranked list of estimator names.

        The system prompt encodes all hard rules as structured data;
        the LLM's job is purely to *rank* the permitted models, not to
        gate them.
        """
        system_prompt = (
            "You are a time-series model selection expert embedded in an automated ML pipeline. "
            "Your sole output must be a JSON array of estimator class names, ranked from most "
            "preferred to least preferred. Do not include any explanation, markdown, or text "
            "outside the JSON array.\n\n"
            "Rules you must follow:\n"
            "1. Only recommend estimators from the permitted_models list in the complexity budget.\n"
            "2. Never recommend an estimator that appears in failed_estimators with failure_count > 1 "
            "   unless all permitted alternatives have also failed.\n"
            "3. If the series has a structural break (break_detected=true), prefer models that handle "
            "   changepoints natively (Prophet) or are robust to level shifts (NaiveForecaster, "
            "   ExponentialSmoothing) over ARIMA-family models.\n"
            "4. If the series is non-stationary (is_stationary=false), prefer models that do not "
            "   require stationarity (Prophet, ExponentialSmoothing, NaiveForecaster) unless "
            "   AutoARIMA is permitted and no structural break is present.\n"
            "5. If seasonality is strong (seasonality_class=strong), prefer models that model "
            "   seasonality explicitly (Prophet, TBATS, ExponentialSmoothing with seasonal_periods, "
            "   AutoARIMA with seasonal=True).\n"
            "6. Always include at least one simple baseline (NaiveForecaster or "
            "   PolynomialTrendForecaster) at the end of the list as a last-resort fallback.\n"
            "7. Return between 2 and 5 estimators."
        )

        # Build a single, self-contained user message with all evidence
        evidence = {
            "dataset_id":        profile.dataset_id,
            "n_observations":    profile.n_observations,
            "narrative":         profile.narrative,
            "stationarity":      profile.stationarity,
            "seasonality":       profile.seasonality,
            "structural_break":  profile.structural_break,
            "complexity_budget": profile.complexity_budget,
            "dataset_history":   profile.dataset_history,
            "mlflow_context":    mlflow_context or {},
        }

        user_message = (
            "Select and rank estimators for the following dataset profile.\n\n"
            f"{json.dumps(evidence, indent=2)}\n\n"
            "Return ONLY a JSON array of estimator class names."
        )

        response = await self._llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text: str = response.content[0].text.strip()

        try:
            candidates = json.loads(raw_text)
            if not isinstance(candidates, list):
                raise ValueError("Expected a JSON array")
            return [str(c) for c in candidates]
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "ModelSelectorAgent._llm_select: failed to parse LLM response for %s: %s\n"
                "Raw response: %s",
                profile.dataset_id, exc, raw_text,
            )
            # Graceful degradation: return the full permitted list in complexity order
            return profile.complexity_budget.get("permitted_models", ["NaiveForecaster"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_profile(self, dataset_id: str) -> DataProfile:
        """Load DataProfile written by PipelineArchitectAgent from Valkey."""
        key = _PROFILE_KEY.format(dataset_id=dataset_id)
        raw = await self.valkey.get(key)
        if not raw:
            raise RuntimeError(
                f"ModelSelectorAgent: no profile found in Valkey for dataset_id={dataset_id}. "
                "Ensure PipelineArchitectAgent ran successfully first."
            )
        data = json.loads(raw)
        return DataProfile(**data)

    def _fetch_mlflow_context(self, dataset_id: str) -> dict[str, Any]:
        """
        Pull any registered model versions for this dataset from MLflow.

        Returns a dict with:
          - registered_versions: list of {version, run_id, metrics, tags}
          - best_metric: the best validation MAE seen so far (or None)
        """
        try:
            versions = self.mlflow.search_model_versions(f"tags.dataset_id='{dataset_id}'")
            parsed = []
            best_mae = None

            for v in versions:
                run = self.mlflow.get_run(v.run_id)
                mae = run.data.metrics.get("val_mae")
                parsed.append({
                    "version":    v.version,
                    "run_id":     v.run_id,
                    "estimator":  run.data.tags.get("estimator", "unknown"),
                    "val_mae":    mae,
                    "status":     v.status,
                })
                if mae is not None and (best_mae is None or mae < best_mae):
                    best_mae = mae

            return {"registered_versions": parsed, "best_mae": best_mae}

        except Exception as exc:
            logger.warning("ModelSelectorAgent: MLflow query failed for %s: %s", dataset_id, exc)
            return {"registered_versions": [], "best_mae": None}