"""
PipelineArchitectAgent
======================
The first agent to run for any dataset. It profiles the dataset using the MCP
client, fetches production history, and writes a DataProfile to Valkey so all
downstream agents (ModelSelectorAgent, TrainingAgent, etc.) share a single
source of truth.

Responsibilities
----------------
1. Call mcp.profile_dataset() to get stationarity, seasonality, structural-break
   and complexity-budget results in one shot.
2. Call mcp.get_dataset_history() to attach production memory (failures, drifts).
3. Optionally call mcp.estimate_training_cost() for each permitted model when the
   caller requests cost-aware pipeline construction.
4. Serialize the assembled DataProfile to Valkey (TTL = 1 h).
5. Return the DataProfile to the caller so the orchestrator can immediately hand
   it off to ModelSelectorAgent without a second Valkey round-trip.

Design constraints
------------------
- This agent must NEVER make a model selection decision. It assembles evidence
  only; ModelSelectorAgent makes the choice.
- profile_dataset() is always called first, giving the LLM (and subsequent agents)
  the full picture in a single call. Individual MCP tools are used only for
  targeted drill-down when the profile alone is ambiguous.
- The agent is idempotent: re-profiling the same dataset overwrites the Valkey key.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.schemas import DataProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valkey key helpers
# ---------------------------------------------------------------------------
_PROFILE_KEY = "profile:{dataset_id}"
_PROFILE_TTL = 3600  # seconds – long enough for a full training run


class PipelineArchitectAgent:
    """
    Parameters
    ----------
    valkey      : async Valkey/Redis client
    mcp_client  : app.mcp.client.MCPClient
    settings    : app.config.Settings
    """

    def __init__(self, valkey, mcp_client, settings):
        self.valkey   = valkey
        self.mcp      = mcp_client
        self.settings = settings

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def construct_pipeline(self, dataset_id: str) -> DataProfile:
        """
        Build the full DataProfile for *dataset_id* and cache it in Valkey.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the time-series dataset.

        Returns
        -------
        DataProfile
            The assembled profile, ready to pass to ModelSelectorAgent.
        """
        logger.info("PipelineArchitectAgent.construct_pipeline: starting for %s", dataset_id)

        # ---- 1. Full diagnostic profile (single MCP call) ----
        freq = self._infer_freq(dataset_id)
        profile_raw: dict[str, Any] = self.mcp.profile_dataset(dataset_id, freq=freq)
        logger.debug("PipelineArchitectAgent: profile_dataset returned for %s: %s", dataset_id, profile_raw.get("narrative"))

        # ---- 2. Production history ----
        history: dict[str, Any] = self.mcp.get_dataset_history(dataset_id)
        logger.debug(
            "PipelineArchitectAgent: history status=%s for %s",
            history.get("status"), dataset_id,
        )

        # ---- 3. Optional: targeted drill-down when profile signals ambiguity ----
        stationarity    = profile_raw["stationarity"]
        structural_break = profile_raw["structural_break"]

        # If both stationarity tests conflict AND there's a borderline break signal,
        # re-run the structural-break test in isolation to get full tool output.
        ambiguous_stationarity = stationarity["conclusion"] in (
            "trend_stationary", "difference_stationary"
        )
        borderline_break = (
            structural_break["break_detected"]
            and structural_break.get("confidence", 0.0) < 0.3
        )
        if ambiguous_stationarity and borderline_break:
            logger.info(
                "PipelineArchitectAgent: running targeted structural-break drill-down for %s",
                dataset_id,
            )
            refined_break = self.mcp.check_structural_break(dataset_id)
            # Only override if the refined result disagrees with the profile result
            if refined_break["break_detected"] != structural_break["break_detected"]:
                profile_raw["structural_break"] = refined_break

        # ---- 4. Training-cost estimates for all permitted models ----
        complexity_budget   = profile_raw["complexity_budget"]
        permitted_models    = complexity_budget.get("permitted_models", [])
        seasonality_period  = profile_raw["seasonality"].get("period") or 1
        training_costs      = self._estimate_costs(dataset_id, permitted_models, seasonality_period)

        # ---- 5. Assemble DataProfile ----
        profile = DataProfile(
            dataset_id       = dataset_id,
            n_observations   = profile_raw["n_observations"],
            narrative        = profile_raw["narrative"],
            stationarity     = profile_raw["stationarity"],
            seasonality      = profile_raw["seasonality"],
            structural_break = profile_raw["structural_break"],
            complexity_budget= profile_raw["complexity_budget"],
            dataset_history  = history,
            training_costs   = training_costs,
        )

        # ---- 6. Persist to Valkey ----
        key = _PROFILE_KEY.format(dataset_id=dataset_id)
        await self.valkey.setex(key, _PROFILE_TTL, profile.model_dump_json())
        logger.info("PipelineArchitectAgent: DataProfile cached at key=%s (TTL=%ds)", key, _PROFILE_TTL)

        return profile

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_freq(self, dataset_id: str) -> str | None:
        """
        Infer the sampling frequency from the dataset_id naming convention.

        Convention: "sales_weekly_store42" → "W"
                    "iot_hourly_sensor7"  → "H"
                    anything else         → None (let MCP auto-detect)
        """
        lowered = dataset_id.lower()
        freq_hints = {
            "hourly":   "H",
            "daily":    "D",
            "weekly":   "W",
            "monthly":  "M",
        }
        for hint, freq in freq_hints.items():
            if hint in lowered:
                return freq
        return None

    def _estimate_costs(
        self,
        dataset_id: str,
        permitted_models: list[str],
        seasonality_period: int,
    ) -> dict[str, dict[str, Any]]:
        """
        Call estimate_training_cost for each permitted model.

        Returns a dict keyed by model class name, e.g.:
        {
            "AutoARIMA": {"estimated_minutes": 1.2, "estimated_cost_usd": 0.00060, ...},
            "Prophet":   {"estimated_minutes": 0.4, "estimated_cost_usd": 0.00020, ...},
        }
        Failures are logged and skipped gracefully.
        """
        costs: dict[str, dict[str, Any]] = {}
        for model_class in permitted_models:
            try:
                result = self.mcp.estimate_training_cost(
                    dataset_id=dataset_id,
                    model_class=model_class,
                    seasonality_period=seasonality_period,
                )
                costs[model_class] = result
            except Exception as exc:
                logger.warning(
                    "PipelineArchitectAgent: cost estimation failed for %s / %s: %s",
                    dataset_id, model_class, exc,
                )
        return costs