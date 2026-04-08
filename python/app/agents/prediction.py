"""
PredictionAgent
===============
Serves online forecast requests from the model registered by TrainingAgent.

Key fixes vs original
---------------------
- _run_inference now passes a ForecastingHorizon to model.predict(), not a
  bare np.ndarray. sktime forecasters require ForecastingHorizon.
- asyncio.get_event_loop() replaced with asyncio.get_running_loop() (3.10+).
- Model loading tries mlflow.sklearn first, then mlflow.pyfunc as fallback.
- Prediction counter uses a pipeline() context manager correctly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
from sktime.forecasting.base import ForecastingHorizon

from app.schemas import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)

_MODEL_VER_KEY  = "model_version:{dataset_id}"
_PRED_COUNT_KEY = "pred_count:{dataset_id}"
_PRED_COUNT_TTL = 86_400   # 24 h
_DEFAULT_HORIZON = 10


class PredictionAgent:
    """
    Parameters
    ----------
    valkey        : async Valkey/Redis client
    mlflow_client : synchronous MlflowClient (for registry lookups)
    settings      : app.config.Settings
    """

    def __init__(self, valkey, mlflow_client, settings):
        self.valkey   = valkey
        self.mlflow   = mlflow_client
        self.settings = settings

        # In-process model cache keyed by (dataset_id, model_version).
        # Shared by all requests in the same process to avoid repeated
        # MLflow artifact downloads.
        self._local_cache: dict[tuple[str, str], Any] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def predict(
        self,
        job: ForecastRequest,
        model_version: str | None = None,
        model_cache: dict | None = None,
    ) -> ForecastResponse:
        """
        Produce forecasts for the given ForecastRequest.

        Parameters
        ----------
        job           : ForecastRequest — must have .dataset_id and .fh
        model_version : If provided, skip Valkey lookup (pinned deployment /
                        A/B testing).
        model_cache   : Optional shared cache dict[(dataset_id, version)] → model.
                        Falls back to self._local_cache if not provided.

        Returns
        -------
        ForecastResponse
        """
        dataset_id: str  = job.dataset_id
        fh_values: list[int] = list(getattr(job, "fh", []) or [])
        if not fh_values:
            default_horizon = int(
                getattr(self.settings, "default_horizon", _DEFAULT_HORIZON)
            )
            fh_values = list(range(1, default_horizon + 1))

        cache = model_cache if model_cache is not None else self._local_cache

        # ---- 1. Resolve model version ----
        if model_version is None:
            model_version = await self._resolve_model_version(dataset_id)

        if model_version is None:
            raise RuntimeError(
                f"PredictionAgent: no registered model version found for "
                f"dataset_id={dataset_id}. Run TrainingAgent first."
            )

        # ---- 2. Load model (cache → MLflow) ----
        cache_key = (dataset_id, model_version)
        cache_hit = cache_key in cache
        model     = await self._load_model(dataset_id, model_version, cache)

        # ---- 3. Run inference off the event loop (sktime is not async-safe) ----
        t0 = time.monotonic()
        try:
            predictions = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._run_inference(model, fh_values),
            )
        except Exception as exc:
            logger.error(
                "PredictionAgent: inference failed for %s v%s: %s",
                dataset_id, model_version, exc,
            )
            raise

        elapsed_ms = (time.monotonic() - t0) * 1000

        # ---- 4. Build response ----
        response = ForecastResponse(
            dataset_id    = dataset_id,
            predictions   = predictions,
            prediction_intervals = None,
            model_version = model_version,
            model_class   = type(model).__name__,
            model_status  = "ok",
            drift_score   = None,
            drift_method  = None,
            warning       = None,
            llm_rationale = f"served_in_ms={round(elapsed_ms, 2)}",
            cache_hit     = cache_hit,
            correlation_id= job.correlation_id,
        )

        # ---- 5. Increment prediction counter (fire-and-forget) ----
        asyncio.ensure_future(self._increment_pred_count(dataset_id))

        logger.info(
            "PredictionAgent: served %d-step forecast for %s v%s "
            "in %.1f ms (cache_hit=%s)",
            len(fh_values), dataset_id, model_version, elapsed_ms, cache_hit,
        )
        return response

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, model: Any, fh_values: list[int]) -> list[float]:
        """
        Call model.predict() with a proper ForecastingHorizon.

        sktime requires a ForecastingHorizon (not a raw np.ndarray).
        is_relative=True means "1 step ahead, 2 steps ahead, …" relative to
        the end of the training series.
        """
        fh  = ForecastingHorizon(fh_values, is_relative=True)
        raw = model.predict(fh)

        if hasattr(raw, "values"):
            return [float(v) for v in raw.values]
        return [float(v) for v in raw]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def _load_model(
        self,
        dataset_id: str,
        model_version: str,
        cache: dict,
    ) -> Any:
        """
        Return the fitted model, using the shared cache to avoid repeated
        MLflow artifact downloads.
        """
        cache_key = (dataset_id, model_version)
        if cache_key in cache:
            logger.debug(
                "PredictionAgent: cache hit for %s v%s", dataset_id, model_version
            )
            return cache[cache_key]

        logger.info(
            "PredictionAgent: loading model from MLflow for %s v%s",
            dataset_id, model_version,
        )

        model = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._mlflow_load(dataset_id, model_version),
        )
        cache[cache_key] = model
        return model

    def _mlflow_load(self, dataset_id: str, model_version: str) -> Any:
        """
        Download and return the fitted model artifact from MLflow.

        Tries mlflow.sklearn first (logged by TrainingAgent), then
        mlflow.pyfunc as a generic fallback.
        """
        model_name = f"ts-forecaster-{dataset_id}"
        model_uri  = f"models:/{model_name}/{model_version}"

        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception:
            pass

        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as exc:
            raise RuntimeError(
                f"PredictionAgent: cannot load model {model_name} v{model_version} "
                f"from MLflow: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Version resolution
    # ------------------------------------------------------------------

    async def _resolve_model_version(self, dataset_id: str) -> str | None:
        """
        Resolve the active model version:
        1. Valkey (set by TrainingAgent after promotion — fastest path).
        2. MLflow registry fallback (latest version in any stage).
        """
        # -- Valkey --
        try:
            key = _MODEL_VER_KEY.format(dataset_id=dataset_id)
            raw = await self.valkey.get(key)
            if raw:
                version = raw.decode() if isinstance(raw, bytes) else raw
                logger.debug(
                    "PredictionAgent: resolved version %s for %s from Valkey",
                    version, dataset_id,
                )
                return version
        except Exception as exc:
            logger.warning(
                "PredictionAgent: Valkey version lookup failed for %s: %s",
                dataset_id, exc,
            )

        # -- MLflow fallback --
        try:
            model_name = f"ts-forecaster-{dataset_id}"
            versions   = self.mlflow.get_latest_versions(
                model_name, stages=["Production", "Staging", "None"]
            )
            if versions:
                latest = sorted(
                    versions, key=lambda v: int(v.version), reverse=True
                )[0]
                logger.debug(
                    "PredictionAgent: resolved version %s for %s from MLflow",
                    latest.version, dataset_id,
                )
                return str(latest.version)
        except Exception as exc:
            logger.warning(
                "PredictionAgent: MLflow version lookup failed for %s: %s",
                dataset_id, exc,
            )

        return None

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    async def _increment_pred_count(self, dataset_id: str) -> None:
        """Increment the per-dataset prediction counter in Valkey."""
        key = _PRED_COUNT_KEY.format(dataset_id=dataset_id)
        try:
            async with self.valkey.pipeline(transaction=False) as pipe:
                await pipe.incr(key)
                await pipe.expire(key, _PRED_COUNT_TTL)
                await pipe.execute()
        except Exception as exc:
            logger.debug(
                "PredictionAgent: failed to increment pred_count for %s: %s",
                dataset_id, exc,
            )