import asyncio
import logging

from app.agents.model_selector import ModelSelectorAgent
from app.agents.pipeline_architect import PipelineArchitectAgent
from app.agents.prediction import PredictionAgent
from app.agents.training import TrainingAgent
from app.monitoring.drift_monitor import DriftMonitor
from app.schemas import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, valkey, mlflow_client, mcp_client, settings):
        self.valkey = valkey
        self.mlflow = mlflow_client
        self.mcp = mcp_client
        self.settings = settings
        self.model_cache = {}

        self.pipeline_architect = PipelineArchitectAgent(valkey, mcp_client, settings)
        self.model_selector = ModelSelectorAgent(valkey, mlflow_client, mcp_client, settings)
        self.training_agent = TrainingAgent(valkey, mlflow_client, settings)
        self.prediction_agent = PredictionAgent(valkey, mlflow_client, settings)
        self.drift_monitor = DriftMonitor(valkey, settings)

    async def handle_job(self, job: ForecastRequest) -> ForecastResponse:
        dataset_id = job.dataset_id

        await self._maybe_reload_model(dataset_id)

        model_version = await self._get_cached_model_version(dataset_id)
        if not model_version:
            logger.info("Orchestrator: cold start flow for dataset_id=%s", dataset_id)
            await self.pipeline_architect.construct_pipeline(dataset_id)
            await self.model_selector.select(job)
            model_version = await self.training_agent.handle_retrain_job(
                {"dataset_id": dataset_id, "reason": "cold_start"}
            )

        if not model_version:
            raise RuntimeError(
                f"Orchestrator: unable to resolve/train model version for dataset_id={dataset_id}"
            )

        result = await self.prediction_agent.predict(
            job,
            model_version=model_version,
            model_cache=self.model_cache,
        )

        # Non-blocking drift check; prediction response should not wait on this.
        asyncio.create_task(self.drift_monitor.check(job, result))
        return result

    async def _maybe_reload_model(self, dataset_id: str) -> None:
        signal_key = f"model_updated:{dataset_id}"
        try:
            signal = await self.valkey.get(signal_key)
            if not signal:
                return

            stale_keys = [key for key in self.model_cache if key[0] == dataset_id]
            for key in stale_keys:
                self.model_cache.pop(key, None)

            await self.valkey.delete(signal_key)
            logger.info(
                "Orchestrator: model_updated signal processed for %s (invalidated %d cached entries)",
                dataset_id,
                len(stale_keys),
            )
        except Exception as exc:
            logger.warning("Orchestrator: failed to process model_updated signal for %s: %s", dataset_id, exc)

    async def _get_cached_model_version(self, dataset_id: str) -> str | None:
        key = f"model_version:{dataset_id}"
        try:
            raw = await self.valkey.get(key)
            if raw:
                return raw.decode() if isinstance(raw, bytes) else str(raw)
        except Exception as exc:
            logger.warning("Orchestrator: failed reading model version for %s: %s", dataset_id, exc)
        return None

    async def startup_cleanup(self) -> None:
        patterns = ("model_lock:*", "retrain_lock:*")

        for pattern in patterns:
            try:
                async for key in self.valkey.scan_iter(pattern):
                    ttl = await self.valkey.ttl(key)
                    if ttl is not None and int(ttl) < 0:
                        await self.valkey.delete(key)
                        logger.info("Orchestrator: removed orphaned lock %s", key)
            except Exception as exc:
                logger.warning("Orchestrator: startup cleanup failed for pattern %s: %s", pattern, exc)