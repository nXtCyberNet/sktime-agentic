from __future__ import annotations

import importlib
import json
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valkey key helpers
# ---------------------------------------------------------------------------
_CANDIDATE_KEY   = "candidates:{dataset_id}"
_MODEL_VER_KEY   = "model_version:{dataset_id}"
_MODEL_VER_TTL   = 86_400          # 24 h
_VALIDATION_FRAC = 0.2             # last 20 % of data used for evaluation
_EARLY_STOP_MAE  = None            # set via settings if desired; None = no early stop


class TrainingAgent:
    """
    Parameters
    ----------
    valkey        : async Valkey/Redis client
    mlflow_client : synchronous MLflow tracking client
    settings      : app.config.Settings
    """

    def __init__(self, valkey, mlflow_client, settings):
        self.valkey   = valkey
        self.mlflow   = mlflow_client
        self.settings = settings

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def handle_retrain_job(self, job) -> str | None:
        dataset_id: str = job.dataset_id if hasattr(job, "dataset_id") else job["dataset_id"]
        reason: str     = getattr(job, "reason", job.get("reason", "scheduled") if isinstance(job, dict) else "scheduled")

        logger.info("TrainingAgent.handle_retrain_job: dataset_id=%s reason=%s", dataset_id, reason)

        # ---- 1. Fetch ranked candidate list ----
        candidates: list[str] = await self._load_candidates(dataset_id)
        if not candidates:
            logger.error("TrainingAgent: no candidates found for %s – aborting", dataset_id)
            return None

        # ---- 2. Load dataset ----
        y_train, y_val = self._load_data(dataset_id)
        if len(y_train) < 5:
            logger.error("TrainingAgent: insufficient training data for %s (%d obs)", dataset_id, len(y_train))
            return None

        # ---- 3. Ensure MLflow experiment exists ----
        experiment_name = f"ts-{dataset_id}"
        try:
            experiment = self.mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.mlflow.create_experiment(experiment_name, tags={"dataset_id": dataset_id})
        except Exception as exc:
            logger.warning("TrainingAgent: MLflow experiment setup warning for %s: %s", dataset_id, exc)

        # ---- 4. Train and evaluate each candidate ----
        results: list[dict[str, Any]] = []

        for estimator_name in candidates:
            result = self._train_one(
                dataset_id=dataset_id,
                estimator_name=estimator_name,
                y_train=y_train,
                y_val=y_val,
                experiment_name=experiment_name,
                reason=reason,
            )
            if result is not None:
                results.append(result)
                # Early-stop: if first-choice model meets target, don't bother training the rest
                early_stop_mae = getattr(self.settings, "early_stop_mae", None)
                if early_stop_mae and result["val_mae"] <= early_stop_mae:
                    logger.info(
                        "TrainingAgent: early stop triggered for %s (mae=%.4f ≤ threshold=%.4f)",
                        estimator_name, result["val_mae"], early_stop_mae,
                    )
                    break

        if not results:
            logger.error("TrainingAgent: all candidates failed for %s", dataset_id)
            return None

        # ---- 5. Pick the best model (lowest val_mae) ----
        best = min(results, key=lambda r: r["val_mae"])
        logger.info(
            "TrainingAgent: best model for %s is %s (val_mae=%.4f)",
            dataset_id, best["estimator_name"], best["val_mae"],
        )

        # ---- 6. Register in MLflow model registry ----
        model_version = self._register_model(dataset_id, best)
        if model_version is None:
            return None

        # ---- 7. Persist new model version to Valkey ----
        key = _MODEL_VER_KEY.format(dataset_id=dataset_id)
        await self.valkey.setex(key, _MODEL_VER_TTL, model_version)
        logger.info("TrainingAgent: promoted model version %s for %s", model_version, dataset_id)

        return model_version

    # ------------------------------------------------------------------
    # Per-estimator training loop
    # ------------------------------------------------------------------

    def _train_one(
        self,
        dataset_id: str,
        estimator_name: str,
        y_train: np.ndarray,
        y_val: np.ndarray,
        experiment_name: str,
        reason: str,
    ) -> dict[str, Any] | None:
        logger.info("TrainingAgent: fitting %s for %s", estimator_name, dataset_id)
        t0 = time.monotonic()

        try:
            estimator = self._instantiate_estimator(estimator_name, len(y_train))
        except (ImportError, ValueError) as exc:
            logger.error("TrainingAgent: cannot instantiate %s: %s", estimator_name, exc)
            return None

        try:
            estimator.fit(y_train)
        except Exception as exc:
            logger.error("TrainingAgent: fit failed for %s on %s: %s", estimator_name, dataset_id, exc)
            return None

        elapsed_fit = time.monotonic() - t0

        # ---- Evaluate ----
        try:
            n_val = len(y_val)
            preds = estimator.predict(np.arange(n_val))
            val_mae  = float(np.mean(np.abs(preds - y_val)))
            val_mape = float(np.mean(np.abs((preds - y_val) / (np.abs(y_val) + 1e-8))))
            val_rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
        except Exception as exc:
            logger.error("TrainingAgent: predict/eval failed for %s on %s: %s", estimator_name, dataset_id, exc)
            return None

        # ---- Log to MLflow ----
        run_id = None
        try:
            with self.mlflow.start_run(experiment_id=self._get_experiment_id(experiment_name)) as run:
                run_id = run.info.run_id
                self.mlflow.log_params({
                    "estimator":   estimator_name,
                    "dataset_id":  dataset_id,
                    "n_train":     len(y_train),
                    "n_val":       n_val,
                    "retrain_reason": reason,
                })
                self.mlflow.log_metrics({
                    "val_mae":        val_mae,
                    "val_mape":       val_mape,
                    "val_rmse":       val_rmse,
                    "fit_seconds":    elapsed_fit,
                })
                self.mlflow.set_tags({
                    "estimator": estimator_name,
                    "dataset_id": dataset_id,
                })
        except Exception as exc:
            logger.warning("TrainingAgent: MLflow logging failed for %s: %s", estimator_name, exc)

        logger.info(
            "TrainingAgent: %s → val_mae=%.4f val_rmse=%.4f fit_seconds=%.1f",
            estimator_name, val_mae, val_rmse, elapsed_fit,
        )

        return {
            "estimator_name": estimator_name,
            "estimator_obj":  estimator,
            "val_mae":        val_mae,
            "val_mape":       val_mape,
            "val_rmse":       val_rmse,
            "fit_seconds":    elapsed_fit,
            "run_id":         run_id,
        }

    # ------------------------------------------------------------------
    # Estimator factory
    # ------------------------------------------------------------------

    def _instantiate_estimator(self, name: str, n_train: int) -> Any:
        # Module paths for each known estimator class name
        _ESTIMATOR_MAP = {
            "NaiveForecaster": (
                "sktime.forecasting.naive", "NaiveForecaster",
                {"strategy": "last"},
            ),
            "PolynomialTrendForecaster": (
                "sktime.forecasting.trend", "PolynomialTrendForecaster",
                {"degree": 1},
            ),
            "ThetaForecaster": (
                "sktime.forecasting.theta", "ThetaForecaster",
                {},
            ),
            "ExponentialSmoothing": (
                "sktime.forecasting.exp_smoothing", "ExponentialSmoothing",
                {"trend": "add", "damped_trend": True},
            ),
            "AutoARIMA": (
                "sktime.forecasting.arima", "AutoARIMA",
                {"sp": 1, "suppress_warnings": True, "error_action": "ignore"},
            ),
            "AutoETS": (
                "sktime.forecasting.ets", "AutoETS",
                {"auto": True, "information_criterion": "aic"},
            ),
            "Prophet": (
                "sktime.forecasting.fbprophet", "Prophet",
                {"seasonality_mode": "additive"},
            ),
            "BATS": (
                "sktime.forecasting.bats", "BATS",
                {"use_box_cox": None, "use_trend": True},
            ),
            "TBATS": (
                "sktime.forecasting.tbats", "TBATS",
                {"use_box_cox": None, "use_trend": True},
            ),
        }

        if name not in _ESTIMATOR_MAP:
            raise ValueError(f"Unknown estimator: {name!r}. Add it to _ESTIMATOR_MAP.")

        module_path, class_name, default_kwargs = _ESTIMATOR_MAP[name]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(**default_kwargs)

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _get_experiment_id(self, experiment_name: str) -> str | None:
        try:
            exp = self.mlflow.get_experiment_by_name(experiment_name)
            return exp.experiment_id if exp else None
        except Exception:
            return None

    def _register_model(self, dataset_id: str, best: dict[str, Any]) -> str | None:
        model_name = f"ts-forecaster-{dataset_id}"
        run_id     = best.get("run_id")

        if not run_id:
            logger.warning("TrainingAgent: no run_id for best model – skipping registry")
            return None

        try:
            model_uri = f"runs:/{run_id}/model"
            mv = self.mlflow.register_model(model_uri, model_name)
            self.mlflow.update_registered_model(
                model_name,
                description=(
                    f"Best forecaster for dataset '{dataset_id}' "
                    f"(estimator={best['estimator_name']}, val_mae={best['val_mae']:.4f})"
                ),
            )
            self.mlflow.set_registered_model_tag(model_name, "dataset_id", dataset_id)
            self.mlflow.set_registered_model_tag(model_name, "estimator", best["estimator_name"])
            return str(mv.version)
        except Exception as exc:
            logger.error("TrainingAgent: MLflow registration failed for %s: %s", dataset_id, exc)
            return None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
        # Reuse settings-injected loader if available
        loader = getattr(self.settings, "data_loader", None)
        if loader is not None:
            y = loader(dataset_id)
        else:
            # Deterministic mock for testing
            rng = np.random.default_rng(seed=int(hash(dataset_id) % 2**31))
            y = rng.standard_normal(200).cumsum()

        split = max(5, int(len(y) * (1 - _VALIDATION_FRAC)))
        return y[:split], y[split:]

    # ------------------------------------------------------------------
    # Candidate loading
    # ------------------------------------------------------------------

    async def _load_candidates(self, dataset_id: str) -> list[str]:
        """Read the ranked candidate list from Valkey."""
        key = _CANDIDATE_KEY.format(dataset_id=dataset_id)
        raw = await self.valkey.get(key)
        if not raw:
            logger.warning("TrainingAgent: no candidates key in Valkey for %s", dataset_id)
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("TrainingAgent: corrupt candidates payload for %s: %s", dataset_id, exc)
            return []