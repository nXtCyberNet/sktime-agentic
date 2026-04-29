   python python/scripts/run_demo.py --dataset_id airline --valkey_url redis://localhost:6379
C:\Users\cyber\oss\sktime-forge\.venv\Lib\site-packages\mlflow\pyfunc\utils\data_validation.py:187: UserWarning: Add type hints to the `predict` method to enable data validation and automatic signature inference during model logging. Check https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel for more details.
  color_warning(
Using Valkey URL: redis://localhost:6379
Using MLflow tracking URI: http://localhost:5000
Running demo for dataset: airline
Forecast horizon: [1, 2, 3, 4, 5, 6]
C:\Users\cyber\oss\sktime-forge\python\app\orchestrator.py:425: FutureWarning: ``mlflow.tracking.client.MlflowClient.get_latest_versions`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages
  versions = self.mlflow.get_latest_versions(
WARNING:app.orchestrator:Orchestrator: failed MLflow version fallback for airline: RESOURCE_DOES_NOT_EXIST: Registered Model with name=ts-forecaster-airline not found
INFO:app.orchestrator:Orchestrator: cold start flow for dataset_id=airline
INFO:app.agents.pipeline_architect:PipelineArchitectAgent.construct_pipeline: starting for airline
INFO:app.agents.pipeline_architect:PipelineArchitectAgent: DataProfile cached at key=profile:airline (TTL=3600s)
INFO:app.agents.model_selector:ModelSelectorAgent.select: starting for dataset_id=airline
INFO:httpx:HTTP Request: POST https://ai.hackclub.com/proxy/v1/chat/completions "HTTP/1.1 200 OK"
INFO:app.agents.training:TrainingAgent.handle_retrain_job: dataset_id=airline reason=cold_start
INFO:app.agents.training:TrainingAgent: fitting Prophet for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate Prophet: Prophet requires package 'prophet' to be present in the python environment, but 'prophet' was not found. 'prophet' is a dependency of Prophet and required to construct it. To install the requirement 'prophet', please run: `pip install prophet` 
INFO:app.agents.training:TrainingAgent: fitting TBATS for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate TBATS: TBATS requires package 'tbats' to be present in the python environment, but 'tbats' was not found. 'tbats' is a dependency of TBATS and required to construct it. To install the requirement 'tbats', please run: `pip install tbats` 
INFO:app.agents.training:TrainingAgent: fitting BATS for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate BATS: BATS requires package 'tbats' to be present in the python environment, but 'tbats' was not found. 'tbats' is a dependency of BATS and required to construct it. To install the requirement 'tbats', please run: `pip install tbats` 
INFO:app.agents.training:TrainingAgent: fitting AutoETS for airline
WARNING:app.agents.training:TrainingAgent: failed to load profile for airline: Task <Task pending name='Task-11' coro=<Redis.execute_command() running at C:\Users\cyber\oss\sktime-forge\.venv\Lib\site-packages\redis\asyncio\client.py:781> cb=[_run_until_complete_cb() at C:\Users\cyber\AppData\Local\Programs\Python\Python313\Lib\asyncio\base_events.py:181]> got Future <Future pending> attached to a different loop
2026/04/29 11:46:46 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:46:46 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged AutoETS as sklearn pipeline
🏃 View run thoughtful-sow-779 at: http://localhost:5000/#/experiments/1/runs/b641e32f86f24fd09ddd36ebf870f590
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: AutoETS → val_mae=25.1520 val_rmse=29.7411 fit_seconds=2.1
INFO:app.agents.training:TrainingAgent: fitting AutoARIMA for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate AutoARIMA: AutoARIMA requires package 'pmdarima' to be present in the python environment, but 'pmdarima' was not found. 'pmdarima' is a dependency of AutoARIMA and required to construct it. To install the requirement 'pmdarima', please run: `pip install pmdarima` 
INFO:app.agents.training:TrainingAgent: fitting ExponentialSmoothing for airline
2026/04/29 11:46:54 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:46:54 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged ExponentialSmoothing as sklearn pipeline
🏃 View run monumental-mare-275 at: http://localhost:5000/#/experiments/1/runs/1d500b73f61643bb93747a900e424fec
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: ExponentialSmoothing → val_mae=43.4961 val_rmse=51.9158 fit_seconds=0.2
INFO:app.agents.training:TrainingAgent: fitting ThetaForecaster for airline
WARNING:app.agents.training:TrainingAgent: failed to load profile for airline: Event loop is closed
2026/04/29 11:46:58 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:46:59 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged ThetaForecaster as sklearn pipeline
🏃 View run colorful-horse-931 at: http://localhost:5000/#/experiments/1/runs/6e63ee0ae5a047068d09b605945da47a
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: ThetaForecaster → val_mae=91.3702 val_rmse=102.4195 fit_seconds=0.0
INFO:app.agents.training:TrainingAgent: fitting PolynomialTrendForecaster for airline
2026/04/29 11:47:03 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:47:04 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged PolynomialTrendForecaster as sklearn pipeline
🏃 View run brawny-whale-483 at: http://localhost:5000/#/experiments/1/runs/6f47b1cbd2694856a624354d1be5a21b
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: PolynomialTrendForecaster → val_mae=34.5551 val_rmse=48.1882 fit_seconds=0.0
INFO:app.agents.training:TrainingAgent: fitting NaiveForecaster for airline
WARNING:app.agents.training:TrainingAgent: failed to load profile for airline: Event loop is closed
2026/04/29 11:47:12 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:47:12 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged NaiveForecaster as sklearn pipeline
🏃 View run skillful-shoat-878 at: http://localhost:5000/#/experiments/1/runs/d008500d2e704357bfec096232199f71
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: NaiveForecaster → val_mae=81.4483 val_rmse=93.1339 fit_seconds=0.0
INFO:app.agents.training:TrainingAgent: best model for airline is AutoETS (val_mae=25.1520)
Successfully registered model 'ts-forecaster-airline'.
2026/04/29 11:47:16 WARNING mlflow.tracking._model_registry.fluent: Run with id b641e32f86f24fd09ddd36ebf870f590 has no artifacts at artifact path 'model', registering model based on models:/m-cd05f96f29a747ac82060574a5d21c51 instead
2026/04/29 11:47:17 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: ts-forecaster-airline, version 1
Created version '1' of model 'ts-forecaster-airline'.
INFO:app.agents.training:TrainingAgent: promoted model version 1 for airline
INFO:app.agents.prediction:PredictionAgent: loading model from MLflow for airline v1
INFO:app.agents.watchdog:Watchdog: starting post-promotion monitoring for airline v1 (baseline_mae=25.1520, ttl=3600s)
INFO:app.agents.prediction:PredictionAgent: served 6-step forecast for airline v1 in 25.8 ms (cache_hit=False)
Forecast result:
{
  "dataset_id": "airline",
  "predictions": [
    483.7564244857241,
    429.041125509465,
    373.51584265216655,
    326.0116381435012,
    370.04631685386715,
    375.1702170511556
  ],
  "prediction_intervals": {
    "lower": [
      457.3183478997306,
      399.3248085996499,
      344.0507669451156,
      293.40917665636346,
      330.63026342005463,
      333.3896399070918
    ],
    "upper": [
      510.6893239968434,
      458.5687141934848,
      405.01485118160423,
      357.596458855407,
      411.4126654856508,
      417.6872264454395
    ]
  },
  "model_version": "1",
  "model_class": "TransformedTargetForecaster",
  "model_status": "ok",
  "drift_score": null,
  "drift_method": null,
  "warning": null,
  "llm_rationale": "Forecast generated for dataset airline using TransformedTargetForecaster (version 1) over 6 horizon steps. First predictions: 483.756, 429.041, 373.516. Prediction intervals are included to show forecast uncertainty. No active drift signal is attached to this response.",
  "cache_hit": false,
  "correlation_id": "demo-run"
}
C:\Users\cyber\oss\sktime-forge\python\scripts\run_demo.py:160: DeprecationWarning: Call to deprecated close. (Use aclose() instead) -- Deprecated since version 5.0.1.
  await valkey.close()

