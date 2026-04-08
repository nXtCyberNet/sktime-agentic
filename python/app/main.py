from fastapi import FastAPI
from fastapi import HTTPException
import mlflow
from mlflow.tracking import MlflowClient
import redis.asyncio as redis

from app.config import Settings
from app.mcp.client import MCPClient
from app.orchestrator import Orchestrator
from app.schemas import ForecastRequest, ForecastResponse

app = FastAPI(title="sktime-agentic")
settings = Settings()


@app.on_event("startup")
async def startup_event() -> None:
    valkey = redis.from_url(settings.valkey_url, decode_responses=False)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    data_loader = getattr(settings, "data_loader", None)
    memory_loader = getattr(settings, "memory_loader", None)
    mcp_client = MCPClient(data_loader=data_loader, memory_loader=memory_loader)

    orchestrator = Orchestrator(valkey, mlflow_client, mcp_client, settings)
    await orchestrator.startup_cleanup()

    app.state.valkey = valkey
    app.state.mlflow_client = mlflow_client
    app.state.mcp_client = mcp_client
    app.state.orchestrator = orchestrator


@app.on_event("shutdown")
async def shutdown_event() -> None:
    valkey = getattr(app.state, "valkey", None)
    if valkey is not None:
        if hasattr(valkey, "aclose"):
            await valkey.aclose()
        else:
            await valkey.close()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest) -> ForecastResponse:
    orchestrator: Orchestrator = app.state.orchestrator
    try:
        return await orchestrator.handle_job(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc