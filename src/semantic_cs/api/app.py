from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from semantic_cs.bootstrap import build_pipeline
from semantic_cs.config import ROOT_DIR
from semantic_cs.evaluation import OfflineEvaluator


class ChatRequestDTO(BaseModel):
    question: str = Field(..., min_length=1)
    user_id: str | None = None


class ChatResponseDTO(BaseModel):
    answer: str
    route: str
    latency_ms: float
    llm_calls: int
    cache_level: str | None = None
    decision: str | None = None
    sources: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Semantic Commerce Customer Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = build_pipeline(preload=True)
frontend_dir = ROOT_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


@app.get("/", response_model=None)
def index():
    if (frontend_dir / "index.html").exists():
        return RedirectResponse("/static/index.html")
    return {"status": "frontend not found"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponseDTO)
def chat(request: ChatRequestDTO) -> dict[str, Any]:
    response = pipeline.ask(request.question)
    return to_jsonable(response)


@app.get("/metrics/eval")
def eval_metrics() -> dict[str, Any]:
    """离线评测指标。

    不传入全局 pipeline：评测使用独立实例并在每轮前重置缓存，
    否则重复调用会因缓存回写而让命中率一轮比一轮高（旧实现不幂等）。
    """
    return to_jsonable(OfflineEvaluator().run())
