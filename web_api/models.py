from typing import Any

from pydantic import BaseModel


class SuiteInfo(BaseModel):
    id: str
    name: str
    description: str


class AnalysisRequest(BaseModel):
    duration: float = 10.0
    config: dict[str, Any] = {}


class AnalysisResult(BaseModel):
    signal_name: str
    status: str
    metrics: dict[str, Any] | None = None
    error: str | None = None
