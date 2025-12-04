from pydantic import BaseModel
from typing import Optional, Dict, Any


class SuiteInfo(BaseModel):
    id: str
    name: str
    description: str


class AnalysisRequest(BaseModel):
    duration: float = 10.0
    config: Dict[str, Any] = {}


class AnalysisResult(BaseModel):
    signal_name: str
    status: str
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
