from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime


class ModelListResponse(BaseModel):
    """모델 목록 응답 스키마"""

    models: List[str]
    current_model: Optional[str]
    message: str


class ModelChangeResponse(BaseModel):
    """모델 변경 응답 스키마"""

    model_name: str
    message: str
