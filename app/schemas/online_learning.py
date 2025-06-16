from typing import Dict, Optional, List
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


class InteractionData(BaseModel):
    """사용자 상호작용 데이터를 위한 Pydantic 모델."""

    user_id: str
    item_id: str
    action: str
    reward: float
    done: bool


class InteractionResponse(BaseModel):
    """상호작용 처리 결과를 위한 Pydantic 모델."""

    q_value: float
    loss: float


class BatchInteractionData(BaseModel):
    """배치 상호작용 데이터를 위한 Pydantic 모델."""

    interactions: List[InteractionData]


class BatchInteractionResponse(BaseModel):
    """배치 상호작용 처리 결과를 위한 Pydantic 모델."""

    results: List[InteractionResponse]
    total_loss: float


class TaskStatus(BaseModel):
    """작업 상태를 위한 Pydantic 모델."""

    task_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    total_interactions: int
    processed_interactions: int
    results: Optional[List[Dict[str, float]]] = None
    total_loss: Optional[float] = None
    error: Optional[str] = None
    save_path: Optional[str] = None
    message: Optional[str] = None

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class AsyncBatchResponse(BaseModel):
    """비동기 배치 처리 응답을 위한 Pydantic 모델."""

    task_id: str
    status: str
    message: str


class TrainingDataResponse(BaseModel):
    """학습 데이터 수집 응답을 위한 Pydantic 모델."""

    message: str
    data_count: int


class TrainingResponse(BaseModel):
    """학습 처리 응답을 위한 Pydantic 모델."""

    message: str
    processed_count: int
    total_loss: float
