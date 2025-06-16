from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelListResponse(BaseModel):
    """
    사용 가능한 모델 목록 및 현재 모델 정보 응답 스키마.

    Attributes:
        models (List[str]): 모델 파일명 리스트
        current_model (Optional[str]): 현재 선택된 모델 파일명
        message (str): 처리 결과 메시지
    """

    models: List[str] = Field(..., description="모델 파일명 리스트")
    current_model: Optional[str] = Field(None, description="현재 선택된 모델 파일명")
    message: str = Field(..., description="처리 결과 메시지")


class ModelChangeResponse(BaseModel):
    """
    모델 변경 결과 응답 스키마.

    Attributes:
        model_name (str): 변경된 모델 파일명
        message (str): 처리 결과 메시지
    """

    model_name: str = Field(..., description="변경된 모델 파일명")
    message: str = Field(..., description="처리 결과 메시지")


class InteractionData(BaseModel):
    """
    단일 사용자-아이템 상호작용 데이터 모델.

    Attributes:
        user_id (str): 사용자 식별자
        item_id (str): 아이템(콘텐츠) 식별자
        action (str): 상호작용 유형
        reward (float): 보상 값
        done (bool): 에피소드 종료 여부
    """

    user_id: str = Field(..., description="사용자 식별자")
    item_id: str = Field(..., description="아이템(콘텐츠) 식별자")
    action: str = Field(..., description="상호작용 유형")
    reward: float = Field(..., description="보상 값")
    done: bool = Field(..., description="에피소드 종료 여부")


class InteractionResponse(BaseModel):
    """
    단일 상호작용 처리 결과 모델.

    Attributes:
        q_value (float): 예측된 Q-값
        loss (float): 계산된 손실 값
    """

    q_value: float = Field(..., description="예측된 Q-값")
    loss: float = Field(..., description="계산된 손실 값")


class BatchInteractionData(BaseModel):
    """
    배치 단위 상호작용 데이터 모델.

    Attributes:
        interactions (List[InteractionData]): 상호작용 데이터 목록
    """

    interactions: List[InteractionData] = Field(..., description="상호작용 데이터 목록")


class BatchInteractionResponse(BaseModel):
    """
    배치 상호작용 처리 결과 모델.

    Attributes:
        results (List[InteractionResponse]): 상호작용 결과 목록
        total_loss (float): 배치 전체 손실 합계
    """

    results: List[InteractionResponse] = Field(..., description="상호작용 결과 목록")
    total_loss: float = Field(..., description="배치 전체 손실 합계")


class TaskStatus(BaseModel):
    """
    비동기 작업 상태 모델.

    Attributes:
        task_id (str): 작업 UUID
        status (str): 작업 상태
        start_time (str): 작업 시작 시각 (ISO 포맷)
        end_time (Optional[str]): 작업 종료 시각 (ISO 포맷)
        total_interactions (int): 총 요청 개수
        processed_interactions (int): 처리된 요청 개수
        results (Optional[List[Dict[str, float]]]): 처리 결과 데이터
        total_loss (Optional[float]): 총 손실
        error (Optional[str]): 오류 메시지
        save_path (Optional[str]): 모델 저장 경로
        message (Optional[str]): 상태 메시지
    """

    task_id: str = Field(..., description="작업 UUID")
    status: str = Field(..., description="작업 상태")
    start_time: str = Field(..., description="작업 시작 시각 (ISO 포맷)")
    end_time: Optional[str] = Field(None, description="작업 종료 시각 (ISO 포맷)")
    total_interactions: int = Field(..., description="총 요청 개수")
    processed_interactions: int = Field(..., description="처리된 요청 개수")
    results: Optional[List[Dict[str, float]]] = Field(
        None, description="처리 결과 데이터(Optional)"
    )
    total_loss: Optional[float] = Field(None, description="총 손실(Optional)")
    error: Optional[str] = Field(None, description="오류 메시지(Optional)")
    save_path: Optional[str] = Field(None, description="모델 저장 경로(Optional)")
    message: Optional[str] = Field(None, description="상태 메시지(Optional)")

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class AsyncBatchResponse(BaseModel):
    """
    비동기 배치 작업 시작 응답 스키마.

    Attributes:
        task_id (str): 생성된 작업 UUID
        status (str): 초기 상태
        message (str): 처리 결과 메시지
    """

    task_id: str = Field(..., description="생성된 작업 UUID")
    status: str = Field(..., description="초기 상태")
    message: str = Field(..., description="처리 결과 메시지")


class TrainingDataResponse(BaseModel):
    """
    학습 데이터 수집 응답 스키마.

    Attributes:
        message (str): 처리 결과 메시지
        data_count (int): 수집된 데이터 샘플 수
    """

    message: str = Field(..., description="처리 결과 메시지")
    data_count: int = Field(..., description="수집된 데이터 샘플 수")


class TrainingResponse(BaseModel):
    """
    모델 학습 완료 응답 스키마.

    Attributes:
        message (str): 처리 결과 메시지
        processed_count (int): 처리된 샘플 수
        total_loss (float): 학습 손실 값
    """

    message: str = Field(..., description="처리 결과 메시지")
    processed_count: int = Field(..., description="처리된 샘플 수")
    total_loss: float = Field(..., description="학습 손실 값")
