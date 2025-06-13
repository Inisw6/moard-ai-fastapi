from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
import json
import redis
from datetime import datetime
import asyncio
import uuid

from app.ml.online_learning import OnlineLearner
from app.core.config import settings

router = APIRouter()

# Redis 연결
redis_client = redis.Redis(
    host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True
)

# 전역 OnlineLearner 인스턴스
learner = OnlineLearner(model_path=settings.PRETRAINED_MODEL_PATH)


class InteractionData(BaseModel):
    """사용자 상호작용 데이터를 위한 Pydantic 모델.

    Attributes:
        user_id: 사용자 ID
        item_id: 아이템 ID
        action: 사용자 행동
        reward: 보상값
        done: 에피소드 종료 여부
    """

    user_id: str
    item_id: str
    action: str
    reward: float
    done: bool


class InteractionResponse(BaseModel):
    """상호작용 처리 결과를 위한 Pydantic 모델.

    Attributes:
        q_value: 예측된 Q-value
        loss: 학습 손실값
    """

    q_value: float
    loss: float


class BatchInteractionData(BaseModel):
    """배치 상호작용 데이터를 위한 Pydantic 모델.

    Attributes:
        interactions: 상호작용 데이터 리스트
    """

    interactions: List[InteractionData]


class BatchInteractionResponse(BaseModel):
    """배치 상호작용 처리 결과를 위한 Pydantic 모델.

    Attributes:
        results: 각 상호작용의 처리 결과 리스트
        total_loss: 전체 배치의 평균 손실값
    """

    results: List[InteractionResponse]
    total_loss: float


class TaskStatus(BaseModel):
    """작업 상태를 위한 Pydantic 모델.

    Attributes:
        task_id: 작업 ID
        status: 작업 상태 (processing, completed, failed)
        start_time: 시작 시간
        end_time: 종료 시간
        total_interactions: 전체 상호작용 수
        processed_interactions: 처리된 상호작용 수
        results: 처리 결과
        total_loss: 전체 손실값
        error: 오류 메시지
        save_path: 저장된 모델 경로
    """

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


class AsyncBatchResponse(BaseModel):
    """비동기 배치 처리 응답을 위한 Pydantic 모델.

    Attributes:
        task_id: 작업 ID
        status: 작업 상태
        message: 상태 메시지
    """

    task_id: str
    status: str
    message: str


async def update_task_status(task_id: str, **kwargs):
    """Redis에 작업 상태를 업데이트합니다."""
    current_status = redis_client.get(f"task:{task_id}")
    if current_status:
        status_dict = json.loads(current_status)
        status_dict.update(kwargs)
        redis_client.set(f"task:{task_id}", json.dumps(status_dict))
    else:
        status_dict = {
            "task_id": task_id,
            "status": "processing",
            "start_time": datetime.now().isoformat(),
            "total_interactions": 0,
            "processed_interactions": 0,
            **kwargs,
        }
        redis_client.set(f"task:{task_id}", json.dumps(status_dict))


async def get_task_status(task_id: str) -> Optional[TaskStatus]:
    """Redis에서 작업 상태를 조회합니다."""
    status_data = redis_client.get(f"task:{task_id}")
    if status_data:
        try:
            return TaskStatus(**json.loads(status_data))
        except Exception as e:
            print(f"Error parsing task status: {e}")
            return None
    return None


async def process_batch_interaction_task(
    task_id: str, data: BatchInteractionData, save_model: bool = True
):
    """비동기로 배치 상호작용을 처리하는 작업 함수."""
    try:
        results = []
        total_loss = 0.0

        # 초기 상태 설정
        await update_task_status(
            task_id,
            status="processing",
            start_time=datetime.now().isoformat(),
            total_interactions=len(data.interactions),
            processed_interactions=0,
        )

        for interaction in data.interactions:
            # TODO: 실제 임베딩 생성 로직 구현
            user_embedding = torch.randn(1, 20)
            content_embedding = torch.randn(1, 15)
            action = torch.tensor([[1]], dtype=torch.long)
            reward = torch.tensor([interaction.reward])

            # 상호작용 처리
            q_value, loss = learner.process_interaction(
                user_embedding=user_embedding,
                content_embedding=content_embedding,
                action=action,
                reward=reward,
                done=interaction.done,
            )

            results.append({"q_value": q_value.item(), "loss": loss})
            total_loss += loss

            # 진행 상황 업데이트
            await update_task_status(task_id, processed_interactions=len(results))
            await asyncio.sleep(0)  # 다른 작업에 CPU 시간 양보

        # 평균 손실값 계산
        avg_loss = total_loss / len(data.interactions) if data.interactions else 0.0

        # 모델 저장
        save_path = None
        if save_model:
            save_path = learner.save_model()

        # 작업 완료 상태 업데이트
        await update_task_status(
            task_id,
            status="completed",
            end_time=datetime.now().isoformat(),
            results=results,
            total_loss=avg_loss,
            save_path=save_path,
        )

    except Exception as e:
        await update_task_status(
            task_id, status="failed", end_time=datetime.now().isoformat(), error=str(e)
        )


@router.post("/interaction", response_model=InteractionResponse)
async def process_interaction(
    data: InteractionData,
) -> InteractionResponse:
    """단일 사용자 상호작용을 처리하고 Q-value를 예측합니다.

    Args:
        data: 사용자 상호작용 데이터

    Returns:
        예측된 Q-value와 손실값을 포함한 응답

    Raises:
        HTTPException: 처리 중 오류가 발생한 경우
    """
    try:
        # TODO: 실제 임베딩 생성 로직 구현
        # 현재는 예시로 랜덤 텐서 사용
        user_embedding = torch.randn(1, 20)  # [batch_size, user_dim]
        content_embedding = torch.randn(1, 15)  # [batch_size, content_dim]
        action = torch.tensor([[1]], dtype=torch.long)  # [[action_index]]
        reward = torch.tensor([data.reward])

        # 상호작용 처리
        q_value, loss = learner.process_interaction(
            user_embedding=user_embedding,
            content_embedding=content_embedding,
            action=action,
            reward=reward,
            done=data.done,
        )

        return InteractionResponse(
            q_value=q_value.item(),
            loss=loss,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing interaction: {str(e)}",
        )


@router.post("/batch-interaction", response_model=BatchInteractionResponse)
async def process_batch_interaction(
    data: BatchInteractionData,
) -> BatchInteractionResponse:
    """여러 사용자 상호작용을 배치로 처리하고 Q-value를 예측합니다.

    Args:
        data: 배치 상호작용 데이터

    Returns:
        각 상호작용의 처리 결과와 전체 평균 손실값을 포함한 응답

    Raises:
        HTTPException: 처리 중 오류가 발생한 경우
    """
    try:
        results = []
        total_loss = 0.0

        for interaction in data.interactions:
            # TODO: 실제 임베딩 생성 로직 구현
            # 현재는 예시로 랜덤 텐서 사용
            user_embedding = torch.randn(1, 20)  # [batch_size, user_dim]
            content_embedding = torch.randn(1, 15)  # [batch_size, content_dim]
            action = torch.tensor([[1]], dtype=torch.long)  # [[action_index]]
            reward = torch.tensor([interaction.reward])

            # 상호작용 처리
            q_value, loss = learner.process_interaction(
                user_embedding=user_embedding,
                content_embedding=content_embedding,
                action=action,
                reward=reward,
                done=interaction.done,
            )

            results.append(
                InteractionResponse(
                    q_value=q_value.item(),
                    loss=loss,
                )
            )
            total_loss += loss

        # 평균 손실값 계산
        avg_loss = total_loss / len(data.interactions) if data.interactions else 0.0

        return BatchInteractionResponse(
            results=results,
            total_loss=avg_loss,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch interaction: {str(e)}",
        )


@router.post("/save")
async def save_model(
    filename: Optional[str] = None,
) -> Dict[str, str]:
    """현재 모델 상태를 저장합니다.

    Args:
        filename: 저장할 파일명 (None이면 자동 생성)

    Returns:
        저장된 모델 파일의 경로를 포함한 응답

    Raises:
        HTTPException: 저장 중 오류가 발생한 경우
    """
    try:
        save_path = learner.save_model(filename=filename)
        return {"message": "Model saved successfully", "path": save_path}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving model: {str(e)}",
        )


@router.post("/async-batch-interaction", response_model=AsyncBatchResponse)
async def process_async_batch_interaction(
    data: BatchInteractionData,
    background_tasks: BackgroundTasks,
    save_model: bool = True,
) -> AsyncBatchResponse:
    """여러 사용자 상호작용을 비동기로 처리하고 Q-value를 예측합니다.

    Args:
        data: 배치 상호작용 데이터
        background_tasks: FastAPI 백그라운드 작업
        save_model: 처리 완료 후 모델 저장 여부

    Returns:
        작업 ID와 상태를 포함한 응답

    Raises:
        HTTPException: 처리 중 오류가 발생한 경우
    """
    try:
        task_id = str(uuid.uuid4())

        # 백그라운드 작업 시작
        background_tasks.add_task(
            process_batch_interaction_task,
            task_id=task_id,
            data=data,
            save_model=save_model,
        )

        return AsyncBatchResponse(
            task_id=task_id, status="processing", message="Batch processing started"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting async batch processing: {str(e)}",
        )


@router.get("/async-batch-status/{task_id}", response_model=TaskStatus)
async def get_async_batch_status(task_id: str) -> TaskStatus:
    """비동기 배치 처리 작업의 상태를 조회합니다.

    Args:
        task_id: 작업 ID

    Returns:
        작업 상태 정보

    Raises:
        HTTPException: 작업을 찾을 수 없는 경우
    """
    status = await get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return status


@router.delete("/async-batch-status/{task_id}")
async def delete_async_batch_status(task_id: str) -> Dict[str, str]:
    """비동기 배치 처리 작업의 상태를 삭제합니다.

    Args:
        task_id: 작업 ID

    Returns:
        삭제 결과 메시지

    Raises:
        HTTPException: 작업을 찾을 수 없는 경우
    """
    if redis_client.exists(f"task:{task_id}"):
        redis_client.delete(f"task:{task_id}")
        return {"message": f"Task {task_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
