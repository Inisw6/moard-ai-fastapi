import asyncio
from typing import Dict, List
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
import redis
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.schemas.online_learning import (
    AsyncBatchResponse,
    TaskStatus,
    TrainingDataResponse,
    TrainingResponse,
)
from app.services.online_learning_service import (
    collect_training_data,
    get_all_tasks,
    get_task_status,
    process_batch_interaction_task,
    process_training_batch,
)


router: APIRouter = APIRouter()

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    decode_responses=True,
)


@router.get("/tasks", response_model=List[TaskStatus])
async def list_all_tasks() -> List[TaskStatus]:
    """
    모든 작업의 상태를 조회합니다.

    Returns:
        List[TaskStatus]: 등록된 모든 작업의 상태 리스트

    Raises:
        HTTPException: 작업 조회 중 오류 발생 시
    """
    try:
        tasks = await get_all_tasks()
        return [TaskStatus(**task) for task in tasks]
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"작업 조회 중 오류 발생: {err}")


@router.post("/async-train", response_model=AsyncBatchResponse)
async def start_async_training(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    batch_size: int = 100,
    save_model: bool = True,
) -> AsyncBatchResponse:
    """
    비동기로 모델 학습을 시작합니다.

    Args:
        background_tasks (BackgroundTasks): 백그라운드 작업 관리 객체
        db (Session): 데이터베이스 세션
        batch_size (int): 한 번에 처리할 배치 크기
        save_model (bool): 학습 완료 후 모델 저장 여부

    Returns:
        AsyncBatchResponse: 생성된 작업 ID 및 초기 상태 정보

    Raises:
        HTTPException: 작업 시작 중 오류 발생 시
    """
    try:
        task_id: str = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        background_tasks.add_task(
            lambda: loop.create_task(
                process_batch_interaction_task(
                    task_id=task_id,
                    batch_size=batch_size,
                    save_model=save_model,
                    db=db,
                )
            )
        )
        return AsyncBatchResponse(
            task_id=task_id,
            status="processing",
            message="Training started",
        )
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"비동기 학습 시작 중 오류 발생: {err}"
        )


@router.get("/async-batch-status/{task_id}", response_model=TaskStatus)
async def get_async_batch_status(task_id: str) -> TaskStatus:
    """
    비동기 배치 처리 작업의 상태를 조회합니다.

    Args:
        task_id (str): 조회할 작업의 UUID

    Returns:
        TaskStatus: 해당 작업의 상태 정보

    Raises:
        HTTPException: 작업을 찾을 수 없거나 조회 중 오류 발생 시
    """
    try:
        status = await get_task_status(task_id)
        if not status:
            raise HTTPException(
                status_code=404, detail=f"작업을 찾을 수 없습니다: {task_id}"
            )
        return TaskStatus(**status)
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"상태 조회 중 오류 발생: {err}")


@router.delete("/async-batch-status/{task_id}")
async def delete_async_batch_status(task_id: str) -> Dict[str, str]:
    """
    비동기 배치 처리 작업의 상태 정보를 삭제합니다.

    Args:
        task_id (str): 삭제할 작업의 UUID

    Returns:
        Dict[str, str]: 삭제 결과 메시지

    Raises:
        HTTPException: 작업이 존재하지 않거나 삭제 중 오류 발생 시
    """
    key = f"task:{task_id}"
    if redis_client.exists(key):
        redis_client.delete(key)
        return {"message": f"작업이 삭제되었습니다: {task_id}"}
    raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {task_id}")


@router.post("/collect-training-data", response_model=TrainingDataResponse)
async def collect_data(
    batch_size: int = 100,
    db: Session = Depends(get_db),
) -> TrainingDataResponse:
    """
    학습에 사용할 데이터를 수집합니다.

    Args:
        batch_size (int): 한 번에 처리할 배치 크기
        db (Session): 데이터베이스 세션

    Returns:
        TrainingDataResponse: 수집된 데이터 개수 및 메시지

    Raises:
        HTTPException: 데이터 수집 중 오류 발생 시
    """
    try:
        training_data = await collect_training_data(db, batch_size)
        return TrainingDataResponse(
            message="학습 데이터 수집 완료",
            data_count=len(training_data),
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"데이터 수집 중 오류 발생: {err}")


@router.post("/train", response_model=TrainingResponse)
async def train_model(
    batch_size: int = 100,
    db: Session = Depends(get_db),
) -> TrainingResponse:
    """
    수집된 데이터를 이용해 모델을 학습합니다.

    Args:
        batch_size (int): 한 번에 처리할 배치 크기
        db (Session): 데이터베이스 세션

    Returns:
        TrainingResponse: 학습 결과(처리된 샘플 수 및 손실 값)

    Raises:
        HTTPException: 학습 중 오류 발생 시
    """
    try:
        training_data = await collect_training_data(db, batch_size)
        if not training_data:
            return TrainingResponse(
                message="학습 데이터가 없습니다",
                processed_count=0,
                total_loss=0.0,
            )
        processed_count, total_loss = await process_training_batch(db, training_data)
        return TrainingResponse(
            message="학습 완료",
            processed_count=processed_count,
            total_loss=total_loss,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"학습 중 오류 발생: {err}")
