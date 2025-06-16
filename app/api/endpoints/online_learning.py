from typing import Dict, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
import asyncio
import uuid
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.online_learning_service import (
    get_task_status,
    get_all_tasks,
    collect_training_data,
    process_training_batch,
    process_batch_interaction_task,
)
from app.schemas.online_learning import (
    TaskStatus,
    AsyncBatchResponse,
    TrainingDataResponse,
    TrainingResponse,
)

router = APIRouter()


@router.get("/tasks", response_model=List[TaskStatus])
async def list_all_tasks() -> List[TaskStatus]:
    """모든 작업의 상태를 조회합니다."""
    try:
        tasks = await get_all_tasks()
        return [TaskStatus(**task) for task in tasks]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving tasks: {str(e)}")


@router.post("/async-train", response_model=AsyncBatchResponse)
async def start_async_training(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    batch_size: int = 100,
    save_model: bool = True,
) -> AsyncBatchResponse:
    """비동기로 모델 학습을 시작합니다."""
    try:
        task_id = str(uuid.uuid4())
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
            task_id=task_id, status="processing", message="Training started"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting async training: {str(e)}"
        )


@router.get("/async-batch-status/{task_id}", response_model=TaskStatus)
async def get_async_batch_status(task_id: str) -> TaskStatus:
    """비동기 배치 처리 작업의 상태를 조회합니다."""
    status = await get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskStatus(**status)


@router.delete("/async-batch-status/{task_id}")
async def delete_async_batch_status(task_id: str) -> Dict[str, str]:
    """비동기 배치 처리 작업의 상태를 삭제합니다."""
    if redis_client.exists(f"task:{task_id}"):
        redis_client.delete(f"task:{task_id}")
        return {"message": f"Task {task_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.post("/collect-training-data", response_model=TrainingDataResponse)
async def collect_data(
    batch_size: int = 100, db: Session = Depends(get_db)
) -> TrainingDataResponse:
    """학습 데이터를 수집합니다."""
    try:
        training_data = await collect_training_data(db, batch_size)
        return TrainingDataResponse(
            message="Training data collected successfully",
            data_count=len(training_data),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error collecting training data: {str(e)}"
        )


@router.post("/train", response_model=TrainingResponse)
async def train_model(
    batch_size: int = 100, db: Session = Depends(get_db)
) -> TrainingResponse:
    """수집된 데이터로 모델을 학습합니다."""
    try:
        print("Starting training process...")
        print("Collecting training data...")
        training_data = await collect_training_data(db, batch_size)

        if not training_data:
            print("No training data available")
            return TrainingResponse(
                message="No training data available", processed_count=0, total_loss=0.0
            )

        print(f"Collected {len(training_data)} training samples")
        processed_count, total_loss = await process_training_batch(db, training_data)

        return TrainingResponse(
            message="Training completed successfully",
            processed_count=processed_count,
            total_loss=total_loss,
        )

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")
