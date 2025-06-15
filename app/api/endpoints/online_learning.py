from typing import Dict, Optional, List, Tuple
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
import torch
import json
import redis
from datetime import datetime
import asyncio
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import and_
import os

from app.ml.online_learning import OnlineLearner
from app.core.config import settings
from app.core.database import get_db
from app.models.orm import UserLog, Recommendation, Content, EventType

router = APIRouter()

# Redis 연결
redis_client = redis.Redis(
    host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True
)


# OnlineLearner 초기화
def initialize_learner():
    """OnlineLearner를 초기화합니다.

    Returns:
        OnlineLearner 인스턴스
    """
    # 저장된 모델이 있는지 확인
    if os.path.exists(settings.PRETRAINED_MODEL_PATH):
        print(f"Loading existing model from {settings.PRETRAINED_MODEL_PATH}")
        return OnlineLearner(model_path=settings.PRETRAINED_MODEL_PATH)
    else:
        print("No existing model found. Creating new model.")
        return OnlineLearner(model_path=settings.PRETRAINED_MODEL_PATH)


# 전역 OnlineLearner 인스턴스
learner = initialize_learner()


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


class TrainingDataResponse(BaseModel):
    """학습 데이터 수집 응답을 위한 Pydantic 모델.

    Attributes:
        message: 상태 메시지
        data_count: 수집된 데이터 수
    """

    message: str
    data_count: int


class TrainingResponse(BaseModel):
    """학습 처리 응답을 위한 Pydantic 모델.

    Attributes:
        message: 상태 메시지
        processed_count: 처리된 데이터 수
        total_loss: 전체 손실값
    """

    message: str
    processed_count: int
    total_loss: float


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


async def get_all_tasks() -> List[TaskStatus]:
    """Redis에서 모든 작업 상태를 조회합니다."""
    tasks = []
    for key in redis_client.keys("task:*"):
        try:
            status_data = redis_client.get(key)
            if status_data:
                task = TaskStatus(**json.loads(status_data))
                tasks.append(task)
        except Exception as e:
            print(f"Error parsing task status for {key}: {e}")
            continue
    return tasks


@router.get("/tasks", response_model=List[TaskStatus])
async def list_all_tasks() -> List[TaskStatus]:
    """모든 작업의 상태를 조회합니다.

    Returns:
        작업 상태 목록
    """
    try:
        tasks = await get_all_tasks()
        return tasks
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving tasks: {str(e)}"
        )


async def process_batch_interaction_task(
    task_id: str,
    batch_size: int = 100,
    save_model: bool = True,
    db: Session = Depends(get_db),
):
    """비동기로 배치 상호작용을 처리하는 작업 함수."""
    try:
        # 초기 상태 설정
        await update_task_status(
            task_id,
            status="processing",
            start_time=datetime.now().isoformat(),
            total_interactions=0,
            processed_interactions=0,
        )

        # 학습 데이터 수집
        training_data = await collect_training_data(db, batch_size)

        if not training_data:
            await update_task_status(
                task_id,
                status="completed",
                end_time=datetime.now().isoformat(),
                total_interactions=0,
                processed_interactions=0,
                total_loss=0.0,
                message="No training data available",
            )
            return

        # 진행 상황 업데이트
        await update_task_status(task_id, total_interactions=len(training_data))

        # 배치 처리
        total_loss = 0.0
        processed_count = 0
        results = []

        for i, (user_embedding, content_embedding, action, reward) in enumerate(
            training_data
        ):
            try:
                # 학습 진행
                q_value, loss = learner.process_interaction(
                    user_embedding=user_embedding,
                    content_embedding=content_embedding,
                    action=action,
                    reward=reward,
                    done=True,
                )

                total_loss += loss
                processed_count += 1
                results.append({"q_value": q_value.item(), "loss": loss})

                # 진행 상황 업데이트 (10개 샘플마다)
                if (i + 1) % 10 == 0:
                    await update_task_status(
                        task_id,
                        processed_interactions=i + 1,
                        total_loss=total_loss / (i + 1),
                    )
                    # 딜레이 제거
                    await asyncio.sleep(0)

            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

        # 평균 손실값 계산
        avg_loss = total_loss / processed_count if processed_count > 0 else 0.0

        # 모델 저장
        save_path = None
        if save_model:
            save_path = learner.save_model()

        # 작업 완료 상태 업데이트
        await update_task_status(
            task_id,
            status="completed",
            end_time=datetime.now().isoformat(),
            processed_interactions=processed_count,
            results=results,
            total_loss=avg_loss,
            save_path=save_path,
            message="Training completed successfully",
        )

    except Exception as e:
        await update_task_status(
            task_id,
            status="failed",
            end_time=datetime.now().isoformat(),
            error=str(e),
            message=f"Training failed: {str(e)}",
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


@router.post("/async-train", response_model=AsyncBatchResponse)
async def start_async_training(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    batch_size: int = 100,
    save_model: bool = True,
) -> AsyncBatchResponse:
    """비동기로 모델 학습을 시작합니다.

    Args:
        background_tasks: FastAPI 백그라운드 작업
        db: 데이터베이스 세션
        batch_size: 한 번에 처리할 배치 크기
        save_model: 처리 완료 후 모델 저장 여부

    Returns:
        작업 ID와 상태를 포함한 응답
    """
    try:
        task_id = str(uuid.uuid4())

        # 백그라운드 작업 시작
        background_tasks.add_task(
            process_batch_interaction_task,
            task_id=task_id,
            batch_size=batch_size,
            save_model=save_model,
            db=db,
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


async def collect_training_data(
    db: Session, batch_size: int = 100
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """학습을 위한 데이터를 수집합니다.

    Args:
        db: 데이터베이스 세션
        batch_size: 수집할 데이터의 최대 개수

    Returns:
        학습 데이터 리스트 (user_embedding, content_embedding, action, reward) 튜플의 리스트
    """
    # Recommendation에서 추천 데이터 수집 (가장 최근 데이터부터)
    recommendations = (
        db.query(Recommendation)
        .filter(Recommendation.embedding.isnot(None))  # 임베딩이 있는 추천만
        .order_by(Recommendation.recommended_at.desc())  # 최신순 정렬
        .limit(batch_size)  # 배치 사이즈만큼만 가져오기
        .all()
    )

    training_data = []
    for recommendation in recommendations:
        try:
            # 사용자 임베딩
            user_embedding = torch.tensor(json.loads(recommendation.embedding))

            # 추천된 모든 컨텐츠에 대해 처리
            for rec_content in recommendation.contents:
                content = (
                    db.query(Content)
                    .filter(Content.id == rec_content.content_id)
                    .first()
                )

                if not content or not content.embedding:
                    continue

                # 컨텐츠 임베딩
                content_embedding = torch.tensor(json.loads(content.embedding))

                # 액션 (추천된 컨텐츠의 인덱스)
                action = torch.tensor([[rec_content.content_id]], dtype=torch.long)

                # 해당 컨텐츠에 대한 사용자 로그 확인
                user_log = (
                    db.query(UserLog)
                    .filter(
                        and_(
                            UserLog.recommendation_id == recommendation.id,
                            UserLog.content_id == content.id,
                            UserLog.event_type == EventType.CLICK,
                        )
                    )
                    .first()
                )

                # 보상 계산 (클릭했으면 1.0, 아니면 0.0)
                reward = 1.0 if user_log else 0.0
                reward = torch.tensor([reward])

                training_data.append(
                    (user_embedding, content_embedding, action, reward)
                )

        except Exception as e:
            print(f"Error processing recommendation {recommendation.id}: {str(e)}")
            continue

    return training_data


@router.post("/collect-training-data", response_model=TrainingDataResponse)
async def collect_data(
    batch_size: int = 100, db: Session = Depends(get_db)
) -> TrainingDataResponse:
    """학습 데이터를 수집합니다.

    Args:
        batch_size: 한 번에 처리할 배치 크기
        db: 데이터베이스 세션

    Returns:
        수집된 데이터 수와 상태 메시지
    """
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


async def process_training_batch(
    db: Session,
    training_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[int, float]:
    """배치 데이터를 처리하고 학습합니다.

    Args:
        db: 데이터베이스 세션
        training_data: 학습 데이터 리스트

    Returns:
        처리된 데이터 수와 평균 손실값
    """
    total_loss = 0.0
    processed_count = 0

    print(f"Processing {len(training_data)} training samples...")

    for i, (user_embedding, content_embedding, action, reward) in enumerate(
        training_data
    ):
        try:
            # 학습 진행
            q_value, loss = learner.process_interaction(
                user_embedding=user_embedding,
                content_embedding=content_embedding,
                action=action,
                reward=reward,
                done=True,  # 각 상호작용은 독립적으로 처리
            )

            total_loss += loss
            processed_count += 1

            if (i + 1) % 10 == 0:  # 10개 샘플마다 진행상황 출력
                print(
                    f"Processed {i + 1}/{len(training_data)} samples. Current loss: {loss:.4f}"
                )

        except Exception as e:
            print(f"Error processing training data at index {i}: {str(e)}")
            continue

    avg_loss = total_loss / processed_count if processed_count > 0 else 0.0
    print(
        f"Training completed. Processed {processed_count} samples. Average loss: {avg_loss:.4f}"
    )
    return processed_count, avg_loss


@router.post("/train", response_model=TrainingResponse)
async def train_model(
    batch_size: int = 100, db: Session = Depends(get_db)
) -> TrainingResponse:
    """수집된 데이터로 모델을 학습합니다.

    Args:
        batch_size: 한 번에 처리할 배치 크기
        db: 데이터베이스 세션

    Returns:
        학습 처리 결과
    """
    try:
        print("Starting training process...")

        # 학습 데이터 수집
        print("Collecting training data...")
        training_data = await collect_training_data(db, batch_size)

        if not training_data:
            print("No training data available")
            return TrainingResponse(
                message="No training data available", processed_count=0, total_loss=0.0
            )

        print(f"Collected {len(training_data)} training samples")

        # 배치 처리
        processed_count, total_loss = await process_training_batch(db, training_data)

        # 모델 저장
        save_path = learner.save_model()
        print(f"Model saved at {save_path}")

        return TrainingResponse(
            message=f"Training completed. Model saved at {save_path}",
            processed_count=processed_count,
            total_loss=total_loss,
        )

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")
