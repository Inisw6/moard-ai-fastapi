from typing import Dict, Optional, List, Tuple
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
from app.models.orm import UserLog, Recommendation, Content, EventType

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
    if os.path.exists(settings.PRETRAINED_MODEL_PATH):
        print(f"Loading existing model from {settings.PRETRAINED_MODEL_PATH}")
        return OnlineLearner(model_path=settings.PRETRAINED_MODEL_PATH)
    else:
        print("No existing model found. Creating new model.")
        return OnlineLearner(model_path=settings.PRETRAINED_MODEL_PATH)

# 전역 OnlineLearner 인스턴스
learner = initialize_learner()

async def update_task_status(task_id: str, **kwargs):
    """Redis에 작업 상태를 업데이트합니다."""
    try:
        key = f"task:{task_id}"
        current_status = redis_client.get(key)

        if current_status:
            status_dict = json.loads(current_status)
            status_dict.update(kwargs)
            new_status = json.dumps(status_dict)
            redis_client.set(key, new_status)
        else:
            status_dict = {
                "task_id": task_id,
                "status": "processing",
                "start_time": datetime.now().isoformat(),
                "total_interactions": 0,
                "processed_interactions": 0,
                **kwargs,
            }
            new_status = json.dumps(status_dict)
            redis_client.set(key, new_status)

    except Exception as e:
        raise e

async def get_task_status(task_id: str) -> Optional[Dict]:
    """Redis에서 작업 상태를 조회합니다."""
    status_data = redis_client.get(f"task:{task_id}")
    if status_data:
        try:
            return json.loads(status_data)
        except Exception as e:
            print(f"Error parsing task status: {e}")
            return None
    return None

async def get_all_tasks() -> List[Dict]:
    """Redis에서 모든 작업 상태를 조회합니다."""
    tasks = []
    try:
        keys = redis_client.keys("task:*")
        for key in keys:
            try:
                status_data = redis_client.get(key)
                if status_data:
                    task = json.loads(status_data)
                    tasks.append(task)
            except Exception as e:
                continue
    except Exception as e:
        raise e
    return tasks

async def collect_training_data(
    db: Session, batch_size: int = 100
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """학습을 위한 데이터를 수집합니다."""
    recommendations = (
        db.query(Recommendation)
        .filter(Recommendation.embedding.isnot(None))
        .order_by(Recommendation.recommended_at.desc())
        .limit(batch_size)
        .all()
    )

    training_data = []
    for recommendation in recommendations:
        try:
            user_embedding = torch.tensor(json.loads(recommendation.embedding))

            for rec_content in recommendation.contents:
                content = (
                    db.query(Content)
                    .filter(Content.id == rec_content.content_id)
                    .first()
                )

                if not content or not content.embedding:
                    continue

                content_embedding = torch.tensor(json.loads(content.embedding))
                action = torch.tensor([[rec_content.content_id]], dtype=torch.long)

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

                reward = 1.0 if user_log else 0.0
                reward = torch.tensor([reward])

                training_data.append(
                    (user_embedding, content_embedding, action, reward)
                )

        except Exception as e:
            print(f"Error processing recommendation {recommendation.id}: {str(e)}")
            continue

    return training_data

async def process_training_batch(
    db: Session,
    training_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[int, float]:
    """배치 데이터를 처리하고 학습합니다."""
    total_loss = 0.0
    processed_count = 0

    print(f"Processing {len(training_data)} training samples...")

    for i, (user_embedding, content_embedding, action, reward) in enumerate(
        training_data
    ):
        try:
            q_value, loss = learner.process_interaction(
                user_embedding=user_embedding,
                content_embedding=content_embedding,
                action=action,
                reward=reward,
                done=True,
            )

            total_loss += loss
            processed_count += 1

            if (i + 1) % 10 == 0:
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

async def process_batch_interaction_task(
    task_id: str,
    batch_size: int = 100,
    save_model: bool = True,
    db: Session = None,
):
    """비동기로 배치 상호작용을 처리하는 작업 함수."""
    try:
        await update_task_status(
            task_id,
            status="processing",
            start_time=datetime.now().isoformat(),
            total_interactions=0,
            processed_interactions=0,
        )

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

        await update_task_status(task_id, total_interactions=len(training_data))

        total_loss = 0.0
        processed_count = 0
        results = []

        for i, (user_embedding, content_embedding, action, reward) in enumerate(
            training_data
        ):
            try:
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

                if (i + 1) % 10 == 0:
                    await update_task_status(
                        task_id,
                        processed_interactions=i + 1,
                        total_loss=total_loss / (i + 1),
                    )

            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

        avg_loss = total_loss / processed_count if processed_count > 0 else 0.0

        save_path = None
        if save_model:
            save_path = learner.save_model()

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
        print(f"Task {task_id}: Failed with error: {str(e)}")
        await update_task_status(
            task_id,
            status="failed",
            end_time=datetime.now().isoformat(),
            error=str(e),
            message=f"Training failed: {str(e)}",
        ) 