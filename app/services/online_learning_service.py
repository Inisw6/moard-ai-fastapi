import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import redis
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.ml.online_learning import OnlineLearner
from app.models.orm import Content, Recommendation, UserLog, EventType

redis_client: redis.Redis = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD or None,
    decode_responses=True,
)


def initialize_learner() -> OnlineLearner:
    """OnlineLearner를 초기화합니다.

    로컬 파일 시스템에 사전 학습된 모델이 존재하면 이를 로드하고,
    없으면 새 인스턴스를 생성합니다.

    Returns:
        OnlineLearner: 초기화된 OnlineLearner 인스턴스
    """
    pretrained_path: str = settings.PRETRAINED_MODEL_PATH
    if os.path.exists(pretrained_path):
        print(f"Loading existing model from {pretrained_path}")
        return OnlineLearner(model_path=pretrained_path)
    print("No existing model found. Creating new model.")
    return OnlineLearner(model_path=pretrained_path)


learner: OnlineLearner = initialize_learner()


async def update_task_status(task_id: str, **kwargs: Any) -> None:
    """Redis에 학습 작업 상태를 생성 또는 업데이트합니다.

    Args:
        task_id (str): 작업 식별자(UUID)
        **kwargs: 상태 필드(예: status, processed_interactions 등)
    """
    key: str = f"task:{task_id}"
    try:
        raw: Optional[str] = redis_client.get(key)
        if raw:
            status: Dict[str, Any] = json.loads(raw)
            status.update(kwargs)
        else:
            status = {
                "task_id": task_id,
                "status": "processing",
                "start_time": datetime.now().isoformat(),
                "total_interactions": 0,
                "processed_interactions": 0,
                **kwargs,
            }
        redis_client.set(key, json.dumps(status))
    except Exception as err:
        raise RuntimeError(f"작업 상태 업데이트 실패: {err}") from err


async def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Redis에서 작업 상태를 조회합니다.

    Args:
        task_id (str): 작업 식별자(UUID)

    Returns:
        Optional[Dict[str, Any]]: 상태 정보 또는 None
    """
    key: str = f"task:{task_id}"
    raw: Optional[str] = redis_client.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


async def get_all_tasks() -> List[Dict[str, Any]]:
    """Redis에 저장된 모든 작업 상태를 반환합니다.

    Returns:
        List[Dict[str, Any]]: 모든 작업 상태 리스트
    """
    tasks: List[Dict[str, Any]] = []
    try:
        for key in redis_client.keys("task:*"):
            raw: Optional[str] = redis_client.get(key)
            if raw:
                try:
                    tasks.append(json.loads(raw))
                except Exception:
                    continue
    except Exception as err:
        raise RuntimeError(f"모든 작업 조회 실패: {err}") from err
    return tasks


async def collect_training_data(
    db: Session, batch_size: int = 100
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """데이터베이스에서 추천 기록을 조회하여 학습 데이터를 생성합니다.

    Args:
        db (Session): 데이터베이스 세션
        batch_size (int): 최대 샘플 수

    Returns:
        List[Tuple[Tensor, Tensor, Tensor, Tensor]]: 학습용 튜플 리스트
    """
    recs = (
        db.query(Recommendation)
        .filter(Recommendation.embedding.isnot(None))
        .order_by(Recommendation.recommended_at.desc())
        .limit(batch_size)
        .all()
    )
    data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for rec in recs:
        try:
            user_emb = torch.tensor(json.loads(rec.embedding), dtype=torch.float32)
            for rc in rec.contents:
                content_obj = (
                    db.query(Content).filter(Content.id == rc.content_id).first()
                )
                if not content_obj or not content_obj.embedding:
                    continue
                content_emb = torch.tensor(
                    json.loads(content_obj.embedding), dtype=torch.float32
                )
                action = torch.tensor([[rc.content_id]], dtype=torch.long)
                user_log = (
                    db.query(UserLog)
                    .filter(
                        and_(
                            UserLog.recommendation_id == rec.id,
                            UserLog.content_id == content_obj.id,
                            UserLog.event_type == EventType.CLICK,
                        )
                    )
                    .first()
                )
                reward_val = 1.0 if user_log else 0.0
                reward = torch.tensor([reward_val], dtype=torch.float32)
                data.append((user_emb, content_emb, action, reward))
        except Exception:
            continue
    return data


async def process_training_batch(
    db: Session,
    training_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[int, float]:
    """수집된 학습 데이터를 순회하며 OnlineLearner로 학습을 수행합니다.

    Args:
        db (Session): 데이터베이스 세션(사용되지 않음)
        training_data: collect_training_data 반환 값

    Returns:
        Tuple[int, float]: 처리된 샘플 수 및 평균 손실
    """
    total_loss = 0.0
    count = 0
    for i, (u, c, a, r) in enumerate(training_data):
        try:
            _, loss = learner.process_interaction(
                user_embedding=u,
                content_embedding=c,
                action=a,
                reward=r,
                done=True,
            )
            total_loss += loss
            count += 1
        except Exception:
            continue
    avg_loss = total_loss / count if count else 0.0
    return count, avg_loss


async def process_batch_interaction_task(
    task_id: str,
    batch_size: int = 100,
    save_model: bool = True,
    db: Session = None,
) -> None:
    """비동기 배치 상호작용 학습 작업을 수행합니다.

    Args:
        task_id (str): 작업 식별자(UUID)
        batch_size (int): 처리할 최대 샘플 수
        save_model (bool): 학습 완료 후 모델 저장 여부
        db (Session): 데이터베이스 세션
    """
    try:
        await update_task_status(
            task_id,
            status="processing",
            total_interactions=0,
            processed_interactions=0,
        )
        data = await collect_training_data(db, batch_size)
        if not data:
            await update_task_status(
                task_id,
                status="completed",
                processed_interactions=0,
                total_loss=0.0,
                message="No training data available",
            )
            return
        await update_task_status(task_id, total_interactions=len(data))
        total_loss = 0.0
        results: List[Dict[str, Any]] = []
        processed = 0
        for i, (u, c, a, r) in enumerate(data):
            try:
                qv, loss = learner.process_interaction(
                    user_embedding=u,
                    content_embedding=c,
                    action=a,
                    reward=r,
                    done=True,
                )
                total_loss += loss
                processed += 1
                results.append({"q_value": qv.item(), "loss": loss})
                if processed % 10 == 0:
                    await update_task_status(
                        task_id,
                        processed_interactions=processed,
                        total_loss=total_loss / processed,
                    )
            except Exception:
                continue
        avg_loss = total_loss / processed if processed else 0.0
        save_path: Optional[str] = None
        if save_model:
            save_path = learner.save_model()
        await update_task_status(
            task_id,
            status="completed",
            processed_interactions=processed,
            results=results,
            total_loss=avg_loss,
            save_path=save_path,
            message="Training completed successfully",
        )
    except Exception as err:
        logging.error("Task %s 실패: %s", task_id, err)
        await update_task_status(
            task_id,
            status="failed",
            error=str(err),
            message=f"Training failed: {err}",
        )
