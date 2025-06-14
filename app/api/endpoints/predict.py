from typing import List
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.predict import InferenceRequest, InferenceResponse
from app.services.model_service import ModelService, get_model_service
from app.models.orm import User, Content, UserLog
from sqlalchemy.orm import Session
from app.core.database import get_db
from pydantic import BaseModel
from uuid import UUID
import numpy as np
import json
from datetime import datetime, timezone
import pandas as pd

router = APIRouter()


@router.post("/bulk", response_model=InferenceResponse)
async def predict(
    request: InferenceRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """사용자 임베딩과 콘텐츠 임베딩 리스트로부터 Q-Value를 예측합니다.

    Args:
        request (InferenceRequest): `user_embedding`과 `content_embeddings`를 포함하는
            요청 본문입니다.
        model_service (ModelService): 의존성 주입을 통해 제공되는 모델 서비스
            인스턴스입니다.

    Returns:
        InferenceResponse: 각 콘텐츠에 대한 예측 Q-value 리스트를 포함하는
            응답 객체입니다.

    Raises:
        HTTPException: 모델이 로드되지 않았거나 추론 중 오류가 발생한 경우.
    """
    try:
        q_values = model_service.predict(
            user_embedding=request.user_embedding,
            content_embeddings=request.content_embeddings,
        )
        return InferenceResponse(q_values=q_values)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


class TopContentsRequest(BaseModel):
    user_id: UUID
    content_ids: List[int]

    def get_uuid_bytes(self) -> bytes:
        return self.user_id.bytes


class TopContentsResponse(BaseModel):
    """상위 6개의 콘텐츠 ID를 반환하는 응답 모델입니다.

    Attributes:
        content_ids (List[int]): 상위 6개의 콘텐츠 ID 리스트입니다.
        user_embedding (List[float]): 사용자의 임베딩 벡터입니다.
    """

    content_ids: List[int]
    user_embedding: List[float]


def get_user_embedding(user_id: int, db: Session, time_decay_factor: float = 0.9, max_logs: int = 10) -> np.ndarray:
    """사용자의 최근 활동 로그를 기반으로 임베딩을 생성합니다.

    Args:
        user_id: 사용자 ID
        db: 데이터베이스 세션
        time_decay_factor: 시간 감쇠 계수
        max_logs: 사용할 최대 로그 수

    Returns:
        np.ndarray: 사용자 임베딩 벡터
    """
    # 사용자의 최근 로그 조회
    user_logs = (
        db.query(UserLog)
        .filter(UserLog.user_id == user_id)
        .order_by(UserLog.timestamp.desc())
        .limit(max_logs)
        .all()
    )

    if not user_logs:
        return np.zeros(300, dtype=np.float32)

    # 현재 시간 (UTC 기준)
    current_time = datetime.now(timezone.utc)
    
    weighted_embeddings = []
    for log in user_logs:
        # 콘텐츠 임베딩 가져오기
        content = db.query(Content).filter(Content.id == log.content_id).first()
        if not content or not content.embedding:
            continue

        try:
            # JSON 문자열을 파싱하여 float 리스트로 변환
            embedding = json.loads(content.embedding)
            if not isinstance(embedding, list) or not all(
                isinstance(x, (int, float)) for x in embedding
            ):
                continue

            # timestamp가 timezone 정보가 없는 경우 UTC로 가정
            log_time = log.timestamp
            if log_time.tzinfo is None:
                log_time = log_time.replace(tzinfo=timezone.utc)

            # 시간 가중치 계산
            hours_diff = (current_time - log_time).total_seconds() / 3600
            weight = time_decay_factor ** hours_diff

            weighted_embeddings.append(np.array(embedding) * weight)
        except (json.JSONDecodeError, ValueError):
            continue

    if not weighted_embeddings:
        return np.zeros(300, dtype=np.float32)

    # 가중 평균 계산
    weighted_embedding = np.mean(weighted_embeddings, axis=0)
    
    # 길이 맞춰 패딩 또는 자르기
    if len(weighted_embedding) < 300:
        weighted_embedding = np.pad(
            weighted_embedding, (0, 300 - len(weighted_embedding)), "constant"
        )
    elif len(weighted_embedding) > 300:
        weighted_embedding = weighted_embedding[:300]

    return weighted_embedding.astype(np.float32)


@router.post("/top-contents", response_model=TopContentsResponse)
async def get_top_contents(
    request: TopContentsRequest,
    model_service: ModelService = Depends(get_model_service),
    db: Session = Depends(get_db),
):
    """사용자 ID와 콘텐츠 ID 리스트를 받아 상위 6개의 콘텐츠 ID와 사용자 임베딩을 반환합니다.

    Args:
        request (TopContentsRequest): 사용자 ID와 콘텐츠 ID 리스트를 포함하는 요청 본문입니다.
        model_service (ModelService): 의존성 주입을 통해 제공되는 모델 서비스 인스턴스입니다.
        db (Session): 데이터베이스 세션입니다.

    Returns:
        TopContentsResponse: 상위 6개의 콘텐츠 ID와 사용자 임베딩을 포함하는 응답 객체입니다.

    Raises:
        HTTPException: 사용자나 콘텐츠를 찾을 수 없거나, 추론 중 오류가 발생한 경우.
    """
    try:
        # 사용자 조회
        user = db.query(User).filter(User.uuid == request.get_uuid_bytes()).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # 콘텐츠 조회
        contents = db.query(Content).filter(Content.id.in_(request.content_ids)).all()
        if not contents:
            raise HTTPException(status_code=404, detail="No contents found")

        # 콘텐츠 임베딩 추출 및 변환
        content_embeddings = []
        for content in contents:
            try:
                # JSON 문자열을 파싱하여 float 리스트로 변환
                embedding = json.loads(content.embedding)
                if not isinstance(embedding, list) or not all(
                    isinstance(x, (int, float)) for x in embedding
                ):
                    raise ValueError("Invalid embedding format")
                content_embeddings.append(embedding)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing embedding for content {content.id}: {e}")
                # 임베딩 파싱 실패 시 임시 벡터 사용
                content_embeddings.append(np.random.rand(300).tolist())

        # 사용자 임베딩 생성
        user_embedding = get_user_embedding(user.id, db).tolist()

        # Q-value 예측
        q_values = model_service.predict(
            user_embedding=user_embedding,
            content_embeddings=content_embeddings,
        )

        # Q-value와 콘텐츠를 함께 정렬
        content_q_pairs = list(zip(contents, q_values))
        sorted_pairs = sorted(content_q_pairs, key=lambda x: x[1], reverse=True)
        
        # 상위 6개 선택
        top_pairs = sorted_pairs[:6]
        
        return TopContentsResponse(
            content_ids=[content.id for content, _ in top_pairs],
            user_embedding=user_embedding
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )
