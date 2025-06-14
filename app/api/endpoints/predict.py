from typing import List
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.predict import InferenceRequest, InferenceResponse
from app.services.model_service import ModelService, get_model_service
from app.schemas.content import ContentResponse
from app.models.orm import User, Content
from sqlalchemy.orm import Session
from app.core.database import get_db
from pydantic import BaseModel
from uuid import UUID
import numpy as np

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
    content_ids: List[int]

@router.post("/top-contents", response_model=TopContentsResponse)
async def get_top_contents(
    request: TopContentsRequest,
    model_service: ModelService = Depends(get_model_service),
    db: Session = Depends(get_db)
):
    """사용자 ID와 콘텐츠 ID 리스트를 받아 상위 6개의 콘텐츠 ID를 반환합니다.

    Args:
        request (TopContentsRequest): 사용자 ID와 콘텐츠 ID 리스트를 포함하는 요청 본문입니다.
        model_service (ModelService): 의존성 주입을 통해 제공되는 모델 서비스 인스턴스입니다.
        db (Session): 데이터베이스 세션입니다.

    Returns:
        TopContentsResponse: 상위 6개의 콘텐츠 ID를 포함하는 응답 객체입니다.

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

        # TODO: 실제 콘텐츠 임베딩 추출 로직
        # content_embeddings = [content.embedding for content in contents]
        
        # 임시: 5차원 콘텐츠 임베딩 생성 (각 콘텐츠마다 랜덤 벡터)
        content_embeddings = [np.random.rand(5).tolist() for _ in range(len(contents))]

        # TODO: 실제 사용자 임베딩 생성 로직
        # user_embedding = [0.0] * len(content_embeddings[0]) if content_embeddings else []
        
        # 임시: 30차원 사용자 임베딩 생성 (랜덤 벡터)
        user_embedding = np.random.rand(30).tolist()

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
            content_ids=[content.id for content, _ in top_pairs]
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )
