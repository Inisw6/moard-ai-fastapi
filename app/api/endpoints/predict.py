from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.orm import User
from app.schemas.predict import (
    InferenceRequest,
    InferenceResponse,
    TopContentsRequest,
    TopContentsResponse,
)
from app.services.model_service import get_model_service, ModelService
from app.services.predict_service import PredictService

router: APIRouter = APIRouter()


def get_predict_service(
    model_service: ModelService = Depends(get_model_service),
) -> PredictService:
    """
    PredictService 인스턴스를 생성하여 반환하는 FastAPI 의존성 함수입니다.

    Args:
        model_service (ModelService): 의존성 주입된 모델 서비스 인스턴스.

    Returns:
        PredictService: 예측 로직을 수행하는 서비스 인스턴스.
    """
    return PredictService(model_service)


@router.post("/bulk", response_model=InferenceResponse)
async def predict(
    request: InferenceRequest,
    predict_service: PredictService = Depends(get_predict_service),
) -> InferenceResponse:
    """
    사용자 임베딩과 콘텐츠 임베딩 리스트로부터 Q-Value를 예측합니다.

    Args:
        request (InferenceRequest): user_embedding 및 content_embeddings 정보를 포함.
        predict_service (PredictService): 예측 서비스 인스턴스.

    Returns:
        InferenceResponse: 예측된 Q-value 리스트를 반환.

    Raises:
        HTTPException(500): 예측 중 오류가 발생한 경우.
    """
    try:
        q_values = predict_service.model_service.predict(
            user_embedding=request.user_embedding,
            content_embeddings=request.content_embeddings,
        )
        return InferenceResponse(q_values=q_values)
    except RuntimeError as err:
        raise HTTPException(status_code=500, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"예상치 못한 오류 발생: {err}",
        )


@router.post("/top-contents", response_model=TopContentsResponse)
async def get_top_contents(
    request: TopContentsRequest,
    predict_service: PredictService = Depends(get_predict_service),
    db: Session = Depends(get_db),
) -> TopContentsResponse:
    """
    사용자 ID와 콘텐츠 ID 리스트를 받아 상위 콘텐츠를 조회합니다.

    Args:
        request (TopContentsRequest): user_id 및 content_ids 리스트를 포함.
        predict_service (PredictService): 예측 서비스 인스턴스.
        db (Session): 데이터베이스 세션.

    Returns:
        TopContentsResponse: 상위 콘텐츠 ID 리스트, 사용자 임베딩, 모델명을 반환.

    Raises:
        HTTPException(404): 사용자나 콘텐츠를 찾을 수 없는 경우.
        HTTPException(500): 예측 중 오류가 발생한 경우.
    """
    try:
        # 사용자 조회
        user = db.query(User).filter(User.uuid == request.get_uuid_bytes()).first()
        if not user:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

        # 상위 콘텐츠 예측
        content_ids, user_embedding, model_name = predict_service.get_top_contents(
            user_id=user.id,
            content_ids=request.content_ids,
            db=db,
        )

        return TopContentsResponse(
            content_ids=content_ids,
            user_embedding=user_embedding,
            model_name=model_name,
        )
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except RuntimeError as err:
        raise HTTPException(status_code=500, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"예상치 못한 오류 발생: {err}",
        )
