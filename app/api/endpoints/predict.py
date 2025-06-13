from fastapi import APIRouter, Depends, HTTPException
from app.schemas.predict import InferenceRequest, InferenceResponse
from app.services.model_service import ModelService, get_model_service

router = APIRouter()


@router.post("/", response_model=InferenceResponse)
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
