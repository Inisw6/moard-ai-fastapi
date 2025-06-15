from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import os
from app.core.database import get_db
from app.services.model_service import ModelService
from app.core import config
from app.schemas.online_learning import ModelListResponse, ModelChangeResponse

router = APIRouter()


@router.get("/list", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """사용 가능한 모델 목록을 조회합니다.

    Returns:
        ModelListResponse: 모델 목록과 현재 사용 중인 모델 정보
    """
    try:
        models_dir = "models"
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]

        # 현재 사용 중인 모델 정보 가져오기
        model_service = ModelService()
        current_model = (
            os.path.basename(config.MODEL_PATH) if model_service.model else None
        )

        return ModelListResponse(
            models=model_files,
            current_model=current_model,
            message="모델 목록 조회 성공",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"모델 목록 조회 중 오류 발생: {str(e)}"
        )


@router.post("/change", response_model=ModelChangeResponse)
async def change_model(
    model_name: str, db: Session = Depends(get_db)
) -> ModelChangeResponse:
    """현재 사용 중인 모델을 변경합니다.

    Args:
        model_name: 변경할 모델 파일명
        db: 데이터베이스 세션

    Returns:
        ModelChangeResponse: 모델 변경 결과
    """
    try:
        # 모델 파일 존재 확인
        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, detail=f"모델 파일을 찾을 수 없습니다: {model_name}"
            )

        # config의 MODEL_PATH 업데이트
        config.MODEL_PATH = model_path

        # 모델 서비스 재로드
        model_service = ModelService()
        model_service._load_model()

        if model_service.model is None:
            raise HTTPException(status_code=500, detail="모델 로드 실패")

        return ModelChangeResponse(model_name=model_name, message="모델 변경 성공")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 변경 중 오류 발생: {str(e)}")
