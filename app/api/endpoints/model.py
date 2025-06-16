from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import os
from app.core.database import get_db
from app.services.model_service import ModelService
from app.core import config
from app.schemas.online_learning import ModelListResponse, ModelChangeResponse
from typing import Dict, List
from app.core.config import settings

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


@router.delete("/{model_path:path}")
async def delete_model(
    model_path: str,
) -> Dict[str, str]:
    """저장된 모델을 삭제합니다.

    Args:
        model_path: 삭제할 모델 파일의 경로 (models/ 디렉토리 내의 상대 경로)

    Returns:
        삭제 결과 메시지

    Raises:
        HTTPException: 모델 파일이 존재하지 않거나 삭제 중 오류가 발생한 경우
    """
    try:
        # models 디렉토리 내의 파일만 삭제 가능하도록 경로 검증
        if not model_path.startswith("models/"):
            model_path = f"models/{model_path}"

        # 프로젝트 루트 디렉토리 기준으로 경로 생성
        full_path = os.path.abspath(
            os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                ),
                model_path,
            )
        )

        if not os.path.exists(full_path):
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_path}",
            )

        # 모델 파일 삭제
        os.remove(full_path)

        return {
            "message": f"Model deleted successfully: {model_path}",
            "deleted_path": model_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting model: {str(e)}",
        )
