# uvicorn main:app --reload
from fastapi import FastAPI
from app.api.router import api_router
from app.services.model_service import get_model_service
from app.services.embedding_service import get_embedding_service

app = FastAPI(
    title="Q-Network Inference API",
    description="사용자 및 콘텐츠 임베딩으로부터 Q-Value를 예측하는 API입니다.",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    """FastAPI 애플리케이션 시작 시 필요한 서비스들을 초기화합니다."""
    get_model_service()
    get_embedding_service()

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """API 루트 경로. 서비스 상태를 확인하는 데 사용됩니다.

    Returns:
        dict: 환영 메시지가 담긴 딕셔너리.
    """
    return {"message": "Welcome to Q-Network Inference API. Visit /docs for details."}