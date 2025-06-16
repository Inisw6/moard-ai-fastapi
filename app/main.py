from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typing import AsyncGenerator, Dict

from app.api.router import api_router
from app.services.embedding_service import get_embedding_service
from app.services.model_service import get_model_service


# uvicorn app.main:app --reload
app: FastAPI = FastAPI(
    title="Q-Network Inference API",
    description="사용자 및 콘텐츠 임베딩으로부터 Q-Value를 예측하는 API입니다.",
    version="1.0.0",
)

# CORS 설정: 모든 도메인/헤더/메서드 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    애플리케이션 수명주기(lifespan) 관리용 컨텍스트 매니저입니다.

    FastAPI 앱 시작 시 모델 및 임베딩 서비스를 초기화하고,
    종료 시 필요한 정리 작업을 수행할 수 있습니다.

    Args:
        app (FastAPI): FastAPI 애플리케이션 인스턴스.

    Yields:
        None
    """
    # 서비스 초기화
    get_model_service()
    get_embedding_service()
    yield
    # (필요 시 종료 작업 추가)


# API 라우터 등록
app.include_router(api_router, prefix="/api/v1")


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """
    API 루트 엔드포인트입니다.

    서비스 상태 확인용으로, 간단한 환영 메시지를 반환합니다.

    Returns:
        Dict[str, str]: 환영 메시지를 담은 딕셔너리.
    """
    return {"message": "Welcome to Q-Network Inference API. Visit /docs for details."}
