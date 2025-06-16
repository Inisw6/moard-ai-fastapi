from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.embedding import (
    EmbeddingBulkRequest,
    EmbeddingBulkResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from app.services.embedding_service import EmbeddingService, get_embedding_service


router: APIRouter = APIRouter()


@router.post("/doc2vec", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingResponse:
    """
    Doc2Vec 모델을 사용하여 단일 콘텐츠의 임베딩을 생성합니다.

    Args:
        request (EmbeddingRequest): 'title'과 'description'을 포함한 요청 데이터.
        embedding_service (EmbeddingService): 의존성 주입된 임베딩 서비스 인스턴스.

    Returns:
        EmbeddingResponse: 생성된 임베딩 벡터를 포함한 응답 객체.

    Raises:
        HTTPException(503): 서비스 준비가 안 되었거나 런타임 오류 발생 시.
        HTTPException(500): 기타 예외 발생 시.
    """
    try:
        embedder = embedding_service.get_embedder()
        content_payload: dict = {
            "title": request.title,
            "description": request.description,
        }
        embedding_vector: List[float] = embedder.embed_content(content_payload).tolist()
        return EmbeddingResponse(embedding=embedding_vector)

    except RuntimeError as err:
        # 외부 서비스나 리소스 문제로 임베딩을 생성할 수 없을 때
        raise HTTPException(status_code=503, detail=str(err))
    except Exception as err:
        # 의도하지 않은 예외 처리
        raise HTTPException(
            status_code=500,
            detail=f"임베딩 생성 중 오류 발생: {err}",
        )


@router.post("/doc2vec/bulk", response_model=EmbeddingBulkResponse)
async def create_embeddings_bulk(
    request: EmbeddingBulkRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingBulkResponse:
    """
    Doc2Vec 모델을 사용하여 여러 콘텐츠의 임베딩을 일괄 생성합니다.

    Args:
        request (EmbeddingBulkRequest): 임베딩 대상 콘텐츠 리스트를 포함한 요청 데이터.
        embedding_service (EmbeddingService): 의존성 주입된 임베딩 서비스 인스턴스.

    Returns:
        EmbeddingBulkResponse: 생성된 임베딩 벡터 리스트를 포함한 응답 객체.

    Raises:
        HTTPException(503): 서비스 준비가 안 되었거나 런타임 오류 발생 시.
        HTTPException(500): 기타 예외 발생 시.
    """
    try:
        # Pydantic 모델 리스트를 dict 리스트로 변환
        content_list: List[dict] = [item.model_dump() for item in request.contents]
        embeddings_list: List[List[float]] = embedding_service.embed_bulk(content_list)
        return EmbeddingBulkResponse(embeddings=embeddings_list)

    except RuntimeError as err:
        raise HTTPException(status_code=503, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"벌크 임베딩 생성 중 오류 발생: {err}",
        )
