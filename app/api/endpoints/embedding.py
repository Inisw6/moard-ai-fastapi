from fastapi import APIRouter, Depends, HTTPException
from app.schemas.embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingBulkRequest,
    EmbeddingBulkResponse,
)
from app.services.embedding_service import EmbeddingService, get_embedding_service

router = APIRouter()


@router.post("/doc2vec", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Doc2Vec 모델을 사용하여 텍스트로부터 콘텐츠 임베딩을 생성합니다.

    Args:
        request (EmbeddingRequest): 'title'과 'description'을 포함하는 요청 본문.
        embedding_service (EmbeddingService): 의존성 주입으로 제공되는 임베딩 서비스.

    Returns:
        EmbeddingResponse: 생성된 임베딩 벡터가 담긴 응답 객체.

    Raises:
        HTTPException: 임베딩 과정에서 오류 발생 시 500 Internal Server Error.
    """
    try:
        embedder = embedding_service.get_embedder()
        content_dict = {"title": request.title, "description": request.description}
        embedding = embedder.embed_content(content_dict)
        return EmbeddingResponse(embedding=embedding.tolist())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"임베딩 생성 중 오류 발생: {str(e)}"
        )


@router.post("/doc2vec/bulk", response_model=EmbeddingBulkResponse)
async def create_embeddings_bulk(
    request: EmbeddingBulkRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Doc2Vec 모델을 사용하여 여러 콘텐츠를 한번에 임베딩합니다.

    Args:
        request (EmbeddingBulkRequest): 임베딩할 콘텐츠 리스트를 포함하는 요청 본문.
        embedding_service (EmbeddingService): 의존성 주입으로 제공되는 임베딩 서비스.

    Returns:
        EmbeddingBulkResponse: 생성된 임베딩 벡터 리스트가 담긴 응답 객체.

    Raises:
        HTTPException: 임베딩 과정에서 오류 발생 시.
    """
    try:
        # Pydantic 모델을 dict 리스트로 변환합니다.
        content_dicts = [c.model_dump() for c in request.contents]
        embeddings = embedding_service.embed_bulk(content_dicts)
        return EmbeddingBulkResponse(embeddings=embeddings)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"벌크 임베딩 생성 중 오류 발생: {str(e)}"
        )
