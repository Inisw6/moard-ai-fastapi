from fastapi import APIRouter, Depends, HTTPException
from app.schemas.embedding import EmbeddingRequest, EmbeddingResponse
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