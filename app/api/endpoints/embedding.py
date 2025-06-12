from fastapi import APIRouter, Depends, HTTPException
from app.schemas.embedding import EmbeddingRequest, EmbeddingResponse
from app.ml.base_embedder import BaseContentEmbedder
from app.ml.registry import get_embedder

router = APIRouter()

def doc2vec_embedder_dependency() -> BaseContentEmbedder:
    """Doc2Vec 임베더 의존성을 제공하는 함수.

    Returns:
        BaseContentEmbedder: 'doc2vec_content' 이름으로 등록된 임베더 인스턴스.

    Raises:
        HTTPException: 임베더를 로드할 수 없는 경우 503 Service Unavailable 오류 발생.
    """
    try:
        # get_embedder는 싱글톤 인스턴스를 반환하지 않으므로 매번 새로 생성될 수 있습니다.
        # 성능이 중요하다면, 서비스 계층에서 싱글톤으로 관리하는 것을 고려해야 합니다.
        return get_embedder("doc2vec_content")
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(
            status_code=503, detail=f"Doc2Vec 임베더를 로드할 수 없습니다: {e}"
        )

@router.post("/doc2vec", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    embedder: BaseContentEmbedder = Depends(doc2vec_embedder_dependency),
):
    """Doc2Vec 모델을 사용하여 텍스트로부터 콘텐츠 임베딩을 생성합니다.

    Args:
        request (EmbeddingRequest): 'title'과 'description'을 포함하는 요청 본문.
        embedder (BaseContentEmbedder): 의존성 주입으로 제공되는 Doc2Vec 임베더.

    Returns:
        EmbeddingResponse: 생성된 임베딩 벡터가 담긴 응답 객체.

    Raises:
        HTTPException: 임베딩 과정에서 오류 발생 시 500 Internal Server Error.
    """
    try:
        content_dict = {"title": request.title, "description": request.description}
        embedding = embedder.embed_content(content_dict)
        return EmbeddingResponse(embedding=embedding.tolist())
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"임베딩 생성 중 오류 발생: {str(e)}"
        ) 