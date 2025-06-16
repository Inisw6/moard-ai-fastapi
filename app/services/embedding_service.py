import logging
from typing import Any, Dict, List

from app.ml.doc2vec_embedder import Doc2VecContentEmbedder


class EmbeddingService:
    """
    싱글턴 패턴으로 Doc2VecContentEmbedder를 관리하는 서비스 클래스.

    Attributes:
        _instance (Optional[EmbeddingService]): 싱글턴 인스턴스 캐시
        _embedder (Optional[Doc2VecContentEmbedder]): Doc2Vec 임베더 인스턴스
    """

    _instance: "EmbeddingService" = None  # type: ignore[name-defined]
    _embedder: Doc2VecContentEmbedder = None  # type: ignore[assignment]

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                cls._embedder = Doc2VecContentEmbedder()
            except Exception as err:
                logging.error("Doc2Vec 임베더 초기화 실패: %s", err)
                cls._embedder = None  # type: ignore[assignment]
        return cls._instance

    def get_embedder(self) -> Doc2VecContentEmbedder:
        """
        로드된 Doc2VecContentEmbedder 인스턴스를 반환합니다.

        Returns:
            Doc2VecContentEmbedder: 임베더 인스턴스

        Raises:
            RuntimeError: 임베더 로딩에 실패했을 경우
        """
        if self._embedder is None:
            raise RuntimeError("Doc2Vec 임베더가 성공적으로 로드되지 않았습니다.")
        return self._embedder

    def embed_bulk(self, contents: List[Dict[str, Any]]) -> List[List[float]]:
        """
        여러 콘텐츠를 임베딩하여 결과 벡터 리스트를 반환합니다.

        Args:
            contents (List[Dict[str, Any]]): 임베딩할 콘텐츠 리스트

        Returns:
            List[List[float]]: 생성된 임베딩 벡터 리스트

        Raises:
            RuntimeError: 임베딩 중 오류가 발생한 경우
        """
        embedder = self.get_embedder()
        try:
            return embedder.embed_contents(contents).tolist()
        except Exception as err:
            logging.error("벌크 임베딩 중 오류 발생: %s", err)
            raise RuntimeError("벌크 임베딩 과정에서 오류가 발생했습니다.") from err


def get_embedding_service() -> EmbeddingService:
    """
    FastAPI 의존성 주입용 함수로 EmbeddingService 싱글턴을 반환합니다.

    Returns:
        EmbeddingService: 임베딩 서비스 인스턴스
    """
    return EmbeddingService()
