import logging
from app.ml.doc2vec_embedder import Doc2VecContentEmbedder


class EmbeddingService:
    """Doc2Vec 임베더 모델을 관리하는 싱글턴 서비스 클래스."""

    _instance = None
    _embedder = None

    def __new__(cls):
        """싱글턴 인스턴스를 생성하고 반환합니다."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            try:
                # 애플리케이션 시작 시 임베더를 한 번만 로드합니다.
                cls._embedder = Doc2VecContentEmbedder()
            except Exception as e:
                logging.error(f"Doc2Vec 임베더 초기화 실패: {e}")
                cls._embedder = None
        return cls._instance

    def get_embedder(self) -> Doc2VecContentEmbedder:
        """로드된 Doc2Vec 임베더 인스턴스를 반환합니다.

        Returns:
            Doc2VecContentEmbedder: Doc2Vec 임베더 인스턴스.

        Raises:
            RuntimeError: 임베더 로딩에 실패했을 경우 발생합니다.
        """
        if self._embedder is None:
            raise RuntimeError("Doc2Vec 임베더가 성공적으로 로드되지 않았습니다.")
        return self._embedder


def get_embedding_service() -> EmbeddingService:
    """EmbeddingService의 싱글턴 인스턴스를 반환하는 의존성 주입용 함수.

    Returns:
        EmbeddingService: 임베딩 서비스의 싱글턴 인스턴스.
    """
    return EmbeddingService() 