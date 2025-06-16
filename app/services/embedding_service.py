import logging
from typing import List, Dict
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

    def embed_bulk(self, contents: List[Dict]) -> List[List[float]]:
        """여러 콘텐츠를 한번에 임베딩합니다.

        Args:
            contents (ListDict]): 임베딩할 콘텐츠 딕셔너리 리스트.

        Returns:
            List[List[float]]: 생성된 임베딩 벡터들의 리스트.

        Raises:
            RuntimeError: 임베더가 로드되지 않았거나 추론 중 오류가 발생한 경우.
        """
        embedder = self.get_embedder()
        try:
            embeddings_array = embedder.embed_contents(contents)
            return embeddings_array.tolist()
        except Exception as e:
            logging.error(f"벌크 임베딩 중 오류 발생: {e}")
            raise RuntimeError("벌크 임베딩 과정에서 오류가 발생했습니다.")


def get_embedding_service() -> EmbeddingService:
    """EmbeddingService의 싱글턴 인스턴스를 반환하는 의존성 주입용 함수.

    Returns:
        EmbeddingService: 임베딩 서비스의 싱글턴 인스턴스.
    """
    return EmbeddingService()
