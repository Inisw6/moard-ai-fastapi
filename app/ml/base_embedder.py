from abc import ABC, abstractmethod
import numpy as np


class BaseContentEmbedder(ABC):
    """콘텐츠 임베더를 위한 추상 기본 클래스.

    모든 콘텐츠 임베더는 이 클래스를 상속받아 `output_dim`과 `embed_content`
    메서드를 구현해야 합니다.
    """

    @abstractmethod
    def output_dim(self) -> int:
        """임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 콘텐츠 임베딩 벡터의 차원.
        """
        pass

    @abstractmethod
    def embed_content(self, content: dict) -> np.ndarray:
        """단일 콘텐츠 아이템을 임베딩 벡터로 변환합니다.

        Args:
            content (dict): 임베딩할 콘텐츠 정보를 담은 딕셔너리.

        Returns:
            np.ndarray: 콘텐츠의 임베딩 벡터.
        """
        pass 