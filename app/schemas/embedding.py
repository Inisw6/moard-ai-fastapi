from pydantic import BaseModel
from typing import List, Optional


class EmbeddingContent(BaseModel):
    """임베딩될 단일 콘텐츠 항목 스키마.

    Attributes:
        title (str): 임베딩할 콘텐츠의 제목.
        description (Optional[str]): 임베딩할 콘텐츠의 설명. 기본값은 빈 문자열.
    """
    title: str
    description: Optional[str] = ""


class EmbeddingRequest(BaseModel):
    """임베딩 생성 요청 스키마.

    Attributes:
        title (str): 임베딩할 콘텐츠의 제목.
        description (Optional[str]): 임베딩할 콘텐츠의 설명. 기본값은 빈 문자열.
    """

    title: str
    description: Optional[str] = ""


class EmbeddingResponse(BaseModel):
    """임베딩 생성 응답 스키마.

    Attributes:
        embedding (List[float]): 생성된 임베딩 벡터.
    """

    embedding: List[float]


class EmbeddingBulkRequest(BaseModel):
    """벌크 임베딩 생성 요청 스키마.

    Attributes:
        contents (List[EmbeddingContent]): 임베딩할 콘텐츠 객체의 리스트.
    """
    contents: List[EmbeddingContent]


class EmbeddingBulkResponse(BaseModel):
    """벌크 임베딩 생성 응답 스키마.

    Attributes:
        embeddings (List[List[float]]): 생성된 임베딩 벡터의 리스트.
    """
    embeddings: List[List[float]] 