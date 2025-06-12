from pydantic import BaseModel
from typing import List, Optional


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