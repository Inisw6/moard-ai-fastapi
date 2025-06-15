from pydantic import BaseModel
from typing import List


class InferenceRequest(BaseModel):
    """/predict 엔드포인트에 대한 요청 스키마.

    Attributes:
        user_embedding (List[float]): 사용자 임베딩 벡터.
        content_embeddings (List[List[float]]): 콘텐츠 임베딩 벡터의 리스트.
    """

    user_embedding: List[float]
    content_embeddings: List[List[float]]


class InferenceResponse(BaseModel):
    """/predict 엔드포인트에 대한 응답 스키마.

    Attributes:
        q_values (List[float]): 예측된 Q-value의 리스트.
    """

    q_values: List[float]
