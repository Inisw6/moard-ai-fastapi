from typing import List
from uuid import UUID

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """/bulk 예측 요청 스키마.

    Attributes:
        user_embedding (List[float]): 사용자 임베딩 벡터
        content_embeddings (List[List[float]]): 콘텐츠 임베딩 벡터 리스트
    """

    user_embedding: List[float] = Field(..., description="사용자 임베딩 벡터")
    content_embeddings: List[List[float]] = Field(
        ..., description="콘텐츠 임베딩 벡터의 리스트"
    )


class InferenceResponse(BaseModel):
    """/bulk 예측 응답 스키마.

    Attributes:
        q_values (List[float]): 예측된 Q-value 리스트
    """

    q_values: List[float] = Field(..., description="예측된 Q-value 리스트")


class TopContentsRequest(BaseModel):
    """/top-contents 요청 스키마.

    Attributes:
        user_id (UUID): 사용자 UUID
        content_ids (List[int]): 콘텐츠 ID 리스트
    """

    user_id: UUID = Field(..., description="사용자 UUID")
    content_ids: List[int] = Field(..., description="콘텐츠 ID 리스트")

    def get_uuid_bytes(self) -> bytes:
        """
        UUID를 바이트 형식으로 반환합니다.

        Returns:
            bytes: UUID 바이트 표현
        """
        return self.user_id.bytes


class TopContentsResponse(BaseModel):
    """/top-contents 응답 스키마.

    Attributes:
        content_ids (List[int]): 상위 6개 콘텐츠 ID 리스트
        user_embedding (List[float]): 사용자 임베딩 벡터
        model_name (str): 사용된 모델 이름
    """

    content_ids: List[int] = Field(..., description="상위 콘텐츠 ID 리스트")
    user_embedding: List[float] = Field(..., description="사용자 임베딩 벡터")
    model_name: str = Field(..., description="사용된 모델 이름")
