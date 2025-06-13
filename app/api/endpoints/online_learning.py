"""온라인 학습을 위한 API 엔드포인트.

이 모듈은 Q-Network의 온라인 학습을 위한 REST API 엔드포인트를 제공합니다.
"""

from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch

from app.ml.online_learning import OnlineLearner
from app.core.config import settings

router = APIRouter()

# 전역 OnlineLearner 인스턴스
learner = OnlineLearner(
    model_path=settings.PRETRAINED_MODEL_PATH
)


class InteractionData(BaseModel):
    """사용자 상호작용 데이터를 위한 Pydantic 모델.

    Attributes:
        user_id: 사용자 ID
        item_id: 아이템 ID
        action: 사용자 행동
        reward: 보상값
        done: 에피소드 종료 여부
    """
    user_id: str
    item_id: str
    action: str
    reward: float
    done: bool


class InteractionResponse(BaseModel):
    """상호작용 처리 결과를 위한 Pydantic 모델.

    Attributes:
        q_value: 예측된 Q-value
        loss: 학습 손실값
    """
    q_value: float
    loss: float


class BatchInteractionData(BaseModel):
    """배치 상호작용 데이터를 위한 Pydantic 모델.

    Attributes:
        interactions: 상호작용 데이터 리스트
    """
    interactions: List[InteractionData]


class BatchInteractionResponse(BaseModel):
    """배치 상호작용 처리 결과를 위한 Pydantic 모델.

    Attributes:
        results: 각 상호작용의 처리 결과 리스트
        total_loss: 전체 배치의 평균 손실값
    """
    results: List[InteractionResponse]
    total_loss: float


@router.post("/interaction", response_model=InteractionResponse)
async def process_interaction(
    data: InteractionData,
) -> InteractionResponse:
    """단일 사용자 상호작용을 처리하고 Q-value를 예측합니다.

    Args:
        data: 사용자 상호작용 데이터

    Returns:
        예측된 Q-value와 손실값을 포함한 응답

    Raises:
        HTTPException: 처리 중 오류가 발생한 경우
    """
    try:
        # TODO: 실제 임베딩 생성 로직 구현
        # 현재는 예시로 랜덤 텐서 사용
        user_embedding = torch.randn(1, 20)  # [batch_size, user_dim]
        content_embedding = torch.randn(1, 15)  # [batch_size, content_dim]
        action = torch.tensor([[1]], dtype=torch.long)  # [[action_index]]
        reward = torch.tensor([data.reward])

        # 상호작용 처리
        q_value, loss = learner.process_interaction(
            user_embedding=user_embedding,
            content_embedding=content_embedding,
            action=action,
            reward=reward,
            done=data.done,
        )

        return InteractionResponse(
            q_value=q_value.item(),
            loss=loss,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing interaction: {str(e)}",
        )


@router.post("/batch-interaction", response_model=BatchInteractionResponse)
async def process_batch_interaction(
    data: BatchInteractionData,
) -> BatchInteractionResponse:
    """여러 사용자 상호작용을 배치로 처리하고 Q-value를 예측합니다.

    Args:
        data: 배치 상호작용 데이터

    Returns:
        각 상호작용의 처리 결과와 전체 평균 손실값을 포함한 응답

    Raises:
        HTTPException: 처리 중 오류가 발생한 경우
    """
    try:
        results = []
        total_loss = 0.0

        for interaction in data.interactions:
            # TODO: 실제 임베딩 생성 로직 구현
            # 현재는 예시로 랜덤 텐서 사용
            user_embedding = torch.randn(1, 20)  # [batch_size, user_dim]
            content_embedding = torch.randn(1, 15)  # [batch_size, content_dim]
            action = torch.tensor([[1]], dtype=torch.long)  # [[action_index]]
            reward = torch.tensor([interaction.reward])

            # 상호작용 처리
            q_value, loss = learner.process_interaction(
                user_embedding=user_embedding,
                content_embedding=content_embedding,
                action=action,
                reward=reward,
                done=interaction.done,
            )

            results.append(
                InteractionResponse(
                    q_value=q_value.item(),
                    loss=loss,
                )
            )
            total_loss += loss

        # 평균 손실값 계산
        avg_loss = total_loss / len(data.interactions) if data.interactions else 0.0

        return BatchInteractionResponse(
            results=results,
            total_loss=avg_loss,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch interaction: {str(e)}",
        )


@router.post("/save")
async def save_model(
    filename: Optional[str] = None,
) -> Dict[str, str]:
    """현재 모델 상태를 저장합니다.

    Args:
        filename: 저장할 파일명 (None이면 자동 생성)

    Returns:
        저장된 모델 파일의 경로를 포함한 응답

    Raises:
        HTTPException: 저장 중 오류가 발생한 경우
    """
    try:
        save_path = learner.save_model(filename=filename)
        return {"message": "Model saved successfully", "path": save_path}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving model: {str(e)}",
        ) 