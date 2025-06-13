"""Q-Network를 사용한 온라인 학습 파이프라인.

이 모듈은 사용자 상호작용 데이터를 기반으로 Q-Network를 온라인으로 학습하는
파이프라인을 제공합니다. 사전 학습된 모델을 불러와 파인튜닝하는 기능을 포함합니다.

Typical usage example:
    python online_learning_example.py --save q_network_finetuned.pth
"""

from typing import Dict, Optional
import torch
from torch import Tensor
from app.ml.q_network import QNetwork
import os
from datetime import datetime

# --- 사용자 데이터 연동을 위한 Placeholder 함수들 ---


def get_user_embedding(interaction_data: Dict[str, str]) -> Tensor:
    """사용자 상호작용 데이터로부터 사용자 임베딩 벡터를 생성합니다.

    Args:
        interaction_data: 사용자 상호작용 데이터를 담은 딕셔너리.
            예: {"user_id": "user123", "item_id": "item456", "action": "click"}

    Returns:
        사용자 임베딩 텐서. Shape: [batch_size, user_dim]
    """
    print("Getting user embedding...")
    # 실제로는 interaction_data를 사용해 사용자 임베딩을 만들어야 합니다.
    placeholder_user_embedding = torch.randn(1, 20)  # (batch_size, user_dim)
    return placeholder_user_embedding


def get_content_embedding(interaction_data: Dict[str, str]) -> Tensor:
    """사용자 상호작용 데이터로부터 콘텐츠 임베딩 벡터를 생성합니다.

    Args:
        interaction_data: 사용자 상호작용 데이터를 담은 딕셔너리.
            예: {"user_id": "user123", "item_id": "item456", "action": "click"}

    Returns:
        콘텐츠 임베딩 텐서. Shape: [batch_size, content_dim]
    """
    print("Getting content embedding...")
    # 실제로는 interaction_data를 사용해 콘텐츠 임베딩을 만들어야 합니다.
    placeholder_content_embedding = torch.randn(1, 15)  # (batch_size, content_dim)
    return placeholder_content_embedding


def get_action_from_data(interaction_data: Dict[str, str]) -> Tensor:
    """사용자가 어떤 행동(action)을 했는지 데이터로부터 추출합니다.

    Args:
        interaction_data: 사용자 상호작용 데이터를 담은 딕셔너리.

    Returns:
        행동 인덱스 텐서. Shape: [batch_size, 1]
    """
    print("Getting action from data...")
    placeholder_action = torch.tensor([[1]], dtype=torch.long)  # [[action_index]]
    return placeholder_action


def get_reward_from_data(interaction_data: Dict[str, str]) -> Tensor:
    """사용자의 행동에 대한 보상(reward)을 계산합니다.

    Args:
        interaction_data: 사용자 상호작용 데이터를 담은 딕셔너리.

    Returns:
        보상 텐서. Shape: [batch_size]
    """
    print("Getting reward from data...")
    placeholder_reward = torch.tensor([1.0])  # [reward]
    return placeholder_reward


def is_session_done(interaction_data: Dict[str, str]) -> bool:
    """사용자 세션이 종료되었는지 여부를 판단합니다.

    Args:
        interaction_data: 사용자 상호작용 데이터를 담은 딕셔너리.

    Returns:
        세션 종료 여부를 나타내는 불리언 값.
    """
    print("Checking if session is done...")
    return False  # 실제 로직으로 대체 필요


def save_model(
    policy_net: QNetwork,
    target_net: QNetwork,
    save_dir: str = "models",
    filename: Optional[str] = None,
) -> None:
    """학습된 모델을 저장합니다.

    Args:
        policy_net: 학습된 정책 네트워크.
        target_net: 타겟 네트워크.
        save_dir: 모델을 저장할 디렉토리.
        filename: 저장할 파일명. None이면 현재 시간을 포함한 이름이 생성됩니다.
    """
    # 저장 디렉토리가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    # 파일명이 주어지지 않으면 현재 시간을 포함한 이름 생성
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"q_network_{timestamp}.pth"

    # 전체 경로 생성
    save_path = os.path.join(save_dir, filename)

    # 모델 상태 저장
    checkpoint = {
        "q_net_state": policy_net.state_dict(),
        "target_net_state": target_net.state_dict(),
        "user_dim": policy_net.user_dim,
        "content_dim": policy_net.content_dim,
        "hidden_dim": policy_net.hidden_dim,
        "timestamp": timestamp,
    }

    # 모델 저장
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def online_learning_pipeline(save_filename: Optional[str] = None) -> None:
    """실제 사용자 상호작용이 발생할 때마다 호출될 함수.

    이 함수는 다음과 같은 단계로 실행됩니다:
    1. Q-Network 초기화
    2. 사전 학습된 모델 로드 (있는 경우)
    3. 사용자 상호작용 데이터 처리
    4. Q-value 예측
    5. 모델 최적화
    6. 모델 저장

    Args:
        save_filename: 모델을 저장할 때 사용할 파일명.
    """
    # 1. QNetwork 초기화
    user_dim = 20  # 사용자 임베딩 차원
    content_dim = 15  # 콘텐츠 임베딩 차원
    hidden_dim = 128  # 은닉층 크기
    policy_net = QNetwork(user_dim, content_dim, hidden_dim)
    target_net = QNetwork(user_dim, content_dim, hidden_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    print("QNetwork initialized.")

    # 2. 사전 학습된 모델 로드 (Fine-tuning)
    model_path = "models/dqn_model_seed0_final.pth"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)

        # 저장된 딕셔너리에서 가중치를 불러옵니다.
        policy_net.load_state_dict(checkpoint["q_net_state"])
        target_net.load_state_dict(checkpoint["target_net_state"])

        # 파인튜닝을 위해 모델을 학습 모드로 설정합니다.
        policy_net.train()
        target_net.eval()
        print(
            f"Successfully loaded pre-trained model from {model_path} for fine-tuning."
        )
    except FileNotFoundError:
        print(
            f"Warning: Pre-trained model not found at {model_path}. Starting training from scratch."
        )
    except Exception as e:
        print(f"Error loading pre-trained model: {e}. Starting training from scratch.")

    # 3. 사용자 상호작용 데이터가 발생했다고 가정
    sample_interaction_data = {
        "user_id": "user123",
        "item_id": "item456",
        "action": "click",
    }

    # 4. 데이터로부터 학습에 필요한 요소 추출
    user_embedding = get_user_embedding(sample_interaction_data)  # [1, 20]
    content_embedding = get_content_embedding(sample_interaction_data)  # [1, 15]
    action = get_action_from_data(sample_interaction_data)
    reward = get_reward_from_data(sample_interaction_data)
    done = is_session_done(sample_interaction_data)

    # 5. Q-value 예측
    q_value = policy_net(user_embedding, content_embedding)
    print(f"Predicted Q-value: {q_value.item():.4f}")

    # 6. 모델 최적화 (학습)
    print("Optimizing model...")
    # TODO: QNetwork에 맞는 최적화 로직 구현 필요
    # optimize_model(agent)  # 이 부분은 QNetwork 구조에 맞게 수정 필요

    # 7. 모델 저장
    save_model(policy_net, target_net, filename=save_filename)


def main() -> None:
    """메인 함수. 명령행 인자를 파싱하고 학습 파이프라인을 실행합니다."""
    import argparse

    parser = argparse.ArgumentParser(description="Q-Network Online Learning")
    parser.add_argument(
        "--save", type=str, help="모델을 저장할 파일명 (예: q_network_finetuned.pth)"
    )
    args = parser.parse_args()

    online_learning_pipeline(save_filename=args.save)


if __name__ == "__main__":
    main()
