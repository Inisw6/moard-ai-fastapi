"""온라인 학습을 위한 모듈.

이 모듈은 Q-Network의 온라인 학습을 위한 클래스와 함수들을 제공합니다.
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from app.ml.q_network import QNetwork
import os
from datetime import datetime


class OnlineLearner:
    """Q-Network의 온라인 학습을 관리하는 클래스.

    이 클래스는 사용자 상호작용 데이터를 받아 Q-Network를 온라인으로 학습시키는
    기능을 제공합니다.

    Attributes:
        policy_net: 학습할 정책 네트워크
        target_net: 타겟 네트워크
        device: 학습에 사용할 디바이스 (CPU/GPU)
    """

    def __init__(
        self,
        user_dim: int = 20,
        content_dim: int = 15,
        hidden_dim: int = 128,
        model_path: Optional[str] = None,
    ) -> None:
        """OnlineLearner를 초기화합니다.

        Args:
            user_dim: 사용자 임베딩 차원
            content_dim: 콘텐츠 임베딩 차원
            hidden_dim: 은닉층 크기
            model_path: 사전 학습된 모델 경로 (있는 경우)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 네트워크 초기화
        self.policy_net = QNetwork(user_dim, content_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(user_dim, content_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 사전 학습된 모델이 있다면 로드
        if model_path and os.path.exists(model_path):
            self._load_pretrained_model(model_path)

    def _load_pretrained_model(self, model_path: str) -> None:
        """사전 학습된 모델을 로드합니다.

        Args:
            model_path: 사전 학습된 모델 파일 경로
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["q_net_state"])
            self.target_net.load_state_dict(checkpoint["target_net_state"])
            self.policy_net.train()
            self.target_net.eval()
            print(f"Successfully loaded pre-trained model from {model_path}")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}. Starting from scratch.")

    def process_interaction(
        self,
        user_embedding: Tensor,
        content_embedding: Tensor,
        action: Tensor,
        reward: Tensor,
        done: bool,
    ) -> Tuple[Tensor, float]:
        """사용자 상호작용을 처리하고 Q-value를 예측합니다.

        Args:
            user_embedding: 사용자 임베딩 텐서
            content_embedding: 콘텐츠 임베딩 텐서
            action: 선택된 행동 텐서
            reward: 보상 텐서
            done: 에피소드 종료 여부

        Returns:
            예측된 Q-value와 손실값의 튜플
        """
        # 데이터를 디바이스로 이동
        user_embedding = user_embedding.to(self.device)
        content_embedding = content_embedding.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)

        # Q-value 예측
        q_value = self.policy_net(user_embedding, content_embedding)
        
        # TODO: 실제 학습 로직 구현
        # 현재는 예시로 0을 반환
        loss = torch.tensor(0.0, device=self.device)

        return q_value, loss.item()

    def save_model(
        self,
        save_dir: str = "models",
        filename: Optional[str] = None,
    ) -> str:
        """현재 모델 상태를 저장합니다.

        Args:
            save_dir: 모델을 저장할 디렉토리
            filename: 저장할 파일명 (None이면 자동 생성)

        Returns:
            저장된 모델 파일의 전체 경로
        """
        os.makedirs(save_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"q_network_{timestamp}.pth"

        save_path = os.path.join(save_dir, filename)

        checkpoint = {
            "q_net_state": self.policy_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "user_dim": self.policy_net.user_dim,
            "content_dim": self.policy_net.content_dim,
            "hidden_dim": self.policy_net.hidden_dim,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
        
        return save_path 