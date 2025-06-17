import os
import logging
from datetime import datetime
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from app.ml.q_network import DuelingQNetwork


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
        user_dim: int = 300,
        content_dim: int = 300,
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
        self.policy_net = DuelingQNetwork(user_dim, content_dim, hidden_dim).to(
            self.device
        )
        self.target_net = DuelingQNetwork(user_dim, content_dim, hidden_dim).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 옵티마이저 초기화
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

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
            logging.info(f"Successfully loaded pre-trained model from {model_path}")
        except Exception as e:
            logging.warning(
                f"Error loading pre-trained model: {e}. Starting from scratch."
            )

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
            user_embedding: 사용자 임베딩 텐서 [batch_size, user_dim]
            content_embedding: 콘텐츠 임베딩 텐서 [batch_size, content_dim]
            action: 선택된 행동 텐서 [batch_size, 1]
            reward: 보상 텐서 [batch_size, 1]
            done: 에피소드 종료 여부

        Returns:
            예측된 Q-value와 손실값의 튜플
        """
        # 데이터를 디바이스로 이동
        user_embedding = user_embedding.to(self.device)
        content_embedding = content_embedding.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)

        # 현재 상태의 Q-value 예측 [batch_size, 1]
        current_q_value = self.policy_net(user_embedding, content_embedding)

        # 타겟 Q-value 계산
        with torch.no_grad():
            next_q_value = self.target_net(user_embedding, content_embedding)
            target_q_value = reward + (0.99 * next_q_value * (1 - done))

        # 손실 계산 및 역전파
        loss = torch.nn.functional.smooth_l1_loss(current_q_value, target_q_value)

        # 옵티마이저로 학습
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 타겟 네트워크 업데이트 (매 10번째 스텝마다)
        if done:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return current_q_value, loss.item()

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
        logging.info(f"Model saved to {save_path}")

        return save_path
