import torch
import torch.nn as nn


class DQN(nn.Module):
    """심층 강화학습용 DQN 네트워크 클래스입니다.

    선형 레이어와 ReLU 활성화 함수로 구성된 MLP입니다.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Args:
            input_dim (int): 입력 상태의 차원.
            output_dim (int): 가능한 행동의 개수.
        """
        super(DQN, self).__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파 계산을 수행합니다.

        Args:
            x (torch.Tensor): 입력 상태 텐서.

        Returns:
            torch.Tensor: 각 행동에 대한 Q-값 텐서.
        """
        return self.net(x)
