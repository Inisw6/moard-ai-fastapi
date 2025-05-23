import torch
import torch.nn as nn

class ContentDQN(nn.Module):
    """
    하위 계층 콘텐츠 선택을 위한 DQN
    입력: 사용자 + goal + 후보 콘텐츠 벡터 (텍스트 + 감정 포함)
    출력: Q값 (후보 콘텐츠 수 만큼)
    """
    def __init__(self, state_dim: int, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        last_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))  # Q값 1개 출력
        self.q_net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            state_batch: [B, state_dim]
        Returns:
            q_values: [B, 1] → squeeze() 필요 시 적용
        """
        return self.q_net(state_batch)