import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Q-Value를 예측하는 MLP(Multi-Layer Perceptron) 네트워크.

    Attributes:
        user_dim (int): 사용자 임베딩 벡터의 차원.
        content_dim (int): 콘텐츠 임베딩 벡터의 차원.
        hidden_dim (int): 은닉층의 크기.
        net (nn.Sequential): 실제 Q-value를 계산하는 신경망.
    """

    def __init__(self, user_dim: int, content_dim: int, hidden_dim: int = 128) -> None:
        """QNetwork 인스턴스를 초기화합니다.

        Args:
            user_dim (int): 사용자 임베딩 벡터의 차원.
            content_dim (int): 콘텐츠 임베딩 벡터의 차원.
            hidden_dim (int): 은닉층의 크기. 기본값은 128.
        """
        super().__init__()
        self.user_dim = user_dim
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(user_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """사용자 및 콘텐츠 벡터로부터 Q-value를 예측합니다.

        두 입력 텐서의 배치 크기(batch size)는 동일해야 합니다.

        Args:
            user (torch.Tensor): 사용자 임베딩 텐서. Shape: `[batch_size, user_dim]`.
            content (torch.Tensor): 콘텐츠 임베딩 텐서. Shape: `[batch_size, content_dim]`.

        Returns:
            torch.Tensor: 예측된 Q-value 텐서. Shape: `[batch_size, 1]`.

        Raises:
            ValueError: 사용자 텐서와 콘텐츠 텐서의 배치 크기 또는 임베딩 차원이
                모델 초기화 시의 값과 일치하지 않을 경우 발생합니다.
        """
        if user.shape[0] != content.shape[0]:
            raise ValueError(
                f"Batch size mismatch: user.shape={user.shape}, content.shape={content.shape}"
            )
        if user.shape[1] != self.user_dim or content.shape[1] != self.content_dim:
            raise ValueError(
                f"Input dim mismatch: user_dim={user.shape[1]}, expected={self.user_dim}; "
                f"content_dim={content.shape[1]}, expected={self.content_dim}"
            )
        x = torch.cat([user, content], dim=1)
        return self.net(x)  # [batch, 1] 