import random
import math

import torch
import torch.optim as optim

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN 에이전트를 구현하는 클래스입니다.

    정책 네트워크와 타겟 네트워크를 유지하며,
    Epsilon-greedy 방식으로 행동을 선택하고,
    경험 리플레이 버퍼에서 배치를 샘플링해 학습을 수행합니다.

    Args:
        state_dim (int): 상태(state)의 차원.
        action_dim (int): 행동(action)의 개수.
        replay_capacity (int, optional): 리플레이 버퍼 최대 크기. Defaults to 10000.
        batch_size (int, optional): 학습 시 샘플 배치 크기. Defaults to 128.
        gamma (float, optional): 할인율. Defaults to 0.99.
        lr (float, optional): 학습률(learning rate). Defaults to 1e-4.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay_capacity: int = 10000,
        batch_size: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-4,
    ) -> None:
        # 환경 및 학습 파라미터
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.batch_size: int = batch_size
        self.gamma: float = gamma

        # 네트워크 초기화
        self.policy_net: DQN = DQN(state_dim, action_dim)
        self.target_net: DQN = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 평가 모드 유지

        # 옵티마이저 및 리플레이 버퍼 설정
        self.optimizer: optim.Optimizer = optim.Adam(
            self.policy_net.parameters(), lr=lr
        )
        self.memory: ReplayBuffer = ReplayBuffer(replay_capacity)

        # Epsilon-greedy 정책 관련 변수
        self.steps_done: int = 0
        self.eps_start: float = 0.9
        self.eps_end: float = 0.05
        self.eps_decay: float = 200

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Epsilon-greedy 정책으로 행동을 선택합니다.

        Args:
            state (torch.Tensor): 현재 상태 텐서.

        Returns:
            torch.Tensor: 선택된 행동 인덱스 (shape: [1,1]).
        """
        sample: float = random.random()
        eps_threshold: float = self.eps_end + (
            self.eps_start - self.eps_end
        ) * math.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            # 정책 네트워크를 이용해 행동 선택
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # 무작위 행동 선택
            return torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        """경험 리플레이 버퍼에 트랜지션을 저장합니다.

        Args:
            state (torch.Tensor): 현재 상태.
            action (torch.Tensor): 선택한 행동.
            next_state (torch.Tensor): 다음 상태.
            reward (float): 받은 보상.
            done (bool): 에피소드 종료 여부.
        """
        self.memory.push(state, action, next_state, reward, done)

    def update_target_net(self) -> None:
        """정책 네트워크의 파라미터를 타겟 네트워크로 복사합니다."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
