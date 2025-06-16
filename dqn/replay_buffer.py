import random
from collections import deque
from typing import Deque, List, NamedTuple


class Transition(NamedTuple):
    state: any
    action: any
    next_state: any
    reward: float
    done: bool


class ReplayBuffer:
    """경험 리플레이 버퍼를 관리하는 클래스입니다.

    deque를 사용해 고정된 용량(capacity)의 메모리를 유지하며,
    샘플링을 통해 무작위로 배치를 반환합니다.

    Args:
        capacity (int): 버퍼가 보유할 최대 트랜지션 수.

    Attributes:
        memory (Deque[Transition]): 저장된 트랜지션을 담은 deque.
    """

    def __init__(self, capacity: int) -> None:
        """
        ReplayBuffer 객체를 초기화합니다.

        Args:
            capacity (int): 최대 버퍼 크기.

        Returns:
            None
        """
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self, state: any, action: any, next_state: any, reward: float, done: bool
    ) -> None:
        """하나의 트랜지션을 버퍼에 저장합니다.

        Args:
            state (any): 현재 상태.
            action (any): 수행한 행동.
            next_state (any): 다음 상태.
            reward (float): 받은 보상.
            done (bool): 에피소드 종료 여부.

        Returns:
            None
        """
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """버퍼에서 무작위로 배치 크기만큼의 트랜지션을 샘플링합니다.

        Args:
            batch_size (int): 샘플할 트랜지션 개수.

        Returns:
            List[Transition]: 선택된 트랜지션 리스트.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """현재 버퍼에 저장된 트랜지션 수를 반환합니다.

        Returns:
            int: 버퍼 내 트랜지션 개수.
        """
        return len(self.memory)
