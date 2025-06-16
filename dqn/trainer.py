from typing import List

import torch
import torch.nn.functional as F

from dqn.agent import DQNAgent
from dqn.replay_buffer import Transition


def optimize_model(agent: DQNAgent) -> None:
    """DQN 에이전트의 한 번의 최적화 단계를 수행합니다.

    이 과정에서는 에이전트의 리플레이 버퍼에서 배치를 샘플링하고,
    현재 상태와 다음 상태에 대한 Q-값을 계산한 뒤,
    정책 네트워크를 Huber 손실을 최소화하도록 업데이트합니다.

    Args:
        agent (DQNAgent): 네트워크, 옵티마이저, 메모리를 포함하는 DQN 에이전트.

    Returns:
        None
    """
    # 메모리에 배치 사이즈만큼의 샘플이 없다면 최적화 건너뛰기
    if len(agent.memory) < agent.batch_size:
        return

    # 트랜지션 배치 샘플링
    transitions: List[Transition] = agent.memory.sample(agent.batch_size)
    batch: Transition = Transition(*zip(*transitions))

    # 최종 상태가 아닌 다음 상태들에 대한 마스크 생성
    non_final_mask: torch.Tensor = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
    )
    # 최종 상태가 아닌 다음 상태들만 결합
    non_final_next_states: torch.Tensor = torch.cat(
        [s for s in batch.next_state if s is not None]
    )

    # 배치 내 상태, 행동, 보상 텐서 결합
    state_batch: torch.Tensor = torch.cat(batch.state)
    action_batch: torch.Tensor = torch.cat(batch.action)
    reward_batch: torch.Tensor = torch.cat(batch.reward)

    # 정책 네트워크로부터 Q(s_t, a) 계산
    state_action_values: torch.Tensor = agent.policy_net(state_batch).gather(
        1, action_batch
    )

    # 타겟 네트워크로부터 비최종 다음 상태들의 V(s_{t+1}) 계산
    next_state_values: torch.Tensor = torch.zeros(agent.batch_size)
    next_state_values[non_final_mask] = (
        agent.target_net(non_final_next_states).max(1)[0].detach()
    )

    # 기대 Q-값 계산
    expected_state_action_values: torch.Tensor = (
        next_state_values * agent.gamma
    ) + reward_batch

    # Huber 손실(loss) 계산
    loss: torch.Tensor = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # 역전파 및 파라미터 업데이트
    agent.optimizer.zero_grad()
    loss.backward()
    for param in agent.policy_net.parameters():
        if param.grad is not None:
            # 그래디언트 클램핑
            param.grad.data.clamp_(-1, 1)
    agent.optimizer.step()
