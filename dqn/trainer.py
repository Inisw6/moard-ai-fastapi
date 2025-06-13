import torch
import torch.nn.functional as F
from dqn.replay_buffer import Transition


def optimize_model(agent):
    if len(agent.memory) < agent.batch_size:
        return

    transitions = agent.memory.sample(agent.batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = agent.policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(agent.batch_size)
    next_state_values[non_final_mask] = (
        agent.target_net(non_final_next_states).max(1)[0].detach()
    )

    expected_state_action_values = (next_state_values * agent.gamma) + reward_batch

    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    agent.optimizer.zero_grad()
    loss.backward()
    for param in agent.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    agent.optimizer.step()
