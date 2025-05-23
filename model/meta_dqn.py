import torch
import torch.nn as nn

class MetaDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        last_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))
        self.q_net = nn.Sequential(*layers)

    def forward(self, state):
        return self.q_net(state)
