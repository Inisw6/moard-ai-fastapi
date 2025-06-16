import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # net.0
            nn.ReLU(),
            nn.Linear(128, 128),  # net.2
            nn.ReLU(),
            nn.Linear(128, output_dim),  # net.4
        )

    def forward(self, x):
        return self.net(x)
