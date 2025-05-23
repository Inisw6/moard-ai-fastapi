import torch.nn as nn

class MultiStockDQN(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q(s, a)
        )

    def forward(self, x):
        return self.net(x)
