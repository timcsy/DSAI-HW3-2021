import torch
import torch.nn as nn

import numpy as np


class DQN(nn.Module):
    def __init__(self, n_states=336, n_actions=72):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_states, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.fc(x)
