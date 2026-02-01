import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_mu = nn.Linear(64, action_size)

        self.fc_sigma = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.fc_mu(x))

        sigma = F.softplus(self.fc_sigma(x)) + 1e-5

        return mu, sigma
