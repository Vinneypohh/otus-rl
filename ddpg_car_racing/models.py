import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=3):
        super(Actor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.cnn_out_size = 7 * 7 * 64

        # 2. Полносвязная часть
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_size, 265), nn.ReLU(), nn.Linear(265, action_dim)
        )

    def forward(self, state):
        x = state.float() / 255.0
        x = self.fc(self.cnn(x))
        outs = F.relu(x)
        means = self.mean(outs)
        stds = self.logstd.exp()
        return means, stds


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=3):
        super(Critic, self).__init__()

        # 1. Сверточная часть для картинки (такая же, как у Актора)
        self.cnn = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cnn_out_size = 7 * 7 * 64

        # 2. Полносвязная часть
        # На вход она принимает: Фичи картинки + Действие
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_size + action_dim, 265),  # ВАЖНО: + action_dim
            nn.ReLU(),
            nn.Linear(265, 1),  # Выход Q-value
        )

    def forward(self, state, action):
        x = state.float() / 255.0
        cnn_features = self.cnn(x)

        # 2. Объединяем картинку и действие
        x = torch.cat([cnn_features, action], dim=1)

        # 3. Считаем Q-value
        q_value = self.fc(x)
        return q_value
