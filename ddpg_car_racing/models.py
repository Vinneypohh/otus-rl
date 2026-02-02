# Actor:
# CNN как раньше (Conv2d -> ReLU).
# Потом Flatten.
# Потом Linear слои.
# На выходе 3 нейрона.
# Для руля используй tanh (диапазон -1..1).
# Для газа/тормоза используй sigmoid (диапазон 0..1).

# Critic:
# CNN для обработки картинки.
# Flatten.
# Внимание: В этот момент у тебя есть вектор фичей картинки (например, размер 256) и вектор действия (размер 3).
# Ты должен их склеить: torch.cat([cnn_out, action], dim=1).
# Дальше идут Linear слои.
# Выход: 1 нейрон (без активации).
import torch
import torch.nn as nn


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
        x = self.fc(self.cnn(state))
        steering = torch.tanh(x[:, 0]).unsqueeze(1)  # [-1, 1]
        gas = torch.sigmoid(x[:, 1]).unsqueeze(1)  # [0, 1]
        brake = torch.sigmoid(x[:, 2]).unsqueeze(1)  # [0, 1]
        return torch.cat([steering, gas, brake], dim=1)


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
        # 1. Прогоняем картинку
        cnn_features = self.cnn(state)

        # 2. Объединяем картинку и действие
        x = torch.cat([cnn_features, action], dim=1)

        # 3. Считаем Q-value
        q_value = self.fc(x)
        return q_value
