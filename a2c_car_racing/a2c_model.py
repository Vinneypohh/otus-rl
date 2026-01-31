import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, action_dim=3):
        super(ActorCritic, self).__init__()

        # === ТЕЛО (CNN) ===
        # Вход: (4, 84, 84)
        # 1. Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)

        # 2. Conv2d(32 -> 64, kernel=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)

        # 3. Conv2d(64 -> 64, kernel=3, stride=1)
        # Твой код:
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        # 4. Полносвязный слой (Linear).
        # Чтобы узнать входной размер, нужно посчитать выход CNN.
        # Подсказка: 84 -> (k=8,s=4) -> 20 -> (k=4,s=2) -> 9 -> (k=3,s=1) -> 7
        # Итоговая карта признаков: 64 канала * 7 * 7
        self.fc_input_dim = 64 * 7 * 7

        self.fc1 = nn.Linear(self.fc_input_dim, 512)

        # === ГОЛОВА ACTOR (Как раньше) ===
        # Для continuous actions нам нужны mu (среднее) и sigma (разброс)
        self.actor_mu = nn.Linear(512, action_dim)
        self.actor_sigma = nn.Linear(512, action_dim)

        # === ГОЛОВА CRITIC ===
        # Выдает одно число V(s)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        # x shape: (Batch, 4, 84, 84)
        # Не забываем делить на 255! (нормализация)
        x = x / 255.0

        # Проход через CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten (выпрямление)
        x = x.view(-1, self.fc_input_dim)

        # Общий скрытый слой
        x = F.relu(self.fc1(x))

        # Головы
        # Mu - это tanh (так как действия ограничены -1..1 или 0..1)
        # Но у нас разные диапазоны! Руль [-1, 1], Газ [0, 1].
        # Пока сделаем просто Tanh, а потом разберемся с диапазонами в агенте.
        mu = torch.tanh(self.actor_mu(x))

        # Sigma - должна быть положительной (Softplus)
        sigma = F.softplus(self.actor_sigma(x)) + 1e-5

        # Value
        value = self.critic(x)

        return mu, sigma, value
