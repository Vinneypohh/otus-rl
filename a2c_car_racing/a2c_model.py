import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, action_dim=3):
        super(ActorCritic, self).__init__()

        # === 1. ОБЩИЙ ЭНКОДЕР (CNN) ===
        # Обрабатывает картинку (4, 84, 84) в вектор признаков
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Размер выхода CNN: 64 канала * 7 * 7 = 3136
        self.fc_input_dim = 64 * 7 * 7
        
        # Общий полносвязный слой (можно разделить, но часто делают общим)
        self.fc_common = nn.Linear(self.fc_input_dim, 512)

        # === 2. ГОЛОВА ACTOR (Среднее и Std) ===
        # mean: предсказывает среднее значение действий
        self.actor_mean = nn.Linear(512, action_dim)
        
        # logstd: предсказывает логарифм стандартного отклонения (ширину колокола)
        # Можно сделать его зависимым от картинки (nn.Linear), а можно глобальным (nn.Parameter).
        # Препод в примере использовал nn.Parameter(torch.zeros), что делает шум одинаковым для всех картинок,
        # но обучаемым. Это классический подход (State-Independent LogStd).
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # === 3. ГОЛОВА CRITIC (Value) ===
        # value: предсказывает ценность состояния V(s) - одно число
        self.critic_value = nn.Linear(512, 1)

    def forward(self, x):
        # x: (Batch, 4, 84, 84)
        
        # 1. Нормализация (важно!)
        x = x / 255.0
        
        # 2. Проход по CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 3. Выпрямление
        x = x.view(-1, self.fc_input_dim)
        
        # 4. Общий слой
        x = F.relu(self.fc_common(x))
        
        # 5. Головы
        
        # -- Actor --
        # Mean не ограничиваем Tanh-ом тут! Tanh делаем ПОСЛЕ сэмплирования (в агенте),
        # иначе распределение Normal(tanh(mu), std) будет искаженным.
        # Пусть нейросеть выдает "сырые" значения (логиты).
        mu = self.actor_mean(x)
        
        # Std берем через экспоненту, чтобы было всегда > 0
        sigma = self.actor_logstd.exp()
        # Если logstd - параметр, то он имеет размер (3,), а нам нужно (Batch, 3).
        # PyTorch сам сделает broadcasting, так что всё ок.

        # -- Critic --
        value = self.critic_value(x)

        return mu, sigma, value
