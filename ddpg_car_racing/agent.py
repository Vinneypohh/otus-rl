import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import Actor, Critic
from utils import ReplayBuffer, OUNoise


class DDPGAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim

        # 1. Основные сети
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)

        # 2. Target сети (клоны)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        # Копируем веса 1-в-1 в начале
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 3. Оптимизаторы
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=1e-3
        )  # Critic учится быстрее

        self.memory = ReplayBuffer(100000)
        self.noise = OUNoise(action_dim)

        self.gamma = 0.99
        self.tau = 0.005  # Коэффициент мягкого обновления (0.5%)

    def act(self, state, add_noise=True):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().data.numpy()[0]
        self.actor.train()

        if add_noise:
            noise = self.noise.sample()
            action += noise

        # Обязательно обрезаем (clip), чтобы не выйти за рамки физики
        # Руль: -1..1, Газ/Тормоз: 0..1 (хотя sigmoid и так 0..1, но шум может вывести за пределы)
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0, 1)
        action[2] = np.clip(action[2], 0, 1)

        return action

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            batch_size, self.device
        )

        # --- 1. Обновление Critic ---

        with torch.no_grad():
            # а) Какое действие выберет TargetActor для next_state?
            next_actions = self.actor_target(next_states)

            # б) Какую оценку даст TargetCritic этому действию?
            target_Q_values = self.critic_target(next_states, next_actions)

            # в) Считаем Target Q (Беллман)
            # target = reward + gamma * target_Q * (1 - done)
            target_Q = rewards + (self.gamma * target_Q_values * (1 - dones))

        # г) Текущая оценка Critic-а
        current_Q = self.critic(states, actions)

        # д) Loss Критика (MSE)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 2. Обновление Actor ---

        # а) Какое действие СЕЙЧАС выбрал бы Actor для этих states?
        # Важно: тут нам НУЖНЫ градиенты, поэтому без no_grad()
        actions_pred = self.actor(states)

        # б) Какую оценку даст Critic этим действиям?
        # Нам нужно МАКСИМИЗИРОВАТЬ это число. Значит минимизировать минус.
        actor_loss = -self.critic(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 3. Soft Update Target Networks ---
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, local_model, target_model):
        """
        θ_target = τ*θ_local + (1 - τ)*θ_target
        local_model: модель, которую мы только что обучили (быстрая)
        target_model: модель, которая служит эталоном (медленная)
        """
        # Итерируемся по всем весам (параметрам) обеих сетей
        for local_param, target_param in zip(
            local_model.parameters(), target_model.parameters()
        ):
            # target_param.data — доступ к тензору весов напрямую
            # copy_ — копирует значения внутрь тензора
            # Формула: берем 0.5% (tau) от новых весов и 99.5% от старых весов Target-а.
            # Это делает изменение Target-сети очень плавным.
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
