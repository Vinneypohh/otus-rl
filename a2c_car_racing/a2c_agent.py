import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy


class A2CAgent:
    """
    A2C с целевой сетью для критика (target network).
    Bootstrap: V_target(s') вместо V(s') — убирает «chasing the target» и стабилизирует обучение.
    По идее из Practical RL: follow-up papers используют target networks для value estimation.
    """

    def __init__(
        self,
        model,
        device,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        tau=0.005,
    ):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.tau = tau  # коэффициент мягкого обновления target: θ_target = τ*θ_target + (1-τ)*θ

        # Целевая сеть (копия модели) — используется только для bootstrap value
        self.target_model = copy.deepcopy(model).to(device)
        for p in self.target_model.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def act(self, state):
        """
        Принимает: state (4, 84, 84) - numpy array
        Возвращает: action (3,) - numpy array для среды
        """
        # 1. Готовим тензор (добавляем Batch dimension: 1, 4, 84, 84)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 2. Прогон через модель
        # model возвращает (mu, sigma, value)
        mu, sigma, value = self.model(state_t)

        # 3. Создаем распределение
        # Используем Normal (Гаусс) для непрерывных действий
        dist = torch.distributions.Normal(mu, sigma)

        # 4. Сэмплируем действие (с gradients!)
        # Используем rsample() или sample()?
        # Для A2C достаточно sample(), так как градиент идет через log_prob.
        action = dist.sample()

        # 5. Считаем log_prob и entropy
        # action имеет форму (1, 3). Нам нужно одно число для всего шага.
        # Поэтому суммируем логарифмы вероятностей руля, газа и тормоза.
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # 6. Сохраняем в память
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)

        # 7. Возвращаем действие для среды
        # Важно: Tanh уже был в модели (для mu), но сэмплирование могло выкинуть за -1..1
        # Поэтому для надежности можно сделать clamp, но только для numpy версии!
        # Не ломай тензор action, он нужен для градиентов (хотя в A2C градиент идет через log_prob, так что тут безопасно).
        action_np = action.cpu().detach().numpy()[0]

        # ВАЖНО: CarRacing ожидает:
        # 0: Руль [-1, 1]
        # 1: Газ [0, 1]
        # 2: Тормоз [0, 1]
        # У нас нейросеть выдает всё в диапазоне Tanh (-1..1) или около того.
        # Газ и тормоз не могут быть отрицательными!
        # Давай сделаем простой хак здесь:
        action_np[0] = np.clip(action_np[0], -1, 1)  # Руль
        action_np[1] = np.clip(action_np[1], 0, 1)  # Газ
        action_np[2] = np.clip(action_np[2], 0, 1)  # Тормоз

        return action_np

    def bootstrap_value(self, state: np.ndarray) -> float:
        """Считает V_target(s) для bootstrap — без градиентов, через целевую сеть."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, v = self.target_model(state_t)
        return v.item()

    def update_target(self):
        """Мягкое обновление целевой сети (Polyak): θ_target = τ*θ_target + (1-τ)*θ."""
        for target_p, p in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_p.data.copy_(self.tau * target_p.data + (1.0 - self.tau) * p.data)

    def learn(self, next_value=0):
        """
        Считает Loss и обновляет веса.
        next_value: V(s_last), предсказание для последнего состояния (если не Done)
        """

        # --- 1. Считаем Returns (G_t) ---
        returns = []
        R = next_value  # Начинаем с хвоста (bootstrap)

        # Идем с конца в начало по self.rewards
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)

        # Превращаем списки в тензоры
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze()  # (N, 1) -> (N)
        entropies = torch.stack(self.entropies)

        # --- 2. Считаем Advantage ---
        advantage = returns - values.detach()
        # Нормализация стабилизирует градиенты (best practice A2C/PPO)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # --- 3. Считаем Лоссы ---

        # Actor Loss: - mean(log_prob * advantage)
        actor_loss = -(log_probs * advantage).mean()

        # Critic Loss: MSE(returns, values)
        # Тут values нужны С градиентом!
        # Можно использовать F.mse_loss(values, returns) или ручками
        critic_loss = F.mse_loss(returns, values)

        # Entropy Loss: - mean(entropy)
        entropy_loss = -entropies.mean()

        # Total Loss
        # loss = actor + 0.5 * critic + coef * entropy_loss
        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss

        # --- 4. Обновление ---
        self.optimizer.zero_grad()
        total_loss.backward()

        # Обрезаем градиенты (защита от взрыва)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

        self.optimizer.step()

        # Очистка памяти
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()

        return total_loss.item()
