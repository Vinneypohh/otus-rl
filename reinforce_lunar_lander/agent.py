import numpy as np
import torch
import torch.optim as optim


class Agent:
    def __init__(self, policy, device, lr=1e-3, gamma=0.99):
        self.policy = policy
        self.device = device
        self.gamma = gamma

        # Оптимизатор Adam (как мы и решили)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Память для текущего эпизода
        self.saved_log_probs = []
        self.rewards = []

    def act(self, state):
        """Выбирает действие и запоминает его log_prob"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Получаем параметры распределения
        mu, sigma = self.policy(state)

        # Создаем нормальное распределение
        dist = torch.distributions.Normal(mu, sigma)

        # Сэмплируем действие (кидаем кубик)
        action = dist.sample()

        # Сохраняем log_prob для обучения
        # sum() нужен, так как у нас вектор действий, а нужна одна скалярная вероятность
        self.saved_log_probs.append(dist.log_prob(action).sum())

        # Возвращаем действие для среды (обрезаем, чтобы не сломать физику)
        return torch.clamp(action, -1, 1).cpu().numpy()[0]

    def train(self):
        """Обучение после окончания эпизода"""
        R = 0
        policy_loss = []
        returns = []

        # --- Шаг 1: Считаем Discounted Returns (G_t) ---
        # Идем с конца эпизода в начало!
        # Так проще считать: G_t = r_t + gamma * G_{t+1}
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            # Вставляем в начало списка (так как идем с конца)
            returns.insert(0, R)

        # Превращаем в тензор
        returns = torch.tensor(returns).to(self.device)

        # --- (Опционально, но важно) Нормализация наград ---
        # Это стабилизирует обучение. Вычитаем среднее, делим на стд.
        # Чтобы награды были примерно около 0 (плюс-минус).
        eps = np.finfo(np.float32).eps.item()  # маленькое число, чтобы не делить на 0
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # --- Шаг 2: Считаем Loss ---
        # Loss = - sum(log_prob * G_t)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        # Суммируем потери со всех шагов
        # cat превращает список тензоров в один тензор, потом sum()
        optimizer_loss = torch.stack(policy_loss).sum()

        # --- Шаг 3: Обновление весов (Backprop) ---
        self.optimizer.zero_grad()
        optimizer_loss.backward()
        self.optimizer.step()

        # --- Шаг 4: Очистка памяти ---
        del self.saved_log_probs[:]
        del self.rewards[:]
