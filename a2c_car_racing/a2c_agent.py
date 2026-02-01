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
        self.rewards = []  # список массивов (nenvs,) на каждый шаг
        self.dones = []  # список массивов (nenvs,) bool на каждый шаг
        self.entropies = []

    def act(self, state):
        """
        Один env: state (4, 84, 84) -> action (3,).
        """
        actions = self.act_batch(state[None, ...])
        return actions[0]

    def act_batch(self, states):
        """
        Батч env: states (nenvs, 4, 84, 84) -> actions (nenvs, 3).
        Сохраняет log_probs, values, entropies для последующего learn().
        """
        state_t = torch.FloatTensor(states).to(self.device)
        mu, sigma, value = self.model(state_t)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        self.log_probs.append(log_prob)
        self.values.append(value.squeeze(-1))
        self.entropies.append(entropy)

        action_np = action.cpu().detach().numpy()
        action_np[..., 0] = np.clip(action_np[..., 0], -1, 1)
        action_np[..., 1] = np.clip(action_np[..., 1], 0, 1)
        action_np[..., 2] = np.clip(action_np[..., 2], 0, 1)
        return action_np

    def bootstrap_value(self, state: np.ndarray) -> float:
        """Один env: V_target(s) для bootstrap."""
        return float(self.bootstrap_value_batch(state[None, ...])[0])

    def bootstrap_value_batch(self, states: np.ndarray) -> np.ndarray:
        """Батч: states (nenvs, 4, 84, 84) -> V_target(s) (nenvs,)."""
        state_t = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            _, _, v = self.target_model(state_t)
        return v.cpu().numpy().ravel()

    def update_target(self):
        """Мягкое обновление целевой сети (Polyak): θ_target = τ*θ_target + (1-τ)*θ."""
        for target_p, p in zip(self.target_model.parameters(), self.model.parameters()):
            target_p.data.copy_(self.tau * target_p.data + (1.0 - self.tau) * p.data)

    def push_rewards_dones(self, rewards: np.ndarray, dones: np.ndarray):
        """Добавить награды и флаги done за шаг (для батча). rewards, dones: (nenvs,)."""
        self.rewards.append(rewards)
        self.dones.append(dones)

    def learn(self, next_value=0, next_values=None):
        """
        Считает Loss и обновляет веса.
        Один env: next_value — скаляр V(s_last), rewards уже в self.rewards как скаляры.
        Батч: next_values (nenvs,) — bootstrap по env (0 если done), rewards/dones через push_rewards_dones.
        """
        if next_values is not None:
            return self._learn_batch(next_values)
        # Один env: rewards — список скаляров, dones нет (считаем все False)
        nsteps = len(self.rewards)
        rewards = np.array(self.rewards, dtype=np.float32).reshape(nsteps, 1)
        dones = np.zeros((nsteps, 1), dtype=np.float32)
        next_vals = np.array([next_value], dtype=np.float32)
        self.rewards.clear()
        self.dones.clear()
        return self._learn_batch_from_arrays(rewards, dones, next_vals)

    def _learn_batch(self, next_values: np.ndarray):
        """next_values: (nenvs,) — bootstrap для каждого env в конце rollout."""
        rewards = np.stack(self.rewards, axis=0)
        dones = np.stack(self.dones, axis=0)
        self.rewards.clear()
        self.dones.clear()
        return self._learn_batch_from_arrays(rewards, dones, next_values)

    def _learn_batch_from_arrays(
        self, rewards: np.ndarray, dones: np.ndarray, next_values: np.ndarray
    ):
        """rewards, dones: (nsteps, nenvs); next_values: (nenvs,)."""
        nsteps, nenvs = rewards.shape

        returns = np.zeros((nsteps, nenvs), dtype=np.float32)
        R = next_values.copy()
        for t in reversed(range(nsteps)):
            R = rewards[t] + self.gamma * R * (1.0 - dones[t].astype(np.float32))
            returns[t] = R

        returns_t = torch.FloatTensor(returns).to(self.device)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)
        returns_flat = returns_t.view(-1)
        log_probs_flat = log_probs.view(-1)
        values_flat = values.view(-1)
        entropies_flat = entropies.view(-1)

        advantage = returns_flat - values_flat.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(log_probs_flat * advantage).mean()
        critic_loss = F.mse_loss(returns_flat, values_flat)
        entropy_loss = -entropies_flat.mean()
        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        self.log_probs.clear()
        self.values.clear()
        self.entropies.clear()
        return total_loss.item()
