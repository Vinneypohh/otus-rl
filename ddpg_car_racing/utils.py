from collections import deque
import cv2
import gymnasium as gym
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        # 1. Взять случайные N элементов
        batch = random.sample(self.buffer, batch_size)

        # 2. Распаковать их в отдельные списки
        states, actions, rewards, next_states, dones = zip(*batch)

        # 3. Превратить в torch.Tensor и закинуть на device
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class CarRacingWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames=4):
        super().__init__(env)
        self.stack_frames = stack_frames
        self.frames = deque(maxlen=stack_frames)

        # На выходе будет массив (4, 84, 84) - 4 кадра по 84x84
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(stack_frames, 84, 84), dtype=np.uint8
        )

    def _process_frame(self, frame):
        frame = frame[:84, :, :]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, (84, 84))

        return frame

    # --- Остальное я написал за тебя, это стандартная логика стекинга ---
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        processed = self._process_frame(state)
        for _ in range(self.stack_frames):  # Заполняем буфер дублями первого кадра
            self.frames.append(processed)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        processed = self._process_frame(state)
        self.frames.append(processed)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info
