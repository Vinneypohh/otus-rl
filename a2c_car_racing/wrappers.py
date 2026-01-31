import gymnasium as gym
import numpy as np
import cv2
from collections import deque


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
