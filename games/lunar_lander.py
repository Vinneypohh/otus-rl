import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np


def manual_control():
    print("Запускаем ручное управление LunarLander Continuous!")
    print("Управление: [W] - Газ, [A] - Влево, [D] - Вправо")

    # Создаем среду
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")

    # ВАЖНО: dtype=np.float32, иначе gymnasium ругается на несовпадение типов
    mapping = {
        (ord("w"),): np.array([1.0, 0.0], dtype=np.float32),
        (ord("a"),): np.array([0.0, -1.0], dtype=np.float32),
        (ord("d"),): np.array([0.0, 1.0], dtype=np.float32),
        (ord("w"), ord("a")): np.array([1.0, -1.0], dtype=np.float32),
        (ord("w"), ord("d")): np.array([1.0, 1.0], dtype=np.float32),
    }

    play(
        env,
        keys_to_action=mapping,
        noop=np.array([-1.0, 0.0], dtype=np.float32),
        fps=30,
    )


if __name__ == "__main__":
    manual_control()
