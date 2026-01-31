import gymnasium as gym
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
from typing import List


from a2c_model import ActorCritic
from a2c_agent import A2CAgent
from wrappers import CarRacingWrapper

ROOT_DIR = "/Users/nabandurko/repos/otus-rl/a2c_car_racing"


# === КОНФИГУРАЦИЯ ===
class Config:
    EPISODES = 2000
    ROLLOUT_STEPS = 20
    SCORES_WINDOW_SIZE = 100
    CHECKPOINT_PATH = str(os.path.join(ROOT_DIR, "checkpoint_a2c_car_racing.pth"))
    PLOT_PATH = str(os.path.join(ROOT_DIR, "training_plot.png"))
    LR = 3e-4
    GAMMA = 0.99
    ENTROPY_COEF = 0.05
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_SCORE = 500  # Цель для сохранения


def create_env():
    """Создает и оборачивает среду."""
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,  # False для начала, True для усложнения потом
        continuous=True,
    )
    env = CarRacingWrapper(env, stack_frames=4)
    return env


def plot_training_results(scores: List[float], filename: str):
    """Рисует и сохраняет график обучения."""
    sns.set_theme(style="darkgrid")

    data = pd.DataFrame({"Score": scores})
    data["Average"] = data["Score"].rolling(window=100, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(data["Score"], label="Episode Score", alpha=0.3, color="cyan")
    plt.plot(data["Average"], label="Moving Average (100)", color="blue", linewidth=2)
    plt.axhline(y=Config.TARGET_SCORE, color="red", linestyle="--", label="Target")

    plt.title("A2C CarRacing Training")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # Закрываем, чтобы не висело в памяти
    print(f"Plot saved to {filename}")


# === ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ ===


def train(cfg: Config):
    env = create_env()

    # Инициализация модели и агента
    # Переносим на девайс сразу
    model = ActorCritic(action_dim=3).to(cfg.DEVICE)

    agent = A2CAgent(
        model=model,
        device=cfg.DEVICE,
        lr=cfg.LR,
        gamma=cfg.GAMMA,
        entropy_coef=cfg.ENTROPY_COEF,
    )

    scores = []  # История всех очков
    scores_window = deque(maxlen=cfg.SCORES_WINDOW_SIZE)
    best_avg_score = -np.inf

    print(f"Starting training on {cfg.DEVICE}...")

    try:
        for i_episode in range(1, cfg.EPISODES + 1):
            state, _ = env.reset()
            score = 0.0
            done = False

            while not done:
                # 1. Сбор Rollout
                for _ in range(cfg.ROLLOUT_STEPS):
                    action = agent.act(state)

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # Важно: В CarRacing за каждый кадр дают -0.1 очко.
                    # Можно чуть подкрутить награду тут (reward clipping), если хочется.
                    # Но пока оставим как есть.
                    agent.rewards.append(reward)

                    state = next_state
                    score += reward

                    if done:
                        break

                # 2. Bootstrap (Хвост)
                if done:
                    next_val = 0
                else:
                    # state -> Tensor -> Model -> Value
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(cfg.DEVICE)
                    # Используем torch.no_grad(), чтобы не копить градиенты тут
                    with torch.no_grad():
                        _, _, next_val = model(state_t)
                        next_val = next_val.item()

                # 3. Обучение
                agent.learn(next_value=next_val)

            # Эпизод завершен
            scores_window.append(score)
            scores.append(score)
            avg_score = np.mean(scores_window)

            # Логирование
            print(
                f"\rEpisode {i_episode}\tScore: {score:.2f}\tAvg: {avg_score:.2f}",
                end="",
            )

            # Сохранение и отчет каждые 20 эпизодов
            if i_episode % 20 == 0:
                print(
                    f"\rEpisode {i_episode}\tScore: {score:.2f}\tAvg: {avg_score:.2f}"
                )
                plot_training_results(scores, cfg.PLOT_PATH)

            # Сохраняем лучшую модель (если она реально хорошая, > 0)
            if avg_score > best_avg_score and avg_score > 0:
                best_avg_score = avg_score
                torch.save(model.state_dict(), cfg.CHECKPOINT_PATH)
                print(f"\nNew Best Model Saved! Avg Score: {best_avg_score:.2f}")

            if avg_score >= cfg.TARGET_SCORE:
                print(f"\nSolved in {i_episode} episodes!")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving plot...")
        plot_training_results(scores, cfg.PLOT_PATH)

    return scores


if __name__ == "__main__":
    config = Config()
    train(config)
