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
from env_batch import ParallelEnvBatch, EnvBatch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# === КОНФИГУРАЦИЯ ===
class Config:
    EPISODES = 2000
    NENVS = 8  # число параллельных сред (как в Practical RL: nenvs=8)
    USE_PARALLEL = (
        True  # True — ParallelEnvBatch (процессы), False — EnvBatch (последовательно)
    )
    ROLLOUT_STEPS = 20
    SCORES_WINDOW_SIZE = 100
    CHECKPOINT_PATH = str(os.path.join(ROOT_DIR, "checkpoint_a2c_car_racing.pth"))
    PLOT_PATH = str(os.path.join(ROOT_DIR, "training_plot.png"))
    LR = 3e-4
    GAMMA = 0.99
    ENTROPY_COEF = 0.05
    TAU = 0.005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_SCORE = 500


def make_env():
    """Фабрика одной среды (для батча вызывается nenvs раз)."""
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
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
    if cfg.USE_PARALLEL:
        env = ParallelEnvBatch(make_env, cfg.NENVS)
    else:
        env = EnvBatch(make_env, cfg.NENVS)
    nenvs = env.nenvs

    model = ActorCritic(action_dim=3).to(cfg.DEVICE)
    agent = A2CAgent(
        model=model,
        device=cfg.DEVICE,
        lr=cfg.LR,
        gamma=cfg.GAMMA,
        entropy_coef=cfg.ENTROPY_COEF,
        tau=cfg.TAU,
    )

    scores = []
    scores_window = deque(maxlen=cfg.SCORES_WINDOW_SIZE)
    best_avg_score = -np.inf
    running_rewards = np.zeros(nenvs)
    iteration = 0

    print(
        f"Starting training on {cfg.DEVICE} with {nenvs} envs (parallel={cfg.USE_PARALLEL})..."
    )

    try:
        obs, _ = env.reset()
        while len(scores) < cfg.EPISODES:
            for _ in range(cfg.ROLLOUT_STEPS):
                actions = agent.act_batch(obs)
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                done = terminated | truncated
                agent.push_rewards_dones(rewards, done.astype(np.float32))
                running_rewards += rewards
                for j in range(nenvs):
                    if done[j]:
                        scores.append(float(running_rewards[j]))
                        scores_window.append(float(running_rewards[j]))
                        running_rewards[j] = 0.0
                obs = next_obs

            bootstrap = agent.bootstrap_value_batch(obs)
            next_values = np.where(done, 0.0, bootstrap).astype(np.float32)
            agent.learn(next_values=next_values)
            agent.update_target()

            iteration += 1
            if len(scores) == 0:
                avg_score = 0.0
            else:
                avg_score = (
                    np.mean(scores_window) if scores_window else np.mean(scores[-100:])
                )
            n_ep = len(scores)
            print(f"\rIter {iteration}\tEpisodes {n_ep}\tAvg: {avg_score:.2f}", end="")

            if n_ep >= 20 and n_ep % 20 < nenvs:
                print(f"\rIter {iteration}\tEpisodes {n_ep}\tAvg: {avg_score:.2f}")
                plot_training_results(scores, cfg.PLOT_PATH)

            if avg_score > best_avg_score and avg_score > 0:
                best_avg_score = avg_score
                torch.save(model.state_dict(), cfg.CHECKPOINT_PATH)
                print(f"\nNew Best Model Saved! Avg Score: {best_avg_score:.2f}")

            if avg_score >= cfg.TARGET_SCORE:
                print(f"\nSolved in {n_ep} episodes!")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving plot...")
        plot_training_results(scores, cfg.PLOT_PATH)
    finally:
        if hasattr(env, "close"):
            env.close()

    return scores


if __name__ == "__main__":
    config = Config()
    train(config)
