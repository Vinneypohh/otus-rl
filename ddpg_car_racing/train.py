import os
import sys
import numpy as np
import gymnasium as gym
from loguru import logger
import torch
import matplotlib.pyplot as plt
from collections import deque

from agent import DDPGAgent
from utils import CarRacingWrapper
from env_batch import ParallelEnvBatch, EnvBatch

# Конфиг
BATCH_SIZE = 64
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = 100_000
EPISODES = 1000
MAX_STEPS = 1000
TARGET_SCORE = 500
CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(CHECKPOINT_DIR, "checkpoints", "ddpg_training.log")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def setup_logging():
    """Логи в консоль (stderr) и в файл."""
    logger.remove()
    fmt_console = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    fmt_file = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    logger.add(sys.stderr, format=fmt_console, level="INFO")
    logger.add(
        LOG_FILE,
        format=fmt_file,
        level="DEBUG",
        rotation="10 MB",
        retention="3 days",
    )
    logger.info(f"Logging: console + file {LOG_FILE}")

# Ускорение: батч сред и несколько шагов обучения за rollout
NENVS = 8
USE_PARALLEL = True
ROLLOUT_STEPS = 10
LEARN_STEPS = 4


def make_env():
    env = gym.make(
        "CarRacing-v3",
        continuous=True,
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
    )
    return CarRacingWrapper(env, stack_frames=4)


def train():
    setup_logging()

    if USE_PARALLEL:
        env = ParallelEnvBatch(make_env, NENVS)
    else:
        env = EnvBatch(make_env, NENVS)
    nenvs = env.nenvs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, nenvs: {nenvs}, parallel: {USE_PARALLEL}")

    agent = DDPGAgent(
        state_dim=4,
        action_dim=3,
        device=device,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        tau=TAU,
        memory_size=MEMORY_SIZE,
    )

    best_score = -100
    scores = []
    scores_window = deque(maxlen=100)
    running_rewards = np.zeros(nenvs)

    try:
        obs, _ = env.reset()
        total_episodes = 0

        while total_episodes < EPISODES:
            for _ in range(ROLLOUT_STEPS):
                actions = agent.act_batch(obs, add_noise=True)
                action_list = [actions[j] for j in range(nenvs)]
                next_obs, rewards, terminated, truncated, _ = env.step(action_list)
                done = terminated | truncated

                for j in range(nenvs):
                    agent.memory.add(
                        obs[j].copy(),
                        actions[j].copy(),
                        float(rewards[j]),
                        next_obs[j].copy(),
                        bool(done[j]),
                    )
                    running_rewards[j] += rewards[j]
                    if done[j]:
                        scores.append(float(running_rewards[j]))
                        scores_window.append(float(running_rewards[j]))
                        total_episodes += 1
                        running_rewards[j] = 0.0

                obs = next_obs

            for _ in range(LEARN_STEPS):
                agent.learn(BATCH_SIZE)

            if total_episodes >= 20:
                agent.noise.sigma = max(0.1, agent.noise.sigma * 0.9995)

            if len(scores_window) > 0:
                avg_score = np.mean(scores_window)
                print(f"\rEpisodes {total_episodes}\tAvg: {avg_score:.2f}", end="", flush=True)

                if total_episodes % 20 < nenvs:
                    print(f"\rEpisodes {total_episodes}\tAvg: {avg_score:.2f}")
                    logger.info(f"Episodes {total_episodes}, Avg: {avg_score:.2f}")
                    torch.save(
                        agent.actor.state_dict(),
                        os.path.join(CHECKPOINT_DIR, "actor_checkpoint.pth"),
                    )
                    torch.save(
                        agent.critic.state_dict(),
                        os.path.join(CHECKPOINT_DIR, "critic_checkpoint.pth"),
                    )

                if avg_score > best_score:
                    best_score = avg_score
                    torch.save(
                        agent.actor.state_dict(),
                        os.path.join(CHECKPOINT_DIR, "best_actor.pth"),
                    )
                    torch.save(
                        agent.critic.state_dict(),
                        os.path.join(CHECKPOINT_DIR, "best_critic.pth"),
                    )
                    logger.info(f"New Best Score: {best_score:.2f} -> Model Saved!")

                if avg_score >= TARGET_SCORE:
                    logger.success(
                        f"Solved! Average score over 100 episodes: {avg_score:.2f}"
                    )
                    torch.save(
                        agent.actor.state_dict(),
                        os.path.join(CHECKPOINT_DIR, "solved_actor.pth"),
                    )
                    break

        if len(scores_window) > 0 and np.mean(scores_window) < TARGET_SCORE:
            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
        logger.warning("Training interrupted by user")
    finally:
        if hasattr(env, "close"):
            env.close()

    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.3, color="cyan")
    if len(scores) >= 100:
        avg = np.convolve(scores, np.ones(100) / 100, mode="valid")
        plt.plot(range(99, len(scores)), avg, color="blue", linewidth=2, label="MA(100)")
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "ddpg_scores.png"))
    plt.close()
    logger.info("Plot saved to ddpg_scores.png")

    return scores


if __name__ == "__main__":
    train()
