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
LOG_FILE = os.path.join(CHECKPOINT_DIR, "ddpg_training.log")
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


def train():
    setup_logging()

    env = gym.make(
        "CarRacing-v3",
        continuous=True,
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
    )
    env = CarRacingWrapper(env, stack_frames=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

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

    try:
        for i_episode in range(1, EPISODES + 1):
            state, _ = env.reset()
            score = 0
            agent.noise.reset()

            for t in range(MAX_STEPS):
                action = agent.act(state, add_noise=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.memory.add(state, action, reward, next_state, done)
                agent.learn(BATCH_SIZE)

                state = next_state
                score += reward

                if done:
                    break

            scores_window.append(score)
            scores.append(score)
            avg_score = np.mean(scores_window)

            print(f"\rEpisode {i_episode}\tScore: {score:.2f}\tAvg: {avg_score:.2f}", end="", flush=True)

            if i_episode % 20 == 0:
                print(f"\rEpisode {i_episode}\tScore: {score:.2f}\tAvg: {avg_score:.2f}")
                logger.info(f"Episode {i_episode}, Avg: {avg_score:.2f}")
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

            agent.noise.sigma = max(0.1, agent.noise.sigma * 0.999)

        if np.mean(scores_window) < TARGET_SCORE:
            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
        logger.warning("Training interrupted by user")

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
