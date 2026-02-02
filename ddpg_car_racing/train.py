import os
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
MEMORY_SIZE = 100000
EPISODES = 1000
MAX_STEPS = 1000
TARGET_SCORE = 500  
CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train():
    # Создаем среду
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    # Важно: DDPG тоже нужен StackFrames, чтобы понимать скорость и поворот
    env = CarRacingWrapper(env, stack_frames=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Создаем агента
    agent = DDPGAgent(state_dim=4, action_dim=3, device=device)

    best_score = -100
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        score = 0
        agent.noise.reset()  # Сбрасываем шум в начале эпизода

        for t in range(MAX_STEPS):
            # 1. Выбор действия (с шумом)
            action = agent.act(state, add_noise=True)

            # 2. Шаг среды
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. Сохраняем в память
            agent.memory.add(state, action, reward, next_state, done)

            # 4. Учимся (если накопили данных)
            # Можно учиться не каждый шаг, а например раз в 2 шага, но обычно каждый шаг ок.
            agent.learn(BATCH_SIZE)

            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)
        avg_score = np.mean(scores_window)

        print(f"\rEpisode {i_episode}\tScore: {score:.2f}\tAvg Score: {avg_score:.2f}", end="")

        if i_episode % 20 == 0:
            logger.info(f"Checkpoint saved at episode {i_episode}")
            torch.save(
                agent.actor.state_dict(), f"{CHECKPOINT_DIR}/actor_checkpoint.pth"
            )
            torch.save(
                agent.critic.state_dict(), f"{CHECKPOINT_DIR}/critic_checkpoint.pth"
            )

        if avg_score > best_score:
            best_score = avg_score
            torch.save(agent.actor.state_dict(), f"{CHECKPOINT_DIR}/best_actor.pth")
            torch.save(agent.critic.state_dict(), f"{CHECKPOINT_DIR}/best_critic.pth")
            logger.info(f"New Best Score: {best_score:.2f} -> Model Saved!")
        
        
        if avg_score >= TARGET_SCORE:
            logger.success(f"Solved! Average score over 100 episodes: {avg_score:.2f}")
            torch.save(agent.actor.state_dict(), f"{CHECKPOINT_DIR}/solved_actor.pth")
            break

        # Уменьшаем шум со временем
        agent.noise.sigma = max(0.1, agent.noise.sigma * 0.999)

    # Рисуем график
    plt.plot(scores)
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.savefig("ddpg_scores.png")
    plt.show()


if __name__ == "__main__":
    train()
