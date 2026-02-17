import gymnasium as gym
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
from typing import List, Callable

from a2c_model import ActorCritic
from a2c_agent import A2CAgent
from wrappers import CarRacingWrapper

# === КОНФИГУРАЦИЯ ===
class Config:
    EPISODES = 2000
    NENVS = 8              
    ROLLOUT_STEPS = 20     # N-step learning
    SCORES_WINDOW_SIZE = 100
    CHECKPOINT_PATH = "checkpoint_a2c_car_racing.pth"
    PLOT_PATH = "training_plot.png"
    LR = 3e-4
    GAMMA = 0.99
    ENTROPY_COEF = 0.05
    TAU = 0.005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_SCORE = 500

# === ФАБРИКА СРЕД ===
def make_env() -> Callable:
    def _thunk():
        env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=True,
        )
        env = CarRacingWrapper(env, stack_frames=4)
        return env
    return _thunk

def plot_training_results(scores: List[float], filename: str):
    sns.set_theme(style="darkgrid")
    data = pd.DataFrame({"Score": scores})
    data["Average"] = data["Score"].rolling(window=100, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(data["Score"], label="Episode Score", alpha=0.3, color="cyan")
    plt.plot(data["Average"], label="Moving Average (100)", color="blue", linewidth=2)
    plt.axhline(y=Config.TARGET_SCORE, color="red", linestyle="--", label="Target")
    plt.title("A2C CarRacing Training")
    plt.legend()
    plt.savefig(filename)
    plt.close()

# === ГЛАВНЫЙ ЦИКЛ ===
def train(cfg: Config):
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(cfg.NENVS)])
    
    print(f"Training on {cfg.DEVICE} with {cfg.NENVS} envs...")

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
    
    # Трекер текущих наград для каждой среды
    current_episode_rewards = np.zeros(cfg.NENVS)
    
    # Счетчик общих эпизодов (сумма по всем средам)
    total_episodes_done = 0

    try:
        # Сброс всех сред сразу
        obs, _ = envs.reset()
        
        while total_episodes_done < cfg.EPISODES:
            # 1. Сбор данных (Rollout)
            for _ in range(cfg.ROLLOUT_STEPS):
                # Агент выбирает действия для ВСЕХ сред сразу
                actions = agent.act_batch(obs)
                
                # Шаг во всех средах
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                dones = terminations | truncations
                
                # Сохраняем награды для обучения
                agent.push_rewards_dones(rewards, dones.astype(np.float32))
                
                # Обновляем статистику
                current_episode_rewards += rewards
                
                # Если среда завершилась, gym.vector.AsyncVectorEnv САМ делает reset!
                # Нам нужно только записать результат.
                for i, done in enumerate(dones):
                    if done:
                        final_score = current_episode_rewards[i]
                        scores.append(final_score)
                        scores_window.append(final_score)
                        current_episode_rewards[i] = 0.0
                        total_episodes_done += 1
                        
                        # Логирование
                        avg = np.mean(scores_window)
                        print(f"\rEp: {total_episodes_done}\tScore: {final_score:.1f}\tAvg: {avg:.1f}", end="")
                
                obs = next_obs

            # 2. Обучение (Update)
            # Считаем Bootstrap Value для последних состояний
            bootstrap_values = agent.bootstrap_value_batch(obs)
            
            # Если среда только что закончилась (done=True), то V(s') = 0.
            # Но векторный env уже сбросил среду и вернул s_new (начало новой игры).
            # В идеале нужно брать V(s_terminal) из infos, но для A2C часто просто берут V(s_new)*(1-done).
            # Пока оставим твою логику:
            # next_values = np.where(dones, 0.0, bootstrap_values)
            # Внимание: dones тут от ПОСЛЕДНЕГО шага роллаута.
            
            # Правильнее так:
            # Если на последнем шаге done, то мы не бутстрапим (0).
            # Если не done, то бутстрапим от V(obs).
            # Переменная dones у нас есть с конца цикла for.
            next_values = np.where(dones, 0.0, bootstrap_values).astype(np.float32)

            loss = agent.learn(next_values=next_values)
            agent.update_target()
            
            # Периодические проверки
            if total_episodes_done > 0 and total_episodes_done % 20 == 0:
                plot_training_results(scores, cfg.PLOT_PATH)
                avg = np.mean(scores_window)
                if avg > best_avg_score and avg > 0:
                    best_avg_score = avg
                    torch.save(model.state_dict(), cfg.CHECKPOINT_PATH)
                    print(f"\nSaved Best Model: {best_avg_score:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        envs.close()
        plot_training_results(scores, cfg.PLOT_PATH)

if __name__ == "__main__":
    train(Config())
