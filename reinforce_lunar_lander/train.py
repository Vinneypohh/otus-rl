import gymnasium as gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from policy_network import PolicyNetwork
from agent import Agent


CHECKPOINT_PATH = "/Users/nabandurko/repos/otus-rl/reinforce_lunar_lander/checkpoint_lunar_continuous.pth"

def train():
    # 1. Создаем среду (ОБЯЗАТЕЛЬНО continuous=True для нашей сети)
    env = gym.make('LunarLander-v3', continuous=True)
    
    # 2. Определяем параметры
    state_size = env.observation_space.shape[0]  # 8
    action_size = env.action_space.shape[0]      # 2 (Main Engine, Side Engine)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 3. Инициализируем сеть и агента
    policy = PolicyNetwork(state_size, action_size).to(device)
    agent = Agent(policy, device, lr=3e-4, gamma=0.99)

    # 4. Параметры обучения
    n_episodes = 3000          # Сколько игр сыграем
    max_t = 1000               # Максимум шагов в одной игре (чтобы не зависал)
    print_every = 100           # Как часто печатать прогресс
    
    scores_deque = deque(maxlen=100) # Очередь для хранения последних 100 результатов
    scores = []                      # Для графика

    # --- ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ ---
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(max_t):
            # Агент выбирает действие
            action = agent.act(state)
            
            # Среда реагирует
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Сохраняем награду в память агента (ВАЖНО для REINFORCE)
            agent.rewards.append(reward)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # --- МОМЕНТ ИСТИНЫ: ОБУЧЕНИЕ ---
        # Эпизод закончен. Запускаем обучение на собранной траектории
        agent.train()
        
        # Логирование
        scores_deque.append(episode_reward)
        scores.append(episode_reward)
        
        mean_score = np.mean(scores_deque)

        
        if i_episode % print_every == 0:
            print(f'\rEpisode {i_episode}\tScore: {episode_reward:.2f}\tAverage Score: {mean_score:.2f}')
        
        # Условие победы (LunarLander считается решенным при 200+)
        if mean_score >= 200.0:
            print(f'\nEnvironment solved in {i_episode} episodes! Average Score: {mean_score:.2f}')
            torch.save(policy.state_dict(), CHECKPOINT_PATH)
            break

    return scores


def plot_training_results(scores, rolling_window=100, goal=200):
    """
    Рисует красивый график обучения.
    scores: список очков за эпизоды
    rolling_window: окно для сглаживания (обычно 100)
    goal: целевое значение (для LunarLander 200)
    """
    
    # Используем стиль seaborn для красоты
    sns.set_theme(style="darkgrid")
    
    # Создаем DataFrame для удобства
    data = pd.DataFrame({'Score': scores})
    
    # Считаем скользящее среднее
    data['Average'] = data['Score'].rolling(window=rolling_window, min_periods=1).mean()
    
    # Создаем фигуру
    plt.figure(figsize=(12, 6))
    
    # 1. Рисуем "шумные" данные (реальные очки)
    # alpha=0.3 делает их прозрачными, чтобы не перекрывать главное
    plt.plot(data['Score'], label='Episode Score', color='cyan', alpha=0.3, linewidth=1)
    
    # 2. Рисуем тренд (скользящее среднее)
    # Это главная линия прогресса
    plt.plot(data['Average'], label=f'Moving Average ({rolling_window})', color='blue', linewidth=2.5)
    
    # 3. Рисуем линию цели (если достигли)
    plt.axhline(y=goal, color='red', linestyle='--', alpha=0.8, label=f'Goal ({goal})')
    
    # 4. Оформление
    plt.title('REINFORCE Training Progress', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    
    # Добавляем заливку, если награда стала положительной (визуально приятно)
    plt.fill_between(data.index, data['Average'], 0, where=(data['Average'] > 0), 
                     interpolate=True, color='green', alpha=0.1)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    scores = train()
    
    plot_training_results(scores)
