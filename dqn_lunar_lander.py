import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# from agents.dqn_agent import Agent

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

# from models.dqn_model import DQN
from utils.replay_buffer import ReplayBuffer

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

from models.dqn_model import DQN
from utils.replay_buffer import ReplayBuffer


import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        
        # 1. Первый слой: принимает 'state_size', отдает 'hidden_size'
        self.fc1 = nn.Linear(state_size, hidden_size)
        
        # 2. Второй слой: принимает 'hidden_size', отдает 'hidden_size'
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # 3. Выходной слой: принимает 'hidden_size', отдает 'action_size'
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        """
        Проход данных через сеть.
        x - тензор состояния (batch_size, state_size)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) # На выходе чистые значения Q-values

import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=10000):
        """
        capacity: максимальный размер буфера.
        Когда переполняется, старые воспоминания удаляются.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Добавляет один переход (опыт) в память.
        
        state: текущее состояние (numpy array или list)
        action: выбранное действие (int)
        reward: полученная награда (float)
        next_state: следующее состояние
        done: закончилась ли игра (bool)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Берет случайную пачку (batch) из памяти.
        
        Возвращает:
        - states: тензор состояний (batch_size, state_size)
        - actions: тензор действий (batch_size,)
        - rewards: тензор наград (batch_size,)
        - next_states: тензор следующих состояний
        - dones: тензор флагов завершения (batch_size,)
        """
        # Берем batch_size случайных элементов
        batch = random.sample(self.buffer, batch_size)
        
        # Распаковываем в отдельные списки
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Конвертируем в PyTorch тензоры
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Возвращает текущий размер буфера."""
        return len(self.buffer)



class Agent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Гиперпараметры
        self.gamma = 0.99    # дисконт
        self.tau = 0.001     # мягкое обновление
        self.lr = 0.001     # скорость обучения
        self.batch_size = 64
        
        # 1. Создаем сети
        self.policy_net: DQN = DQN(state_size, action_size).to(device)
        self.target_net: DQN = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Копируем веса
        self.target_net.eval() # Замораживаем
        
        # 2. Оптимизатор
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # 3. Память
        self.memory = ReplayBuffer(capacity=100000)

    def act(self, state, eps=0.0): 
        if random.random() < eps:
            return random.choice(np.arange(self.action_size))
        
        if isinstance(state, np.ndarray):
             state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # 3. Теперь state - это точно Тензор. Можно кормить сети.
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        return np.argmax(action_values.cpu().data.numpy())
            

    def step(self, state, action, reward, next_state, done):
        """
        1. Сохраняет опыт в память.
        2. Если в памяти > batch_size элементов -> запускает обучение.
        """
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_expected = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            # Берем max Q для следующих состояний
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            # Формула Беллмана: R + gamma * max(Q_next)
            q_targets = rewards.unsqueeze(1) + (self.gamma * q_next * (1 - dones.unsqueeze(1)))

        loss = F.mse_loss(q_expected, q_targets)
        
        self.optimizer.zero_grad()  # 1. Сбросить старые градиенты
        loss.backward()             # 2. Посчитать новые (кто виноват?)
        self.optimizer.step()       # 3. Подкрутить веса - обучаем policy_net (Ученика)
        
        # Подкручиваем учителя
        self.soft_update()

            
    def soft_update(self): 
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            new_val = self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            target_param.data.copy_(new_val)


def train_dqn(n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.997):
    """
    n_episodes: сколько игр сыграть
    max_t: макс. шагов в одной игре
    eps_start: начальная вероятность случайного действия (100%)
    eps_end: минимальная вероятность (1%)
    eps_decay: скорость уменьшения случайности
    """
    scores = []                        # список очков за каждый эпизод
    scores_window = deque(maxlen=100)  # последние 100 очков
    eps = eps_start                    # текущий epsilon
    
    print("Начинаем обучение... (Это может занять 10-20 минут)")
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            # 1. Выбрать действие
            action = agent.act(state, eps)
            
            # 2. Сделать шаг
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Сообщить агенту (он сохранит и поучится)
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break 
        
        # Сохраняем очки
        scores_window.append(score)
        scores.append(score)
        
        # Уменьшаем epsilon
        eps = max(eps_end, eps * eps_decay)
        
        # Вывод прогресса
        print(f'\rЭпизод {i_episode}\tСредний счет (100): {np.mean(scores_window):.2f}\tEpsilon: {eps:.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rЭпизод {i_episode}\tСредний счет (100): {np.mean(scores_window):.2f}')
        
        # Условие победы (если средний счет > 200)
        if np.mean(scores_window) >= 200.0:
            print(f'\nЗадача решена за {i_episode-100} эпизодов! \tСредний счет: {np.mean(scores_window):.2f}')
            torch.save(agent.policy_net.state_dict(), 'checkpoint.pth')
            break
            
    return scores

def save_plot(scores):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.3, color='blue', label='Raw Score')
    window_size = 100
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(scores)), moving_avg, color='red', linewidth=2, label='Avg (100 eps)')
    
    plt.title('DQN LunarLander Training')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/dqn_training_curve.png')
    print("График сохранен в images/dqn_training_curve.png")


if __name__ == "__main__":
    env = gym.make('LunarLander-v3')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    
    agent = Agent(state_size=8, action_size=4, device=device)
    scores = train_dqn()
    
    save_plot(scores)