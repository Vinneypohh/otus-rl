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
