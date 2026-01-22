import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

from models.dqn_model import DQN
from utils.replay_buffer import ReplayBuffer

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

            
def soft_update(self, local_model, target_model):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        # ВНИМАНИЕ НА СКОБКИ!
        new_val = self.tau * local_param.data + (1.0 - self.tau) * target_param.data
        target_param.data.copy_(new_val)
