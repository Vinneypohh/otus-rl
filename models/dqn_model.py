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
