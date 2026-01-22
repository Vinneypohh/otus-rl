"""Deep Q-Network (DQN) neural network model.

This module implements a feedforward neural network for approximating
the Q-function in Deep Q-Learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Deep Q-Network model for value function approximation.
    
    The network consists of three fully connected layers with ReLU activations.
    The output layer produces Q-values for each possible action.
    
    Attributes:
        fc1: First fully connected layer (state_size -> hidden_size).
        fc2: Second fully connected layer (hidden_size -> hidden_size).
        fc3: Output layer (hidden_size -> action_size).
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64) -> None:
        """Initialize the DQN model.
        
        Args:
            state_size: Dimension of the state space.
            action_size: Number of possible actions.
            hidden_size: Number of neurons in the hidden layers. Defaults to 64.
        """
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, state_size) containing state observations.
            
        Returns:
            Tensor of shape (batch_size, action_size) containing Q-values for each action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
