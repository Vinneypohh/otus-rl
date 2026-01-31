"""Experience replay buffer for Deep Q-Network (DQN) training.

This module implements a replay buffer that stores and samples transitions
for off-policy reinforcement learning algorithms.
"""
import random
from collections import deque
from typing import Tuple, Union, List

import numpy as np
import torch


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions.
    
    The buffer uses a deque with a maximum capacity. When the buffer is full,
    the oldest transitions are automatically removed when new ones are added.
    
    Attributes:
        buffer: Deque storing transitions as tuples of (state, action, reward, next_state, done).
        capacity: Maximum number of transitions to store.
    """
    
    def __init__(self, capacity: int = 10000) -> None:
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum size of the buffer. When exceeded, oldest transitions are removed.
        """
        self.buffer: deque = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        state: Union[np.ndarray, List[float]],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, List[float]],
        done: bool
    ) -> None:
        """Add a transition to the replay buffer.
        
        Args:
            state: Current state observation (numpy array or list).
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            next_state: Next state observation after taking the action.
            done: Whether the episode terminated after this transition.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            A tuple containing:
                - states: Tensor of shape (batch_size, state_size) with current states.
                - actions: Tensor of shape (batch_size,) with actions taken.
                - rewards: Tensor of shape (batch_size,) with rewards received.
                - next_states: Tensor of shape (batch_size, state_size) with next states.
                - dones: Tensor of shape (batch_size,) with done flags.
                
        Raises:
            ValueError: If batch_size exceeds the current buffer size.
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Batch size {batch_size} exceeds buffer size {len(self.buffer)}")
        
        # Sample random batch of transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to PyTorch tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        dones_tensor = torch.FloatTensor(dones)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
    
    def __len__(self) -> int:
        """Return the current number of transitions in the buffer.
        
        Returns:
            Current buffer size.
        """
        return len(self.buffer)
