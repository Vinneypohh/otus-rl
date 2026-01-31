"""Deep Q-Network (DQN) agent implementation.

This module implements a DQN agent with experience replay, target network,
and epsilon-greedy exploration strategy.
"""
import random
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import DQN
from .replay_buffer import ReplayBuffer


class Agent:
    """DQN agent for reinforcement learning.
    
    The agent uses two neural networks: a policy network for action selection
    and a target network for stable Q-value estimation. It implements experience
    replay and soft target network updates.
    
    Attributes:
        state_size: Dimension of the state space.
        action_size: Number of possible actions.
        device: Computing device (CPU or CUDA).
        gamma: Discount factor for future rewards.
        tau: Soft update coefficient for target network.
        lr: Learning rate for the optimizer.
        batch_size: Batch size for training.
        policy_net: Policy network for action selection.
        target_net: Target network for stable Q-value estimation.
        optimizer: Adam optimizer for training the policy network.
        memory: Experience replay buffer.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: torch.device,
        gamma: float = 0.99,
        tau: float = 0.001,
        lr: float = 0.001,
        batch_size: int = 64,
        memory_capacity: int = 100000
    ) -> None:
        """Initialize the DQN agent.
        
        Args:
            state_size: Dimension of the state space.
            action_size: Number of possible actions.
            device: Computing device (CPU or CUDA).
            gamma: Discount factor for future rewards. Defaults to 0.99.
            tau: Soft update coefficient for target network. Defaults to 0.001.
            lr: Learning rate for the optimizer. Defaults to 0.001.
            batch_size: Batch size for training. Defaults to 64.
            memory_capacity: Maximum size of the replay buffer. Defaults to 100000.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size
        
        # Initialize networks
        self.policy_net: DQN = DQN(state_size, action_size).to(device)
        self.target_net: DQN = DQN(state_size, action_size).to(device)
        # Copy weights from policy network to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Freeze target network (only update via soft updates)
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Experience replay buffer
        self.memory = ReplayBuffer(capacity=memory_capacity)

    def act(self, state: Union[np.ndarray, torch.Tensor], eps: float = 0.0) -> int:
        """Select an action using epsilon-greedy policy.
        
        With probability eps, selects a random action (exploration).
        Otherwise, selects the action with the highest Q-value (exploitation).
        
        Args:
            state: Current state observation.
            eps: Exploration probability (epsilon). Defaults to 0.0 (greedy).
            
        Returns:
            Selected action index.
        """
        if random.random() < eps:
            return random.choice(np.arange(self.action_size))
        
        # Convert state to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Get Q-values from policy network
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        return int(np.argmax(action_values.cpu().data.numpy()))

    def step(
        self,
        state: Union[np.ndarray, List[float]],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, List[float]],
        done: bool
    ) -> None:
        """Process a step in the environment.
        
        Stores the transition in the replay buffer and triggers learning
        if enough samples are available.
        
        Args:
            state: Current state observation.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            next_state: Next state observation.
            done: Whether the episode terminated.
        """
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self) -> None:
        """Perform one step of Q-learning update.
        
        Samples a batch from the replay buffer and updates the policy network
        using the Bellman equation. Also performs a soft update of the target network.
        """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move tensors to the appropriate device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute Q-values for taken actions
        q_expected = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values using target network
        with torch.no_grad():
            # Get maximum Q-value for next states
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            # Bellman equation: Q_target = R + gamma * max(Q_next) * (1 - done)
            q_targets = rewards.unsqueeze(1) + (self.gamma * q_next * (1 - dones.unsqueeze(1)))

        # Compute loss and update policy network
        loss = F.mse_loss(q_expected, q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()

    def soft_update(self) -> None:
        """Perform soft update of target network.
        
        Updates target network parameters as a weighted average of current
        target network parameters and policy network parameters.
        """
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            new_val = self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            target_param.data.copy_(new_val)
