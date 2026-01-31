"""Training script for Q-Learning agent on Taxi-v3 environment.

This module provides functions for training a tabular Q-Learning agent
on the Taxi-v3 environment from Gymnasium.
"""
import os
import pickle
import time
from typing import List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


# Training constants
DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA = 0.99
DEFAULT_EPSILON = 1.0
DEFAULT_EPSILON_DECAY = 0.9996
DEFAULT_EPSILON_MIN = 0.01
DEFAULT_EPISODES = 10000
PROGRESS_INTERVAL = 500
WINDOW_SIZE = 100
PLOT_PATH = 'taxi/images/taxi_training.png'
AGENT_PATH = "taxi/checkpoints/taxi_agent.pkl"


class QLearningAgent:
    """Tabular Q-Learning agent for discrete state-action spaces.
    
    This agent maintains a Q-table that stores Q-values for all state-action pairs.
    It uses epsilon-greedy exploration and updates Q-values using the Q-learning algorithm.
    
    Attributes:
        n_states: Number of states in the environment.
        n_actions: Number of actions in the environment.
        alpha: Learning rate for Q-value updates.
        gamma: Discount factor for future rewards.
        epsilon: Current exploration probability.
        epsilon_decay: Rate at which epsilon decreases after each episode.
        epsilon_min: Minimum exploration probability.
        q_table: Q-table storing Q-values for all state-action pairs.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        epsilon: float = DEFAULT_EPSILON,
        epsilon_decay: float = DEFAULT_EPSILON_DECAY,
        epsilon_min: float = DEFAULT_EPSILON_MIN
    ) -> None:
        """Initialize the Q-Learning agent.
        
        Args:
            n_states: Number of states in the environment.
            n_actions: Number of actions in the environment.
            alpha: Learning rate for Q-value updates. Defaults to 0.1.
            gamma: Discount factor for future rewards. Defaults to 0.99.
            epsilon: Initial exploration probability. Defaults to 1.0.
            epsilon_decay: Rate at which epsilon decreases. Defaults to 0.9996.
            epsilon_min: Minimum exploration probability. Defaults to 0.01.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state: int, is_training: bool = True) -> int:
        """Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state index.
            is_training: Whether to use exploration. If False, always uses greedy policy.
            
        Returns:
            Selected action index.
        """
        eps = self.epsilon if is_training else 0.0
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)  # Exploration
        return int(np.argmax(self.q_table[state]))  # Exploitation

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Update Q-table using the Q-learning update rule.
        
        Args:
            state: Current state index.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            next_state: Next state index after taking the action.
        """
        best_next_q = np.max(self.q_table[next_state, :])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Decay exploration probability after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename: str) -> None:
        """Save the learned Q-table to disk.
        
        Args:
            filename: Path to save the Q-table.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Agent saved to {filename}")

    def load(self, filename: str) -> None:
        """Load a previously saved Q-table from disk.
        
        Args:
            filename: Path to load the Q-table from.
        """
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Agent loaded from {filename}")
        else:
            print("Save file not found.")


def train_agent(
    episodes: int = DEFAULT_EPISODES,
    agent_path: str = AGENT_PATH,
    render: bool = False
) -> List[float]:
    """Train a Q-learning agent in Taxi-v3 environment.
    
    Args:
        episodes: Number of training episodes. Defaults to 10000.
        agent_path: Path to save the trained agent. Defaults to AGENT_PATH.
        render: Whether to render the environment during training. Defaults to False.
        
    Returns:
        List of total rewards for each episode.
    """
    render_mode = 'human' if render else None
    env = gym.make('Taxi-v3', render_mode=render_mode)

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n
    )

    rewards_history: List[float] = []

    print(f"Training started ({episodes} episodes)...")
    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not terminated and not truncated:
            action = agent.choose_action(state, is_training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (i + 1) % PROGRESS_INTERVAL == 0:
            avg_rew = np.mean(rewards_history[-WINDOW_SIZE:])
            print(
                f"Episode {i+1}, "
                f"Epsilon: {agent.epsilon:.3f}, "
                f"Avg reward (last {WINDOW_SIZE}): {avg_rew:.2f}"
            )

    env.close()
    agent.save(agent_path)
    return rewards_history


def plot_results(rewards: List[float], plot_path: str = PLOT_PATH) -> None:
    """Plot moving average reward and save the figure.
    
    Args:
        rewards: List of episode rewards.
        plot_path: Path to save the plot. Defaults to PLOT_PATH.
    """
    moving_avg = np.convolve(
        rewards,
        np.ones(WINDOW_SIZE) / WINDOW_SIZE,
        mode='valid'
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        moving_avg,
        label=f'Average reward ({WINDOW_SIZE}-episode window)',
        color='orange'
    )
    plt.title('Taxi-v3 Q-learning training')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.show()


def test_agent(episodes: int, agent_path: str) -> None:
    """Run the trained agent with human rendering to visually inspect behavior.
    
    Args:
        episodes: Number of test episodes to run.
        agent_path: Path to load the trained agent from.
    """
    print("\n--- TEST RUN (HUMAN RENDER) ---")
    env = gym.make('Taxi-v3', render_mode='human')

    agent = QLearningAgent(
        env.observation_space.n,
        env.action_space.n
    )
    agent.load(agent_path)

    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        print(f"Episode {i+1} started...")

        while not terminated and not truncated:
            # Use greedy policy (no exploration)
            action = agent.choose_action(state, is_training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        print(f"Episode {i+1} finished. Reward: {total_reward}")
        time.sleep(1)  # Pause between episodes

    env.close()


def main() -> None:
    """Main training function."""
    rewards = train_agent(episodes=DEFAULT_EPISODES, agent_path=AGENT_PATH, render=False)
    plot_results(rewards, PLOT_PATH)
    test_agent(episodes=100, agent_path=AGENT_PATH)


if __name__ == '__main__':
    main()
