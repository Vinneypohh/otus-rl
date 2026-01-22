"""Training script for DQN agent on LunarLander-v3 environment.

This module provides functions for training a Deep Q-Network agent
on the LunarLander-v3 environment from Gymnasium.
"""
import os
from collections import deque
from typing import List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from .agent import Agent


# Training constants
DEFAULT_N_EPISODES = 1000
DEFAULT_MAX_T = 1000
DEFAULT_EPS_START = 1.0
DEFAULT_EPS_END = 0.01
DEFAULT_EPS_DECAY = 0.997
WINNING_SCORE = 200.0
SCORES_WINDOW_SIZE = 100
CHECKPOINT_PATH = 'lunar_lander/checkpoints/checkpoint.pth'
PLOT_PATH = 'lunar_lander/images/dqn_training_curve.png'


def train_dqn(
    env: gym.Env,
    agent: Agent,
    n_episodes: int = DEFAULT_N_EPISODES,
    max_t: int = DEFAULT_MAX_T,
    eps_start: float = DEFAULT_EPS_START,
    eps_end: float = DEFAULT_EPS_END,
    eps_decay: float = DEFAULT_EPS_DECAY,
    winning_score: float = WINNING_SCORE
) -> List[float]:
    """Train a DQN agent on the given environment.
    
    Args:
        env: Gymnasium environment to train on.
        agent: DQN agent to train.
        n_episodes: Maximum number of training episodes. Defaults to 1000.
        max_t: Maximum number of steps per episode. Defaults to 1000.
        eps_start: Initial exploration probability. Defaults to 1.0.
        eps_end: Minimum exploration probability. Defaults to 0.01.
        eps_decay: Exploration decay rate per episode. Defaults to 0.997.
        winning_score: Average score threshold to consider the task solved. Defaults to 200.0.
        
    Returns:
        List of scores for each episode.
    """
    scores: List[float] = []
    scores_window = deque(maxlen=SCORES_WINDOW_SIZE)
    eps = eps_start
    
    print("Starting training...")
    
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0.0
        
        for t in range(max_t):
            # Select action using epsilon-greedy policy
            action = agent.act(state, eps)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition and learn
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break
        
        # Record score
        scores_window.append(score)
        scores.append(score)
        
        # Decay exploration probability
        eps = max(eps_end, eps * eps_decay)
        
        # Print progress
        print(
            f'\rEpisode {i_episode}\t'
            f'Average Score (100): {np.mean(scores_window):.2f}\t'
            f'Epsilon: {eps:.2f}',
            end=""
        )
        if i_episode % 100 == 0:
            print(
                f'\rEpisode {i_episode}\t'
                f'Average Score (100): {np.mean(scores_window):.2f}'
            )
        
        # Check if task is solved
        if np.mean(scores_window) >= winning_score:
            print(
                f'\nTask solved in {i_episode - 100} episodes! '
                f'\tAverage Score: {np.mean(scores_window):.2f}'
            )
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save(agent.policy_net.state_dict(), CHECKPOINT_PATH)
            break
            
    return scores


def save_plot(scores: List[float], plot_path: str = PLOT_PATH) -> None:
    """Save training curve plot.
    
    Creates a plot showing raw scores and moving average, then saves it to disk.
    
    Args:
        scores: List of episode scores.
        plot_path: Path to save the plot. Defaults to PLOT_PATH.
    """
    window_size = SCORES_WINDOW_SIZE
    moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
    
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.3, color='blue', label='Raw Score')
    plt.plot(
        range(window_size - 1, len(scores)),
        moving_avg,
        color='red',
        linewidth=2,
        label='Average (100 episodes)'
    )
    
    plt.title('DQN LunarLander Training')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


def main() -> None:
    """Main training function."""
    env = gym.make('LunarLander-v3')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = Agent(state_size=8, action_size=4, device=device)
    scores = train_dqn(env, agent)
    
    save_plot(scores)


if __name__ == "__main__":
    main()
