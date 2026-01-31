"""Inference script for trained DQN agent on LunarLander-v3 environment.

This module provides functions to load a trained agent and run it
with visual rendering to observe the agent's behavior.
"""
import argparse
import os
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch

from .agent import Agent


# Default paths
DEFAULT_CHECKPOINT_PATH = 'lunar_lander/checkpoints/checkpoint.pth'
DEFAULT_N_EPISODES = 10
DEFAULT_MAX_STEPS = 1000


def load_agent(
    checkpoint_path: str,
    state_size: int = 8,
    action_size: int = 4,
    device: Optional[torch.device] = None
) -> Agent:
    """Load a trained agent from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint.
        state_size: Dimension of the state space. Defaults to 8.
        action_size: Number of possible actions. Defaults to 4.
        device: Computing device (CPU or CUDA). If None, auto-detects.
        
    Returns:
        Agent with loaded weights.
        
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create agent
    agent = Agent(state_size=state_size, action_size=action_size, device=device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Please train the agent first using train.py"
        )
    
    agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.policy_net.eval()  # Set to evaluation mode
    
    print(f"Agent loaded from {checkpoint_path}")
    print(f"Using device: {device}")
    
    return agent


def run_inference(
    agent: Agent,
    n_episodes: int = DEFAULT_N_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    render_mode: str = 'human',
    verbose: bool = True
) -> list[float]:
    """Run inference with the trained agent.
    
    Args:
        agent: Trained DQN agent.
        n_episodes: Number of episodes to run. Defaults to 10.
        max_steps: Maximum steps per episode. Defaults to 1000.
        render_mode: Rendering mode ('human', 'rgb_array', or None). Defaults to 'human'.
        verbose: Whether to print episode information. Defaults to True.
        
    Returns:
        List of scores for each episode.
    """
    env = gym.make('LunarLander-v3', render_mode=render_mode)
    scores: List[float] = []
    
    if verbose:
        print(f"\nRunning inference for {n_episodes} episodes...")
        print("=" * 50)
    
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0.0
        steps = 0
        
        for step in range(max_steps):
            # Select action using greedy policy (no exploration)
            action = agent.act(state, eps=0.0)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            score += reward
            steps += 1
            
            if done:
                break
        
        scores.append(score)
        
        if verbose:
            print(
                f"Episode {episode:3d} | "
                f"Score: {score:8.2f} | "
                f"Steps: {steps:4d}"
            )
    
    env.close()
    
    if verbose:
        print("=" * 50)
        print(f"Average score: {np.mean(scores):.2f}")
        print(f"Std deviation: {np.std(scores):.2f}")
        print(f"Min score: {np.min(scores):.2f}")
        print(f"Max score: {np.max(scores):.2f}")
    
    return scores


def main() -> None:
    """Main inference function with command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with trained DQN agent on LunarLander-v3'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f'Path to checkpoint file (default: {DEFAULT_CHECKPOINT_PATH})'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=DEFAULT_N_EPISODES,
        help=f'Number of episodes to run (default: {DEFAULT_N_EPISODES})'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f'Maximum steps per episode (default: {DEFAULT_MAX_STEPS})'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering (faster execution)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output messages'
    )
    
    args = parser.parse_args()
    
    # Determine render mode
    render_mode = None if args.no_render else 'human'
    
    # Load agent
    try:
        agent = load_agent(args.checkpoint)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Run inference
    scores = run_inference(
        agent=agent,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        render_mode=render_mode,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
