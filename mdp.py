import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time


class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        agent_path,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9996,
        epsilon_min=0.01
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((n_states, n_actions))


    def choose_action(self, state, is_training=True):
        """Choose an action using epsilon-greedy policy."""
        eps = self.epsilon if is_training else 0.0
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)  # Exploration
        return np.argmax(self.q_table[state])          # Exploitation


    def update(self, state, action, reward, next_state):
        """Update Q-table using the Q-learning update rule."""
        best_next_q = np.max(self.q_table[next_state, :])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def save(self, filename):
        """Save the learned Q-table to disk."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Agent saved to {filename}")


    def load(self, filename):
        """Load a previously saved Q-table from disk."""
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Agent loaded from {filename}")
        else:
            print("Save file not found.")


def train_agent(episodes, agent_path, render=False):
    """Train a Q-learning agent in Taxi-v3."""
    render_mode = 'human' if render else None
    env = gym.make('Taxi-v3', render_mode=render_mode)

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n
    )

    rewards_history = []

    print(f"Training started ({episodes} episodes)...")
    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            action = agent.choose_action(state, is_training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (i + 1) % 500 == 0:
            avg_rew = np.mean(rewards_history[-100:])
            print(f"Episode {i+1}, Epsilon: {agent.epsilon:.3f}, Avg reward (last 100): {avg_rew:.2f}")

    env.close()
    agent.save(agent_path)
    return rewards_history


def plot_results(rewards, plot_path):
    """Plot moving average reward (window=100) and save the figure."""
    window_size = 100
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, label='Average reward (100-episode window)', color='orange')
    plt.title('Taxi-v3 Q-learning training')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path)
    plt.show()


def test_agent(episodes, agent_path):
    """Run the trained agent with human rendering to visually inspect behavior."""
    print("\n--- TEST RUN (HUMAN RENDER) ---")
    env = gym.make('Taxi-v3', render_mode='human')

    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    agent.load(agent_path)

    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        print(f"Episode {i+1} started...")

        while not terminated and not truncated:
            # is_training=False => purely greedy actions (no exploration)
            action = agent.choose_action(state, is_training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            # time.sleep(0.1)  # Uncomment if it runs too fast

        print(f"Episode {i+1} finished. Reward: {total_reward}")
        time.sleep(1)  # Pause between episodes

    env.close()


if __name__ == '__main__':
    plot_path = 'taxi/taxi_training.png'
    agent_path = "taxi/taxi_agent.pkl"
    rewards = train_agent(episodes=10000, agent_path=agent_path, render=False)

    plot_results(rewards)

    test_agent(episodes=100, agent_path=agent_path)
