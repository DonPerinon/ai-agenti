import numpy as np
import random
import pickle
import os
from typing import Tuple, List, Optional
from grid_environment import GridWorld


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def get_action(self, state: int) -> int:
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_value(self, state: int, action: int, reward: float,
                      next_state: int, done: bool):
        # Q-learning update rule
        current_q = self.q_table[state, action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])

        # Update Q-value
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_policy(self) -> np.ndarray:
        # Return the greedy policy (best action for each state)
        return np.argmax(self.q_table, axis=1)

    def train(self, env: GridWorld, episodes: int = 1000, max_steps: int = 200,
              verbose: bool = True) -> Tuple[List[float], List[int]]:
        episode_rewards = []
        episode_lengths = []

        for episode in range(episodes):
            state = env.reset()
            state_idx = env.get_state_index(state)
            total_reward = 0
            steps = 0

            for step in range(max_steps):
                # Choose action
                action = self.get_action(state_idx)

                # Take action
                next_state, reward, done = env.step(action)
                next_state_idx = env.get_state_index(next_state)

                # Update Q-value
                self.update_q_value(state_idx, action, reward, next_state_idx, done)

                # Update state
                state_idx = next_state_idx
                total_reward += reward
                steps += 1

                if done:
                    break

            # Decay epsilon
            self.decay_epsilon()

            # Store statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")

        self.episode_rewards = episode_rewards
        self.episode_lengths = episode_lengths
        return episode_rewards, episode_lengths

    def evaluate(self, env: GridWorld, episodes: int = 100) -> Tuple[float, float]:
        total_rewards = []
        total_steps = []
        old_epsilon = self.epsilon
        self.epsilon = 0  # No exploration during evaluation

        for _ in range(episodes):
            state = env.reset()
            state_idx = env.get_state_index(state)
            episode_reward = 0
            steps = 0

            while steps < 200:
                action = self.get_action(state_idx)
                next_state, reward, done = env.step(action)
                state_idx = env.get_state_index(next_state)
                episode_reward += reward
                steps += 1

                if done:
                    break

            total_rewards.append(episode_reward)
            total_steps.append(steps)

        self.epsilon = old_epsilon  # Restore epsilon
        return np.mean(total_rewards), np.mean(total_steps)

    def save_model(self, filepath: str = "q_table_model.pkl"):
        """Save the trained Q-table and agent parameters to a file"""
        model_data = {
            'q_table': self.q_table,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to '{filepath}'")

    def load_model(self, filepath: str = "q_table_model.pkl") -> bool:
        """Load a trained Q-table and agent parameters from a file"""
        if not os.path.exists(filepath):
            print(f"❌ Model file '{filepath}' not found")
            return False

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            # Restore Q-table and parameters
            self.q_table = model_data['q_table']
            self.n_states = model_data['n_states']
            self.n_actions = model_data['n_actions']
            self.learning_rate = model_data['learning_rate']
            self.discount_factor = model_data['discount_factor']
            self.epsilon = model_data['epsilon']
            self.epsilon_decay = model_data['epsilon_decay']
            self.epsilon_min = model_data['epsilon_min']
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.episode_lengths = model_data.get('episode_lengths', [])

            print(f"✅ Model loaded from '{filepath}'")
            print(f"   Q-table shape: {self.q_table.shape}")
            print(f"   Final epsilon: {self.epsilon:.3f}")
            if self.episode_rewards:
                print(f"   Training episodes: {len(self.episode_rewards)}")
                print(f"   Final avg reward: {np.mean(self.episode_rewards[-100:]):.2f}")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    @staticmethod
    def model_exists(filepath: str = "q_table_model.pkl") -> bool:
        """Check if a saved model file exists"""
        return os.path.exists(filepath)