import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
import random
from collections import deque
import pickle
import os
import matplotlib.pyplot as plt


class PPONetwork(nn.Module):
    """PPO Actor-Critic Network with goal conditioning"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPONetwork, self).__init__()

        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        shared_features = self.shared_layers(obs)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value

    def get_action(self, obs):
        """Get action with exploration"""
        with torch.no_grad():
            action_probs, _ = self.forward(obs)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def evaluate_actions(self, obs, actions):
        """Evaluate actions for PPO update"""
        action_probs, values = self.forward(obs)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, values.squeeze(), entropy


class PPOAgent:
    """Proximal Policy Optimization Agent with goal conditioning"""

    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, eps_clip: float = 0.2,
                 k_epochs: int = 4, hidden_dim: int = 128):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # Networks
        self.policy = PPONetwork(obs_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Storage for trajectories
        self.memory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def select_action(self, observation):
        """Select action using current policy"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        action, log_prob = self.policy.get_action(obs_tensor)

        # Store for training
        with torch.no_grad():
            _, value = self.policy(obs_tensor)

        self.memory['observations'].append(observation)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value.item())

        return action

    def store_transition(self, reward, done):
        """Store reward and done flag"""
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

    def compute_returns_and_advantages(self, next_value=0):
        """Compute returns and advantages using GAE"""
        rewards = self.memory['rewards']
        values = self.memory['values']
        dones = self.memory['dones']

        returns = []
        advantages = []

        # Add next value for bootstrap
        values = values + [next_value]
        gae = 0

        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.gamma * values[step + 1] - values[step]
                gae = delta + self.gamma * 0.95 * gae  # GAE lambda = 0.95

            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return returns, advantages

    def update(self):
        """Update policy using PPO"""
        if len(self.memory['observations']) == 0:
            return

        # Convert to tensors (optimize by converting to numpy first)
        observations = torch.from_numpy(np.array(self.memory['observations'], dtype=np.float32))
        actions = torch.from_numpy(np.array(self.memory['actions'], dtype=np.int64))
        old_log_probs = torch.from_numpy(np.array(self.memory['log_probs'], dtype=np.float32))

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        returns = torch.from_numpy(np.array(returns, dtype=np.float32))
        advantages = torch.from_numpy(np.array(advantages, dtype=np.float32))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy outputs
            log_probs, values, entropy = self.policy.evaluate_actions(observations, actions)

            # Compute policy ratio
            ratios = torch.exp(log_probs - old_log_probs)

            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Entropy bonus for exploration
            entropy_loss = -entropy.mean()

            # Total loss
            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # Clear memory
        self.clear_memory()

    def clear_memory(self):
        """Clear trajectory memory"""
        for key in self.memory:
            self.memory[key] = []

    def train_on_environment(self, env, episodes: int = 1000, max_steps: int = 200,
                           update_freq: int = 20, verbose: bool = True):
        """Train PPO agent on environment with multiple goals"""

        print(f"🚀 Training PPO Agent for {episodes} episodes")
        print("=" * 50)

        for episode in range(episodes):
            # Sample random goal for this episode
            goal = env.sample_random_goal()
            env.set_goal(goal)

            obs = env.reset()
            episode_reward = 0
            step_count = 0

            for step in range(max_steps):
                action = self.select_action(obs)
                next_obs, reward, done, info = env.step(action)

                self.store_transition(reward, done)
                episode_reward += reward
                step_count += 1

                obs = next_obs

                if done:
                    break

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step_count)

            # Update policy
            if episode % update_freq == 0 and episode > 0:
                self.update()

            # Logging
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                success_rate = sum(1 for r in self.episode_rewards[-100:] if r > 8) / 100
                print(f"Episode {episode + 1}/{episodes}: "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.1f}, "
                      f"Success Rate: {success_rate:.2f}")

        print("✅ Training completed!")

    def test_on_goal(self, env, goal: Tuple[int, int], episodes: int = 10,
                    max_steps: int = 200, visualize: bool = False):
        """Test agent on specific goal"""
        env.set_goal(goal)
        results = []

        print(f"\n🎯 Testing PPO Agent on goal {goal}")

        for episode in range(episodes):
            obs = env.reset()
            episode_reward = 0
            step_count = 0
            path = [env.current_pos]
            done = False

            for step in range(max_steps):
                # Use policy without exploration
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_probs, _ = self.policy(obs_tensor)
                    action = torch.argmax(action_probs, dim=1).item()

                next_obs, reward, done, info = env.step(action)
                path.append(env.current_pos)
                episode_reward += reward
                step_count += 1

                obs = next_obs

                if done:
                    break

            results.append({
                'success': done,
                'steps': step_count,
                'reward': episode_reward,
                'path': path
            })

            status = "✅" if done else "❌"
            print(f"  Episode {episode+1}: {status} {step_count} steps, reward {episode_reward:.2f}")

            # Visualize first episode
            if visualize and episode == 0:
                fig, ax = env.visualize(path)
                fig.suptitle(f'PPO Agent - Goal {goal} (Episode 1)', fontsize=16)
                fig.savefig(f'ppo_approach/ppo_test_{goal[0]}_{goal[1]}.png',
                           dpi=150, bbox_inches='tight')
                plt.close(fig)

        # Calculate statistics
        successes = [r for r in results if r['success']]
        success_rate = len(successes) / len(results)
        avg_steps = np.mean([r['steps'] for r in successes]) if successes else float('inf')
        avg_reward = np.mean([r['reward'] for r in successes]) if successes else -float('inf')

        print(f"\n📊 Results for goal {goal}:")
        print(f"  Success rate: {success_rate:.2f} ({len(successes)}/{len(results)})")
        if successes:
            print(f"  Average steps: {avg_steps:.1f}")
            print(f"  Average reward: {avg_reward:.2f}")

        return {
            'goal': goal,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
            'episodes': results
        }

    def save_model(self, filepath: str = "robust_ppo_model.pkl"):
        """Save trained model"""
        model_data = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ PPO model saved to '{filepath}'")

    def load_model(self, filepath: str = "ppo_model.pkl") -> bool:
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.policy.load_state_dict(model_data['policy_state_dict'])
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.episode_lengths = model_data.get('episode_lengths', [])

            print(f"✅ PPO model loaded from '{filepath}'")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False