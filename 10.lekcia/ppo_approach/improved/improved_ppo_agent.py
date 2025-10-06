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
    """PPO Actor-Critic Network with improved architecture"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPONetwork, self).__init__()

        # Shared feature extractor with dropout for better generalization
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Remove softmax - will be applied later
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        shared_features = self.shared_layers(obs)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value

    def get_action(self, obs, valid_actions=None, exploration_rate=0.0):
        """Get action with optional action masking and exploration"""
        with torch.no_grad():
            action_logits, _ = self.forward(obs)

            # Apply action masking if provided
            if valid_actions is not None and len(valid_actions) > 0:
                # Create mask for invalid actions
                mask = torch.full_like(action_logits, float('-inf'))
                mask[0, valid_actions] = 0  # Fix indexing for batch dimension
                action_logits = action_logits + mask

            # Add exploration noise during training
            if exploration_rate > 0:
                noise = torch.normal(0, exploration_rate, action_logits.shape)
                action_logits = action_logits + noise

            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def evaluate_actions(self, obs, actions, valid_actions_batch=None):
        """Evaluate actions for PPO update with action masking"""
        action_logits, values = self.forward(obs)

        # Apply action masking if provided
        if valid_actions_batch is not None:
            for i, valid_actions in enumerate(valid_actions_batch):
                if valid_actions is not None and len(valid_actions) > 0:
                    mask = torch.full_like(action_logits[i], float('-inf'))
                    mask[valid_actions] = 0  # This is correct since action_logits[i] is 1D
                    action_logits[i] = action_logits[i] + mask

        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, values.squeeze(), entropy


class ImprovedPPOAgent:
    """Improved PPO Agent with better exploration and obstacle handling"""

    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, eps_clip: float = 0.2,
                 k_epochs: int = 4, hidden_dim: int = 128):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.action_dim = action_dim

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
            'dones': [],
            'valid_actions': []
        }

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []

        # Exploration parameters
        self.exploration_rate = 0.1
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

    def select_action(self, observation, valid_actions=None, training=True):
        """Select action using current policy with exploration"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

        # Use exploration during training
        exploration = self.exploration_rate if training else 0.0
        action, log_prob = self.policy.get_action(obs_tensor, valid_actions, exploration)

        # Store for training
        with torch.no_grad():
            _, value = self.policy(obs_tensor)

        if training:
            self.memory['observations'].append(observation)
            self.memory['actions'].append(action)
            self.memory['log_probs'].append(log_prob)
            self.memory['values'].append(value.item())
            self.memory['valid_actions'].append(valid_actions)

        return action

    def select_action_with_fallback(self, observation, valid_actions, training=True):
        """Select action with fallback to random valid action if stuck"""
        action = self.select_action(observation, valid_actions, training)

        # If selected action is not valid, choose random valid action
        if valid_actions and action not in valid_actions:
            action = random.choice(valid_actions) if valid_actions else 0

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
        """Update policy using PPO with action masking"""
        if len(self.memory['observations']) == 0:
            return

        # Convert to tensors
        observations = torch.from_numpy(np.array(self.memory['observations'], dtype=np.float32))
        actions = torch.from_numpy(np.array(self.memory['actions'], dtype=np.int64))
        old_log_probs = torch.from_numpy(np.array(self.memory['log_probs'], dtype=np.float32))
        valid_actions_batch = self.memory['valid_actions']

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        returns = torch.from_numpy(np.array(returns, dtype=np.float32))
        advantages = torch.from_numpy(np.array(advantages, dtype=np.float32))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy outputs
            log_probs, values, entropy = self.policy.evaluate_actions(
                observations, actions, valid_actions_batch)

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
            total_loss = policy_loss + 0.5 * value_loss + 0.02 * entropy_loss

            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

        # Clear memory
        self.clear_memory()

    def clear_memory(self):
        """Clear trajectory memory"""
        for key in self.memory:
            self.memory[key] = []

    def train_on_environment(self, env, episodes: int = 1000, max_steps: int = 200,
                           update_freq: int = 20, verbose: bool = True):
        """Train PPO agent on environment with improved exploration"""

        print(f"🚀 Training Improved PPO Agent for {episodes} episodes")
        print("=" * 50)

        for episode in range(episodes):
            # Sample random goal for this episode
            goal = env.sample_random_goal()
            env.set_goal(goal)

            obs = env.reset()
            episode_reward = 0
            step_count = 0
            stuck_counter = 0

            for step in range(max_steps):
                # Get valid actions from environment
                valid_actions = env.get_valid_actions()

                # Select action with fallback
                action = self.select_action_with_fallback(obs, valid_actions, training=True)

                next_obs, reward, done, info = env.step(action)

                # Add anti-stagnation bonus
                if 'stagnation_counter' in info and info['stagnation_counter'] > 3:
                    stuck_counter += 1
                    reward -= 0.1 * stuck_counter  # Increasing penalty for being stuck
                else:
                    stuck_counter = 0

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
                      f"Success Rate: {success_rate:.2f}, "
                      f"Exploration: {self.exploration_rate:.3f}")

        print("✅ Training completed!")

    def test_on_goal(self, env, goal: Tuple[int, int], episodes: int = 10,
                    max_steps: int = 200, visualize: bool = False,
                    use_exploration: bool = False):
        """Test agent on specific goal with optional exploration"""
        env.set_goal(goal)
        results = []

        print(f"\n🎯 Testing Improved PPO Agent on goal {goal}")

        for episode in range(episodes):
            obs = env.reset()
            episode_reward = 0
            step_count = 0
            path = [env.current_pos]
            done = False
            stuck_positions = set()

            for step in range(max_steps):
                # Get valid actions
                valid_actions = env.get_valid_actions()

                if use_exploration:
                    # Use epsilon-greedy exploration
                    if np.random.random() < 0.1:  # 10% random exploration
                        if valid_actions:
                            action = np.random.choice(valid_actions)
                        else:
                            action = np.random.randint(0, self.action_dim)
                    else:
                        action = self.select_action(obs, valid_actions, training=False)
                else:
                    # Use policy without exploration
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action_logits, _ = self.policy(obs_tensor)

                        # Apply action masking
                        if valid_actions and len(valid_actions) > 0:
                            mask = torch.full_like(action_logits, float('-inf'))
                            mask[0, valid_actions] = 0
                            action_logits = action_logits + mask

                        action = torch.argmax(action_logits, dim=1).item()

                # Anti-stagnation: if stuck in same position too long, try random valid action
                if env.current_pos in stuck_positions and step > 10:
                    if valid_actions and np.random.random() < 0.3:
                        action = np.random.choice(valid_actions)

                next_obs, reward, done, info = env.step(action)
                path.append(env.current_pos)
                episode_reward += reward
                step_count += 1

                # Track stuck positions
                if len(path) > 5 and path[-1] == path[-5]:
                    stuck_positions.add(path[-1])

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
                fig.suptitle(f'Improved PPO Agent - Goal {goal} (Episode 1)', fontsize=16)
                fig.savefig(f'visualizations/improved_ppo_test_{goal[0]}_{goal[1]}.png',
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

    def save_model(self, filepath: str = "improved_ppo_model.pkl"):
        """Save trained model"""
        model_data = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'exploration_rate': self.exploration_rate,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Improved PPO model saved to '{filepath}'")

    def load_model(self, filepath: str = "improved_ppo_model.pkl") -> bool:
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.policy.load_state_dict(model_data['policy_state_dict'])
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.episode_lengths = model_data.get('episode_lengths', [])
            self.exploration_rate = model_data.get('exploration_rate', self.exploration_rate)

            print(f"✅ Improved PPO model loaded from '{filepath}'")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False