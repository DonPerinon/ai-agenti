#!/usr/bin/env python3
"""
Robust PPO Training with Domain Randomization for Obstacle Changes
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv
from ppo_agent import PPOAgent
import random


def generate_random_obstacles(num_obstacles=5, width=8, height=8, start=(0, 0)):
    """Generate random obstacle layout"""
    obstacles = []
    forbidden = {start}  # Don't place obstacles on start

    while len(obstacles) < num_obstacles:
        pos = (random.randint(0, height-1), random.randint(0, width-1))
        if pos not in forbidden and pos not in obstacles:
            obstacles.append(pos)

    return obstacles


def train_robust_agent(episodes=5000):
    """Train agent with domain randomization - multiple obstacle layouts"""
    print("🎯 Robust PPO Training with Domain Randomization")
    print("=" * 60)

    # Base environment setup
    base_obstacles = [(1, 0), (3, 4), (5, 1), (6, 5), (1, 6)]

    # Create agent
    env = GridWorldEnv(obstacles=base_obstacles)
    agent = PPOAgent(15, 4, lr=3e-4)  # 15-dimensional observation space

    print(f"🤖 Training for {episodes} episodes with varying obstacles...")

    episode_rewards = []

    for episode in range(episodes):
        # Every 10 episodes, change obstacle layout
        if episode % 10 == 0:
            if random.random() < 0.7:  # 70% use base layout
                obstacles = base_obstacles
            else:  # 30% use random layout
                obstacles = generate_random_obstacles(8, 8, 8, (0, 0))

            env = GridWorldEnv(obstacles=obstacles)

        # Sample random goal for this episode
        goal = env.sample_random_goal()
        env.set_goal(goal)

        obs = env.reset()
        episode_reward = 0

        for step in range(200):
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store_transition(reward, done)
            episode_reward += reward
            obs = next_obs

            if done:
                break

        episode_rewards.append(episode_reward)

        # Update policy every 20 episodes
        if episode % 20 == 0 and episode > 0:
            agent.update()

        # Progress logging
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-500:])
            success_rate = sum(1 for r in episode_rewards[-500:] if r > 8) / 500
            print(f"Episode {episode + 1}: Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")

    agent.episode_rewards = episode_rewards
    agent.save_model('robust_ppo_model.pkl')
    print("✅ Robust training complete!")

    return agent


def test_obstacle_robustness(agent):
    """Test agent on different obstacle layouts"""
    print("\n🧪 Testing Obstacle Robustness...")

    test_layouts = [
        # Original training layout
        [(1, 0), (3, 4), (5, 1), (6, 5), (1, 6)],
        # Moved obstacles
        [(2, 2), (3, 4), (5, 1), (6, 5), (1, 6)],
        [(0, 3), (4, 2), (6, 1), (5, 5), (2, 6)],
        # Different pattern
        [(1, 1), (2, 3), (4, 4), (5, 2), (7, 6)]
    ]

    test_goals = [(7, 7), (5, 7), (0, 7), (3, 3)]

    for i, obstacles in enumerate(test_layouts):
        print(f"\nLayout {i+1}: {obstacles}")

        successes = 0
        total_tests = 0

        for goal in test_goals:
            env = GridWorldEnv(obstacles=obstacles)
            try:
                result = agent.test_on_goal(env, goal, episodes=3, visualize=False)
                successes += result['episodes'].count(True) if isinstance(result['episodes'], list) else int(result['success_rate'] * 3)
                total_tests += 3
            except:
                total_tests += 3

        success_rate = successes / total_tests if total_tests > 0 else 0
        print(f"  Success rate: {success_rate:.2f}")


if __name__ == "__main__":
    # Train robust agent
    agent = train_robust_agent(episodes=40000)  # Shorter for demo

    # Test robustness
    test_obstacle_robustness(agent)

    print("\n💡 Usage:")
    print("This agent should handle obstacle changes much better!")
    print("Load with: agent.load_model('robust_ppo_model.pkl')")