#!/usr/bin/env python3
"""
Training script for Improved PPO Agent
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv
from improved_ppo_agent import ImprovedPPOAgent
import os


def plot_training_results(rewards, lengths, save_path="ppo_approach/improved_ppo_training.png"):
    """Plot and save training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot rewards
    ax1.plot(rewards, alpha=0.6, label='Episode Reward')
    window_size = 100
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(rewards)), moving_avg,
                color='red', linewidth=2, label='Moving Average (100 episodes)')
        ax1.legend()
    ax1.set_title('Episode Rewards During Improved PPO Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)

    # Plot episode lengths
    ax2.plot(lengths, alpha=0.6, label='Episode Length')
    if len(lengths) >= window_size:
        moving_avg = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(lengths)), moving_avg,
                color='red', linewidth=2, label='Moving Average (100 episodes)')
        ax2.legend()
    ax2.set_title('Episode Lengths During Improved PPO Training')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Goal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training plots saved to '{save_path}'")
    plt.close()


def main():
    print("🎯 Improved PPO Training Script")
    print("=" * 40)

    # Environment setup
    print("\n🏗️  Setting up environment...")
    env = GridWorldEnv(width=8, height=8, start=(0, 0), goal=(7, 7))
    print(f"Environment: {env.width}×{env.height} grid")
    print(f"Start: {env.start_pos}")
    print(f"Obstacles: {list(env.obstacles)}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")

    # Agent setup
    print("\n🤖 Creating Improved PPO agent...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ImprovedPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        hidden_dim=128
    )

    print(f"Agent configuration:")
    print(f"  • Observation dim: {obs_dim}")
    print(f"  • Action dim: {action_dim}")
    print(f"  • Learning rate: 3e-4")
    print(f"  • Discount factor: 0.99")
    print(f"  • PPO clip ratio: 0.2")
    print(f"  • Initial exploration rate: {agent.exploration_rate}")

    # Check for existing model
    model_file = "improved_ppo_model.pkl"
    if os.path.exists(model_file):
        print(f"\n⚠️  Found existing model '{model_file}'")
        choice = input("Do you want to retrain? (y/N): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("Aborting training. Use existing model.")
            return

    # Training
    episodes = 15000
    print(f"\n🚀 Training for {episodes} episodes...")
    print("Agent will learn to navigate to randomly sampled goals each episode.")
    print("Features:")
    print("  • Action masking to prevent invalid moves")
    print("  • Anti-stagnation penalties")
    print("  • Decaying exploration rate")
    print("  • Improved reward shaping")

    agent.train_on_environment(env, episodes=episodes, verbose=True)

    # Save model
    print(f"\n💾 Saving trained model...")
    agent.save_model(model_file)

    # Training analysis
    print(f"\n📈 Training Analysis:")
    rewards = agent.episode_rewards
    lengths = agent.episode_lengths

    final_100_rewards = rewards[-100:]
    final_100_lengths = lengths[-100:]

    print(f"  • Total episodes: {len(rewards)}")
    print(f"  • Final 100 episodes:")
    print(f"    - Average reward: {np.mean(final_100_rewards):.2f}")
    print(f"    - Average steps: {np.mean(final_100_lengths):.1f}")
    print(f"    - Success rate: {sum(1 for r in final_100_rewards if r > 8) / len(final_100_rewards) * 100:.1f}%")
    print(f"  • Best episode reward: {max(rewards):.2f}")
    print(f"  • Final exploration rate: {agent.exploration_rate:.4f}")

    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    plot_training_results(rewards, lengths)

    # Test on different goals
    print(f"\n🧪 Testing on different goal positions...")
    test_goals = [(7, 7), (0, 7), (5, 7), (3, 3), (2, 1), (6, 2)]

    all_results = []
    for goal in test_goals:
        if env.is_valid_position(goal) and goal not in env.obstacles:
            result = agent.test_on_goal(env, goal, episodes=5, visualize=True, use_exploration=False)
            all_results.append(result)
        else:
            print(f"⚠️  Skipping invalid goal: {goal}")

    # Summary
    print(f"\n📊 Overall Test Results:")
    successful_goals = [r for r in all_results if r['success_rate'] > 0.5]
    print(f"  • Goals with >50% success: {len(successful_goals)}/{len(all_results)}")

    if successful_goals:
        avg_success_rate = np.mean([r['success_rate'] for r in successful_goals])
        avg_steps = np.mean([r['avg_steps'] for r in successful_goals if r['avg_steps'] != float('inf')])
        print(f"  • Average success rate: {avg_success_rate:.2f}")
        print(f"  • Average steps (successful goals): {avg_steps:.1f}")

    # Test with exploration enabled
    print(f"\n🔍 Testing with exploration enabled...")
    print("This helps when the deterministic policy gets stuck.")

    problem_goal = (7, 7)  # The goal that was failing before
    result_with_exploration = agent.test_on_goal(env, problem_goal, episodes=3,
                                                visualize=True, use_exploration=True)

    print(f"Results with exploration for {problem_goal}:")
    print(f"  Success rate: {result_with_exploration['success_rate']:.2f}")

    print(f"\n✅ Improved PPO training complete!")
    print(f"🎮 Use 'python run_improved_ppo.py <goal_x> <goal_y>' to test different goals.")


if __name__ == "__main__":
    main()