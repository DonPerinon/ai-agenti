#!/usr/bin/env python3
"""
Training script for Q-Learning agent in Grid World environment.
Run this once to train and save the model, then use run.py for demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_environment import GridWorld
from q_agent import QLearningAgent


def plot_training_results(rewards, lengths, save_path="training_results.png"):
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
    ax1.set_title('Episode Rewards During Training')
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
    ax2.set_title('Episode Lengths During Training')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Goal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training plots saved to '{save_path}'")
    plt.close()
    return fig


def main():
    print("🎯 Q-Learning Training Script")
    print("=" * 40)

    # Environment setup
    print("\n🏗️  Setting up environment...")
    env = GridWorld(width=8, height=8, start=(0, 0), goal=(5, 7))
    print(f"Environment: {env.width}×{env.height} grid")
    print(f"Start: {env.start_pos} → Goal: {env.goal_pos}")
    print(f"Obstacles: {env.obstacles}")

    # Agent setup
    print("\n🤖 Creating agent...")
    n_states = env.width * env.height
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=env.actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    print(f"Agent configuration:")
    print(f"  • States: {n_states} ({env.width}×{env.height})")
    print(f"  • Actions: {env.actions} (↑→↓←)")
    print(f"  • Learning rate: {agent.learning_rate}")
    print(f"  • Discount factor: {agent.discount_factor}")
    print(f"  • Initial exploration: {agent.epsilon:.1f} (ε-greedy)")

    # Check for existing model
    model_file = "q_table_model.pkl"
    if agent.model_exists(model_file):
        print(f"\n⚠️  Found existing model '{model_file}'")
        choice = input("Do you want to retrain? (y/N): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("Aborting training. Use existing model.")
            return

    # Training
    episodes = 2000
    print(f"\n🚀 Training for {episodes} episodes...")
    print("Progress will be shown every 100 episodes.")

    rewards, lengths = agent.train(env, episodes=episodes, verbose=True)

    # Save model
    print(f"\n💾 Saving trained model...")
    agent.save_model(model_file)

    # Training analysis
    print(f"\n📈 Training Analysis:")
    final_100_rewards = rewards[-100:]
    final_100_lengths = lengths[-100:]

    print(f"  • Total episodes: {len(rewards)}")
    print(f"  • Final 100 episodes:")
    print(f"    - Average reward: {np.mean(final_100_rewards):.2f}")
    print(f"    - Average steps: {np.mean(final_100_lengths):.1f}")
    print(f"    - Success rate: {sum(1 for r in final_100_rewards if r > 9) / len(final_100_rewards) * 100:.1f}%")
    print(f"  • Best episode reward: {max(rewards):.2f}")
    print(f"  • Final exploration rate: {agent.epsilon:.3f}")

    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    plot_training_results(rewards, lengths)

    # Policy visualization
    env.reset()
    policy = agent.get_policy()
    fig, _ = env.visualize(policy=policy)
    fig.savefig('trained_policy.png', dpi=150, bbox_inches='tight')
    print(f"🗺️  Policy visualization saved to 'trained_policy.png'")
    plt.close()

    # Quick evaluation
    print(f"\n🧪 Quick evaluation (100 test episodes)...")
    avg_reward, avg_steps = agent.evaluate(env, episodes=100)
    print(f"  • Average reward: {avg_reward:.2f}")
    print(f"  • Average steps to goal: {avg_steps:.1f}")

    print(f"\n✅ Training complete!")
    print(f"🎮 Use 'python run.py' to see the trained agent in action.")


if __name__ == "__main__":
    main()