#!/usr/bin/env python3
"""
Training script for Goal-Conditioned Q-Learning agent.
This agent can handle different goal positions without retraining.
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_environment import GridWorld
from goal_conditioned_agent import GoalConditionedQLearningAgent


def plot_training_results(rewards, lengths, save_path="goal_conditioned_training.png"):
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
    print("🎯 Goal-Conditioned Q-Learning Training Script")
    print("=" * 50)

    # Environment setup
    print("\n🏗️  Setting up environment...")
    env = GridWorld(width=8, height=8, start=(0, 0), goal=(7, 7))
    print(f"Environment: {env.width}×{env.height} grid")
    print(f"Start: {env.start_pos}")
    print(f"Obstacles: {env.obstacles}")

    # Define training goals - agent will learn to reach any of these
    training_goals = [(7, 7), (0, 7), (7, 0), (3, 3), (5, 2), (2, 6), (1, 1), (6, 3), (5, 7), (4, 4), (1, 3), (6, 1), (2, 1)]
    print(f"Training goals: {training_goals}")

    # Agent setup
    print("\n🤖 Creating goal-conditioned agent...")
    n_positions = env.width * env.height
    agent = GoalConditionedQLearningAgent(
        n_positions=n_positions,
        n_actions=env.actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    print(f"Agent configuration:")
    print(f"  • Positions: {n_positions} ({env.width}×{env.height})")
    print(f"  • Actions: {env.actions} (↑→↓←)")
    print(f"  • Q-table shape: {agent.q_table.shape}")
    print(f"  • Learning rate: {agent.learning_rate}")
    print(f"  • Discount factor: {agent.discount_factor}")
    print(f"  • Initial exploration: {agent.epsilon:.1f} (ε-greedy)")

    # Check for existing model
    model_file = "goal_conditioned_model.pkl"
    if agent.model_exists(model_file):
        print(f"\n⚠️  Found existing model '{model_file}'")
        choice = input("Do you want to retrain? (y/N): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("Aborting training. Use existing model.")
            return

    # Training
    episodes = 3000  # More episodes needed for multiple goals
    print(f"\n🚀 Training for {episodes} episodes...")
    print("Agent will learn to reach multiple different goals.")
    print("Progress will be shown every 100 episodes.")

    rewards, lengths = agent.train(env, episodes=episodes,
                                 goal_positions=training_goals, verbose=True)

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

    # Test on different goals
    print(f"\n🧪 Testing on different goal positions...")
    test_goals = [(7, 7), (0, 7), (5, 2), (4, 4)]  # Mix of trained and new goals

    for goal in test_goals:
        print(f"\nTesting goal {goal}:")
        avg_reward, avg_steps = agent.evaluate_on_goal(env, goal, episodes=50)
        success_rate = 100 if avg_reward > 8 else 0  # Rough estimate
        print(f"  • Average reward: {avg_reward:.2f}")
        print(f"  • Average steps: {avg_steps:.1f}")

        # Generate policy visualization for this goal
        env.goal_pos = goal
        policy = agent.get_policy_for_goal(env.get_state_index(goal))
        fig, _ = env.visualize(policy=policy)
        fig.savefig(f'policy_goal_{goal[0]}_{goal[1]}.png', dpi=150, bbox_inches='tight')
        print(f"  • Policy visualization saved to 'policy_goal_{goal[0]}_{goal[1]}.png'")
        plt.close()

    print(f"\n✅ Goal-conditioned training complete!")
    print(f"🎮 Use 'python run_goal_conditioned.py <goal_x> <goal_y>' to test different goals.")


if __name__ == "__main__":
    main()