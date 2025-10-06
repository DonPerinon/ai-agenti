#!/usr/bin/env python3
"""
PPO Agent Runner - Test trained agent on any goal position
Usage: python run_ppo.py [goal_x] [goal_y]
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from shared.grid_env import GridWorldEnv
from shared.ppo_agent import PPOAgent
import os


def run_single_episode(agent, env, goal, max_steps=200, show_steps=True):
    """Run a single episode with detailed output"""
    env.set_goal(goal)
    obs = env.reset()

    steps = 0
    total_reward = 0
    path = [env.current_pos]
    done = False

    print(f"\n🎮 PPO Agent Demonstration")
    print(f"Starting: {env.current_pos} → Goal: {goal}")
    print(f"Obstacles: {list(env.obstacles)}")

    if show_steps:
        print("\nStep-by-step moves:")

    action_names = ['↑ (up)', '→ (right)', '↓ (down)', '← (left)']

    for step in range(max_steps):
        # Get action from trained policy (no exploration)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_probs, _ = agent.policy(obs_tensor)
            action = torch.argmax(action_probs, dim=1).item()

        # Execute action
        next_obs, reward, done, info = env.step(action)
        path.append(env.current_pos)
        total_reward += reward
        steps += 1

        if show_steps:
            print(f"  Step {steps:2d}: {action_names[action]:<12} → {env.current_pos} (reward: {reward:5.2f})")

        obs = next_obs

        if done:
            print(f"\n🎉 SUCCESS! Goal {goal} reached in {steps} steps!")
            break

    if not done:
        print(f"\n⚠️  Goal {goal} not reached in {max_steps} steps")

    print(f"Total reward: {total_reward:.2f}")
    print(f"Path length: {len(path)} positions")

    return steps, total_reward, done, path


def test_multiple_goals(agent, env, goals):
    """Test agent on multiple goals"""
    print(f"\n🔄 Testing PPO Agent on {len(goals)} different goals...")
    print("-" * 60)

    results = []
    for i, goal in enumerate(goals):
        if not env.is_valid_position(goal) or goal in env.obstacles:
            print(f"⚠️  Skipping invalid goal: {goal}")
            continue

        print(f"\nGoal {i+1}: {goal}")
        steps, reward, success, path = run_single_episode(agent, env, goal, show_steps=False)
        results.append((goal, steps, reward, success, path))

        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"Result: {steps} steps, reward {reward:.2f} - {status}")

        # Visualize path
        fig, ax = env.visualize(path)
        fig.suptitle(f'PPO Agent - Goal {goal}', fontsize=16)
        filename = f'visualizations/ppo_goal_{goal[0]}_{goal[1]}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 Visualization saved to '{filename}'")

    # Summary
    successes = [r for r in results if r[3]]  # r[3] is success flag
    print(f"\n📊 Summary of {len(goals)} goals:")
    print(f"  • Success rate: {len(successes)}/{len(results)} ({len(successes)/len(results)*100:.1f}%)")

    if successes:
        avg_steps = sum(r[1] for r in successes) / len(successes)  # r[1] is steps
        avg_reward = sum(r[2] for r in successes) / len(successes)  # r[2] is reward
        print(f"  • Average steps (successful): {avg_steps:.1f}")
        print(f"  • Average reward (successful): {avg_reward:.2f}")


def main():
    print("🤖 PPO Agent Runner")
    print("=" * 30)

    # Parse command line arguments
    if len(sys.argv) >= 3:
        try:
            goal_x = int(sys.argv[1])
            goal_y = int(sys.argv[2])
            goal = (goal_x, goal_y)
        except ValueError:
            print("❌ Invalid goal coordinates. Use integers.")
            sys.exit(1)
    else:
        # Default goal
        goal = (5, 7)
        print(f"Using default goal: {goal}")

    # Setup environment
    env = GridWorldEnv(width=8, height=8, start=(0, 0), goal=goal)

    # Validate goal position
    if not env.is_valid_position(goal) or goal in env.obstacles:
        print(f"❌ Goal {goal} is invalid or on an obstacle!")
        sys.exit(1)

    # Create agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim)

    # Load trained model
    model_file = "robust_ppo_model.pkl"
    if not agent.load_model(model_file):
        print(f"\n❌ No trained PPO model found!")
        print(f"Please run 'python train_ppo.py' first to train the agent.")
        sys.exit(1)

    # Run demonstration
    print(f"\n🎯 Testing PPO agent on goal: {goal}")

    # Single detailed run
    steps, reward, success, path = run_single_episode(agent, env, goal, show_steps=True)

    # Generate visualization
    print(f"\n🗺️  Generating visualization...")
    fig, ax = env.visualize(path)
    fig.suptitle(f'PPO Agent Navigation to {goal}', fontsize=16)
    filename = f'visualizations/ppo_demo_{goal[0]}_{goal[1]}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to '{filename}'")
    plt.close(fig)

    # Test on multiple goals for comparison
    if len(sys.argv) < 3:  # Only if using default goal
        other_goals = [(7, 7), (0, 7), (3, 3), (2, 1), (6, 2)]
        other_goals = [g for g in other_goals if g != goal]
        if other_goals:
            print(f"\n🔍 Testing adaptability on other goals...")
            test_multiple_goals(agent, env, other_goals[:3])

    print(f"\n💡 Try different goals:")
    print(f"  python run_ppo.py 7 7")
    print(f"  python run_ppo.py 0 7")
    print(f"  python run_ppo.py 3 3")
    print(f"  python run_ppo.py 1 1")


if __name__ == "__main__":
    main()