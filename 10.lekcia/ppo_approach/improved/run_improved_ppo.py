#!/usr/bin/env python3
"""
Improved PPO Agent Runner - Test improved agent on any goal position
Usage: python run_improved_ppo.py [goal_x] [goal_y] [--exploration]
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Get the absolute path to the parent directory of the script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add BASE_DIR to sys.path if not already present
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from shared.grid_env import GridWorldEnv
from improved_ppo_agent import ImprovedPPOAgent
import os


def run_single_episode(agent, env, goal, max_steps=200, show_steps=True, use_exploration=False):
    """Run a single episode with detailed output"""
    env.set_goal(goal)
    obs = env.reset()

    steps = 0
    total_reward = 0
    path = [env.current_pos]
    done = False
    stuck_positions = set()

    print(f"\n🎮 Improved PPO Agent Demonstration")
    print(f"Starting: {env.current_pos} → Goal: {goal}")
    print(f"Obstacles: {list(env.obstacles)}")
    print(f"Exploration mode: {'ON' if use_exploration else 'OFF'}")

    if show_steps:
        print("\nStep-by-step moves:")

    action_names = ['↑ (up)', '→ (right)', '↓ (down)', '← (left)']

    for step in range(max_steps):
        # Get valid actions from environment
        valid_actions = env.get_valid_actions()

        if use_exploration:
            # Use epsilon-greedy exploration
            if np.random.random() < 0.15:  # 15% random exploration
                if valid_actions:
                    action = np.random.choice(valid_actions)
                else:
                    action = np.random.randint(0, 4)
            else:
                action = agent.select_action(obs, valid_actions, training=False)
        else:
            # Use policy without exploration but with action masking
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_logits, _ = agent.policy(obs_tensor)

                # Apply action masking
                if valid_actions and len(valid_actions) > 0:
                    mask = torch.full_like(action_logits, float('-inf'))
                    mask[0, valid_actions] = 0
                    action_logits = action_logits + mask

                action = torch.argmax(action_logits, dim=1).item()

        # Anti-stagnation: if stuck in same position too long, try random valid action
        if env.current_pos in stuck_positions and step > 10:
            if valid_actions and np.random.random() < 0.4:
                action = np.random.choice(valid_actions)
                if show_steps:
                    print(f"  Step {steps+1:2d}: {action_names[action]:<12} → ANTI-STAGNATION MOVE")

        # Execute action
        next_obs, reward, done, info = env.step(action)
        path.append(env.current_pos)
        total_reward += reward
        steps += 1

        if show_steps:
            stagnation_info = ""
            if 'stagnation_counter' in info and info['stagnation_counter'] > 0:
                stagnation_info = f" [STUCK:{info['stagnation_counter']}]"
            valid_info = f" [Valid:{len(valid_actions)}]" if valid_actions else " [No valid actions!]"
            print(f"  Step {steps:2d}: {action_names[action]:<12} → {env.current_pos} (reward: {reward:5.2f}){stagnation_info}{valid_info}")

        # Track stuck positions
        if len(path) > 5 and path[-1] == path[-5]:
            stuck_positions.add(path[-1])

        obs = next_obs

        if done:
            print(f"\n🎉 SUCCESS! Goal {goal} reached in {steps} steps!")
            break

    if not done:
        print(f"\n⚠️  Goal {goal} not reached in {max_steps} steps")
        if len(stuck_positions) > 0:
            print(f"Agent got stuck at positions: {stuck_positions}")

    print(f"Total reward: {total_reward:.2f}")
    print(f"Path length: {len(path)} positions")
    print(f"Unique positions visited: {len(set(path))}")

    return steps, total_reward, done, path


def compare_with_old_agent(improved_agent, env, goals):
    """Compare improved agent with original behavior simulation"""
    print(f"\n📊 Comparing with Original PPO Behavior")
    print("-" * 50)

    from shared.ppo_agent import PPOAgent

    # Try to load old agent
    old_agent = PPOAgent(15, 4)
    if old_agent.load_model("robust_ppo_model.pkl"):
        print("Loaded original PPO model for comparison")

        for goal in goals:
            if not env.is_valid_position(goal) or goal in env.obstacles:
                continue

            print(f"\n🎯 Goal: {goal}")

            # Test improved agent
            improved_result = improved_agent.test_on_goal(env, goal, episodes=3, visualize=False)

            # Test old agent
            old_result = old_agent.test_on_goal(env, goal, episodes=3, visualize=False)

            print(f"  Improved PPO: Success {improved_result['success_rate']:.2f}")
            print(f"  Original PPO: Success {old_result['success_rate']:.2f}")

            if improved_result['success_rate'] > old_result['success_rate']:
                print("  ✅ Improved agent performs better!")
            elif improved_result['success_rate'] == old_result['success_rate']:
                print("  → Same performance")
            else:
                print("  ⚠️  Original agent was better for this goal")


def main():
    print("🤖 Improved PPO Agent Runner")
    print("=" * 35)

    # Parse command line arguments
    use_exploration = '--exploration' in sys.argv or '-e' in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]

    if len(args) >= 2:
        try:
            goal_x = int(args[0])
            goal_y = int(args[1])
            goal = (goal_x, goal_y)
        except ValueError:
            print("❌ Invalid goal coordinates. Use integers.")
            sys.exit(1)
    else:
        # Default goal - the one that was problematic
        goal = (7, 7)
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
    agent = ImprovedPPOAgent(obs_dim=obs_dim, action_dim=action_dim)

    # Load trained model
    model_file = "improved_ppo_model.pkl"
    if not agent.load_model(model_file):
        print(f"\n❌ No trained improved PPO model found!")
        print(f"Please run 'python train_improved_ppo.py' first to train the agent.")
        sys.exit(1)

    # Run demonstration
    print(f"\n🎯 Testing Improved PPO agent on goal: {goal}")

    # Single detailed run
    steps, reward, success, path = run_single_episode(
        agent, env, goal, show_steps=True, use_exploration=use_exploration)

    # Generate visualization
    print(f"\n🗺️  Generating visualization...")
    fig, ax = env.visualize(path)
    mode_str = " (with exploration)" if use_exploration else ""
    fig.suptitle(f'Improved PPO Agent Navigation to {goal}{mode_str}', fontsize=16)
    filename = f'ppo_approach/improved_ppo_demo_{goal[0]}_{goal[1]}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to '{filename}'")
    plt.close(fig)

    # Test multiple runs to show consistency
    print(f"\n🔄 Testing consistency with 5 runs...")
    results = []
    for i in range(5):
        _, _, success, _ = run_single_episode(
            agent, env, goal, show_steps=False, use_exploration=use_exploration)
        results.append(success)
        print(f"  Run {i+1}: {'✅ SUCCESS' if success else '❌ FAILED'}")

    success_rate = sum(results) / len(results)
    print(f"\nConsistency: {success_rate:.1%} success rate over 5 runs")

    # If not successful, try with exploration
    if success_rate < 0.8 and not use_exploration:
        print(f"\n🔍 Low success rate detected. Trying with exploration...")
        exploration_results = []
        for i in range(3):
            _, _, success, _ = run_single_episode(
                agent, env, goal, show_steps=False, use_exploration=True)
            exploration_results.append(success)
            print(f"  Exploration run {i+1}: {'✅ SUCCESS' if success else '❌ FAILED'}")

        exploration_success_rate = sum(exploration_results) / len(exploration_results)
        print(f"With exploration: {exploration_success_rate:.1%} success rate")

    # Test on multiple goals for comparison
    if len(args) < 2:  # Only if using default goal
        other_goals = [(0, 7), (3, 3), (6, 2)]
        compare_with_old_agent(agent, env, [goal] + other_goals)

    print(f"\n💡 Try different goals and modes:")
    print(f"  python run_improved_ppo.py 7 7")
    print(f"  python run_improved_ppo.py 0 7 --exploration")
    print(f"  python run_improved_ppo.py 3 3 -e")


if __name__ == "__main__":
    main()