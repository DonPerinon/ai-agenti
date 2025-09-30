#!/usr/bin/env python3
"""
Goal-conditioned agent runner - demonstrates agent on any goal position.
Usage: python run_goal_conditioned.py [goal_x] [goal_y]
"""

import sys
import matplotlib.pyplot as plt
from grid_environment import GridWorld
from goal_conditioned_agent import GoalConditionedQLearningAgent


def show_run_with_goal(agent, env, goal_pos, show_steps=True):
    """Demonstrate agent reaching a specific goal"""
    env.goal_pos = goal_pos
    env.reset()

    agent_pos_idx = env.get_state_index(env.current_pos)
    goal_pos_idx = env.get_state_index(goal_pos)
    steps = 0
    total_reward = 0
    path = [env.current_pos]

    print(f"\n🎮 Goal-Conditioned Agent Demonstration")
    print(f"Starting: {env.current_pos} → Goal: {goal_pos}")
    print(f"Obstacles: {env.obstacles}")

    if show_steps:
        print("\nStep-by-step moves:")

    action_names = ['↑ (up)', '→ (right)', '↓ (down)', '← (left)']

    # Turn off exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while steps < 150:
        # Get action for current agent position and target goal
        action = agent.get_action(agent_pos_idx, goal_pos_idx)

        # Execute action
        next_state, reward, done = env.step(action)
        agent_pos_idx = env.get_state_index(next_state)
        path.append(env.current_pos)
        steps += 1
        total_reward += reward

        if show_steps:
            print(f"  Step {steps:2d}: {action_names[action]:<12} → {env.current_pos} (reward: {reward:5.2f})")

        if done:
            print(f"\n🎉 SUCCESS! Goal {goal_pos} reached in {steps} steps!")
            break

    agent.epsilon = old_epsilon

    if not done:
        print(f"\n⚠️  Goal {goal_pos} not reached in {steps} steps")

    print(f"Total reward: {total_reward:.2f}")
    print(f"Path: {' → '.join(map(str, path))}")

    return steps, total_reward, done


def show_policy_for_goal(agent, env, goal_pos):
    """Generate and save policy visualization for specific goal"""
    env.goal_pos = goal_pos
    env.reset()

    goal_pos_idx = env.get_state_index(goal_pos)
    policy = agent.get_policy_for_goal(goal_pos_idx)

    print(f"\n🗺️  Generating policy visualization for goal {goal_pos}...")
    fig, _ = env.visualize(policy=policy)
    filename = f'policy_goal_{goal_pos[0]}_{goal_pos[1]}.png'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Policy saved to '{filename}'")

    # Show some policy information
    print(f"\nPolicy info for goal {goal_pos}:")
    print(f"  • Grid size: {env.width}×{env.height}")
    print(f"  • Actions: ↑(0) →(1) ↓(2) ←(3)")

    plt.close()


def test_multiple_goals(agent, env, goals):
    """Test agent on multiple goals"""
    print(f"\n🔄 Testing on {len(goals)} different goals...")
    print("-" * 60)

    results = []
    for i, goal in enumerate(goals):
        print(f"\nGoal {i+1}: {goal}")
        steps, reward, success = show_run_with_goal(agent, env, goal, show_steps=False)
        results.append((goal, steps, reward, success))

        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"Result: {steps} steps, reward {reward:.2f} - {status}")

    # Summary
    print(f"\n📊 Summary of {len(goals)} goals:")
    successes = sum(1 for _, _, _, success in results if success)
    print(f"  • Success rate: {successes}/{len(goals)} ({successes/len(goals)*100:.1f}%)")

    if successes > 0:
        successful_results = [(steps, reward) for _, steps, reward, success in results if success]
        avg_steps = sum(steps for steps, _ in successful_results) / len(successful_results)
        avg_reward = sum(reward for _, reward in successful_results) / len(successful_results)
        print(f"  • Average steps (successful): {avg_steps:.1f}")
        print(f"  • Average reward (successful): {avg_reward:.2f}")


def main():
    print("🤖 Goal-Conditioned Q-Learning Agent Runner")
    print("=" * 45)

    # Parse command line arguments for goal position
    if len(sys.argv) >= 3:
        try:
            goal_x = int(sys.argv[1])
            goal_y = int(sys.argv[2])
            goal_pos = (goal_x, goal_y)
        except ValueError:
            print("❌ Invalid goal coordinates. Use integers.")
            sys.exit(1)
    else:
        # Default goal
        goal_pos = (5, 7)
        print(f"Using default goal: {goal_pos}")

    # Setup environment
    env = GridWorld(width=8, height=8, start=(0, 1), goal=goal_pos)

    # Validate goal position
    if goal_pos[0] < 0 or goal_pos[0] >= env.width or goal_pos[1] < 0 or goal_pos[1] >= env.height:
        print(f"❌ Goal {goal_pos} is outside grid boundaries ({env.width}×{env.height})")
        sys.exit(1)

    if goal_pos in env.obstacles:
        print(f"❌ Goal {goal_pos} is on an obstacle!")
        sys.exit(1)

    # Create agent and load model
    n_positions = env.width * env.height
    agent = GoalConditionedQLearningAgent(n_positions=n_positions, n_actions=env.actions)

    model_file = "goal_conditioned_model.pkl"
    if not agent.load_model(model_file):
        print(f"\n❌ No trained goal-conditioned model found!")
        print(f"Please run 'python train_goal_conditioned.py' first to train the agent.")
        sys.exit(1)

    # Run demonstration
    print(f"\n🎯 Testing goal position: {goal_pos}")

    # Check if this goal is in valid range
    if 0 <= goal_pos[0] < env.width and 0 <= goal_pos[1] < env.height and goal_pos not in env.obstacles:
        show_run_with_goal(agent, env, goal_pos, show_steps=True)
        show_policy_for_goal(agent, env, goal_pos)

    else:
        print(f"❌ Invalid goal position {goal_pos}")



if __name__ == "__main__":
    main()