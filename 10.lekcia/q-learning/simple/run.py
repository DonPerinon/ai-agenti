#!/usr/bin/env python3
"""
Agent runner script - demonstrates trained Q-Learning agent.
Requires a pre-trained model (run train.py first).
"""

import sys
import matplotlib.pyplot as plt
from grid_environment import GridWorld
from q_agent import QLearningAgent


def show_single_run(agent, env, show_steps=True):
    """Demonstrate agent with detailed step-by-step output"""
    env.reset()
    state_idx = env.get_state_index()
    steps = 0
    total_reward = 0
    path = [env.current_pos]

    print(f"\n🎮 Agent Demonstration")
    print(f"Starting: {env.current_pos} → Goal: {env.goal_pos}")
    print(f"Obstacles: {env.obstacles}")

    if show_steps:
        print("\nStep-by-step moves:")

    action_names = ['↑ (up)', '→ (right)', '↓ (down)', '← (left)']

    while steps < 150:
        # Get action from trained agent (no exploration)
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0
        action = agent.get_action(state_idx)
        agent.epsilon = old_epsilon

        # Execute action
        next_state, reward, done = env.step(action)
        state_idx = env.get_state_index(next_state)
        path.append(env.current_pos)
        steps += 1
        total_reward += reward

        if show_steps:
            print(f"  Step {steps:2d}: {action_names[action]:<12} → {env.current_pos} (reward: {reward:5.2f})")

        if done:
            print(f"\n🎉 SUCCESS! Goal reached in {steps} steps!")
            break

    if not done:
        print(f"\n⚠️  Goal not reached in {steps} steps")

    print(f"Total reward: {total_reward:.2f}")
    print(f"Path: {' → '.join(map(str, path))}")

    return steps, total_reward, done


def show_multiple_runs(agent, env, episodes=5):
    """Run multiple episodes and show summary"""
    print(f"\n🔄 Running {episodes} episodes...")
    print("-" * 50)

    results = []
    for episode in range(episodes):
        env.reset()
        state_idx = env.get_state_index()
        steps = 0
        total_reward = 0
        path = [env.current_pos]

        # Turn off exploration
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0

        while steps < 350:
            action = agent.get_action(state_idx)
            next_state, reward, done = env.step(action)
            state_idx = env.get_state_index(next_state)
            path.append(env.current_pos)
            steps += 1
            total_reward += reward

            if done:
                break

        agent.epsilon = old_epsilon
        results.append((steps, total_reward, done))

        status = "✅ SUCCESS" if done else "❌ FAILED"
        print(f"Episode {episode + 1:2d}: {steps:2d} steps, reward {total_reward:5.2f} - {status}")

    # Summary
    successes = sum(1 for _, _, done in results if done)
    avg_steps = sum(steps for steps, _, done in results if done) / max(successes, 1)
    avg_reward = sum(reward for _, reward, done in results if done) / max(successes, 1)

    print(f"\n📊 Summary:")
    print(f"  • Success rate: {successes}/{episodes} ({successes/episodes*100:.1f}%)")
    if successes > 0:
        print(f"  • Average steps (successful): {avg_steps:.1f}")
        print(f"  • Average reward (successful): {avg_reward:.2f}")


def show_policy_visualization(agent, env):
    """Generate and save policy visualization"""
    env.reset()
    policy = agent.get_policy()

    print(f"\n🗺️  Generating policy visualization...")
    fig, _ = env.visualize(policy=policy)
    fig.savefig('current_policy.png', dpi=150, bbox_inches='tight')
    print(f"Policy saved to 'current_policy.png'")

    # Show some policy information
    print(f"\nPolicy info:")
    print(f"  • Grid size: {env.width}×{env.height}")
    print(f"  • Total states: {len(policy)}")
    print(f"  • Actions: ↑(0) →(1) ↓(2) ←(3)")

    plt.close()


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "single"

    print("🤖 Q-Learning Agent Runner")
    print("=" * 30)

    # Setup environment
    env = GridWorld(width=8, height=8, start=(3, 1), goal=(5, 7))

    # Create agent and load model
    n_states = env.width * env.height
    agent = QLearningAgent(n_states=n_states, n_actions=env.actions)

    model_file = "q_table_model.pkl"
    if not agent.load_model(model_file):
        print(f"\n❌ No trained model found!")
        print(f"Please run 'python train.py' first to train the agent.")
        sys.exit(1)

    # Ensure no exploration during demonstration
    agent.epsilon = 0.0

    # Run based on mode
    if mode == "single":
        show_single_run(agent, env, show_steps=True)
        show_policy_visualization(agent, env)

    elif mode == "multi":
        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        show_multiple_runs(agent, env, episodes)

    elif mode == "quiet":
        show_single_run(agent, env, show_steps=False)

    elif mode == "policy":
        show_policy_visualization(agent, env)
        print(f"Policy visualization generated.")

    else:
        print(f"Usage:")
        print(f"  python run.py [mode] [options]")
        print(f"")
        print(f"Modes:")
        print(f"  single    - Single detailed run (default)")
        print(f"  multi N   - Run N episodes (default: 5)")
        print(f"  quiet     - Single run without step details")
        print(f"  policy    - Generate policy visualization only")
        print(f"")
        print(f"Examples:")
        print(f"  python run.py")
        print(f"  python run.py single")
        print(f"  python run.py multi 10")
        print(f"  python run.py quiet")
        print(f"  python run.py policy")


if __name__ == "__main__":
    main()