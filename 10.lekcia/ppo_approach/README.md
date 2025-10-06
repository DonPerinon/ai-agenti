# PPO Approach: Grid World Navigation

This project implements and compares two versions of Proximal Policy Optimization (PPO) for navigating a grid world environment with obstacles. The project demonstrates the evolution from a basic PPO implementation to an improved version with advanced features.

## 🎯 Overview

The project consists of two main approaches:
- **Simple PPO**: Basic implementation following standard PPO algorithms
- **Improved PPO**: Enhanced version with action masking, anti-stagnation mechanisms, and better exploration

Both agents learn to navigate an 8×8 grid world with obstacles to reach randomly sampled goal positions.

## 📁 Project Structure

```
AiKurz/                    
    ├── 10.lekcia/  # Root project with pyproject.toml
    |   ├── pyproject.toml         # UV project configuration
    |   ├── ppo_approach/
    |   │   ├── shared/                 # Common components
    |   │   │   ├── grid_env.py        # Grid world environment
    |   │   │   └── ppo_agent.py       # Basic PPO agent implementation
    |   │   ├── simple/                # Simple PPO approach
    |   │   │   ├── run_ppo.py         # Test simple PPO agent
    |   │   │   └── robust_training.py # Training script for simple PPO
    |   │   ├── improved/              # Improved PPO approach
    |   │   │   ├── improved_ppo_agent.py    # Enhanced PPO agent
    |   │   │   ├── train_improved_ppo.py    # Training script
    |   │   │   └── run_improved_ppo.py      # Test improved PPO agent
    |   │   ├── visualizations/        # Generated plots and visualizations
    |   │   ├── robust_ppo_model.pkl   # Trained simple PPO model
    |   │   └── improved_ppo_model.pkl # Trained improved PPO model
```

## 🚀 Quick Start

### Prerequisites
Dependencies are managed by `uv` from the root `pyproject.toml`. Make sure you're in the project root:

```bash
# From the root directory (where pyproject.toml is located)
uv sync  # Install dependencies
```

### Training Agents

**Simple PPO:**
```bash
# From ppo_approach directory
uv run simple/robust_training.py
```

**Improved PPO:**
```bash
# From ppo_approach directory
uv run python improved/train_improved_ppo.py
```

### Testing Agents

**Test simple PPO:**
```bash
# From ppo_approach directory
uv run python simple/run_ppo.py [goal_x] [goal_y]
# Example: uv run python simple/run_ppo.py 7 7
```

**Test improved PPO:**
```bash
# From ppo_approach directory
uv run python improved/run_improved_ppo.py [goal_x] [goal_y]
# Example: uv run python improved/run_improved_ppo.py 7 7
```

## 🔍 Key Differences Between Approaches

### Simple PPO (`shared/ppo_agent.py`)
- Basic actor-critic architecture
- Standard PPO loss function
- No action masking or constraint handling
- Pure exploration through policy randomness
- May get stuck on obstacles or boundaries

### Improved PPO (`improved/improved_ppo_agent.py`)
- **Action Masking**: Prevents invalid moves by masking unavailable actions
- **Anti-Stagnation**: Detects and penalizes repeated positions
- **Decaying Exploration**: Starts at 10% exploration, decays to 1%
- **Obstacle Awareness**: Uses environment's valid action information
- **Enhanced Architecture**: Includes dropout for better generalization

## 🏗️ Environment Details

### Grid World Features
- **Size**: 8×8 grid
- **Start Position**: (0, 0) - top-left corner
- **Obstacles**: Randomly placed barriers
- **Actions**: 4 directional moves (up, right, down, left)
- **Rewards**:
  - Reaching goal: +10
  - Each step: -0.1
  - Invalid moves: -1
  - Stagnation penalty (improved): -0.1 per stuck step

### Observation Space
The agent receives a feature vector containing:
- Current position (x, y)
- Goal position (x, y)
- Distance to goal
- Obstacle information in neighboring cells

## 📊 Performance Comparison

### Training Configuration
- **Episodes**: 15,000
- **Update Frequency**: Every 20 episodes
- **Learning Rate**: 3e-4
- **Discount Factor**: 0.99
- **PPO Clip Ratio**: 0.2

### Expected Results
- **Simple PPO**: ~60-70% success rate on various goals
- **Improved PPO**: ~80-90% success rate with faster convergence
- **Improved PPO** shows significantly better performance on:
  - Goals near obstacles
  - Corner positions
  - Complex navigation scenarios

## 🧪 Testing Different Goals

Both implementations support testing on custom goal positions:

```bash
# Test different goals (from ppo_approach directory)
uv run python simple/run_ppo.py 7 7        # Bottom-right corner
uv run python simple/run_ppo.py 0 7        # Top-right corner
uv run python simple/run_ppo.py 3 3        # Center position
uv run python simple/run_ppo.py 2 1        # Near start position

# Or test improved version
uv run python improved/run_improved_ppo.py 7 7
```

## 📈 Visualization

The project automatically generates:
- **Training curves**: Episode rewards and lengths over time
- **Path visualizations**: Agent's navigation paths for each test
- **Success rate plots**: Performance across different goals

Visualizations are saved in the `visualizations/` directory.

## 🔧 Customization

### Modifying the Environment
Edit `shared/grid_env.py` to:
- Change grid size
- Adjust obstacle placement
- Modify reward structure
- Add new features

### Tuning Hyperparameters
Key parameters in both agents:
- `lr`: Learning rate (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- `eps_clip`: PPO clipping ratio (default: 0.2)
- `k_epochs`: PPO update epochs (default: 4)
- `exploration_rate`: Initial exploration (improved only)

### Training Configuration
- `episodes`: Number of training episodes
- `max_steps`: Maximum steps per episode
- `update_freq`: How often to update policy

## 🏆 Advanced Features (Improved PPO)

### Action Masking
```python
# Get valid actions from environment
valid_actions = env.get_valid_actions()
# Policy automatically masks invalid actions
action = agent.select_action(obs, valid_actions)
```

### Anti-Stagnation
- Detects when agent revisits same positions
- Applies increasing penalties for stagnation
- Forces exploration through random valid actions

### Exploration Strategies
- **Training**: Decaying exploration rate with noise injection
- **Testing**: Optional epsilon-greedy for handling difficult scenarios

## 📋 Common Issues and Solutions

### Agent Gets Stuck
- **Simple PPO**: May require retraining or different hyperparameters
- **Improved PPO**: Use exploration mode during testing:
  ```python
  result = agent.test_on_goal(env, goal, use_exploration=True)
  ```

### Poor Performance on Specific Goals
- Check if goal is reachable and not on obstacles
- Increase training episodes for better generalization
- Use improved PPO for better obstacle navigation

### Training Not Converging
- Reduce learning rate
- Increase entropy coefficient for more exploration
- Check reward shaping in environment

## 🚀 Development Workflow

```bash
# From project root (where pyproject.toml is)
uv sync                    # Install/update dependencies
cd 10.lekcia/ppo_approach  # Navigate to this subproject

# Train and test
uv run improved/train_improved_ppo.py
uv run improved/run_improved_ppo.py 7 7
```

## 📚 References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [PyTorch RL Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html)

## 📄 License

This project is for educational purposes as part of the AI course curriculum.