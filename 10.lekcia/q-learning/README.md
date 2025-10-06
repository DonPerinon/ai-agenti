# 🧠 Reinforcement Learning: Grid World with Q-Learning

This project implements a complete Reinforcement Learning system featuring a **Grid World** environment and a **Q-Learning** agent.  
The agent learns to navigate through a grid with obstacles to reach a goal position.

---

## ✨ Features

- **Custom Grid World Environment**: 8×8 grid with configurable obstacles  
- **Q-Learning Agent**: Tabular Q-learning with epsilon-greedy exploration  
- **Training Visualization**: Real-time training progress plots  
- **Policy Visualization**: Visual representation of learned policy  
- **Agent Demonstration**: Watch the trained agent navigate the environment  

---

## 📁 Project Structure

```
.
├── main.py              # Full training pipeline
├── run_agent.py         # Run pre-trained agent demonstrations
├── grid_environment.py  # Grid World environment implementation
├── q_agent.py           # Q-Learning agent implementation
├── pyproject.toml       # Project dependencies
└── README.md            # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
uv sync
```

### Full Training & Evaluation

Run the complete training pipeline:

```bash
uv run python main.py
```

This will:
- Train a Q-Learning agent for 2000 episodes  
- Generate training progress visualizations  
- Show policy visualization with directional arrows  
- Demonstrate the trained agent  
- Save results as PNG files  

### Run Pre-trained Agent

For quick demonstrations without retraining:

```bash
# Single detailed run
uv run python run_agent.py

# Multiple runs (5 episodes, no step details)
uv run python run_agent.py multi 5 false

# Multiple runs with detailed steps
uv run python run_agent.py multi 3 true
```

---

## 🌍 Environment Details

### Grid World Setup
- **Size**: 8×8 grid (64 states)  
- **Start Position**: (0, 0) – top-left corner  
- **Goal Position**: (7, 7) – bottom-right corner  
- **Obstacles**: Fixed positions at (2,2), (3,4), (5,1), (6,5), (1,6)

### Actions
| Action | Direction |
|:--:|:--|
| 0 | ↑ (Up) |
| 1 | → (Right) |
| 2 | ↓ (Down) |
| 3 | ← (Left) |

### Rewards
| Condition | Reward |
|:--|:--|
| Reaching the goal | +10.0 |
| Hitting an obstacle | -1.0 |
| Hitting a boundary | -0.1 |
| Each step | -0.01 (encourages shorter paths) |

---

## ⚙️ Agent Configuration

### Q-Learning Parameters
- **States**: 64 (8×8 grid)  
- **Actions**: 4 (up, right, down, left)  
- **Learning Rate**: 0.1  
- **Discount Factor**: 0.95  
- **Initial Epsilon**: 1.0 (100% exploration)  
- **Epsilon Decay**: 0.995  
- **Minimum Epsilon**: 0.01  

### Training Process
- **Episodes**: 2000 (full training)  
- **Max Steps per Episode**: 200  
- **Exploration Strategy**: Epsilon-greedy with decay  

---

## 📈 Results

### Training Performance
| Metric | Value |
|:--|:--|
| Final Average Reward | 9.87 |
| Average Steps to Goal | 14 |
| Success Rate | 100% (after training) |
| Optimal Path Length | 14 steps |

### Generated Files
- `training_results.png`: Training progress (rewards & episode lengths)  
- `grid_world_policy.png`: Environment with learned policy arrows  
- `current_policy.png`: Policy visualization from agent runs  
- `q_table_model.pkl`: Saved trained Q-table and agent parameters  

---

## 🧩 Implementation Details

### `GridWorld` Class (`grid_environment.py`)
```python
# Key methods:
- reset(): Reset environment to start position
- step(action): Execute action, return (state, reward, done)
- visualize(): Create visual representation with policy arrows
```

### `QLearningAgent` Class (`q_agent.py`)
```python
# Key methods:
- get_action(state): Epsilon-greedy action selection
- update_q_value(): Q-learning update rule
- train(): Complete training loop with statistics
- evaluate(): Performance evaluation without exploration
```

---

## 📊 Training Progress

The agent learns progressively:

1. **Episodes 1–100**: Random exploration (ε = 1.0 → 0.6)  
2. **Episodes 100–500**: Rapid improvement (ε = 0.6 → 0.08)  
3. **Episodes 500–2000**: Fine-tuning and convergence (ε = 0.08 → 0.01)

---

## 🧾 Example Output

```
🧠 Reinforcement Learning: Grid World with Q-Learning
==================================================
Environment: 8x8 grid
Start: (0, 0), Goal: (7, 7)
Obstacles: [(2,2), (3,4), (5,1), (6,5), (1,6)]

🚀 Training agent for 2000 episodes...
Episode 2000/2000, Avg Reward: 9.87, Avg Length: 14.09, Epsilon: 0.010

🧮 Evaluating trained agent...
Average reward over 100 episodes: 9.87
Average steps to goal: 14.00

🏁 Goal reached in 14 steps!
Path: (0,0) → (1,0) → (2,1) → (3,1) → (3,2) → (4,2) → (4,3) → (4,4) → 
      (4,5) → (4,6) → (5,6) → (5,7) → (6,7) → (7,7)
```

---

## 🧠 Customization

### Modify Environment
```python
# In main.py or run_agent.py
env = GridWorld(
    width=10,
    height=10,
    start=(0, 0),
    goal=(9, 9),
    obstacles=[(2,2), (5,5)]
)
```

### Adjust Agent Parameters
```python
agent = QLearningAgent(
    learning_rate=0.2,
    discount_factor=0.9,
    epsilon_decay=0.99
)
```

---

## 🧩 Requirements

- Python ≥ 3.12  
- numpy ≥ 1.24.0  
- matplotlib ≥ 3.7.0  
- gymnasium ≥ 0.28.0 *(for future extensions)*  
- torch ≥ 2.0.0 *(for potential DQN implementation)*  

---

## 🎓 Learning Outcomes

This implementation demonstrates:
- **Tabular Q-Learning**: Classic RL algorithm  
- **Exploration vs Exploitation**: Epsilon-greedy strategy  
- **Reward Engineering**: Designing reward signals  
- **Policy Visualization**: Understanding learned behaviors  
- **Training Monitoring**: Progress tracking and evaluation  

---

## 💾 Training Data Storage

The trained Q-table and agent parameters are automatically saved to `q_table_model.pkl` after training.

### Saving and Loading
```python
# Save model
agent.save_model("my_model.pkl")

# Load model
if agent.load_model("my_model.pkl"):
    print("Model loaded successfully!")

# Check if model exists
if QLearningAgent.model_exists("my_model.pkl"):
    print("Model file found!")
```

### Stored Data
- **Q-table**: Complete state-action value matrix  
- **Agent parameters**: Learning rate, discount factor, epsilon values  
- **Training history**: Episode rewards and lengths  
- **Model metadata**: State/action space dimensions  

### File Persistence
- **Location**: Current working directory  
- **Format**: Python pickle file (`.pkl`)  
- **Size**: ~50KB for 8×8 grid (64 states × 4 actions)  
- **Compatibility**: Works across different Python sessions  

---

## 🚀 Future Extensions

Potential improvements:
- **Deep Q-Network (DQN)** – Neural network-based Q-learning  
- **Larger Environments** – Scale to bigger grids  
- **Dynamic Obstacles** – Moving or changing obstacles  
- **Multiple Goals** – Complex navigation tasks  
- **Policy Gradient Methods** – REINFORCE, PPO implementations  

