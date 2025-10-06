# PPO GridWorld Navigation 🚀

This project demonstrates a **PPO (Proximal Policy Optimization)** agent navigating a **GridWorld environment with obstacles**.  
The agent is trained to reach a **goal position** while avoiding obstacles, using reinforcement learning.

---

## 📂 Project Structure
- **`grid_env.py`** – Custom **GridWorld environment** (Gym-compatible).  
- **`robust_training.py`** – Train a PPO agent with **domain randomization** for obstacle robustness.  
- **`run_ppo.py`** – Run and visualize a trained PPO agent on any goal position.  

A trained model is saved as:
- `robust_ppo_model.pkl`

---

## ⚙️ Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

1. Install **uv** (if you don’t already have it):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

Now you’re ready to train and run the PPO agent 🚀

---

## 🎯 Training the Agent
Run robust PPO training (with random obstacle layouts):
```bash
uv run robust_training.py
```

This will:
- Train the agent for several episodes.
- Save the trained model to `robust_ppo_model.pkl`.
- Test the agent on different obstacle layouts.

---

## 🎮 Running a Trained Agent
Test the agent on a specific **goal position**:
```bash
uv run  run_ppo.py [goal_x] [goal_y]
```

Examples:
```bash
uv run run_ppo.py 7 7
uv run run_ppo.py 0 7
uv run run_ppo.py 3 3
```

If no coordinates are provided, it defaults to goal `(5, 7)`.

---

## 📊 Features
- **Custom GridWorld** with obstacles and rewards.  
- **Robust PPO training** with domain randomization.  
- **Visualization**: Generates PNG plots of the agent’s navigation path (saved in `visualizations/`).  
- **Testing multiple goals**: Evaluate agent adaptability across different targets.  

---

## 🗺️ Example Output
- Terminal shows step-by-step actions and rewards.
- A path visualization is saved, e.g.:
  ```
  visualizations/ppo_demo_5_7.png
  ```