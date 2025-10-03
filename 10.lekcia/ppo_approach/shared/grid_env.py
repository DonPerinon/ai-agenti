import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """Grid World environment compatible with gym interface for PPO"""

    def __init__(self, width: int = 8, height: int = 8,
                 start: Tuple[int, int] = (0, 0),
                 goal: Tuple[int, int] = (7, 7),
                 obstacles: List[Tuple[int, int]] = None):
        super(GridWorldEnv, self).__init__()

        self.width = width
        self.height = height
        self.start_pos = start
        self.goal_pos = goal
        self.obstacles = set(obstacles or [(1, 0), (1, 1) , (3, 4), (5, 1), (6, 5), (1, 6) , (4, 3), (2, 5), (5, 6)])

        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Observation space: [agent_x, agent_y, goal_x, goal_y, grid_features...]
        # We'll use a flattened representation including relative position to goal
        # Plus 9 values for 3x3 local obstacle grid around agent
        self.observation_space = spaces.Box(
            low=0, high=max(width, height),
            shape=(15,), dtype=np.float32
        )

        self.action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.action_names = ['↑', '→', '↓', '←']

        self.current_pos = None
        self.last_failed_action = -1  # Track failed actions for better exploration
        self.position_history = []  # Track position history for stagnation detection
        self.stagnation_counter = 0  # Count how many steps agent has been stuck
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_pos = self.start_pos
        self.last_failed_action = -1
        self.position_history = [self.start_pos]
        self.stagnation_counter = 0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current observation as numpy array"""
        agent_x, agent_y = self.current_pos
        goal_x, goal_y = self.goal_pos

        # Relative position to goal
        rel_x = goal_x - agent_x
        rel_y = goal_y - agent_y

        # Distance to goal
        manhattan_dist = abs(rel_x) + abs(rel_y)
        euclidean_dist = np.sqrt(rel_x**2 + rel_y**2)

        # Local obstacle information (3x3 grid around agent)
        local_obstacles = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                check_pos = (agent_y + dy, agent_x + dx)
                if (check_pos[0] < 0 or check_pos[0] >= self.height or
                    check_pos[1] < 0 or check_pos[1] >= self.width):
                    local_obstacles.append(1.0)  # Boundary as obstacle
                elif check_pos in self.obstacles:
                    local_obstacles.append(1.0)  # Obstacle
                else:
                    local_obstacles.append(0.0)  # Free space

        obs = np.array([
            agent_x / self.width,      # Normalized agent position
            agent_y / self.height,
            rel_x / self.width,        # Normalized relative position to goal
            rel_y / self.height,
            manhattan_dist / (self.width + self.height),  # Normalized distances
            euclidean_dist / np.sqrt(self.width**2 + self.height**2)
        ] + local_obstacles, dtype=np.float32)

        return obs

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions from current position"""
        valid_actions = []
        for action in range(4):
            dy, dx = self.action_map[action]
            new_pos = (self.current_pos[0] + dy, self.current_pos[1] + dx)

            # Check if action leads to valid position
            if (0 <= new_pos[0] < self.height and
                0 <= new_pos[1] < self.width and
                new_pos not in self.obstacles):
                valid_actions.append(action)
        return valid_actions

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return (observation, reward, done, info)"""
        old_pos = self.current_pos
        dy, dx = self.action_map[action]
        new_pos = (self.current_pos[0] + dy, self.current_pos[1] + dx)

        # Detect stagnation (staying in same position for too long)
        if len(self.position_history) >= 10:
            recent_positions = self.position_history[-10:]
            if all(pos == self.current_pos for pos in recent_positions):
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

        # Check boundaries and obstacles
        if (new_pos[0] < 0 or new_pos[0] >= self.height or
            new_pos[1] < 0 or new_pos[1] >= self.width):
            # Hit boundary - stay in place, penalty
            reward = -0.5
            # Add extra penalty if agent is stuck
            if self.stagnation_counter > 5:
                reward -= 0.5
        elif new_pos in self.obstacles:
            # Hit obstacle - stay in place, penalty varies by exploration
            if action != self.last_failed_action:
                reward = -0.5  # Less penalty for trying different directions
            else:
                reward = -1.5  # Higher penalty for repeating failed action
            self.last_failed_action = action
            # Add extra penalty if agent is stuck
            if self.stagnation_counter > 5:
                reward -= 1.0
        else:
            # Valid move
            self.current_pos = new_pos
            self.last_failed_action = -1  # Reset failed action tracker
            self.stagnation_counter = 0  # Reset stagnation counter

            if self.current_pos == self.goal_pos:
                reward = 10.0  # Large reward for reaching goal
            else:
                # Distance-based reward with exploration bonus
                old_dist = abs(old_pos[0] - self.goal_pos[0]) + abs(old_pos[1] - self.goal_pos[1])
                new_dist = abs(new_pos[0] - self.goal_pos[0]) + abs(new_pos[1] - self.goal_pos[1])

                reward = -0.01  # Time penalty
                if new_dist < old_dist:
                    reward += 0.15  # Reward for getting closer
                elif new_dist > old_dist:
                    reward -= 0.02  # Smaller penalty for strategic retreating

                # Add exploration bonus for visiting new positions
                if new_pos not in self.position_history[-20:]:  # Check last 20 positions
                    reward += 0.05

        # Update position history
        self.position_history.append(self.current_pos)
        # Keep only recent history to prevent memory issues
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-50:]

        done = self.current_pos == self.goal_pos
        info = {
            'current_pos': self.current_pos,
            'goal_pos': self.goal_pos,
            'manhattan_distance': abs(self.current_pos[0] - self.goal_pos[0]) + abs(self.current_pos[1] - self.goal_pos[1]),
            'valid_actions': self.get_valid_actions(),
            'stagnation_counter': self.stagnation_counter
        }

        return self._get_observation(), reward, done, info

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not obstacle)"""
        return (0 <= pos[0] < self.height and 0 <= pos[1] < self.width and
                pos not in self.obstacles)

    def set_goal(self, goal: Tuple[int, int]):
        """Dynamically change goal position"""
        if self.is_valid_position(goal):
            self.goal_pos = goal
        else:
            raise ValueError(f"Invalid goal position: {goal}")

    def sample_random_goal(self) -> Tuple[int, int]:
        """Sample a random valid goal position"""
        while True:
            goal = (np.random.randint(0, self.height), np.random.randint(0, self.width))
            if goal not in self.obstacles and goal != self.start_pos:
                return goal

    def render(self, mode='human', path: Optional[List[Tuple[int, int]]] = None):
        """Render the environment"""
        if mode == 'human':
            return self.visualize(path)
        return None

    def visualize(self, path: Optional[List[Tuple[int, int]]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize the grid world"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create grid
        grid = np.zeros((self.height, self.width))

        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1

        # Mark goal
        grid[self.goal_pos] = 2

        # Mark start
        grid[self.start_pos] = 1

        # Mark current position (if different from start)
        if self.current_pos != self.start_pos:
            grid[self.current_pos] = 3

        # Color mapping
        colors = ['white', 'lightgreen', 'gold', 'lightblue', 'black']
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        cmap = plt.cm.colors.ListedColormap(colors)
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        ax.imshow(grid, cmap=cmap, norm=norm)

        # Draw path if provided
        if path and len(path) > 1:
            path_y = [p[0] for p in path]
            path_x = [p[1] for p in path]
            ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.7, label='Path')

        # Add grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1, alpha=0.3)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color='black', linewidth=1, alpha=0.3)

        # Add labels
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.start_pos:
                    ax.text(j, i, 'S', ha='center', va='center', fontsize=16, fontweight='bold')
                elif (i, j) == self.goal_pos:
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=16, fontweight='bold')
                elif (i, j) == self.current_pos and (i, j) != self.start_pos:
                    ax.text(j, i, 'A', ha='center', va='center', fontsize=16, fontweight='bold')
                elif (i, j) in self.obstacles:
                    ax.text(j, i, '█', ha='center', va='center', fontsize=16, fontweight='bold')

        ax.set_title(f'Grid World - Goal: {self.goal_pos}', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

        return fig, ax