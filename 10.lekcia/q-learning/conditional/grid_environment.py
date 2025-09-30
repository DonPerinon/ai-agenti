import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class GridWorld:
    def __init__(self, width: int = 8, height: int = 8, start: Tuple[int, int] = (0, 0),
                 goal: Tuple[int, int] = (7, 7), obstacles: List[Tuple[int, int]] = None):
        self.width = width
        self.height = height
        self.start_pos = start
        self.goal_pos = goal
        self.obstacles = obstacles or [(2, 2), (3, 4), (5, 1), (6, 5), (1, 6)]

        # Action space: 0=up, 1=right, 2=down, 3=left
        self.actions = 4
        self.action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        # Calculate new position
        dy, dx = self.action_map[action]
        new_y, new_x = self.current_pos[0] + dy, self.current_pos[1] + dx

        # Check boundaries
        if new_y < 0 or new_y >= self.height or new_x < 0 or new_x >= self.width:
            # Hit boundary, stay in place
            reward = -0.1
        elif (new_y, new_x) in self.obstacles:
            # Hit obstacle, stay in place
            reward = -1.0
        else:
            # Valid move
            self.current_pos = (new_y, new_x)
            if self.current_pos == self.goal_pos:
                reward = 10.0
            else:
                reward = -0.01  # Small penalty for each step

        done = self.current_pos == self.goal_pos
        return self.current_pos, reward, done

    def get_state_index(self, pos: Optional[Tuple[int, int]] = None) -> int:
        if pos is None:
            pos = self.current_pos
        return pos[0] * self.width + pos[1]

    def get_pos_from_index(self, index: int) -> Tuple[int, int]:
        return (index // self.width, index % self.width)

    def visualize(self, q_table: Optional[np.ndarray] = None, policy: Optional[np.ndarray] = None):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create grid
        grid = np.zeros((self.height, self.width))

        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1

        # Mark goal
        grid[self.goal_pos] = 2

        # Mark current position
        grid[self.current_pos] = 1

        # Display grid
        im = ax.imshow(grid, cmap='RdYlBu', alpha=0.7)

        # Add grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color='black', linewidth=1)

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
                    ax.text(j, i, 'X', ha='center', va='center', fontsize=16, fontweight='bold')

        # Show policy arrows if provided
        if policy is not None:
            arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
            for i in range(self.height):
                for j in range(self.width):
                    if (i, j) not in self.obstacles and (i, j) != self.goal_pos:
                        state_idx = self.get_state_index((i, j))
                        action = policy[state_idx]
                        ax.text(j, i-0.3, arrow_map[action], ha='center', va='center',
                               fontsize=12, color='red', fontweight='bold')

        ax.set_title('Grid World Environment', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

        return fig, ax