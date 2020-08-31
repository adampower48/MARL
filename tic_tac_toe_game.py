from typing import List, Tuple

import numpy as np


class TicTackToeGame:
    def __init__(self, grid_size=3, num_agents=2):
        self.grid_size = grid_size
        self.num_agents = num_agents

        self.grid: np.ndarray = None

        self.reset()

    def reset(self):
        self.grid = np.full((self.grid_size, self.grid_size), -1)

    def get_winner(self) -> (None or int):
        # Check for lines
        for i in range(self.num_agents):
            grid = self.grid == i

            if ((grid.sum(axis=1) == self.grid_size).any() or  # Horizontal
                    (grid.sum(axis=0) == self.grid_size).any() or  # Vertical
                    np.trace(grid) == self.grid_size or  # Diagonal (Major)
                    np.trace(grid[::-1]) == self.grid_size):  # Diagonal (Minor)
                return i

        return None

    def is_game_over(self) -> bool:
        return (self.grid != -1).all()

    def perform_action(self, agent_idx, action) -> bool:
        # Returns True if action was performed, False if not performed
        if self.is_action_valid(agent_idx, action):
            self.grid.ravel()[action] = agent_idx
            return True

        return False

    def get_state(self) -> np.ndarray:
        # Agent-agnostic game state
        # Values:
        #   -1: empty
        # 0..n: agent index

        return self.grid

    def get_state_for_agent(self, agent_idx) -> np.ndarray:
        # State of game from perspective of given agent
        # Values:
        #   -1: empty
        #    0: friendly
        #    1: enemy

        return np.where(self.grid_size == -1, -1, np.where(self.grid == agent_idx, 0, 1))

    def is_action_valid(self, agent_idx, action) -> bool:
        # Check that action is legal
        return self.grid.ravel()[action] == -1

    def get_valid_actions(self, agent_idx) -> np.ndarray:
        # Returns list of indices for valid actions
        return np.arange(self.grid_size ** 2)[self.grid.ravel() == -1]

    def get_action_space(self) -> List[Tuple[type, np.ndarray]]:
        # Returns dtype & shape of action space
        return [(bool, self.grid.ravel().shape)]
