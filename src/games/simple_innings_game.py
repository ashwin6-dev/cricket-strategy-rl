from typing import List
from src.interfaces.game import IGame
import numpy as np

"""
Actions
0 - Attack
1 - Neutral
2 - Defend
"""


class SimpleInningsGame(IGame):
    def __init__(self, expected_runs_map, seed: int | None = None):
        self.historical_states = list(expected_runs_map.keys())
        self.rng = np.random.default_rng(seed)

        # Mean runs per over
        self.runs_mean_matrix = np.array([
            [15.0, 11.0, 8.0],
            [11.0,  8.0, 5.0],
            [ 8.0,  5.0, 3.0]
        ])

        # Mean wickets per over
        self.wkts_mean_matrix = np.array([
            [1.00, 1.0, 0.34],
            [1.0, 0.34, 0.00],
            [0.34, 0.00, 0.00]
        ])

        # Standard deviation of runs per over
        self.runs_std_matrix = np.array([
            [4.5, 3.5, 2.5],
            [3.5, 2.5, 1.8],
            [2.5, 1.8, 1.2]
        ])

        # Standard deviation of wickets per over
        self.wkts_std_matrix = np.array([
            [0.60, 0.55, 0.30],
            [0.55, 0.30, 0.10],
            [0.30, 0.10, 0.05]
        ])

    def initialize(self):
        self.state = (1, 0, 0)

    def get_num_agent_actions(self) -> int:
        return 3
    
    def get_current_state(self) -> tuple:
        base_over, base_runs, base_wickets = self.state

        if base_over > 20 or base_wickets >= 10:
            idx = int(self.rng.integers(0, len(self.historical_states)))
            self.state = self.historical_states[idx]

        return self.state
    
    def _effective_bowl_action(self, bowl_action: int, over: int) -> int:
        if over <= 6 and bowl_action == 2:
            return 1
        return bowl_action

    def apply_joint_action(self, joint_action: List[int]) -> tuple:
        base_over, base_runs, base_wickets = self.state
        bat_action, bowl_action = joint_action
        bowl_action = self._effective_bowl_action(bowl_action, base_over)

        runs_mean = self.runs_mean_matrix[bat_action, bowl_action]
        wkts_mean = self.wkts_mean_matrix[bat_action, bowl_action]

        runs_std = self.runs_std_matrix[bat_action, bowl_action]
        wkts_std = self.wkts_std_matrix[bat_action, bowl_action]

        
        over_runs = self.rng.normal(runs_mean, runs_std)
        std = (over_runs - runs_mean) / runs_std
        over_wkts = wkts_mean - std*wkts_std

        # Clamp to sensible domain
        over_runs = max(0.0, over_runs)
        over_wkts = max(0.0, over_wkts)

        new_over = base_over + 1
        new_runs = int(base_runs + over_runs)
        new_wickets = int(base_wickets + over_wkts)

        self.state = (new_over, new_runs, new_wickets)
        return self.state
