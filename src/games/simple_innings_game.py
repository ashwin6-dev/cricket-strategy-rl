from typing import List
from src.interfaces.game import IGame
import numpy as np

"""
Actions
0 - Defend
1 - Neutral
2 - Attack
"""


class SimpleInningsGame(IGame):
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

        self.batter_distributions = {
            0: ((4.0, 1.0), (0.05, 0.05)),
            1: ((6.0, 1.5), (0.12, 0.08)),
            2: ((10.0, 2.5), (0.25, 0.12)),
        }

        self.bowler_distributions = {
            0: ((-3.0, 0.5), (-0.05, 0.04)),
            1: ((0.0, 1.0), (0.0, 0.06)),
            2: ((2.0, 2.0), (0.20, 0.10)),
        }

    def initialize(self):
        pass 

    def get_num_agent_actions(self) -> int:
        return 3

    def get_current_state(self) -> tuple:
        over = int(self.rng.integers(1, 21))
        runs = int(self.rng.integers(0, 301))
        wickets = int(self.rng.integers(0, 11))
        target = int(self.rng.integers(0, 301))

        self.state = (over, runs, wickets, target)
        return self.state

    def apply_joint_action(self, joint_action: List[int]) -> tuple:
        bat_action, bowl_action = joint_action

        (br_mu, br_sigma), (bw_mu, bw_sigma) = self.batter_distributions[bat_action]
        bat_runs = self.rng.normal(br_mu, br_sigma)
        bat_wkts = self.rng.normal(bw_mu, bw_sigma)

        (bo_mu, bo_sigma), (bo_w_mu, bo_w_sigma) = self.bowler_distributions[bowl_action]
        bowl_runs = self.rng.normal(bo_mu, bo_sigma)
        bowl_wkts = self.rng.normal(bo_w_mu, bo_w_sigma)

        over_runs = max(0.0, bat_runs + bowl_runs)
        over_wkts = max(0.0, bat_wkts + bowl_wkts)
        over_wkts = min(int(np.round(over_wkts)), 2)

        base_over = self.state[0]
        base_runs = self.state[1]
        base_wickets = self.state[2]
        base_target = self.state[3]

        new_over = min(base_over + 1, 20)
        new_runs = int(np.round(base_runs + over_runs))
        new_wickets = min(base_wickets + over_wkts, 10)

        return (new_over, new_runs, new_wickets, base_target)
