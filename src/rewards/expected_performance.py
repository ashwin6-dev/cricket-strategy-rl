import random
from src.interfaces.reward_function import IRewardFunction


class ExpectedPerformanceRewardFunction(IRewardFunction):
    def __init__(self, batting: bool):
        self.batting = batting

    def _get_state_expected_runs(self, state):
        over = state[0]
        runs = state[1]
        wickets = max(0, state[2])

        return runs - wickets**2 - 7*over + 164

    def compute_reward(self, prev_state: tuple, new_state: tuple) -> float:
        expected_runs_gain = self._get_state_expected_runs(new_state) - self._get_state_expected_runs(prev_state)
        coeff = 1 if self.batting else -1
        return coeff * expected_runs_gain