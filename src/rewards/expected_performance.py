import random
from src.interfaces.reward_function import IRewardFunction


class ExpectedPerformanceRewardFunction(IRewardFunction):
    def __init__(self, batting: bool):
        self.batting = batting

    def compute_reward(self, prev_state: tuple, new_state: tuple) -> float:
        coeff = 1 if self.batting else -1
        return coeff * random.randint(50, 150)