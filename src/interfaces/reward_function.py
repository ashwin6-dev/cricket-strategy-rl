from typing import Protocol


class IRewardFunction(Protocol):
    def compute_reward(self, prev_state: tuple, new_state: tuple) -> float:
        ...