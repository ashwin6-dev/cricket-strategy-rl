from typing import Protocol


class IRewardFunction(Protocol):
    def compute_reward(self, state: tuple, action: int) -> float:
        ...