from typing import Protocol


class IPolicyHandler(Protocol):
    def initialize(self, num_actions: int):
        ...

    def choose_action(self, state: tuple) -> int:
        ...

    def update_policy(self, state: tuple, new_state: tuple, action: int, reward: float):
        ...

    def save(self, path: str):
        ...