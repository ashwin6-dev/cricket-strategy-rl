from typing import Protocol

class IPolicyHandler:
    def initialize(self, num_actions: int):
        ...

    def choose_action(self, state: tuple) -> int:
        ...

    def update_policy(self, state: tuple, action: int, reward: float):
        ...