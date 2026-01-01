from typing import Protocol, List

class IGame(Protocol):
    def initialize(self):
        ...

    def get_num_agent_actions(self) -> int:
        ...

    def get_current_state(self) -> tuple:
        ...

    def apply_joint_action(self, joint_action: List[int]) -> tuple:
        ...