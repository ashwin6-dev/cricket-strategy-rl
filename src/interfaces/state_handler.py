from typing import Protocol


class IStateHandler(Protocol):
    def get_current_state(self) -> tuple:
        ...

    def apply_action(self, action: int) -> tuple:
        ...