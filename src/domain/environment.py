from typing import Tuple

from src.interfaces.reward_function import IRewardFunction
from src.interfaces.state_handler import IStateHandler


class Environment:
    def __init__(
        self, 
        reward_function: IRewardFunction, 
        state_handler: IStateHandler,
        num_actions: int):
        self.reward_function = reward_function
        self.state_handler = state_handler
        self.num_actions = num_actions

    def get_num_actions(self) -> int:
        return self.num_actions

    def get_current_state(self) -> tuple:
        return self.state_handler.get_current_state()

    def take_action(self, action: int) -> Tuple[tuple, float]:
        current_state = self.state_handler.get_current_state()
        reward = self.reward_function.compute_reward(current_state, action)
        next_state = self.state_hanlder.apply_action(action)

        return (next_state, reward)