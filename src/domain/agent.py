from src.interfaces.policy_handler import IPolicyHandler
from src.interfaces.reward_function import IRewardFunction


class Agent:
    def __init__(
        self,
        reward_function: IRewardFunction,
        policy_handler: IPolicyHandler
    ):
        self.reward_function = reward_function
        self.policy_handler = policy_handler

    def take_action(self, state: tuple) -> int:
        return self.policy_handler.choose_action(state)

    def initialize_policy(self, num_actions: int):
        self.policy_handler.initialize(num_actions)

    def update_policy(self, state: tuple, action: int, new_state: tuple):
        reward = self.reward_function.compute_reward(state, new_state)
        self.policy_handler.update_policy(state, new_state, action, reward)