from src.domain.environment import Environment
from src.interfaces.policy_handler import IPolicyHandler


class Agent:
    def __init__(self, policy_handler: IPolicyHandler):
        self.policy_handler = policy_handler

    def learn_policy(self, env: Environment, n=1000):
        self.policy_handler.initialize(env.get_num_actions())

        current_state = env.get_current_state()
        for _ in range(n):
            action = self.policy_handler.choose_action(current_state)
            new_state, reward = env.take_action(action)
            self.policy_handler.update_policy(current_state, action, reward)
            current_state = new_state