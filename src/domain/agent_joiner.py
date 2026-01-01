from typing import List
from src.domain.agent import Agent
from src.interfaces.game import IGame


class AgentJoiner:
    @staticmethod
    def train_agents(game: IGame, agents: List[Agent], n=1000):
        game.initialize()

        for agent in agents:
            agent.initialize_policy(game.get_num_agent_actions())

        for _ in range(n):
            current_state = game.get_current_state()
            joint_action = [agent.take_action(current_state) for agent in agents]
            new_state = game.apply_joint_action(joint_action)

            for agent, action in zip(agents, joint_action):
                agent.update_policy(current_state, action, new_state)