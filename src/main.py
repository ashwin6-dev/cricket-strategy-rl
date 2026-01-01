
from src.domain.agent import Agent
from src.domain.agent_joiner import AgentJoiner
from src.games.simple_innings_game import SimpleInningsGame
from src.policy_handlers.boltzmann import BoltzmannMyopicQLearningPolicy
from src.rewards.expected_performance import ExpectedPerformanceRewardFunction


def main():
    game = SimpleInningsGame()

    batting_agent = Agent(
        ExpectedPerformanceRewardFunction(batting=True),
        BoltzmannMyopicQLearningPolicy()
    )

    bowling_agent = Agent(
        ExpectedPerformanceRewardFunction(batting=False),
        BoltzmannMyopicQLearningPolicy()
    )

    AgentJoiner.train_agents(game, [batting_agent, bowling_agent])

    print (batting_agent.policy_handler.table)

if __name__ == "__main__":
    main()