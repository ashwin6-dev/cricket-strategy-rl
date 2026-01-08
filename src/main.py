import joblib

from src.domain.agent import Agent
from src.domain.agent_joiner import AgentJoiner
from src.games.simple_innings_game import SimpleInningsGame
from src.policy_handlers.boltzmann import BoltzmannQLearningPolicy
from src.rewards.expected_performance import ExpectedPerformanceRewardFunction

LR = 0.1
DISCOUNT_FACTOR = 0.95

def train_agents(temperature):
    expected_runs_map = joblib.load("./data/expected_final_runs_map.joblib")
    game = SimpleInningsGame(expected_runs_map)

    batting_agent_policy = BoltzmannQLearningPolicy(learning_rate=LR, discount_factor=DISCOUNT_FACTOR, temperature=temperature)
    bowling_agent_policy = BoltzmannQLearningPolicy(learning_rate=LR, discount_factor=DISCOUNT_FACTOR, temperature=temperature)

    batting_agent = Agent(
        ExpectedPerformanceRewardFunction(batting=True),
        batting_agent_policy
    )

    bowling_agent = Agent(
        ExpectedPerformanceRewardFunction(batting=False),
        bowling_agent_policy
    )

    AgentJoiner.train_agents(game, [batting_agent, bowling_agent], n=500000)

    batting_agent_policy.save(f"./data/temperature-{temperature}/batting/")
    bowling_agent_policy.save(f"./data/temperature-{temperature}/bowling/")

def main():
    print ("Training Temperature 1.0...")
    train_agents(1.0)

    print ("Training Temperature 0.5...")
    train_agents(0.5)

    print ("Training Temperature 0.1...")
    train_agents(0.1)

if __name__ == "__main__":
    main()
