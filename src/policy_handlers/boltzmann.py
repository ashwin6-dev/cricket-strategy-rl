import numpy as np
from src.interfaces.policy_handler import IPolicyHandler


class BoltzmannMyopicQLearningPolicy(IPolicyHandler):
    def __init__(self, learning_rate: float = 0.1, temperature: float = 1.0, seed: int | None = None):
        self.alpha = float(learning_rate)
        self.temperature = float(temperature)
        self.table: dict[tuple, np.ndarray] = {}
        self.num_actions = 0
        self.rng = np.random.default_rng(seed)

    def initialize(self, num_actions: int):
        self.num_actions = int(num_actions)

    def _ensure_state(self, state: tuple) -> np.ndarray:
        if state not in self.table:
            self.table[state] = np.zeros(self.num_actions, dtype=np.float64)
        return self.table[state]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        z = ex.sum()
        if z <= 0.0:
            return np.full_like(ex, 1.0 / ex.size)
        return ex / z

    def choose_action(self, state: tuple) -> int:
        q = self._ensure_state(state)
        tau = max(self.temperature, 1e-8)
        probs = self._softmax(q / tau)
        return int(self.rng.choice(self.num_actions, p=probs))

    def update_policy(self, state: tuple, action: int, reward: float):
        q = self._ensure_state(state)
        a = int(action)
        r = float(reward)
        q[a] = q[a] + self.alpha * (r - q[a])
