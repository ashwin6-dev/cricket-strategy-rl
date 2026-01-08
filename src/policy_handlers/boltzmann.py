import numpy as np
import joblib
from src.interfaces.policy_handler import IPolicyHandler
import os

class BoltzmannQLearningPolicy(IPolicyHandler):
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        temperature: float = 1.0,
        seed: int | None = None,
        temp_decay: float = 1.0,
        min_temperature: float = 0.05,
    ):
        self.alpha = float(learning_rate)
        self.gamma = float(discount_factor)

        self.temperature = float(temperature)
        self.initial_temperature = float(temperature)

        self.temp_decay = float(temp_decay)
        self.min_temperature = float(min_temperature)

        self.table: dict[tuple, np.ndarray] = {}
        self.num_actions = 0
        self.rng = np.random.default_rng(seed)

        # ---- convergence diagnostics ----
        self.t = 0
        self.abs_update_history: list[tuple[int, float]] = []

        # ---- NEW: histories ----
        self.q_update_history: dict[str, list] = {
            "t": [],
            "state": [],
            "action": [],
            "old_q": [],
            "new_q": [],
            "reward": [],
            "td_target": [],
            "tau": [],
        }

        self.q_value_history: dict[tuple, list] = {}  # state -> [(t, q_vec)]

    def initialize(self, num_actions: int):
        self.num_actions = int(num_actions)

    def _ensure_state(self, state: tuple) -> np.ndarray:
        if state not in self.table:
            self.table[state] = self.rng.normal(
                loc=0.0,
                scale=1e-3,
                size=self.num_actions
            )
        return self.table[state]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        z = ex.sum()
        if z <= 0.0 or not np.isfinite(z):
            return np.full_like(ex, 1.0 / ex.size)
        return ex / z

    def _current_temperature(self) -> float:
        tau = self.initial_temperature * (self.temp_decay ** self.t)
        return max(self.min_temperature, tau)

    def choose_action(self, state: tuple) -> int:
        q = self._ensure_state(state)
        tau = self._current_temperature()
        probs = self._softmax(q / tau)
        return int(self.rng.choice(self.num_actions, p=probs))

    def update_policy(self, state: tuple, new_state: tuple, action: int, reward: float):
        q_s = self._ensure_state(state)
        q_sp = self._ensure_state(new_state)

        a = int(action)
        r = float(reward)

        old_q = float(q_s[a])
        td_target = r + self.gamma * float(np.max(q_sp))
        new_q = old_q + self.alpha * (td_target - old_q)
        q_s[a] = new_q

        # ---- abs update ----
        self.abs_update_history.append((self.t, abs(new_q - old_q)))

        # ---- per-update log ----
        tau = self._current_temperature()
        self.q_update_history["t"].append(self.t)
        self.q_update_history["state"].append(state)
        self.q_update_history["action"].append(a)
        self.q_update_history["old_q"].append(old_q)
        self.q_update_history["new_q"].append(new_q)
        self.q_update_history["reward"].append(r)
        self.q_update_history["td_target"].append(td_target)
        self.q_update_history["tau"].append(tau)

        # ---- Q-value history (per state) ----
        if state not in self.q_value_history:
            self.q_value_history[state] = []
        self.q_value_history[state].append((self.t, q_s.copy()))

        self.t += 1

    def save(self, path: str):
        """
        Saves THREE files using joblib:

        1) {path}_q_table.joblib
        2) {path}_q_updates.joblib
        3) {path}_q_history.joblib
        """
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        joblib.dump(self.table, f"{path}q_table.joblib")

        joblib.dump(
            {
                **self.q_update_history,
                "abs_update_history": self.abs_update_history,
            },
            f"{path}q_updates.joblib",
        )

        joblib.dump(self.q_value_history, f"{path}q_history.joblib")
