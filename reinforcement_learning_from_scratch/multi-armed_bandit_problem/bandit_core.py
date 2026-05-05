"""
bandit_core.py
──────────────────────────────────────────────────────────────────────────────
Core bandit logic — NO matplotlib dependencies.

Classes
  BanditEnv  – K-armed bandit environment (stochastic Gaussian rewards)
  Agent      – Action selection + belief update for three strategies:
                 "epsilon_greedy", "greedy", "random"

Functions
  run_experiment  – average results over many independent runs (for bandit.py)
  presimulate     – one detailed run recording before/after state (for bandit_scene.py)
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class BanditEnv:
    """
    K-armed bandit environment.
    Each pull samples R ~ N(true_means[action], reward_std).
    """
    def __init__(self, true_means, reward_std=1.0, seed=None):
        self.true_means = np.array(true_means, dtype=float)
        self.k          = len(self.true_means)
        self.reward_std = reward_std
        self.rng        = np.random.default_rng(seed)

    def pull(self, action: int) -> float:
        return float(self.rng.normal(self.true_means[action], self.reward_std))

    @property
    def optimal_action(self) -> int:
        return int(np.argmax(self.true_means))


# ══════════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════════

class Agent:
    """
    Bandit agent with sample-average Q-value estimates.

    Parameters
    ----------
    k        : number of arms
    strategy : "epsilon_greedy" | "greedy" | "random"
    epsilon  : exploration rate (only used by epsilon_greedy)
    seed     : RNG seed
    """
    def __init__(self, k: int, strategy: str = "epsilon_greedy",
                 epsilon: float = 0.1, seed=None):
        self.k        = k
        self.strategy = strategy
        self.epsilon  = epsilon
        self.rng      = np.random.default_rng(seed)
        self.reset()

    # ── State ──────────────────────────────────────────────────────────────

    def reset(self):
        """Reset all estimates and counts to their initial state."""
        self.q            = np.zeros(self.k, dtype=float)   # Q-value estimates
        self.n            = np.zeros(self.k, dtype=int)      # pull counts
        self.last_reward  = [None] * self.k                  # last seen reward per arm
        self.total_reward = 0.0

    # ── Action selection ───────────────────────────────────────────────────

    def select(self) -> int:
        """Choose an arm according to the current strategy."""
        if self.strategy == "epsilon_greedy":
            if self.rng.random() < self.epsilon:
                return int(self.rng.integers(self.k))   # explore
            return self._greedy()                        # exploit

        if self.strategy == "greedy":
            return self._greedy()

        if self.strategy == "random":
            return int(self.rng.integers(self.k))

        raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def _greedy(self) -> int:
        """Return the arm with the highest Q-value; break ties randomly."""
        max_q = np.max(self.q)
        tied  = np.flatnonzero(self.q == max_q)
        return int(self.rng.choice(tied))

    # ── Next-step probabilities ────────────────────────────────────────────

    def probabilities(self) -> np.ndarray:
        """
        Compute the probability that each arm is chosen on the NEXT step.

        epsilon_greedy : ε/k for all arms + (1-ε) shared equally by greedy arm(s)
        greedy         : 1.0 on best arm (split equally among ties)
        random         : uniform 1/k
        """
        if self.strategy == "epsilon_greedy":
            probs       = np.full(self.k, self.epsilon / self.k)
            tied        = np.flatnonzero(self.q == np.max(self.q))
            probs[tied] += (1.0 - self.epsilon) / len(tied)
            return probs

        if self.strategy == "greedy":
            probs       = np.zeros(self.k)
            tied        = np.flatnonzero(self.q == np.max(self.q))
            probs[tied] = 1.0 / len(tied)
            return probs

        if self.strategy == "random":
            return np.full(self.k, 1.0 / self.k)

        raise ValueError(f"Unknown strategy: {self.strategy!r}")

    # ── Belief update ──────────────────────────────────────────────────────

    def update(self, action: int, reward: float):
        """
        Incremental sample-average update:
            Q(a) ← Q(a) + (R - Q(a)) / N(a)
        """
        self.n[action]    += 1
        self.q[action]    += (reward - self.q[action]) / self.n[action]
        self.last_reward[action] = reward
        self.total_reward += reward


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER  (used by bandit.py for multi-run averaging)
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    true_means,
    steps:      int   = 1000,
    runs:       int   = 500,
    epsilon:    float = 0.1,
    reward_std: float = 1.0,
    base_seed:  int   = 42,
):
    """
    Run several independent episodes for three canonical strategies and
    average the results across runs.

    Returns
    -------
    avg_rewards, avg_cumulative_rewards, optimal_action_rate
        Each is a dict keyed by strategy label → np.ndarray of length `steps`.
    """
    k = len(true_means)
    strategies = {
        "Random":         dict(strategy="random"),
        "Greedy":         dict(strategy="greedy"),
        "Epsilon-Greedy": dict(strategy="epsilon_greedy", epsilon=epsilon),
    }

    avg_rewards     = {name: np.zeros(steps) for name in strategies}
    avg_cumulative  = {name: np.zeros(steps) for name in strategies}
    avg_optimal     = {name: np.zeros(steps) for name in strategies}
    optimal_action  = int(np.argmax(true_means))

    for run in range(runs):
        for idx, (name, cfg) in enumerate(strategies.items()):
            seed   = base_seed + run * 1000 + idx
            env    = BanditEnv(true_means, reward_std=reward_std, seed=seed)
            agent  = Agent(k=k, seed=seed + 99991, **cfg)

            rewards = np.zeros(steps)
            actions = np.zeros(steps, dtype=int)

            for t in range(steps):
                action    = agent.select()
                reward    = env.pull(action)
                agent.update(action, reward)
                rewards[t] = reward
                actions[t] = action

            cumulative_avg  = np.cumsum(rewards) / (np.arange(steps) + 1)
            optimal_chosen  = (actions == optimal_action).astype(float)

            avg_rewards[name]    += rewards
            avg_cumulative[name] += cumulative_avg
            avg_optimal[name]    += optimal_chosen

    for name in strategies:
        avg_rewards[name]    /= runs
        avg_cumulative[name] /= runs
        avg_optimal[name]    /= runs

    return avg_rewards, avg_cumulative, avg_optimal


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN RECORDER  (used by bandit_scene.py for the animation)
# ══════════════════════════════════════════════════════════════════════════════

def presimulate(
    true_means,
    strategy:    str   = "epsilon_greedy",
    epsilon:     float = 0.1,
    reward_std:  float = 1.0,
    total_steps: int   = 100,
    seed:        int   = 42,
):
    """
    Run one episode step by step, recording the full before/after agent state
    at every timestep.  Returns a list of dicts consumed by the animation.

    Each dict contains
        t, action, reward,
        q_before, q_after,
        prob_before, prob_after,
        lr_before, lr_after
    """
    env   = BanditEnv(true_means, reward_std=reward_std, seed=seed)
    agent = Agent(k=len(true_means), strategy=strategy,
                  epsilon=epsilon, seed=seed + 1)

    history = []
    for t in range(total_steps):
        q_before    = agent.q.copy()
        prob_before = agent.probabilities()
        lr_before   = list(agent.last_reward)

        action = agent.select()
        reward = env.pull(action)
        agent.update(action, reward)

        history.append(dict(
            t           = t,
            action      = action,
            reward      = reward,
            q_before    = q_before,
            q_after     = agent.q.copy(),
            prob_before = prob_before,
            prob_after  = agent.probabilities(),
            lr_before   = lr_before,
            lr_after    = list(agent.last_reward),
        ))

    return history
