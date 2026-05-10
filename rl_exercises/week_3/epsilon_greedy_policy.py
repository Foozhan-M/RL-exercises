from __future__ import annotations

from typing import DefaultDict

import gymnasium as gym
import numpy as np


class EpsilonGreedyPolicy(object):
    """A Policy doing Epsilon Greedy Exploration."""

    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        seed: int = 0,
    ) -> None:
        """Init

        Parameters
        ----------
        env : gym.Env
            Environment
        epsilon: float
            Exploration rate
        seed : int, optional
            Seed, by default None
        """
        assert 0 <= epsilon <= 1, "ε must be in [0,1]"
        self.env = env
        self.epsilon = epsilon

        # our private RNG, so sampling is reproducible
        self.rng = np.random.default_rng(seed)

    def __call__(self, Q: DefaultDict, state: tuple, evaluate: bool = False) -> int:  # type: ignore # noqa: E501
        """Select action

        Parameters
        ----------
        Q : DefaultDict
            Q Table/Function

        state : tuple
            State

        evaluate: bool
            evaluation mode - if true, exploration should be turned off.

        Returns
        -------
        int
            action
        """

        # Evaluation mode: pure greedy policy
        if evaluate:
            q_values = Q[state]
            max_q = np.max(q_values)
            best_actions = np.flatnonzero(q_values == max_q)
            return int(self.rng.choice(best_actions))

        # Exploration
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.env.action_space.n))

        # Exploitation with random tie-breaking
        q_values = Q[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(self.rng.choice(best_actions))
