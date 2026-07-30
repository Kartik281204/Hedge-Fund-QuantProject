"""A minimal vectorized environment: N independent `HedgingEnv` copies,
stepped in a Python loop (env dynamics are pure numpy and cheap) but with
action *selection* batched into a single JAX call across all N at once.

On a single CPU core this doesn't parallelize compute, but it amortizes the
fixed per-call JAX dispatch overhead across N environment steps instead of
paying it N times -- the dominant cost for a rollout this small (see
README, "why not stable-baselines3 / performance notes").

Each sub-env auto-resets on termination (episodes are fixed-length, so this
never happens mid-call to `reset`, only inside `step`), sampling a fresh
regime via its own independent RNG stream -- this is exactly the domain
randomization used to train one policy across all volatility regimes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from hedging.config import HedgingConfig, MarketConfig, RegimeConfig
from hedging.env.hedging_env import HedgingEnv
from hedging.market.registry import Simulator


class VectorHedgingEnv:
    def __init__(
        self,
        market: MarketConfig,
        hedging: HedgingConfig,
        regimes: Dict[str, RegimeConfig],
        simulators: Dict[str, Simulator],
        n_envs: int,
        seed: int,
    ) -> None:
        self.n_envs = n_envs
        self.envs: List[HedgingEnv] = [
            HedgingEnv(
                market=market,
                hedging=hedging,
                regimes=regimes,
                simulators=simulators,
                rng=np.random.default_rng(seed + i),
            )
            for i in range(n_envs)
        ]

    def reset(self) -> np.ndarray:
        obs = [env.reset()[0] for env in self.envs]
        return np.stack(obs, axis=0)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """actions: shape (n_envs,) or (n_envs, 1). Auto-resets any sub-env
        that terminates; the episode's terminal info (with `total_pnl` etc.)
        is still returned in `infos[i]` for that step even though `obs[i]`
        already reflects the freshly-reset next episode."""
        actions = np.asarray(actions).reshape(self.n_envs, -1)
        obs_out = np.empty((self.n_envs, self.envs[0].observation_space.shape[0]), dtype=np.float32)
        rewards = np.empty(self.n_envs, dtype=np.float32)
        dones = np.empty(self.n_envs, dtype=np.float32)
        infos: List[Dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            next_obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated
            rewards[i] = reward
            dones[i] = float(done)
            infos.append(info)
            if done:
                next_obs, _ = env.reset()
            obs_out[i] = next_obs

        return obs_out, rewards, dones, infos
