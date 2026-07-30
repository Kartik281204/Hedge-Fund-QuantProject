"""Backtest harness.

The key design choice here is *paired* evaluation: for a given regime, all
three methods (RL, static-BS, adaptive-BS) are run on the exact same set of
simulated price paths. This is a standard Monte-Carlo variance-reduction
trick -- differences between methods on paired paths are far less noisy
than differences between methods evaluated on independently-drawn paths,
so a given number of evaluation episodes yields a much more reliable
comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

import numpy as np

from hedging.env.hedging_env import HedgingEnv
from hedging.evaluation.metrics import Metrics, compute_metrics
from hedging.market.registry import Simulator


class Policy(Protocol):
    def act(self, obs: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...


def generate_eval_paths(
    simulators: Dict[str, Simulator], n_episodes: int, seed: int
) -> Dict[str, np.ndarray]:
    """Pre-generate a reproducible, held-out set of paths per regime, shared
    across every method being compared. Shape per regime: (n_episodes, n_steps+1)."""
    rng = np.random.default_rng(seed)
    return {name: sim.simulate_paths(n_episodes, rng) for name, sim in sorted(simulators.items())}


def run_episode(env: HedgingEnv, policy: Policy, regime: str, path: np.ndarray) -> Dict[str, Any]:
    obs, _ = env.reset(options={"regime": regime, "path": path})
    policy.reset()

    total_cost = 0.0
    turnover = 0.0
    prev_h = 0.0
    hedge_ratios: List[float] = []
    terminated = False
    step_info: Dict[str, Any] = {}

    while not terminated:
        action = policy.act(obs)
        h = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        hedge_ratios.append(h)
        obs, _, terminated, _, step_info = env.step(action)
        total_cost += step_info["cost"]
        turnover += abs(h - prev_h)
        prev_h = h

    # Unwinding the final hedge position at expiry is itself a trade.
    total_cost += step_info["final_unwind_cost"]
    turnover += prev_h

    return {
        "total_pnl": step_info["total_pnl"],
        "total_cost": total_cost,
        "turnover": turnover,
        "hedge_ratios": np.array(hedge_ratios, dtype=float),
        "prices": path,
    }


def run_backtest(
    env: HedgingEnv,
    policy: Policy,
    regime: str,
    paths: np.ndarray,
    keep_examples: int = 0,
) -> Dict[str, Any]:
    n_episodes = paths.shape[0]
    total_pnls = np.zeros(n_episodes)
    total_costs = np.zeros(n_episodes)
    turnovers = np.zeros(n_episodes)
    examples: List[Dict[str, Any]] = []

    for i in range(n_episodes):
        result = run_episode(env, policy, regime, paths[i])
        total_pnls[i] = result["total_pnl"]
        total_costs[i] = result["total_cost"]
        turnovers[i] = result["turnover"]
        if i < keep_examples:
            examples.append(result)

    metrics: Metrics = compute_metrics(total_pnls, total_costs, turnovers)
    return {
        "metrics": metrics,
        "total_pnls": total_pnls,
        "total_costs": total_costs,
        "turnovers": turnovers,
        "examples": examples,
    }
