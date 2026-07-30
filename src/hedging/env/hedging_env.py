"""Gymnasium environment for hedging a short European call.

Replication mechanics (self-financing hedge portfolio)
-------------------------------------------------------
The writer sells one call at t=0 for the fair Black-Scholes premium C_0 and
must deliver max(S_T - K, 0) at maturity. At each rebalancing time t_i the
agent picks a target hedge ratio h_i in [0, 1] (shares of the underlying
held long, per unit of the short call). Moving from h_{i-1} to h_i costs
proportional transaction fees. Between t_i and t_{i+1} the hedge P&L is
h_i * (S_{i+1} - S_i). At maturity the position is unwound (one more
transaction cost) and the payoff is settled.

    total_pnl = C_0 + sum_i [h_i (S_{i+1}-S_i) - cost_i] - unwind_cost - payoff

`total_pnl` is the writer's net replication error: 0 would mean the hedge
was perfect, negative means the hedge lost money net of the premium
collected. The RL reward is shaped so the sum of per-step rewards over an
episode equals:

    episode_return = total_pnl - lambda * total_pnl ** 2

i.e. a mean-variance-style, risk-averse objective on terminal P&L, while
individual transaction costs and hedge P&L are still paid out as immediate,
time-separable rewards at every step (so the agent gets dense, informative
signal about costs rather than only a single reward at the very end).

Observation (float32, shape (5,)):
    [log_moneyness, tau_norm, prev_hedge_ratio, vol_proxy, cum_pnl_norm]
    - log_moneyness = log(S_t / K)
    - tau_norm      = (n_steps - t) / n_steps, in [0, 1]
    - vol_proxy     = trailing realized-vol estimate (see pricing.volatility);
                      NOT the true latent instantaneous vol, even under Heston.
    - cum_pnl_norm  = cumulative hedge P&L so far, divided by S0.

Action: Box([0, 1]), target hedge ratio for the current interval.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hedging.config import HedgingConfig, MarketConfig, RegimeConfig
from hedging.exceptions import HedgingEnvError
from hedging.market.registry import Simulator
from hedging.pricing.black_scholes import bs_call_price
from hedging.pricing.volatility import RealizedVolEstimator


class HedgingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        market: MarketConfig,
        hedging: HedgingConfig,
        regimes: Dict[str, RegimeConfig],
        simulators: Dict[str, Simulator],
        rng: Optional[np.random.Generator] = None,
        fixed_regime: Optional[str] = None,
    ) -> None:
        super().__init__()
        if fixed_regime is not None and fixed_regime not in regimes:
            raise HedgingEnvError(f"fixed_regime {fixed_regime!r} not in regimes {list(regimes)}")

        self.market = market
        self.hedging = hedging
        self.regimes = regimes
        self.simulators = simulators
        self.fixed_regime = fixed_regime
        self._rng = rng if rng is not None else np.random.default_rng()

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0, -np.inf], dtype=np.float32),
            high=np.array([np.inf, 1.0, 1.0, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._vol_estimator = RealizedVolEstimator(
            prior_vol=market.prior_vol, window=market.vol_window, dt=market.dt
        )

        # Episode state, set in reset()
        self._path: Optional[np.ndarray] = None
        self._regime_name: Optional[str] = None
        self._t: int = 0
        self._prev_hedge: float = 0.0
        self._cum_pnl: float = 0.0
        self._C0: float = 0.0
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        regime_name = options.get("regime", self.fixed_regime)
        if regime_name is None:
            regime_name = self._rng.choice(list(self.regimes.keys()))
        if regime_name not in self.regimes:
            raise HedgingEnvError(f"unknown regime {regime_name!r}")

        path = options.get("path")
        if path is None:
            path = self.simulators[regime_name].simulate_path(self._rng)
        path = np.asarray(path, dtype=float)
        if path.shape != (self.market.n_steps + 1,):
            raise HedgingEnvError(
                f"path shape {path.shape} != {(self.market.n_steps + 1,)}"
            )

        pricing_vol = self.regimes[regime_name].pricing_vol
        maturity = self.market.maturity
        self._C0 = float(
            bs_call_price(self.market.S0, self.market.K, maturity, self.market.r, pricing_vol)
        )

        self._path = path
        self._regime_name = regime_name
        self._t = 0
        self._prev_hedge = 0.0
        self._cum_pnl = 0.0
        self._vol_estimator.reset()
        self._initialized = True

        obs = self._build_obs()
        info = {"regime": regime_name, "C0": self._C0, "path": path.copy()}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self._initialized:
            raise HedgingEnvError("call reset() before step()")

        h = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        S_t = self._path[self._t]
        S_next = self._path[self._t + 1]
        cost_rate = self.hedging.transaction_cost_rate

        cost = cost_rate * S_t * abs(h - self._prev_hedge)
        hedge_pnl = h * (S_next - S_t)
        reward = hedge_pnl - cost
        self._cum_pnl += reward

        log_return = float(np.log(S_next / S_t))
        self._vol_estimator.update(log_return)

        self._prev_hedge = h
        self._t += 1

        terminated = self._t >= self.market.n_steps
        truncated = False
        info: Dict[str, Any] = {"regime": self._regime_name, "cost": cost, "hedge_pnl": hedge_pnl}

        if terminated:
            final_unwind_cost = cost_rate * S_next * h
            payoff = max(S_next - self.market.K, 0.0)
            total_pnl = self._C0 + self._cum_pnl - final_unwind_cost - payoff
            lam = self.hedging.variance_penalty_lambda
            reward += (self._C0 - final_unwind_cost - payoff) - lam * total_pnl**2
            info.update(
                {
                    "total_pnl": total_pnl,
                    "payoff": payoff,
                    "final_unwind_cost": final_unwind_cost,
                    "C0": self._C0,
                }
            )

        obs = self._build_obs()
        return obs, float(reward), bool(terminated), truncated, info

    def render(self):
        return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_obs(self) -> np.ndarray:
        S_t = self._path[self._t]
        log_moneyness = float(np.log(S_t / self.market.K))
        tau_norm = float(self.market.n_steps - self._t) / self.market.n_steps
        vol_proxy = self._vol_estimator.current_estimate()
        cum_pnl_norm = self._cum_pnl / self.market.S0
        return np.array(
            [log_moneyness, tau_norm, self._prev_hedge, vol_proxy, cum_pnl_norm],
            dtype=np.float32,
        )
