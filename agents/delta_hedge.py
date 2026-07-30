"""Black-Scholes delta-hedge baselines.

Two variants, both sharing the "textbook" closed-form delta formula, so the
comparison against the RL agent isolates a single question at a time:

- `StaticBSBaseline`: assumes one flat, never-updated vol input for the
  whole option life (the classic textbook approach: calibrate an implied
  vol at inception and hedge off it).
- `AdaptiveBSBaseline`: recomputes delta every step using *the same*
  trailing realized-vol estimate the RL policy observes (see
  `pricing.volatility.RealizedVolEstimator`). This is the fairer
  comparison -- same information set as the RL agent, only the mapping
  from information to hedge ratio differs (closed-form formula vs. learned
  policy). Neither baseline "sees" the latent Heston variance directly.

Both expose `.act(obs) -> np.ndarray` with the same signature as the PPO
policy's action method, so a single backtest loop works for all three.
"""

from __future__ import annotations

import numpy as np

from hedging.config import MarketConfig
from hedging.pricing.black_scholes import bs_call_delta


class _DeltaHedgeBaseline:
    def __init__(self, market: MarketConfig) -> None:
        self.market = market

    def _reconstruct_S_tau(self, obs: np.ndarray) -> tuple[float, float]:
        log_moneyness, tau_norm = float(obs[0]), float(obs[1])
        S = self.market.K * np.exp(log_moneyness)
        tau = tau_norm * self.market.maturity
        return S, tau

    def act(self, obs: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self) -> None:
        """No-op for stateless closed-form baselines; kept for interface
        parity with stateful policies."""
        return None


class StaticBSBaseline(_DeltaHedgeBaseline):
    def __init__(self, market: MarketConfig, static_vol: float) -> None:
        super().__init__(market)
        self.static_vol = float(static_vol)

    def act(self, obs: np.ndarray) -> np.ndarray:
        S, tau = self._reconstruct_S_tau(obs)
        delta = bs_call_delta(S, self.market.K, tau, self.market.r, self.static_vol)
        return np.array([delta], dtype=np.float32)


class AdaptiveBSBaseline(_DeltaHedgeBaseline):
    def act(self, obs: np.ndarray) -> np.ndarray:
        S, tau = self._reconstruct_S_tau(obs)
        vol_proxy = float(obs[3])
        delta = bs_call_delta(S, self.market.K, tau, self.market.r, vol_proxy)
        return np.array([delta], dtype=np.float32)
