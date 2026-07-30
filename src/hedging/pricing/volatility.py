"""Rolling realized-volatility estimator.

Neither the RL agent nor the "adaptive" BS baseline is allowed to observe
the true instantaneous variance of a stochastic-vol path directly -- that
would be an unrealistic omniscience assumption (in reality, latent vol
state isn't observable). Instead both use this estimator, which recovers an
annualized vol from a short trailing window of realized log-returns, seeded
by a prior estimate before any returns are observed. Using the *same*
estimator for both the RL policy's observation and the adaptive baseline
keeps the comparison between them on equal informational footing.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class RealizedVolEstimator:
    def __init__(self, prior_vol: float, window: int, dt: float) -> None:
        if prior_vol <= 0:
            raise ValueError("prior_vol must be > 0")
        if window < 1:
            raise ValueError("window must be >= 1")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        self.prior_vol = float(prior_vol)
        self.window = int(window)
        self.dt = float(dt)
        self._returns: deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._returns.clear()

    def current_estimate(self) -> float:
        """Annualized vol estimate given returns observed so far."""
        if len(self._returns) == 0:
            return self.prior_vol
        mean_sq = float(np.mean(np.square(np.asarray(self._returns))))
        return float(np.sqrt(mean_sq / self.dt))

    def update(self, log_return: float) -> float:
        """Feed in the latest log-return, return the updated estimate."""
        self._returns.append(float(log_return))
        return self.current_estimate()
