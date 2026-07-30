"""Heston stochastic volatility simulator.

    dS = mu S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho

Discretized with the Lord-Koekkoek-Van Dijk (2010) "full truncation" Euler
scheme: the drift/diffusion coefficients use v^+ = max(v, 0), and the raw
Euler update for v is floored at 0 before being used in the next step. This
avoids the negative-variance blow-ups a naive Euler scheme produces for
realistic (non-Feller-satisfying) parameter sets, at the cost of a small
discretization bias that's standard and well-documented in the literature.
"""

from __future__ import annotations

import numpy as np

from hedging.exceptions import SimulationError


class HestonSimulator:
    def __init__(
        self,
        S0: float,
        mu: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        dt: float,
        n_steps: int,
    ) -> None:
        if S0 <= 0:
            raise SimulationError("HestonSimulator: S0 must be > 0")
        if v0 <= 0 or kappa <= 0 or theta <= 0 or xi <= 0:
            raise SimulationError("HestonSimulator: v0, kappa, theta, xi must be > 0")
        if not (-1.0 <= rho <= 1.0):
            raise SimulationError("HestonSimulator: rho must be in [-1, 1]")
        if dt <= 0:
            raise SimulationError("HestonSimulator: dt must be > 0")
        if n_steps < 1:
            raise SimulationError("HestonSimulator: n_steps must be >= 1")

        self.S0 = float(S0)
        self.mu = float(mu)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.dt = float(dt)
        self.n_steps = int(n_steps)

    def _simulate(self, n_paths: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if n_paths < 1:
            raise SimulationError("n_paths must be >= 1")

        S = np.empty((n_paths, self.n_steps + 1))
        v = np.empty((n_paths, self.n_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        sqrt_dt = np.sqrt(self.dt)
        sqrt_one_minus_rho2 = np.sqrt(max(1.0 - self.rho**2, 0.0))

        for t in range(self.n_steps):
            v_pos = np.maximum(v[:, t], 0.0)
            z1 = rng.standard_normal(n_paths)
            z_indep = rng.standard_normal(n_paths)
            z2 = self.rho * z1 + sqrt_one_minus_rho2 * z_indep

            vol_term = np.sqrt(v_pos * self.dt)
            S[:, t + 1] = S[:, t] * np.exp((self.mu - 0.5 * v_pos) * self.dt + vol_term * z1)

            v_raw = v[:, t] + self.kappa * (self.theta - v_pos) * self.dt + self.xi * vol_term * z2
            v[:, t + 1] = np.maximum(v_raw, 0.0)

        if not (np.all(np.isfinite(S)) and np.all(np.isfinite(v))):
            raise SimulationError("HestonSimulator produced non-finite values")
        return S, v

    def simulate_paths(self, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        """Return underlying-price paths only, shape (n_paths, n_steps + 1)."""
        S, _ = self._simulate(n_paths, rng)
        return S

    def simulate_paths_with_variance(
        self, n_paths: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (S, v) paths, each shape (n_paths, n_steps + 1). Used for
        diagnostic plotting of the latent vol process; the RL/baseline
        agents never see `v` directly (see pricing.volatility)."""
        return self._simulate(n_paths, rng)

    def simulate_path(self, rng: np.random.Generator) -> np.ndarray:
        return self.simulate_paths(1, rng)[0]
