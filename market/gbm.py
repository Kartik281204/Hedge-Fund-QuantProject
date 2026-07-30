"""Geometric Brownian Motion price path simulator.

dS = mu S dt + sigma S dW, simulated exactly (log-normal) rather than via
Euler discretization, since GBM has a closed-form transition density:

    S_{t+dt} = S_t * exp( (mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z ),  Z ~ N(0,1)
"""

from __future__ import annotations

import numpy as np

from hedging.exceptions import SimulationError


class GBMSimulator:
    def __init__(self, S0: float, mu: float, sigma: float, dt: float, n_steps: int) -> None:
        if S0 <= 0:
            raise SimulationError("GBMSimulator: S0 must be > 0")
        if sigma <= 0:
            raise SimulationError("GBMSimulator: sigma must be > 0")
        if dt <= 0:
            raise SimulationError("GBMSimulator: dt must be > 0")
        if n_steps < 1:
            raise SimulationError("GBMSimulator: n_steps must be >= 1")
        self.S0 = float(S0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.n_steps = int(n_steps)

    def simulate_paths(self, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        """Return an array of shape (n_paths, n_steps + 1)."""
        if n_paths < 1:
            raise SimulationError("n_paths must be >= 1")
        Z = rng.standard_normal(size=(n_paths, self.n_steps))
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        log_increments = drift + diffusion
        log_paths = np.cumsum(log_increments, axis=1)
        S = self.S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1))
        if not np.all(np.isfinite(S)):
            raise SimulationError("GBMSimulator produced non-finite prices")
        return S

    def simulate_path(self, rng: np.random.Generator) -> np.ndarray:
        return self.simulate_paths(1, rng)[0]
