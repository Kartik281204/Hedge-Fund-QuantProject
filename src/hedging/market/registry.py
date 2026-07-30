"""Factory turning a `RegimeConfig` + `MarketConfig` into a simulator instance."""

from __future__ import annotations

from typing import Dict, Union

from hedging.config import Config, MarketConfig, RegimeConfig
from hedging.exceptions import ConfigError
from hedging.market.gbm import GBMSimulator
from hedging.market.heston import HestonSimulator

Simulator = Union[GBMSimulator, HestonSimulator]


def build_simulator(regime: RegimeConfig, market: MarketConfig) -> Simulator:
    if regime.type == "gbm":
        return GBMSimulator(
            S0=market.S0,
            mu=regime.mu,
            sigma=regime.sigma,
            dt=market.dt,
            n_steps=market.n_steps,
        )
    if regime.type == "heston":
        return HestonSimulator(
            S0=market.S0,
            mu=regime.mu,
            v0=regime.v0,
            kappa=regime.kappa,
            theta=regime.theta,
            xi=regime.xi,
            rho=regime.rho,
            dt=market.dt,
            n_steps=market.n_steps,
        )
    raise ConfigError(f"unknown regime type: {regime.type!r}")


def build_all_simulators(config: Config) -> Dict[str, Simulator]:
    return {name: build_simulator(r, config.market) for name, r in config.regimes.items()}
