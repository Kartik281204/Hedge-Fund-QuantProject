"""Typed configuration schema + YAML loader.

Centralizing config here means every numeric constant in the pipeline (vol
levels, cost rates, PPO hyperparameters, ...) lives in one YAML file and one
validated schema, instead of being scattered as magic numbers through the
codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

from hedging.exceptions import ConfigError


@dataclass(frozen=True)
class MarketConfig:
    S0: float
    K: float
    r: float
    dt: float
    n_steps: int
    prior_vol: float
    vol_window: int

    def __post_init__(self) -> None:
        if self.S0 <= 0:
            raise ConfigError(f"market.S0 must be > 0, got {self.S0}")
        if self.K <= 0:
            raise ConfigError(f"market.K must be > 0, got {self.K}")
        if self.dt <= 0:
            raise ConfigError(f"market.dt must be > 0, got {self.dt}")
        if self.n_steps < 1:
            raise ConfigError(f"market.n_steps must be >= 1, got {self.n_steps}")
        if self.prior_vol <= 0:
            raise ConfigError(f"market.prior_vol must be > 0, got {self.prior_vol}")
        if self.vol_window < 1:
            raise ConfigError(f"market.vol_window must be >= 1, got {self.vol_window}")

    @property
    def maturity(self) -> float:
        """Time to maturity in years, T = n_steps * dt."""
        return self.n_steps * self.dt


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    type: str  # "gbm" or "heston"
    mu: float
    pricing_vol: float
    # GBM-only
    sigma: float | None = None
    # Heston-only
    v0: float | None = None
    kappa: float | None = None
    theta: float | None = None
    xi: float | None = None
    rho: float | None = None

    def __post_init__(self) -> None:
        if self.type not in ("gbm", "heston"):
            raise ConfigError(
                f"regimes.{self.name}.type must be 'gbm' or 'heston', got {self.type!r}"
            )
        if self.pricing_vol <= 0:
            raise ConfigError(f"regimes.{self.name}.pricing_vol must be > 0")
        if self.type == "gbm":
            if self.sigma is None or self.sigma <= 0:
                raise ConfigError(f"regimes.{self.name}: gbm regime requires sigma > 0")
        if self.type == "heston":
            missing = [
                p
                for p in ("v0", "kappa", "theta", "xi", "rho")
                if getattr(self, p) is None
            ]
            if missing:
                raise ConfigError(
                    f"regimes.{self.name}: heston regime missing params {missing}"
                )
            if self.v0 <= 0 or self.kappa <= 0 or self.theta <= 0 or self.xi <= 0:
                raise ConfigError(
                    f"regimes.{self.name}: heston v0/kappa/theta/xi must be > 0"
                )
            if not (-1.0 <= self.rho <= 1.0):
                raise ConfigError(f"regimes.{self.name}: heston rho must be in [-1, 1]")


@dataclass(frozen=True)
class HedgingConfig:
    transaction_cost_rate: float
    variance_penalty_lambda: float
    static_baseline_vol: float

    def __post_init__(self) -> None:
        if self.transaction_cost_rate < 0:
            raise ConfigError("hedging.transaction_cost_rate must be >= 0")
        if self.variance_penalty_lambda < 0:
            raise ConfigError("hedging.variance_penalty_lambda must be >= 0")
        if self.static_baseline_vol <= 0:
            raise ConfigError("hedging.static_baseline_vol must be > 0")


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int
    rollout_len: int
    n_envs: int
    n_epochs: int
    n_minibatches: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coef: float
    vf_coef: float
    max_grad_norm: float
    hidden_sizes: List[int]
    log_std_init: float
    seed: int
    log_every_updates: int

    def __post_init__(self) -> None:
        if self.total_timesteps < 1:
            raise ConfigError("training.total_timesteps must be >= 1")
        if self.rollout_len < 1:
            raise ConfigError("training.rollout_len must be >= 1")
        if self.n_envs < 1:
            raise ConfigError("training.n_envs must be >= 1")
        if self.rollout_len % self.n_envs != 0:
            raise ConfigError(
                "training.rollout_len must be divisible by training.n_envs "
                f"(got rollout_len={self.rollout_len}, n_envs={self.n_envs})"
            )
        if self.rollout_len % self.n_minibatches != 0:
            raise ConfigError(
                "training.rollout_len must be divisible by training.n_minibatches "
                f"(got rollout_len={self.rollout_len}, n_minibatches={self.n_minibatches})"
            )
        if not (0.0 < self.clip_range < 1.0):
            raise ConfigError("training.clip_range must be in (0, 1)")


@dataclass(frozen=True)
class EvaluationConfig:
    n_eval_episodes: int
    seed: int

    def __post_init__(self) -> None:
        if self.n_eval_episodes < 1:
            raise ConfigError("evaluation.n_eval_episodes must be >= 1")


@dataclass(frozen=True)
class Config:
    market: MarketConfig
    regimes: Dict[str, RegimeConfig]
    hedging: HedgingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def __post_init__(self) -> None:
        if not self.regimes:
            raise ConfigError("config must define at least one regime under 'regimes'")


def _require(d: dict, key: str, section: str) -> object:
    if key not in d:
        raise ConfigError(f"missing required key '{key}' in section '{section}'")
    return d[key]


def load_config(path: str | Path) -> Config:
    """Load and validate the pipeline config from a YAML file.

    Raises `ConfigError` (never a raw KeyError/TypeError) on any structural
    or semantic problem, so failures are legible instead of a stack trace
    into dict-indexing code.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"failed to parse YAML in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"top-level config in {path} must be a mapping")

    try:
        market_raw = _require(raw, "market", "root")
        market = MarketConfig(**market_raw)

        regimes_raw = _require(raw, "regimes", "root")
        if not isinstance(regimes_raw, dict) or not regimes_raw:
            raise ConfigError("'regimes' must be a non-empty mapping")
        regimes = {
            name: RegimeConfig(name=name, **params) for name, params in regimes_raw.items()
        }

        hedging = HedgingConfig(**_require(raw, "hedging", "root"))
        training = TrainingConfig(**_require(raw, "training", "root"))
        evaluation = EvaluationConfig(**_require(raw, "evaluation", "root"))
    except TypeError as e:
        # dataclass raises TypeError on unexpected/missing kwargs - translate
        # into our own error type so callers only ever see ConfigError.
        raise ConfigError(f"invalid config structure in {path}: {e}") from e

    return Config(
        market=market,
        regimes=regimes,
        hedging=hedging,
        training=training,
        evaluation=evaluation,
    )
