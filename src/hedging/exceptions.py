"""Typed exception hierarchy for the `hedging` package.

Every error raised by this package inherits from `HedgingError`, so callers
(and tests) can catch package-specific failures without accidentally
swallowing unrelated bugs (e.g. a stray `KeyError` from a typo).
"""

from __future__ import annotations


class HedgingError(Exception):
    """Base class for all errors raised by the `hedging` package."""


class ConfigError(HedgingError):
    """Raised when configuration is missing, malformed, or fails validation."""


class SimulationError(HedgingError):
    """Raised when a market simulation produces invalid data (NaN, negative
    price, negative variance that wasn't floored, mismatched shapes, etc.)."""


class HedgingEnvError(HedgingError):
    """Raised on invalid use of the hedging Gymnasium environment (e.g.
    calling `step` before `reset`, or an out-of-bounds action)."""


class ModelNotTrainedError(HedgingError):
    """Raised when an agent is used for inference/evaluation before training
    or loading has completed."""


class EvaluationError(HedgingError):
    """Raised when a backtest/evaluation run cannot be completed as specified."""
