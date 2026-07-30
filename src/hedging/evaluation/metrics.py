"""Risk and P&L metrics for comparing hedging methods.

All metrics operate on arrays collected across many evaluation episodes
(one `total_pnl` / `total_cost` / `turnover` value per episode).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from hedging.exceptions import EvaluationError


@dataclass(frozen=True)
class Metrics:
    n_episodes: int
    mean_pnl: float
    std_pnl: float
    var_5: float
    cvar_5: float
    mean_total_cost: float
    mean_turnover: float
    sharpe_like: float

    def as_dict(self) -> dict:
        return asdict(self)


def _cvar(pnls: np.ndarray, alpha: float = 0.05) -> float:
    """Expected Shortfall at the alpha level: mean of the worst alpha-fraction
    of outcomes (most negative P&L). A coherent tail-risk measure, unlike VaR."""
    sorted_pnls = np.sort(pnls)
    cutoff = max(1, int(np.ceil(alpha * len(sorted_pnls))))
    return float(np.mean(sorted_pnls[:cutoff]))


def _var(pnls: np.ndarray, alpha: float = 0.05) -> float:
    return float(np.percentile(pnls, alpha * 100))


def compute_metrics(
    total_pnls: np.ndarray, total_costs: np.ndarray, turnovers: np.ndarray
) -> Metrics:
    total_pnls = np.asarray(total_pnls, dtype=float)
    total_costs = np.asarray(total_costs, dtype=float)
    turnovers = np.asarray(turnovers, dtype=float)

    if len(total_pnls) == 0:
        raise EvaluationError("cannot compute metrics over zero episodes")
    if not (len(total_pnls) == len(total_costs) == len(turnovers)):
        raise EvaluationError("total_pnls/total_costs/turnovers length mismatch")

    std_pnl = float(np.std(total_pnls))
    mean_pnl = float(np.mean(total_pnls))
    sharpe_like = mean_pnl / std_pnl if std_pnl > 1e-12 else float("nan")

    return Metrics(
        n_episodes=len(total_pnls),
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        var_5=_var(total_pnls, 0.05),
        cvar_5=_cvar(total_pnls, 0.05),
        mean_total_cost=float(np.mean(total_costs)),
        mean_turnover=float(np.mean(turnovers)),
        sharpe_like=sharpe_like,
    )
