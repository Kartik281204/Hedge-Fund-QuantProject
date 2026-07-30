"""Matplotlib plotting helpers. Kept separate from evaluation logic so the
backtest harness has no plotting dependency and can be reused headlessly."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe, consistent per-method palette used across every figure.
METHOD_COLORS = {
    "rl_ppo": "#0072B2",
    "static_bs": "#D55E00",
    "adaptive_bs": "#009E73",
}
METHOD_LABELS = {
    "rl_ppo": "RL (PPO)",
    "static_bs": "Static BS delta",
    "adaptive_bs": "Adaptive BS delta",
}


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curve(history: List[dict], save_path: str | Path) -> None:
    updates = [h["update"] for h in history]
    mean_pnl = [h["mean_recent_total_pnl"] for h in history]
    std_pnl = [h["std_recent_total_pnl"] for h in history]
    entropy = [h["entropy"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(updates, mean_pnl, color="#0072B2")
    axes[0].axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_title("Mean terminal P&L (trailing 200 episodes)")
    axes[0].set_xlabel("PPO update")
    axes[0].set_ylabel("P&L")

    axes[1].plot(updates, std_pnl, color="#D55E00")
    axes[1].set_title("Std of terminal P&L (trailing 200 episodes)")
    axes[1].set_xlabel("PPO update")
    axes[1].set_ylabel("Std P&L")

    axes[2].plot(updates, entropy, color="#009E73")
    axes[2].set_title("Policy entropy")
    axes[2].set_xlabel("PPO update")
    axes[2].set_ylabel("Entropy")

    _save(fig, save_path)


def plot_pnl_distributions(
    total_pnls_by_method: Dict[str, np.ndarray], regime_name: str, save_path: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_vals = np.concatenate(list(total_pnls_by_method.values()))
    bins = np.linspace(np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5), 60)

    for method, vals in total_pnls_by_method.items():
        ax.hist(
            vals,
            bins=bins,
            alpha=0.5,
            density=True,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
        )
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", label="breakeven")
    ax.set_title(f"Terminal hedge P&L distribution -- {regime_name}")
    ax.set_xlabel("Total P&L")
    ax.set_ylabel("Density")
    ax.legend()
    _save(fig, save_path)


def plot_hedge_path_example(
    examples_by_method: Dict[str, dict], regime_name: str, save_path: str | Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[1, 1.4])

    any_example = next(iter(examples_by_method.values()))
    prices = any_example["prices"]
    axes[0].plot(prices, color="black", linewidth=1.3)
    axes[0].set_ylabel("Underlying price")
    axes[0].set_title(f"Example path and hedge ratios -- {regime_name}")

    for method, ex in examples_by_method.items():
        axes[1].step(
            np.arange(len(ex["hedge_ratios"])),
            ex["hedge_ratios"],
            where="post",
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
        )
    axes[1].set_ylabel("Hedge ratio (shares/option)")
    axes[1].set_xlabel("Rebalancing step")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    _save(fig, save_path)


def plot_summary_bars(
    metrics_by_regime_method: Dict[str, Dict[str, dict]],
    metric_key: str,
    ylabel: str,
    title: str,
    save_path: str | Path,
) -> None:
    regimes = list(metrics_by_regime_method.keys())
    methods = list(next(iter(metrics_by_regime_method.values())).keys())

    x = np.arange(len(regimes))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, method in enumerate(methods):
        vals = [metrics_by_regime_method[r][method][metric_key] for r in regimes]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width=width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.axhline(0.0, color="grey", linewidth=0.8)
    _save(fig, save_path)
