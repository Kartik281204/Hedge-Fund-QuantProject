"""Evaluate the trained PPO hedging policy against both Black-Scholes
delta-hedge baselines, separately per volatility regime, on paired price
paths (see `evaluation.backtest` for why paired evaluation matters).

Usage:
    python scripts/evaluate.py [--config config/config.yaml] [--model models/ppo_agent.pkl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hedging.agents.delta_hedge import AdaptiveBSBaseline, StaticBSBaseline
from hedging.agents.ppo_agent import PPOAgent
from hedging.config import load_config
from hedging.env.hedging_env import HedgingEnv
from hedging.evaluation.backtest import generate_eval_paths, run_backtest
from hedging.logging_setup import get_logger
from hedging.market.registry import build_all_simulators
from hedging.viz.plots import plot_hedge_path_example, plot_pnl_distributions, plot_summary_bars

logger = get_logger("hedging.scripts.evaluate")
ROOT = Path(__file__).resolve().parents[1]

METHOD_KEYS = ["rl_ppo", "static_bs", "adaptive_bs"]
METRIC_FIELDS = ["n_episodes", "mean_pnl", "std_pnl", "var_5", "cvar_5", "mean_total_cost", "mean_turnover", "sharpe_like"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--model", default=str(ROOT / "models" / "ppo_agent.pkl"))
    parser.add_argument("--output-tag", default="", help="suffix for output filenames, e.g. 'high_cost'")
    args = parser.parse_args()
    tag = f"_{args.output_tag}" if args.output_tag else ""

    cfg = load_config(args.config)
    simulators = build_all_simulators(cfg)
    ppo_agent = PPOAgent.load(args.model)

    policies = {
        "rl_ppo": ppo_agent,
        "static_bs": StaticBSBaseline(cfg.market, cfg.hedging.static_baseline_vol),
        "adaptive_bs": AdaptiveBSBaseline(cfg.market),
    }

    # One shared env instance is safe to reuse across every episode/method:
    # reset() fully overwrites all episode state before each run_episode().
    env = HedgingEnv(
        market=cfg.market,
        hedging=cfg.hedging,
        regimes=cfg.regimes,
        simulators=simulators,
        rng=np.random.default_rng(cfg.evaluation.seed),
    )

    figures_dir = ROOT / "results" / "figures"
    metrics_dir = ROOT / "results" / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"generating {cfg.evaluation.n_eval_episodes} paired evaluation paths per regime "
        f"(seed={cfg.evaluation.seed})..."
    )
    eval_paths = generate_eval_paths(simulators, cfg.evaluation.n_eval_episodes, cfg.evaluation.seed)

    all_metrics: dict[str, dict[str, dict]] = {}

    for regime_name in sorted(cfg.regimes.keys()):
        logger.info(f"backtesting regime: {regime_name}")
        paths = eval_paths[regime_name]
        pnl_by_method: dict[str, np.ndarray] = {}
        example_by_method: dict[str, dict] = {}
        regime_metrics: dict[str, dict] = {}

        for method_key in METHOD_KEYS:
            result = run_backtest(env, policies[method_key], regime_name, paths, keep_examples=1)
            m = result["metrics"]
            regime_metrics[method_key] = m.as_dict()
            pnl_by_method[method_key] = result["total_pnls"]
            if result["examples"]:
                example_by_method[method_key] = result["examples"][0]

            logger.info(
                f"  {method_key:>12s} | mean_pnl {m.mean_pnl:+8.4f} | std_pnl {m.std_pnl:7.4f} | "
                f"cvar_5 {m.cvar_5:+8.4f} | cost {m.mean_total_cost:7.4f} | "
                f"turnover {m.mean_turnover:6.3f} | sharpe {m.sharpe_like:+6.3f}"
            )

        all_metrics[regime_name] = regime_metrics

        plot_pnl_distributions(pnl_by_method, regime_name, figures_dir / f"pnl_dist_{regime_name}{tag}.png")
        plot_hedge_path_example(example_by_method, regime_name, figures_dir / f"hedge_path_{regime_name}{tag}.png")
        logger.info(f"  wrote pnl_dist_{regime_name}{tag}.png, hedge_path_{regime_name}{tag}.png")

    plot_summary_bars(
        all_metrics, "std_pnl", "Std of terminal P&L",
        "Hedging risk by regime (lower = better)", figures_dir / f"summary_std_pnl{tag}.png",
    )
    plot_summary_bars(
        all_metrics, "cvar_5", "CVaR 5% (Expected Shortfall)",
        "Tail risk by regime (higher = less severe losses)", figures_dir / f"summary_cvar{tag}.png",
    )
    plot_summary_bars(
        all_metrics, "mean_total_cost", "Mean transaction cost",
        "Transaction cost by regime", figures_dir / f"summary_cost{tag}.png",
    )
    logger.info(f"wrote summary_std_pnl{tag}.png, summary_cvar{tag}.png, summary_cost{tag}.png")

    csv_path = metrics_dir / f"evaluation_metrics{tag}.csv"
    with open(csv_path, "w") as f:
        f.write("regime,method," + ",".join(METRIC_FIELDS) + "\n")
        for regime_name, methods in all_metrics.items():
            for method_key in METHOD_KEYS:
                m = methods[method_key]
                f.write(f"{regime_name},{method_key}," + ",".join(str(m[k]) for k in METRIC_FIELDS) + "\n")
    logger.info(f"wrote {csv_path}")

    md_path = metrics_dir / f"evaluation_summary{tag}.md"
    with open(md_path, "w") as f:
        f.write("# Evaluation summary\n\n")
        f.write(
            f"`{cfg.evaluation.n_eval_episodes}` paired paths per regime, seed `{cfg.evaluation.seed}`. "
            "Lower `std_pnl` = less hedging risk. `cvar_5` is the mean P&L of the worst 5% of "
            "outcomes (less negative = milder tail losses). `mean_cost` includes the final unwind.\n\n"
        )
        for regime_name, methods in all_metrics.items():
            f.write(f"## {regime_name}\n\n")
            f.write("| method | mean_pnl | std_pnl | cvar_5 | mean_cost | turnover | sharpe-like |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for method_key in METHOD_KEYS:
                m = methods[method_key]
                f.write(
                    f"| {method_key} | {m['mean_pnl']:.4f} | {m['std_pnl']:.4f} | {m['cvar_5']:.4f} | "
                    f"{m['mean_total_cost']:.4f} | {m['mean_turnover']:.3f} | {m['sharpe_like']:.3f} |\n"
                )
            f.write("\n")
    logger.info(f"wrote {md_path}")
    logger.info("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
