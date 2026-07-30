"""Train a single PPO hedging policy across all configured volatility
regimes (domain randomization: each of the n_envs parallel environments
independently samples a regime uniformly on every reset). The trained
policy is then evaluated separately per regime in `evaluate.py`.

Usage:
    python scripts/train.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hedging.agents.ppo_agent import PPOAgent
from hedging.config import load_config
from hedging.env.vector_env import VectorHedgingEnv
from hedging.logging_setup import get_logger
from hedging.market.registry import build_all_simulators
from hedging.viz.plots import plot_training_curve

logger = get_logger("hedging.scripts.train")

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--model-out", default=str(ROOT / "models" / "ppo_agent.pkl"))
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "checkpoint.pkl"))
    parser.add_argument("--resume", action="store_true", help="continue from --checkpoint if it exists")
    parser.add_argument("--output-tag", default="", help="suffix for output filenames, e.g. 'high_cost'")
    args = parser.parse_args()
    tag = f"_{args.output_tag}" if args.output_tag else ""

    cfg = load_config(args.config)
    logger.info(
        f"training PPO across regimes {list(cfg.regimes.keys())} | "
        f"total_timesteps={cfg.training.total_timesteps} | n_envs={cfg.training.n_envs} | "
        f"rollout_len={cfg.training.rollout_len} | seed={cfg.training.seed}"
    )

    simulators = build_all_simulators(cfg)
    vec_env = VectorHedgingEnv(
        market=cfg.market,
        hedging=cfg.hedging,
        regimes=cfg.regimes,
        simulators=simulators,
        n_envs=cfg.training.n_envs,
        seed=cfg.training.seed,
    )

    agent = PPOAgent(
        obs_dim=5,
        hidden_sizes=cfg.training.hidden_sizes,
        log_std_init=cfg.training.log_std_init,
        seed=cfg.training.seed,
    )

    t0 = time.time()
    history = agent.train(
        vec_env,
        cfg.training,
        checkpoint_path=args.checkpoint,
        checkpoint_every=20,
        resume=args.resume,
    )
    elapsed = time.time() - t0
    steps_done = history[-1]["total_steps"] if history else 0
    target_steps = (cfg.training.total_timesteps // cfg.training.rollout_len) * cfg.training.rollout_len
    logger.info(f"this invocation ran {elapsed:.1f}s, reaching {steps_done}/{target_steps} steps")

    if steps_done < target_steps:
        logger.info(
            "target not yet reached -- rerun with --resume to continue "
            f"(checkpoint at {args.checkpoint})"
        )
        return

    model_out = Path(args.model_out)
    agent.save(model_out)

    metrics_dir = ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / f"training_history{tag}.csv"
    keys = list(history[0].keys())
    with open(history_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in history:
            f.write(",".join(str(row[k]) for k in keys) + "\n")
    logger.info(f"wrote {history_path}")

    fig_path = ROOT / "results" / "figures" / f"training_curve{tag}.png"
    plot_training_curve(history, fig_path)
    logger.info(f"wrote {fig_path}")

    last = history[-1]
    logger.info(
        f"final trailing stats: mean_pnl={last['mean_recent_total_pnl']:.4f} "
        f"std_pnl={last['std_recent_total_pnl']:.4f} entropy={last['entropy']:.4f}"
    )


if __name__ == "__main__":
    main()
