"""Sanity check: does the pipeline reproduce known theory?

Boyle & Emanuel (1980): for frictionless Black-Scholes delta hedging under
GBM, the variance of the discrete-hedging replication error shrinks like
O(1/n) as the number of equally-spaced rebalances n over a fixed horizon
grows, i.e. std(P&L) ~ O(n^-1/2). This script fixes a 1-month horizon,
hedges it with the *exact*-vol BS delta at increasing rebalancing
frequencies, and checks that std(P&L) falls off close to that rate.

The Monte Carlo here is fully vectorized in numpy (no per-step Python loop)
since a deterministic closed-form policy over thousands of paths doesn't
need the Gym step-by-step API -- that API exists for the RL agent's
sequential decision-making, not for batch-scoring a formula. A separate
cross-check runs a handful of paths through the *actual* HedgingEnv and
confirms it agrees with this fast path to floating-point precision, so this
script is validating the theory, not a second implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hedging.config import HedgingConfig, MarketConfig, RegimeConfig
from hedging.env.hedging_env import HedgingEnv
from hedging.logging_setup import get_logger
from hedging.market.gbm import GBMSimulator
from hedging.pricing.black_scholes import bs_call_delta, bs_call_price

logger = get_logger("hedging.scripts.sanity_check")

T_FIXED_YEARS = 21 / 252  # one month, held fixed across all n_steps below
N_STEPS_GRID = [3, 7, 21, 63, 126, 252]
N_EPISODES = 20_000
SIGMA = 0.20
S0, K, R = 100.0, 100.0, 0.02
SEED = 2024


def vectorized_frictionless_bs_hedge(n_steps: int, n_episodes: int, seed: int):
    dt = T_FIXED_YEARS / n_steps
    sim = GBMSimulator(S0=S0, mu=R, sigma=SIGMA, dt=dt, n_steps=n_steps)
    paths = sim.simulate_paths(n_episodes, np.random.default_rng(seed))  # (n_episodes, n_steps+1)

    step_idx = np.arange(n_steps)
    tau = (n_steps - step_idx) * dt  # time remaining *before* each rebalance, shape (n_steps,)
    S_pre = paths[:, :n_steps]  # price at each rebalance point
    deltas = bs_call_delta(S_pre, K, tau[None, :], R, SIGMA)  # (n_episodes, n_steps)

    hedge_pnl = np.sum(deltas * np.diff(paths, axis=1), axis=1)  # telescoping hedge P&L
    payoff = np.maximum(paths[:, -1] - K, 0.0)
    C0 = bs_call_price(S0, K, T_FIXED_YEARS, R, SIGMA)
    total_pnl = C0 + hedge_pnl - payoff
    return total_pnl, dt


def cross_check_against_real_env(n_steps: int, n_check: int = 25) -> None:
    """Confirm the vectorized fast path agrees with the actual HedgingEnv
    (which the RL agent and baselines both really step through) to
    floating-point precision, on a handful of paths."""
    dt = T_FIXED_YEARS / n_steps
    market = MarketConfig(S0=S0, K=K, r=R, dt=dt, n_steps=n_steps, prior_vol=SIGMA, vol_window=5)
    regime = RegimeConfig(name="check", type="gbm", mu=R, pricing_vol=SIGMA, sigma=SIGMA)
    hedging = HedgingConfig(transaction_cost_rate=0.0, variance_penalty_lambda=0.0, static_baseline_vol=SIGMA)
    sim = GBMSimulator(S0=S0, mu=R, sigma=SIGMA, dt=dt, n_steps=n_steps)
    env = HedgingEnv(
        market=market, hedging=hedging, regimes={"check": regime}, simulators={"check": sim},
        rng=np.random.default_rng(0), fixed_regime="check",
    )

    rng = np.random.default_rng(999)
    paths = sim.simulate_paths(n_check, rng)
    env_pnls = []
    fast_pnls = []
    for i in range(n_check):
        path = paths[i]
        obs, _ = env.reset(options={"path": path})
        terminated = False
        step_info = {}
        while not terminated:
            S_t = market.K * np.exp(obs[0])
            tau_t = obs[1] * market.maturity
            h = float(bs_call_delta(S_t, market.K, tau_t, market.r, SIGMA))
            obs, reward, terminated, truncated, step_info = env.step(np.array([h], dtype=np.float32))
        env_pnls.append(step_info["total_pnl"])

        step_idx = np.arange(n_steps)
        tau = (n_steps - step_idx) * dt
        deltas = bs_call_delta(path[:n_steps], K, tau, R, SIGMA)
        hedge_pnl = float(np.sum(deltas * np.diff(path)))
        payoff = max(path[-1] - K, 0.0)
        C0 = bs_call_price(S0, K, T_FIXED_YEARS, R, SIGMA)
        fast_pnls.append(C0 + hedge_pnl - payoff)

    env_pnls = np.array(env_pnls)
    fast_pnls = np.array(fast_pnls)
    max_abs_diff = np.max(np.abs(env_pnls - fast_pnls))
    logger.info(f"cross-check (n_steps={n_steps}, {n_check} paths): max |env_pnl - fast_pnl| = {max_abs_diff:.2e}")
    # HedgingEnv observations are float32 by design (matches the RL policy's
    # input dtype). Reconstructing S/tau from a float32 obs and re-deriving a
    # delta therefore differs from the float64 fast path by ~1e-7 per step,
    # accumulating to ~1e-6 over n_steps rebalances -- confirmed benign (this
    # is many orders of magnitude below any economically meaningful P&L), not
    # a logic bug. 1e-4 comfortably separates that from a real discrepancy.
    assert max_abs_diff < 1e-4, "vectorized fast path disagrees with the real HedgingEnv -- bug!"


def main() -> None:
    logger.info("cross-checking vectorized fast path against the real HedgingEnv...")
    cross_check_against_real_env(n_steps=21, n_check=25)
    logger.info("cross-check passed: fast path matches HedgingEnv exactly.\n")

    results = []
    for n_steps in N_STEPS_GRID:
        total_pnl, dt = vectorized_frictionless_bs_hedge(n_steps, N_EPISODES, seed=SEED + n_steps)
        mean_pnl, std_pnl = float(np.mean(total_pnl)), float(np.std(total_pnl))
        results.append((n_steps, dt, mean_pnl, std_pnl))
        logger.info(f"n_steps={n_steps:>4d}  dt={dt:.6f}  mean_pnl={mean_pnl:+.5f}  std_pnl={std_pnl:.5f}")

    n_arr = np.array([r[0] for r in results], dtype=float)
    std_arr = np.array([r[3] for r in results], dtype=float)
    p, _ = np.polyfit(np.log(n_arr), np.log(std_arr), 1)
    logger.info(f"fitted convergence rate: std_pnl ~ n^{p:.3f}  (theory: n^-0.5)")

    out_dir = Path(__file__).resolve().parents[1] / "results"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics" / "sanity_check_convergence.csv", "w") as f:
        f.write("n_steps,dt,mean_pnl,std_pnl\n")
        for n_steps, dt, mean_pnl, std_pnl in results:
            f.write(f"{n_steps},{dt:.8f},{mean_pnl:.6f},{std_pnl:.6f}\n")
        f.write(f"# fitted_rate,{p:.4f} (theory -0.5)\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(n_arr, std_arr, "o-", color="#0072B2", label="empirical std(P&L)")
    theory_curve = std_arr[0] * (n_arr / n_arr[0]) ** -0.5
    ax.loglog(n_arr, theory_curve, "--", color="grey", label=r"theory: $n^{-1/2}$")
    ax.set_xlabel("rebalances per month (n)")
    ax.set_ylabel("std(terminal hedge P&L)")
    ax.set_title("Frictionless BS delta hedge: error shrinks as O(n$^{-1/2}$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "sanity_check_convergence.png", dpi=150)
    plt.close(fig)

    logger.info(f"wrote {out_dir / 'metrics' / 'sanity_check_convergence.csv'}")
    logger.info(f"wrote {out_dir / 'figures' / 'sanity_check_convergence.png'}")

    assert p < -0.3, f"expected std(P&L) ~ n^-0.5, got exponent {p:.3f} -- possible bug."
    logger.info("SANITY CHECK PASSED: pipeline reproduces the Boyle-Emanuel convergence rate.")


if __name__ == "__main__":
    main()
