import numpy as np
import pytest

from hedging.config import HedgingConfig, MarketConfig, RegimeConfig
from hedging.env.hedging_env import HedgingEnv
from hedging.exceptions import HedgingEnvError
from hedging.market.gbm import GBMSimulator
from hedging.pricing.black_scholes import bs_call_price


def _build_env(cost_rate=0.0, lam=0.0):
    market = MarketConfig(S0=100.0, K=100.0, r=0.02, dt=1 / 252, n_steps=21, prior_vol=0.2, vol_window=5)
    regime = RegimeConfig(name="test_gbm", type="gbm", mu=0.02, pricing_vol=0.2, sigma=0.2)
    hedging = HedgingConfig(
        transaction_cost_rate=cost_rate, variance_penalty_lambda=lam, static_baseline_vol=0.2
    )
    sim = GBMSimulator(S0=market.S0, mu=regime.mu, sigma=regime.sigma, dt=market.dt, n_steps=market.n_steps)
    env = HedgingEnv(
        market=market,
        hedging=hedging,
        regimes={"test_gbm": regime},
        simulators={"test_gbm": sim},
        rng=np.random.default_rng(0),
        fixed_regime="test_gbm",
    )
    return env, market, regime, sim


def _run_constant_hedge(env, path, h_value):
    obs, _ = env.reset(options={"path": path})
    terminated = False
    step_info = {}
    while not terminated:
        obs, reward, terminated, truncated, step_info = env.step(np.array([h_value], dtype=np.float32))
    return step_info


def test_zero_hedge_frictionless_matches_premium_minus_payoff():
    env, market, regime, sim = _build_env(cost_rate=0.0, lam=0.0)
    path = sim.simulate_path(np.random.default_rng(123))
    step_info = _run_constant_hedge(env, path, 0.0)

    C0 = bs_call_price(market.S0, market.K, market.maturity, market.r, regime.pricing_vol)
    payoff = max(path[-1] - market.K, 0.0)
    assert step_info["total_pnl"] == pytest.approx(C0 - payoff, abs=1e-8)


def test_full_hedge_frictionless_matches_telescoping_sum():
    env, market, regime, sim = _build_env(cost_rate=0.0, lam=0.0)
    path = sim.simulate_path(np.random.default_rng(456))
    step_info = _run_constant_hedge(env, path, 1.0)

    C0 = bs_call_price(market.S0, market.K, market.maturity, market.r, regime.pricing_vol)
    payoff = max(path[-1] - market.K, 0.0)
    expected = C0 + (path[-1] - path[0]) - payoff
    assert step_info["total_pnl"] == pytest.approx(expected, abs=1e-8)


def test_transaction_costs_strictly_reduce_pnl_given_same_actions():
    env0, market, regime, sim = _build_env(cost_rate=0.0, lam=0.0)
    env_cost, _, _, _ = _build_env(cost_rate=0.01, lam=0.0)
    path = sim.simulate_path(np.random.default_rng(789))

    def run_alternating(env):
        obs, _ = env.reset(options={"path": path})
        terminated, t, step_info = False, 0, {}
        while not terminated:
            h = 1.0 if t % 2 == 0 else 0.0
            obs, reward, terminated, truncated, step_info = env.step(np.array([h], dtype=np.float32))
            t += 1
        return step_info["total_pnl"]

    assert run_alternating(env_cost) < run_alternating(env0)


def test_reward_sum_equals_mean_variance_shaped_objective():
    lam = 0.07
    env, market, regime, sim = _build_env(cost_rate=0.0008, lam=lam)
    path = sim.simulate_path(np.random.default_rng(11))

    obs, _ = env.reset(options={"path": path})
    terminated, reward_sum, step_info = False, 0.0, {}
    action_rng = np.random.default_rng(22)
    while not terminated:
        h = action_rng.uniform(0, 1)
        obs, reward, terminated, truncated, step_info = env.step(np.array([h], dtype=np.float32))
        reward_sum += reward

    total_pnl = step_info["total_pnl"]
    assert reward_sum == pytest.approx(total_pnl - lam * total_pnl**2, abs=1e-6)


def test_observation_shape_and_bounds():
    env, market, regime, sim = _build_env()
    path = sim.simulate_path(np.random.default_rng(1))
    obs, info = env.reset(options={"path": path})
    assert obs.shape == (5,)
    assert 0.0 <= obs[1] <= 1.0  # tau_norm
    assert 0.0 <= obs[2] <= 1.0  # prev_hedge

    obs2, reward, terminated, truncated, step_info = env.step(np.array([0.4], dtype=np.float32))
    assert obs2.shape == (5,)
    assert isinstance(reward, float)


def test_reset_rejects_wrong_path_shape():
    env, market, regime, sim = _build_env()
    with pytest.raises(HedgingEnvError):
        env.reset(options={"path": np.array([100.0, 101.0])})


def test_step_before_reset_raises():
    env, market, regime, sim = _build_env()
    with pytest.raises(HedgingEnvError):
        env.step(np.array([0.5], dtype=np.float32))


def test_episode_length_matches_n_steps():
    env, market, regime, sim = _build_env()
    path = sim.simulate_path(np.random.default_rng(7))
    obs, _ = env.reset(options={"path": path})
    n = 0
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        n += 1
    assert n == market.n_steps
