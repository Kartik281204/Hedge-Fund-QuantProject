import numpy as np
import pytest

from hedging.agents.delta_hedge import AdaptiveBSBaseline, StaticBSBaseline
from hedging.config import MarketConfig
from hedging.pricing.black_scholes import bs_call_delta


def _market():
    return MarketConfig(S0=100.0, K=100.0, r=0.02, dt=1 / 252, n_steps=21, prior_vol=0.2, vol_window=5)


def test_static_baseline_matches_bs_delta_directly():
    market = _market()
    baseline = StaticBSBaseline(market, static_vol=0.25)

    S, tau_norm = 110.0, 0.5
    log_moneyness = np.log(S / market.K)
    obs = np.array([log_moneyness, tau_norm, 0.3, 0.20, 0.0], dtype=np.float32)

    tau = tau_norm * market.maturity
    expected = bs_call_delta(S, market.K, tau, market.r, 0.25)
    action = baseline.act(obs)
    assert action.shape == (1,)
    assert action[0] == pytest.approx(expected, abs=1e-5)


def test_adaptive_baseline_uses_vol_proxy_from_obs():
    market = _market()
    baseline = AdaptiveBSBaseline(market)

    S, tau_norm, vol_proxy = 95.0, 0.3, 0.35
    obs = np.array([np.log(S / market.K), tau_norm, 0.1, vol_proxy, 0.0], dtype=np.float32)

    tau = tau_norm * market.maturity
    expected = bs_call_delta(S, market.K, tau, market.r, vol_proxy)
    action = baseline.act(obs)
    assert action[0] == pytest.approx(expected, abs=1e-5)


def test_static_and_adaptive_diverge_when_vol_proxy_differs_from_static_vol():
    market = _market()
    static = StaticBSBaseline(market, static_vol=0.15)
    adaptive = AdaptiveBSBaseline(market)

    obs = np.array([0.0, 0.5, 0.5, 0.45, 0.0], dtype=np.float32)  # vol_proxy far from 0.15
    a_static = static.act(obs)[0]
    a_adaptive = adaptive.act(obs)[0]
    assert abs(a_static - a_adaptive) > 1e-3
