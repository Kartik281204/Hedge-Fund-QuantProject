import numpy as np
import pytest

from hedging.market.gbm import GBMSimulator
from hedging.market.heston import HestonSimulator


def test_gbm_paths_shape_and_start_value():
    sim = GBMSimulator(S0=100.0, mu=0.05, sigma=0.2, dt=1 / 252, n_steps=21)
    rng = np.random.default_rng(0)
    paths = sim.simulate_paths(1000, rng)
    assert paths.shape == (1000, 22)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0), "GBM paths must stay strictly positive"


def test_gbm_log_return_moments_match_theory():
    # Over many paths, empirical mean/var of log-returns should match the
    # theoretical (mu - 0.5 sigma^2) dt and sigma^2 dt within Monte Carlo noise.
    mu, sigma, dt, n_steps = 0.03, 0.25, 1 / 252, 21
    sim = GBMSimulator(S0=100.0, mu=mu, sigma=sigma, dt=dt, n_steps=n_steps)
    rng = np.random.default_rng(1)
    paths = sim.simulate_paths(50_000, rng)
    log_returns = np.diff(np.log(paths), axis=1)

    theoretical_mean = (mu - 0.5 * sigma**2) * dt
    theoretical_var = sigma**2 * dt

    assert log_returns.mean() == pytest.approx(theoretical_mean, abs=5e-4)
    assert log_returns.var() == pytest.approx(theoretical_var, rel=0.05)


def test_heston_variance_never_negative():
    sim = HestonSimulator(
        S0=100.0, mu=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, dt=1 / 252, n_steps=21
    )
    rng = np.random.default_rng(2)
    _, v = sim.simulate_paths_with_variance(5000, rng)
    assert np.all(v >= 0.0)


def test_heston_reduces_to_constant_vol_when_vol_of_vol_is_tiny():
    # With xi ~ 0 and v0 = theta, variance barely moves, so realized vol of
    # the S path should be close to sqrt(theta) -- same statistical check as
    # the GBM test above, just via the Heston code path.
    theta = 0.09  # sqrt(theta) = 0.30
    sim = HestonSimulator(
        S0=100.0, mu=0.02, v0=theta, kappa=2.0, theta=theta, xi=1e-4, rho=0.0, dt=1 / 252, n_steps=21
    )
    rng = np.random.default_rng(3)
    paths = sim.simulate_paths(20_000, rng)
    log_returns = np.diff(np.log(paths), axis=1)
    realized_vol = np.sqrt(log_returns.var() / sim.dt)
    assert realized_vol == pytest.approx(np.sqrt(theta), rel=0.05)


def test_heston_leverage_effect_sign():
    # rho < 0 should induce negative correlation between returns and
    # variance changes (the well-known equity "leverage effect").
    sim = HestonSimulator(
        S0=100.0, mu=0.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.8, dt=1 / 252, n_steps=21
    )
    rng = np.random.default_rng(4)
    S, v = sim.simulate_paths_with_variance(20_000, rng)
    log_returns = np.diff(np.log(S), axis=1).ravel()
    d_var = np.diff(v, axis=1).ravel()
    corr = np.corrcoef(log_returns, d_var)[0, 1]
    assert corr < -0.1
