import numpy as np
import pytest

from hedging.pricing.black_scholes import bs_call_delta, bs_call_greeks, bs_call_price


def test_atm_call_price_matches_known_reference():
    # Textbook reference case: S=100, K=100, T=1, r=0.05, sigma=0.2 -> C ~= 10.4506
    price = bs_call_price(S=100.0, K=100.0, tau=1.0, r=0.05, sigma=0.2)
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_delta_bounds_and_monotonicity():
    S = np.linspace(50, 150, 50)
    delta = bs_call_delta(S, K=100.0, tau=0.5, r=0.02, sigma=0.25)
    assert np.all(delta >= 0.0) and np.all(delta <= 1.0)
    assert np.all(np.diff(delta) >= 0.0), "call delta must be non-decreasing in S"


def test_delta_extremes_deep_itm_otm():
    deep_itm_delta = bs_call_delta(S=1000.0, K=100.0, tau=0.25, r=0.02, sigma=0.2)
    deep_otm_delta = bs_call_delta(S=1.0, K=100.0, tau=0.25, r=0.02, sigma=0.2)
    assert deep_itm_delta == pytest.approx(1.0, abs=1e-6)
    assert deep_otm_delta == pytest.approx(0.0, abs=1e-6)


def test_expiry_edge_case_matches_intrinsic_value():
    price_itm = bs_call_price(S=110.0, K=100.0, tau=0.0, r=0.02, sigma=0.2)
    price_otm = bs_call_price(S=90.0, K=100.0, tau=0.0, r=0.02, sigma=0.2)
    delta_itm = bs_call_delta(S=110.0, K=100.0, tau=0.0, r=0.02, sigma=0.2)
    delta_otm = bs_call_delta(S=90.0, K=100.0, tau=0.0, r=0.02, sigma=0.2)

    assert price_itm == pytest.approx(10.0)
    assert price_otm == pytest.approx(0.0)
    assert delta_itm == pytest.approx(1.0)
    assert delta_otm == pytest.approx(0.0)


def test_gamma_and_vega_positive_away_from_expiry():
    greeks = bs_call_greeks(S=100.0, K=100.0, tau=0.5, r=0.02, sigma=0.2)
    assert greeks.gamma > 0
    assert greeks.vega > 0


def test_vectorized_inputs_return_matching_shape():
    S = np.array([80.0, 100.0, 120.0])
    greeks = bs_call_greeks(S, K=100.0, tau=0.25, r=0.02, sigma=0.3)
    assert greeks.price.shape == (3,)
    assert greeks.delta.shape == (3,)
    # in-the-money option should be worth more than out-of-the-money, all else equal
    assert greeks.price[2] > greeks.price[0]
