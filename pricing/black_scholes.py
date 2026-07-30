"""Black-Scholes pricing and Greeks for European calls, no dividends.

All functions are vectorized over numpy arrays (or accept plain floats) and
handle the tau -> 0 (at/past expiry) edge case explicitly rather than
dividing by zero.

Standard formulas (r = risk-free rate, continuously compounded):
    d1 = (ln(S/K) + (r + 0.5 sigma^2) tau) / (sigma sqrt(tau))
    d2 = d1 - sigma sqrt(tau)
    C  = S N(d1) - K exp(-r tau) N(d2)
    Delta = N(d1)
    Gamma = phi(d1) / (S sigma sqrt(tau))
    Vega  = S phi(d1) sqrt(tau)
    Theta = -S phi(d1) sigma / (2 sqrt(tau)) - r K exp(-r tau) N(d2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_MIN_TAU = 1e-8
_MIN_SIGMA = 1e-8


@dataclass(frozen=True)
class Greeks:
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray


def _d1_d2(S, K, tau, r, sigma):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    tau_safe = np.maximum(tau, _MIN_TAU)
    sigma_safe = np.maximum(sigma, _MIN_SIGMA)
    sqrt_tau = np.sqrt(tau_safe)

    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * tau_safe) / (sigma_safe * sqrt_tau)
    d2 = d1 - sigma_safe * sqrt_tau
    return d1, d2, tau_safe, sqrt_tau


def bs_call_price(S, K, tau, r, sigma):
    """European call price. Returns intrinsic value at/after expiry (tau<=0)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)

    d1, d2, tau_safe, _ = _d1_d2(S, K, tau, r, sigma)
    price = S * norm.cdf(d1) - K * np.exp(-r * tau_safe) * norm.cdf(d2)
    intrinsic = np.maximum(S - K, 0.0)
    return np.where(tau <= _MIN_TAU, intrinsic, price)


def bs_call_delta(S, K, tau, r, sigma):
    """Call delta = N(d1). At/after expiry, delta is the indicator 1{S>K}."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)

    d1, _, _, _ = _d1_d2(S, K, tau, r, sigma)
    delta = norm.cdf(d1)
    expiry_delta = (S > K).astype(float)
    return np.where(tau <= _MIN_TAU, expiry_delta, delta)


def bs_call_greeks(S, K, tau, r, sigma) -> Greeks:
    """Full Greek set in one call (avoids recomputing d1/d2 three times)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)

    d1, d2, tau_safe, sqrt_tau = _d1_d2(S, K, tau, r, sigma)
    sigma_safe = np.maximum(sigma_arr, _MIN_SIGMA)

    price = bs_call_price(S, K, tau, r, sigma)
    delta = bs_call_delta(S, K, tau, r, sigma)

    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (S * sigma_safe * sqrt_tau)
    vega = S * pdf_d1 * sqrt_tau
    theta = -(S * pdf_d1 * sigma_safe) / (2 * sqrt_tau) - r * K * np.exp(-r * tau_safe) * norm.cdf(d2)

    at_expiry = tau <= _MIN_TAU
    gamma = np.where(at_expiry, 0.0, gamma)
    vega = np.where(at_expiry, 0.0, vega)
    theta = np.where(at_expiry, 0.0, theta)

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)
