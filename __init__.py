"""
hedging: A research-grade reinforcement-learning derivative hedging package.

Trains an RL agent (PPO) to dynamically hedge a short European call option,
compares it against Black-Scholes delta-hedge baselines, and evaluates both
across multiple volatility regimes (GBM low/high vol, Heston stochastic vol).
"""

__version__ = "0.1.0"
