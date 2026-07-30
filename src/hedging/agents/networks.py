"""Actor-critic networks implemented as pure JAX functions (no framework
dependency beyond jax/jaxlib itself -- no torch, no flax). Parameters are a
plain nested dict/list pytree so `jax.grad` differentiates through them
directly; this keeps the PPO update in `ppo_agent.py` fully transparent
instead of hidden behind a neural-network framework's abstractions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import jax
import jax.numpy as jnp

Params = Dict[str, Any]

_LOG_2PI = math.log(2 * math.pi)


def _init_mlp(key: jax.Array, sizes: Sequence[int], out_scale: float) -> List[Dict[str, jnp.ndarray]]:
    layers = []
    n_layers = len(sizes) - 1
    keys = jax.random.split(key, n_layers)
    for i, k in enumerate(keys):
        in_dim, out_dim = sizes[i], sizes[i + 1]
        is_last = i == n_layers - 1
        scale = out_scale * math.sqrt(1.0 / in_dim) if is_last else math.sqrt(2.0 / in_dim)
        W = jax.random.normal(k, (in_dim, out_dim)) * scale
        b = jnp.zeros((out_dim,))
        layers.append({"W": W, "b": b})
    return layers


def _mlp_forward(layers: List[Dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    n = len(layers)
    for i, layer in enumerate(layers):
        x = x @ layer["W"] + layer["b"]
        if i < n - 1:
            x = jnp.tanh(x)
    return x


def init_params(
    key: jax.Array, obs_dim: int, hidden_sizes: Sequence[int], log_std_init: float
) -> Params:
    k_policy, k_value = jax.random.split(key)
    policy_sizes = [obs_dim, *hidden_sizes, 1]
    value_sizes = [obs_dim, *hidden_sizes, 1]
    return {
        "policy_mlp": _init_mlp(k_policy, policy_sizes, out_scale=0.6),
        "value_mlp": _init_mlp(k_value, value_sizes, out_scale=1.0),
        "log_std": jnp.array(log_std_init, dtype=jnp.float32),
    }


def policy_mean(params: Params, obs: jnp.ndarray) -> jnp.ndarray:
    """Return the action mean in (0, 1), shape (batch,)."""
    raw = _mlp_forward(params["policy_mlp"], obs)
    return jax.nn.sigmoid(raw).squeeze(-1)


def value_fn(params: Params, obs: jnp.ndarray) -> jnp.ndarray:
    """Return V(s), shape (batch,)."""
    raw = _mlp_forward(params["value_mlp"], obs)
    return raw.squeeze(-1)


def gaussian_log_prob(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """log N(x; mean, std^2), elementwise (all args broadcastable)."""
    return -0.5 * (((x - mean) / std) ** 2) - jnp.log(std) - 0.5 * _LOG_2PI


def gaussian_entropy(std: jnp.ndarray) -> jnp.ndarray:
    """Differential entropy of N(0, std^2), elementwise."""
    return 0.5 + 0.5 * _LOG_2PI + jnp.log(std)


def sample_action(
    key: jax.Array, params: Params, obs: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample raw (unclipped) actions for a batch of observations.

    Returns (raw_action, mean, std). Clipping to the env's [0, 1] action
    space happens at the call site, not here, since the log-prob used for
    the PPO objective is evaluated on the *unclipped* Gaussian sample.
    """
    mean = policy_mean(params, obs)
    std = jnp.exp(params["log_std"])
    eps = jax.random.normal(key, shape=mean.shape)
    raw_action = mean + std * eps
    return raw_action, mean, std
