"""A from-scratch PPO implementation (JAX + optax) for the hedging env.

Deliberately not `stable-baselines3`: this sandbox has one CPU core and no
GPU, and SB3 pulls in a full torch install (which, on PyPI without the
CPU-only index, drags in CUDA runtime packages several hundred MB to GB in
size) for what is, here, a 5-dimensional observation and a 1-dimensional
action -- a tiny problem that doesn't need a heavy dependency. Writing PPO
directly also keeps the clipped objective, GAE, and loss terms fully
visible and auditable instead of hidden behind a library.

The algorithm follows Schulman et al. (2017) "Proximal Policy Optimization
Algorithms" with GAE (Schulman et al. 2016) for advantages -- the standard
practical recipe (advantage normalization, global-norm gradient clipping,
minibatch/epoch updates) used in most reference PPO implementations.

Rollouts are collected from `n_envs` parallel `HedgingEnv` copies
(`VectorHedgingEnv`) so each JAX call handles a batch of `n_envs`
observations instead of one -- on a single CPU core this doesn't add true
parallel compute, but it amortizes fixed per-call dispatch overhead across
more environment steps, which dominates wall-clock time for a network this
small (see README perf notes).
"""

from __future__ import annotations

import functools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax

from hedging.agents.networks import gaussian_entropy, gaussian_log_prob, init_params, policy_mean, value_fn
from hedging.config import TrainingConfig
from hedging.env.vector_env import VectorHedgingEnv
from hedging.exceptions import ModelNotTrainedError
from hedging.logging_setup import get_logger

logger = get_logger(__name__)

OBS_DIM = 5


@dataclass
class RolloutBuffer:
    """Buffer shape (T, N, ...): T sequential steps x N parallel envs."""

    T: int
    N: int
    obs_dim: int

    def __post_init__(self) -> None:
        self.obs = np.zeros((self.T, self.N, self.obs_dim), dtype=np.float32)
        self.raw_actions = np.zeros((self.T, self.N), dtype=np.float32)
        self.log_probs = np.zeros((self.T, self.N), dtype=np.float32)
        self.rewards = np.zeros((self.T, self.N), dtype=np.float32)
        self.dones = np.zeros((self.T, self.N), dtype=np.float32)
        self.values = np.zeros((self.T, self.N), dtype=np.float32)

    def add(self, t, obs, raw_action, log_prob, reward, done, value) -> None:
        self.obs[t] = obs
        self.raw_actions[t] = raw_action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value

    def flatten(self, arr: np.ndarray) -> np.ndarray:
        """(T, N, ...) -> (T*N, ...)"""
        return arr.reshape(self.T * self.N, *arr.shape[2:])


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized GAE across the N env dimension. rewards/values/dones have
    shape (T, N); last_values has shape (N,) (bootstrap for the state right
    after the last collected transition of each env)."""
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_value = last_values if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def act_value_logprob(key, params, obs_batch):
    """Fused forward pass: sample actions, their log-probs, and V(s) for a
    whole batch of observations in one jitted call."""
    mean = policy_mean(params, obs_batch)
    std = jnp.exp(params["log_std"])
    eps = jax.random.normal(key, shape=mean.shape)
    raw_action = mean + std * eps
    log_prob = gaussian_log_prob(raw_action, mean, std)
    value = value_fn(params, obs_batch)
    return raw_action, log_prob, value


_jit_act_value_logprob = jax.jit(act_value_logprob)
_jit_value_fn = jax.jit(value_fn)


def ppo_loss(params, obs, raw_actions, old_log_probs, advantages, returns, clip_range, vf_coef, entropy_coef):
    mean = policy_mean(params, obs)
    std = jnp.exp(params["log_std"])
    new_log_probs = gaussian_log_prob(raw_actions, mean, std)

    ratio = jnp.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    values = value_fn(params, obs)
    value_loss = jnp.mean((returns - values) ** 2)

    entropy = jnp.mean(gaussian_entropy(std))
    loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

    approx_kl = jnp.mean(old_log_probs - new_log_probs)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32))
    aux = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }
    return loss, aux


class PPOAgent:
    def __init__(self, obs_dim: int, hidden_sizes: List[int], log_std_init: float, seed: int) -> None:
        self.obs_dim = obs_dim
        self.hidden_sizes = list(hidden_sizes)
        self.log_std_init = log_std_init
        self.seed = seed

        self._key = jax.random.PRNGKey(seed)
        self._key, init_key = jax.random.split(self._key)
        self.params = init_params(init_key, obs_dim, hidden_sizes, log_std_init)
        self._np_rng = np.random.default_rng(seed)
        self._trained = False

    # ------------------------------------------------------------------ #
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Same `.act(obs) -> action` interface as the BS baselines, so the
        backtest harness can treat every method interchangeably."""
        obs_batch = jnp.asarray(obs, dtype=jnp.float32)[None, :]
        mean = policy_mean(self.params, obs_batch)
        if deterministic:
            action = mean
        else:
            self._key, sk = jax.random.split(self._key)
            std = jnp.exp(self.params["log_std"])
            eps = jax.random.normal(sk, shape=mean.shape)
            action = mean + std * eps
        action = jnp.clip(action, 0.0, 1.0)
        return np.asarray(action, dtype=np.float32)

    def reset(self) -> None:
        """No-op; kept for interface parity with the baseline agents."""
        return None

    # ------------------------------------------------------------------ #
    def train(
        self,
        vec_env: VectorHedgingEnv,
        cfg: TrainingConfig,
        log_fn=None,
        checkpoint_path: "str | Path | None" = None,
        checkpoint_every: int = 20,
        resume: bool = False,
    ) -> List[Dict[str, float]]:
        """Run PPO updates until `cfg.total_timesteps` is reached.

        If `checkpoint_path` is given, a checkpoint (params + optimizer
        state + RNG state + history-so-far) is written every
        `checkpoint_every` updates. If `resume=True` and a checkpoint
        already exists there, training continues from it instead of
        starting over -- this sandbox's single foreground command has a
        wall-clock ceiling well under what a full run needs, so long runs
        are split across several `python scripts/train.py --resume` calls.
        """
        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(cfg.learning_rate),
        )

        start_update = 0
        history: List[Dict[str, float]] = []
        recent_pnls: List[float] = []
        total_steps = 0

        ckpt = None
        if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
            ckpt = self._load_checkpoint(checkpoint_path)
            self.params = ckpt["params"]
            self._key = ckpt["key"]
            self._np_rng.bit_generator.state = ckpt["np_rng_state"]
            opt_state = ckpt["opt_state"]
            start_update = ckpt["update"]
            total_steps = ckpt["total_steps"]
            history = ckpt["history"]
            recent_pnls = ckpt["recent_pnls"]
            logger.info(f"resumed from checkpoint at update {start_update} ({total_steps} steps)")
        else:
            opt_state = optimizer.init(self.params)

        loss_fn = functools.partial(
            ppo_loss, clip_range=cfg.clip_range, vf_coef=cfg.vf_coef, entropy_coef=cfg.entropy_coef
        )
        loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

        N = vec_env.n_envs
        T = cfg.rollout_len // N
        minibatch_size = cfg.rollout_len // cfg.n_minibatches
        num_updates = max(cfg.total_timesteps // cfg.rollout_len, 1)

        if start_update >= num_updates:
            logger.info(f"checkpoint already at/past target ({start_update} >= {num_updates} updates); nothing to do")
            self._trained = True
            return history

        obs = vec_env.reset()

        for update in range(start_update + 1, num_updates + 1):
            buf = RolloutBuffer(T, N, self.obs_dim)
            episode_pnls: List[float] = []

            for t in range(T):
                self._key, sk = jax.random.split(self._key)
                obs_j = jnp.asarray(obs, dtype=jnp.float32)
                raw_action, log_prob, value = _jit_act_value_logprob(sk, self.params, obs_j)

                raw_action_np = np.asarray(raw_action)
                env_actions = np.clip(raw_action_np, 0.0, 1.0)
                next_obs, rewards, dones, infos = vec_env.step(env_actions)

                buf.add(t, obs, raw_action_np, np.asarray(log_prob), rewards, dones, np.asarray(value))

                for info, done in zip(infos, dones):
                    if done and "total_pnl" in info:
                        episode_pnls.append(info["total_pnl"])

                obs = next_obs

            last_values = np.asarray(_jit_value_fn(self.params, jnp.asarray(obs, dtype=jnp.float32)))
            advantages, returns = compute_gae(
                buf.rewards, buf.values, buf.dones, last_values, cfg.gamma, cfg.gae_lambda
            )

            flat_obs = buf.flatten(buf.obs)
            flat_actions = buf.flatten(buf.raw_actions)
            flat_log_probs = buf.flatten(buf.log_probs)
            flat_advantages = buf.flatten(advantages)
            flat_returns = buf.flatten(returns)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

            idx = np.arange(cfg.rollout_len)
            epoch_stats: List[Dict[str, float]] = []
            for _ in range(cfg.n_epochs):
                self._np_rng.shuffle(idx)
                for mb in range(cfg.n_minibatches):
                    mb_idx = idx[mb * minibatch_size : (mb + 1) * minibatch_size]
                    (loss_val, aux), grads = loss_grad_fn(
                        self.params,
                        jnp.asarray(flat_obs[mb_idx]),
                        jnp.asarray(flat_actions[mb_idx]),
                        jnp.asarray(flat_log_probs[mb_idx]),
                        jnp.asarray(flat_advantages[mb_idx]),
                        jnp.asarray(flat_returns[mb_idx]),
                    )
                    updates, opt_state = optimizer.update(grads, opt_state, self.params)
                    self.params = optax.apply_updates(self.params, updates)
                    epoch_stats.append({k: float(v) for k, v in aux.items()})

            total_steps += cfg.rollout_len
            recent_pnls.extend(episode_pnls)
            recent_pnls = recent_pnls[-400:]
            mean_stats = {k: float(np.mean([s[k] for s in epoch_stats])) for k in epoch_stats[0]}
            row = {
                "update": update,
                "total_steps": total_steps,
                "episodes_in_rollout": len(episode_pnls),
                "mean_recent_total_pnl": float(np.mean(recent_pnls)) if recent_pnls else float("nan"),
                "std_recent_total_pnl": float(np.std(recent_pnls)) if recent_pnls else float("nan"),
                **mean_stats,
            }
            history.append(row)

            if update % cfg.log_every_updates == 0 or update == 1 or update == num_updates:
                msg = (
                    f"update {update}/{num_updates} | steps {total_steps} | "
                    f"mean_pnl(last {len(recent_pnls)}) {row['mean_recent_total_pnl']:.4f} | "
                    f"std_pnl {row['std_recent_total_pnl']:.4f} | "
                    f"policy_loss {row['policy_loss']:.4f} | value_loss {row['value_loss']:.4f} | "
                    f"entropy {row['entropy']:.4f} | approx_kl {row['approx_kl']:.5f} | "
                    f"clip_frac {row['clip_frac']:.3f}"
                )
                logger.info(msg)
                if log_fn is not None:
                    log_fn(row)

            if checkpoint_path is not None and (update % checkpoint_every == 0 or update == num_updates):
                self._save_checkpoint(
                    checkpoint_path, opt_state, update, total_steps, history, recent_pnls
                )

        self._trained = True
        return history

    # ------------------------------------------------------------------ #
    # Checkpointing (training-only: params + optimizer + RNG state, so a
    # run can be resumed exactly where it left off across process restarts).
    # Distinct from save()/load() below, which persist only what's needed
    # for *inference* (a small, portable artifact for evaluate.py).
    def _save_checkpoint(self, path, opt_state, update, total_steps, history, recent_pnls) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "params": self.params,
            "opt_state": opt_state,
            "key": self._key,
            "np_rng_state": self._np_rng.bit_generator.state,
            "update": update,
            "total_steps": total_steps,
            "history": history,
            "recent_pnls": recent_pnls,
            "obs_dim": self.obs_dim,
            "hidden_sizes": self.hidden_sizes,
            "log_std_init": self.log_std_init,
            "seed": self.seed,
        }
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f)
        tmp_path.replace(path)  # atomic on the same filesystem: never leaves a half-written checkpoint
        logger.info(f"checkpoint saved at update {update} ({total_steps} steps) -> {path}")

    @staticmethod
    def _load_checkpoint(path) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), self.params)
        payload = {
            "params": params_np,
            "obs_dim": self.obs_dim,
            "hidden_sizes": self.hidden_sizes,
            "log_std_init": self.log_std_init,
            "seed": self.seed,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"saved PPO agent to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PPOAgent":
        path = Path(path)
        if not path.exists():
            raise ModelNotTrainedError(f"no saved model at {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        agent = cls(
            obs_dim=payload["obs_dim"],
            hidden_sizes=payload["hidden_sizes"],
            log_std_init=payload["log_std_init"],
            seed=payload["seed"],
        )
        agent.params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), payload["params"])
        agent._trained = True
        return agent
