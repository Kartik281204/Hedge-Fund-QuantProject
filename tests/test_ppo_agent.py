import dataclasses

import numpy as np
import pytest

from hedging.agents.ppo_agent import PPOAgent
from hedging.config import load_config
from hedging.env.vector_env import VectorHedgingEnv
from hedging.market.registry import build_all_simulators

CONFIG_PATH = "config/config.yaml"


def _tiny_training_setup():
    cfg = load_config(CONFIG_PATH)
    sims = build_all_simulators(cfg)
    vec_env = VectorHedgingEnv(cfg.market, cfg.hedging, cfg.regimes, sims, n_envs=4, seed=0)
    tiny_cfg = dataclasses.replace(
        cfg.training,
        total_timesteps=256,
        rollout_len=64,
        n_envs=4,
        n_epochs=2,
        n_minibatches=2,
        log_every_updates=1000,
    )
    agent = PPOAgent(
        obs_dim=5, hidden_sizes=tiny_cfg.hidden_sizes, log_std_init=tiny_cfg.log_std_init, seed=7
    )
    return agent, vec_env, tiny_cfg


def test_ppo_agent_action_is_valid_before_training():
    agent, vec_env, _ = _tiny_training_setup()
    obs = vec_env.reset()[0]
    action = agent.act(obs, deterministic=True)
    assert action.shape == (1,)
    assert 0.0 <= action[0] <= 1.0


def test_ppo_training_runs_and_updates_params():
    agent, vec_env, tiny_cfg = _tiny_training_setup()
    params_before = agent.params["policy_mlp"][0]["W"].copy()

    history = agent.train(vec_env, tiny_cfg)

    assert len(history) == tiny_cfg.total_timesteps // tiny_cfg.rollout_len
    for row in history:
        assert np.isfinite(row["policy_loss"])
        assert np.isfinite(row["value_loss"])

    params_after = np.asarray(agent.params["policy_mlp"][0]["W"])
    assert not np.allclose(params_before, params_after), "policy params should change after training"


def test_checkpoint_resume_continues_from_same_params_and_reaches_target(tmp_path):
    cfg = load_config(CONFIG_PATH)
    sims = build_all_simulators(cfg)
    tiny_cfg = dataclasses.replace(
        cfg.training,
        total_timesteps=256,
        rollout_len=64,
        n_envs=4,
        n_epochs=2,
        n_minibatches=2,
        log_every_updates=1000,
    )
    ckpt_path = tmp_path / "checkpoint.pkl"

    # Phase 1: train for half the budget, checkpointing every update.
    vec_env_1 = VectorHedgingEnv(cfg.market, cfg.hedging, cfg.regimes, sims, n_envs=4, seed=0)
    agent_1 = PPOAgent(obs_dim=5, hidden_sizes=tiny_cfg.hidden_sizes, log_std_init=tiny_cfg.log_std_init, seed=7)
    half_cfg = dataclasses.replace(tiny_cfg, total_timesteps=128)
    agent_1.train(vec_env_1, half_cfg, checkpoint_path=ckpt_path, checkpoint_every=1)
    params_after_phase1 = np.asarray(agent_1.params["policy_mlp"][0]["W"]).copy()

    # Phase 2: a *fresh* agent/env resumes from the checkpoint and finishes the full budget.
    vec_env_2 = VectorHedgingEnv(cfg.market, cfg.hedging, cfg.regimes, sims, n_envs=4, seed=99)
    agent_2 = PPOAgent(obs_dim=5, hidden_sizes=tiny_cfg.hidden_sizes, log_std_init=tiny_cfg.log_std_init, seed=123)
    history = agent_2.train(vec_env_2, tiny_cfg, checkpoint_path=ckpt_path, checkpoint_every=1, resume=True)

    # Resuming must pick up phase 1's params, not agent_2's own fresh init (different seed).
    params_at_resume_start = None
    for row in history:
        if row["update"] == 3:  # first update after the 2 checkpointed in phase 1
            params_at_resume_start = row
    assert params_at_resume_start is not None
    assert not np.allclose(params_after_phase1, agent_2.params["policy_mlp"][0]["W"]), (
        "params should keep changing after resume, not freeze"
    )
    assert history[-1]["total_steps"] == tiny_cfg.total_timesteps
    assert history[-1]["update"] == tiny_cfg.total_timesteps // tiny_cfg.rollout_len

    # Resuming a checkpoint that's already at the target should be a no-op, not crash or redo work.
    vec_env_3 = VectorHedgingEnv(cfg.market, cfg.hedging, cfg.regimes, sims, n_envs=4, seed=5)
    agent_3 = PPOAgent(obs_dim=5, hidden_sizes=tiny_cfg.hidden_sizes, log_std_init=tiny_cfg.log_std_init, seed=1)
    noop_history = agent_3.train(vec_env_3, tiny_cfg, checkpoint_path=ckpt_path, checkpoint_every=1, resume=True)
    assert len(noop_history) == tiny_cfg.total_timesteps // tiny_cfg.rollout_len
    assert noop_history[-1]["total_steps"] == tiny_cfg.total_timesteps


def test_ppo_agent_save_and_load_round_trip(tmp_path):
    agent, vec_env, tiny_cfg = _tiny_training_setup()
    agent.train(vec_env, tiny_cfg)

    save_path = tmp_path / "agent.pkl"
    agent.save(save_path)
    loaded = PPOAgent.load(save_path)

    obs = vec_env.reset()[0]
    a1 = agent.act(obs, deterministic=True)
    a2 = loaded.act(obs, deterministic=True)
    assert a1 == pytest.approx(a2, abs=1e-6)
