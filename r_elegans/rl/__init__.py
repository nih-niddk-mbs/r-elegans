"""Body-direct reinforcement learning for the seven-parameter sensory controller.

This package trains the same controller shape optimized by
:func:`r_elegans.envs.petri_dish.petri_navigation_loss` (via direct gradient
descent through the differentiable body/gait simulator), but through
model-free reinforcement learning against the Gymnax-compatible environment in
:mod:`r_elegans.envs.gymnax_petri_dish`. :func:`r_elegans.rl.train` defaults to
PPO (:func:`r_elegans.rl.make_ppo_train_step`) and also offers a simpler A2C
update rule (:func:`r_elegans.rl.make_a2c_train_step`); see
:mod:`r_elegans.rl.training` for how they differ. It requires the optional
``env`` extra (``gymnax``, ``optax``); it is not imported by
:mod:`r_elegans.envs` or :mod:`r_elegans.demo`, which must keep working
without those dependencies.
"""

from .critic import CriticParams, critic_value, init_critic_params
from .policy import (
    ACTION_HIGH,
    ACTION_LOW,
    ActorParams,
    action_distribution,
    action_log_prob,
    action_mean,
    deterministic_action,
    gaussian_log_prob,
    init_actor_params,
    sample_action,
)
from .training import (
    AgentParams,
    TrainingConfig,
    Transition,
    compute_gae,
    make_a2c_train_step,
    make_ppo_train_step,
    train,
)

__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "ActorParams",
    "AgentParams",
    "CriticParams",
    "Transition",
    "TrainingConfig",
    "action_distribution",
    "action_log_prob",
    "action_mean",
    "compute_gae",
    "critic_value",
    "deterministic_action",
    "gaussian_log_prob",
    "init_actor_params",
    "init_critic_params",
    "make_a2c_train_step",
    "make_ppo_train_step",
    "sample_action",
    "train",
]
