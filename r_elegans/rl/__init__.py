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

from .actor_interface import (
    ANALYTIC_ACTOR_INTERFACE,
    ActorInterface,
    deterministic_rollout,
    make_connectome_actor_interface,
)
from .connectome_actor import (
    RecurrentConnectomeActorParams,
    actor_from_arrays,
    actor_to_arrays,
    connectome_action_mean_and_next_voltage,
    init_connectome_actor_params,
    initial_voltage,
)
from .critic import CriticParams, critic_value, init_critic_params
from .pretrain import (
    collect_teacher_trajectories,
    fit as pretrain_fit,
    pretrain_loss,
    unroll_student,
)
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
    "ANALYTIC_ACTOR_INTERFACE",
    "ActorInterface",
    "ActorParams",
    "AgentParams",
    "CriticParams",
    "RecurrentConnectomeActorParams",
    "Transition",
    "TrainingConfig",
    "action_distribution",
    "action_log_prob",
    "action_mean",
    "actor_from_arrays",
    "actor_to_arrays",
    "collect_teacher_trajectories",
    "compute_gae",
    "connectome_action_mean_and_next_voltage",
    "critic_value",
    "deterministic_action",
    "deterministic_rollout",
    "gaussian_log_prob",
    "init_actor_params",
    "init_connectome_actor_params",
    "init_critic_params",
    "initial_voltage",
    "make_a2c_train_step",
    "make_connectome_actor_interface",
    "make_ppo_train_step",
    "pretrain_fit",
    "pretrain_loss",
    "sample_action",
    "train",
    "unroll_student",
]
