"""A Gymnax-compatible Petri dish for body-direct reinforcement learning.

This wraps the same food field, body-direct muscle/physics pipeline, and
sensory features as :mod:`r_elegans.envs.petri_dish`, but exposes them through
the ``gymnax.environments.environment.Environment`` reset/step interface so an
external policy can be trained with a model-free reinforcement-learning
algorithm instead of by differentiating through the simulator.

Only the body-direct path is exposed here. The action is the same
``[speed, steering]`` command consumed by the fitted traveling-wave gait; the
observation is the same four sensed quantities (adapted log-concentration
response, its time derivative, and sine/cosine of gait phase) plus relative
concentration that the seven-parameter sensory controller consumes in
:mod:`r_elegans.envs.petri_dish`. The controller itself is not part of the
environment -- it lives in :mod:`r_elegans.rl.policy` so that the same
environment can, in principle, drive any policy architecture.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from r_elegans.body import (
    CommandedControllerParams,
    MuscleBodyParams,
    MuscleBodyState,
    build_muscle_projection,
    commanded_muscle_activations,
    initialize_muscle_body,
    muscle_body_step,
    project_muscles_to_joints,
)

from .petri_dish import PetriDishParams, food_concentration, head_position

Array = jax.Array

ACTION_LOW = jnp.asarray([0.0, -1.0])
ACTION_HIGH = jnp.asarray([1.0, 1.0])


@struct.dataclass
class PetriGymnaxState(environment.EnvState):
    """Body pose, gait phase, sensory memory, and episode bookkeeping."""

    body: MuscleBodyState
    phase: Array
    adapted_log_concentration: Array
    previous_log_concentration: Array
    source_position: Array
    time: int


@struct.dataclass
class PetriGymnaxParams(environment.EnvParams):
    """Dish geometry, diffusion, sampling ranges, and reward shaping."""

    dish_radius: float = 1.5
    source_mass: float = 1.0
    diffusion_coefficient: float = 0.005
    initial_variance: float = 0.16
    concentration_floor: float = 1e-12
    adaptation_time_constant: float = 0.75
    body_margin: float = 0.55
    source_radius_min: float = 0.65
    source_radius_max: float = 0.85
    target_radius: float = 0.12
    dt: float = 0.02
    max_steps_in_episode: int = 500
    progress_reward_scale: float = 10.0
    steering_penalty_scale: float = 1e-3
    success_bonus: float = 5.0


class PetriDishGymnaxEnv(environment.Environment[PetriGymnaxState, PetriGymnaxParams]):
    """Body-direct chemotaxis as a Gymnax environment.

    ``body_params`` and ``gait_params`` are fixed physical/gait fits (not
    trained by RL) supplied at construction time, exactly as they are passed
    explicitly to :func:`r_elegans.envs.petri_dish.simulate_petri_dish`. Only
    the mapping from observation to ``[speed, steering]`` is meant to be
    learned, externally to this environment.
    """

    def __init__(
        self,
        body_params: MuscleBodyParams,
        gait_params: CommandedControllerParams,
        *,
        num_segments: int = 12,
    ):
        super().__init__()
        if num_segments < 2:
            raise ValueError("An actuated body requires at least two segments")
        self.body_params = body_params
        self.gait_params = gait_params
        self.num_segments = num_segments
        self.obs_shape = (5,)

    @property
    def default_params(self) -> PetriGymnaxParams:
        return PetriGymnaxParams()

    def _dish_params(
        self, params: PetriGymnaxParams, source_position: Array
    ) -> PetriDishParams:
        return PetriDishParams(
            radius=jnp.asarray(params.dish_radius),
            source_position=jnp.asarray(source_position),
            source_mass=jnp.asarray(params.source_mass),
            diffusion_coefficient=jnp.asarray(params.diffusion_coefficient),
            initial_variance=jnp.asarray(params.initial_variance),
            concentration_floor=jnp.asarray(params.concentration_floor),
            adaptation_time_constant=jnp.asarray(params.adaptation_time_constant),
            body_margin=jnp.asarray(params.body_margin),
        )

    def _sense(
        self, state: PetriGymnaxState, dish: PetriDishParams
    ) -> tuple[Array, Array, Array, Array]:
        """Return concentration, relative concentration, log-concentration, head."""

        head = head_position(state.body, self.body_params)
        concentration, relative = food_concentration(head, state.body.time, dish)
        log_concentration = jnp.log(concentration + dish.concentration_floor)
        return concentration, relative, log_concentration, head

    def get_obs(
        self,
        state: PetriGymnaxState,
        params: PetriGymnaxParams | None = None,
        key: Array | None = None,
    ) -> Array:
        """Return ``[response, derivative, sin(phase), cos(phase), relative]``."""

        if params is None:
            params = self.default_params
        dish = self._dish_params(params, state.source_position)
        _, relative, log_concentration, _ = self._sense(state, dish)
        response = log_concentration - state.adapted_log_concentration
        derivative = (log_concentration - state.previous_log_concentration) / params.dt
        return jnp.array(
            [
                response,
                derivative,
                jnp.sin(state.phase),
                jnp.cos(state.phase),
                relative,
            ]
        )

    def _state_at(
        self, source_position: Array, heading: Array, params: PetriGymnaxParams
    ) -> PetriGymnaxState:
        body = initialize_muscle_body(self.num_segments, heading=heading)
        dish = self._dish_params(params, source_position)
        head = head_position(body, self.body_params)
        concentration, _ = food_concentration(head, body.time, dish)
        log_concentration = jnp.log(concentration + dish.concentration_floor)

        return PetriGymnaxState(
            body=body,
            phase=self.gait_params.forward.phase_offset,
            adapted_log_concentration=log_concentration,
            previous_log_concentration=log_concentration,
            source_position=jnp.asarray(source_position),
            time=0,
        )

    def reset_env(
        self, key: Array, params: PetriGymnaxParams
    ) -> tuple[Array, PetriGymnaxState]:
        angle_key, radius_key, heading_key = jax.random.split(key, 3)
        angle = jax.random.uniform(angle_key, (), minval=-jnp.pi, maxval=jnp.pi)
        radius = jax.random.uniform(
            radius_key, (), minval=params.source_radius_min, maxval=params.source_radius_max
        )
        source_position = radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        heading = jax.random.uniform(heading_key, (), minval=-jnp.pi, maxval=jnp.pi)

        state = self._state_at(source_position, heading, params)
        return self.get_obs(state, params), state

    def reset_at(
        self,
        source_position: Array,
        heading: Array,
        params: PetriGymnaxParams | None = None,
    ) -> tuple[Array, PetriGymnaxState]:
        """Deterministically (re)start an episode at a chosen source and heading.

        Unlike :meth:`reset_env` (used during training, where the source and
        heading are randomized per episode via a PRNG key), this is for
        reproducible evaluation and rendering against a fixed set of held-out
        trials -- the same role :func:`r_elegans.envs.petri_dish.initialize_petri_worm`
        plays for the differentiable simulator.
        """

        if params is None:
            params = self.default_params
        state = self._state_at(source_position, heading, params)
        return self.get_obs(state, params), state

    def step_env(
        self,
        key: Array,
        state: PetriGymnaxState,
        action: Array,
        params: PetriGymnaxParams,
    ) -> tuple[Array, PetriGymnaxState, Array, Array, dict[Any, Any]]:
        command = jnp.clip(jnp.asarray(action), ACTION_LOW, ACTION_HIGH)
        dish = self._dish_params(params, state.source_position)
        _, _, log_concentration, head = self._sense(state, dish)
        previous_distance = jnp.linalg.norm(head - state.source_position)

        muscles, gait = commanded_muscle_activations(
            state.phase, command, self.gait_params
        )
        projection = build_muscle_projection(self.num_segments - 1)
        dorsal, ventral = project_muscles_to_joints(muscles, projection)
        body = muscle_body_step(state.body, self.body_params, dorsal, ventral, params.dt)

        maximum_center_radius = params.dish_radius - params.body_margin
        radius_now = jnp.linalg.norm(body.position)
        bounded_position = body.position * jnp.minimum(
            1.0, maximum_center_radius / jnp.maximum(radius_now, 1e-8)
        )
        body = body._replace(position=bounded_position)

        next_phase = state.phase - 2.0 * jnp.pi * gait.frequency * params.dt
        adaptation_fraction = -jnp.expm1(-params.dt / params.adaptation_time_constant)
        adapted = state.adapted_log_concentration + adaptation_fraction * (
            log_concentration - state.adapted_log_concentration
        )

        next_state = PetriGymnaxState(
            body=body,
            phase=next_phase,
            adapted_log_concentration=adapted,
            previous_log_concentration=log_concentration,
            source_position=state.source_position,
            time=state.time + 1,
        )

        next_head = head_position(body, self.body_params)
        distance = jnp.linalg.norm(next_head - state.source_position)
        progress = previous_distance - distance
        success = distance < params.target_radius
        reward = (
            params.progress_reward_scale * progress
            - params.steering_penalty_scale * command[1] ** 2
            + params.success_bonus * success.astype(jnp.float32)
        )
        terminated = success

        obs = self.get_obs(next_state, params)
        info = {"distance_to_source": distance, "success": success}
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(next_state),
            reward,
            terminated,
            info,
        )

    @property
    def name(self) -> str:
        return "PetriDishBodyDirect-v0"

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: PetriGymnaxParams | None = None) -> spaces.Box:
        return spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: PetriGymnaxParams) -> spaces.Box:
        high = jnp.array([jnp.inf, jnp.inf, 1.0, 1.0, jnp.inf], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(5,), dtype=jnp.float32)


__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "PetriDishGymnaxEnv",
    "PetriGymnaxParams",
    "PetriGymnaxState",
]
