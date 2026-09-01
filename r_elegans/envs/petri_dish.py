"""A circular Petri dish with a diffusing food pulse and head sensing."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from r_elegans.body import (
    CommandedControllerParams,
    MuscleBodyParams,
    MuscleBodyState,
    NeuromuscularParams,
    as_body_state,
    build_muscle_projection,
    commanded_muscle_activations,
    controller_for_command,
    initialize_muscle_body,
    muscle_activations_from_voltage,
    muscle_body_step,
    project_muscles_to_joints,
    world_segment_centers,
)
from r_elegans.brain import motor_features_from_phase, neural_motor_voltage

Array = jax.Array


class PetriDishParams(NamedTuple):
    """Geometry, food diffusion, and sensory adaptation parameters."""

    radius: Array
    source_position: Array
    source_mass: Array
    diffusion_coefficient: Array
    initial_variance: Array
    concentration_floor: Array
    adaptation_time_constant: Array
    body_margin: Array


class SensoryPolicyParams(NamedTuple):
    """Compact klinotaxis policy driven only by head chemosensation."""

    base_speed: Array
    response_sine_gain: Array
    response_cosine_gain: Array
    derivative_sine_gain: Array
    derivative_cosine_gain: Array
    steering_bias: Array
    food_slowing: Array


class PetriWormState(NamedTuple):
    """Body, gait phase, and sensory memory inside the dish."""

    body: MuscleBodyState
    phase: Array
    adapted_log_concentration: Array
    previous_log_concentration: Array


class PetriObservation(NamedTuple):
    """Quantities recorded from a closed-loop environment step."""

    concentration: Array
    relative_concentration: Array
    sensory_response: Array
    command: Array
    muscle_activation: Array
    distance_to_source: Array
    head_position: Array


def default_petri_dish_params(source_position: Array) -> PetriDishParams:
    """Return a normalized dish large enough for a one-body-length worm."""

    return PetriDishParams(
        radius=jnp.asarray(1.5),
        source_position=jnp.asarray(source_position),
        source_mass=jnp.asarray(1.0),
        diffusion_coefficient=jnp.asarray(0.005),
        initial_variance=jnp.asarray(0.16),
        concentration_floor=jnp.asarray(1e-12),
        adaptation_time_constant=jnp.asarray(0.75),
        body_margin=jnp.asarray(0.55),
    )


def decode_sensory_policy(raw: Array) -> SensoryPolicyParams:
    """Map seven unconstrained parameters to a bounded sensory controller."""

    raw = jnp.asarray(raw)
    if raw.shape != (7,):
        raise ValueError("The Petri-dish sensory policy requires seven parameters")
    return SensoryPolicyParams(
        base_speed=0.5 + 0.5 * jax.nn.sigmoid(raw[0]),
        response_sine_gain=20.0 * jnp.tanh(raw[1]),
        response_cosine_gain=20.0 * jnp.tanh(raw[2]),
        derivative_sine_gain=20.0 * jnp.tanh(raw[3]),
        derivative_cosine_gain=20.0 * jnp.tanh(raw[4]),
        steering_bias=jnp.tanh(raw[5]),
        food_slowing=jax.nn.sigmoid(raw[6]),
    )


def food_concentration(
    position: Array,
    time: Array,
    params: PetriDishParams,
) -> tuple[Array, Array]:
    """Evaluate a finite two-dimensional Gaussian diffusion pulse."""

    variance = params.initial_variance + 2.0 * params.diffusion_coefficient * time
    squared_distance = jnp.sum((position - params.source_position) ** 2)
    relative = jnp.exp(-squared_distance / (2.0 * variance))
    concentration = params.source_mass * relative / (2.0 * jnp.pi * variance)
    return concentration, relative


def head_position(state: MuscleBodyState, params: MuscleBodyParams) -> Array:
    """Return the world position of the anterior-most modeled segment."""

    centers = world_segment_centers(as_body_state(state), params.mechanics)
    return centers[-1]


def initialize_petri_worm(
    dish_params: PetriDishParams,
    body_params: MuscleBodyParams,
    gait_params: CommandedControllerParams,
    *,
    heading: Array = jnp.asarray(0.0),
) -> PetriWormState:
    """Create a straight worm with sensory adaptation at local equilibrium."""

    body = initialize_muscle_body(12, heading=heading)
    head = head_position(body, body_params)
    concentration, _ = food_concentration(head, body.time, dish_params)
    log_concentration = jnp.log(concentration + dish_params.concentration_floor)
    return PetriWormState(
        body=body,
        phase=gait_params.forward.phase_offset,
        adapted_log_concentration=log_concentration,
        previous_log_concentration=log_concentration,
    )


def _sensory_command(
    state: PetriWormState,
    dish_params: PetriDishParams,
    body_params: MuscleBodyParams,
    policy: SensoryPolicyParams,
    dt: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    head = head_position(state.body, body_params)
    concentration, relative = food_concentration(
        head, state.body.time, dish_params
    )
    log_concentration = jnp.log(concentration + dish_params.concentration_floor)
    response = log_concentration - state.adapted_log_concentration
    derivative = (log_concentration - state.previous_log_concentration) / dt
    sine, cosine = jnp.sin(state.phase), jnp.cos(state.phase)
    steering = jnp.tanh(
        policy.response_sine_gain * response * sine
        + policy.response_cosine_gain * response * cosine
        + 0.1 * policy.derivative_sine_gain * derivative * sine
        + 0.1 * policy.derivative_cosine_gain * derivative * cosine
        + policy.steering_bias
    )
    speed = policy.base_speed * (1.0 - policy.food_slowing * relative)
    return jnp.stack((speed, steering)), concentration, relative, response, head


def _finish_step(
    state: PetriWormState,
    body: MuscleBodyState,
    command: Array,
    muscle_activation: Array,
    concentration: Array,
    relative_concentration: Array,
    dish_params: PetriDishParams,
    body_params: MuscleBodyParams,
    gait_params: CommandedControllerParams,
    dt: Array,
) -> tuple[PetriWormState, PetriObservation]:
    maximum_center_radius = dish_params.radius - dish_params.body_margin
    radius = jnp.linalg.norm(body.position)
    bounded_position = body.position * jnp.minimum(
        1.0, maximum_center_radius / jnp.maximum(radius, 1e-8)
    )
    body = body._replace(position=bounded_position)
    gait = controller_for_command(command, gait_params)
    next_phase = state.phase - 2.0 * jnp.pi * gait.frequency * dt
    adaptation_fraction = -jnp.expm1(-dt / dish_params.adaptation_time_constant)
    log_concentration = jnp.log(concentration + dish_params.concentration_floor)
    adapted = state.adapted_log_concentration + adaptation_fraction * (
        log_concentration - state.adapted_log_concentration
    )
    next_state = PetriWormState(
        body=body,
        phase=next_phase,
        adapted_log_concentration=adapted,
        previous_log_concentration=log_concentration,
    )
    next_head = head_position(body, body_params)
    observation = PetriObservation(
        concentration=concentration,
        relative_concentration=relative_concentration,
        sensory_response=log_concentration - state.adapted_log_concentration,
        command=command,
        muscle_activation=muscle_activation,
        distance_to_source=jnp.linalg.norm(
            next_head - dish_params.source_position
        ),
        head_position=next_head,
    )
    return next_state, observation


def _direct_petri_step(
    state: PetriWormState,
    dish_params: PetriDishParams,
    body_params: MuscleBodyParams,
    gait_params: CommandedControllerParams,
    policy: SensoryPolicyParams,
    dt: Array,
) -> tuple[PetriWormState, PetriObservation]:
    command, concentration, relative, _, _ = _sensory_command(
        state, dish_params, body_params, policy, dt
    )
    muscles, _ = commanded_muscle_activations(
        state.phase, command, gait_params
    )
    projection = build_muscle_projection(state.body.joint_angles.shape[0])
    dorsal, ventral = project_muscles_to_joints(muscles, projection)
    body = muscle_body_step(state.body, body_params, dorsal, ventral, dt)
    return _finish_step(
        state,
        body,
        command,
        muscles,
        concentration,
        relative,
        dish_params,
        body_params,
        gait_params,
        dt,
    )


def _neural_petri_step(
    state: PetriWormState,
    dish_params: PetriDishParams,
    body_params: MuscleBodyParams,
    gait_params: CommandedControllerParams,
    policy: SensoryPolicyParams,
    neural_coefficients: Array,
    neuromuscular_params: NeuromuscularParams,
    dt: Array,
) -> tuple[PetriWormState, tuple[PetriObservation, Array]]:
    command, concentration, relative, _, _ = _sensory_command(
        state, dish_params, body_params, policy, dt
    )
    features = motor_features_from_phase(command, state.phase)
    voltage = neural_motor_voltage(neural_coefficients, features)
    muscles = muscle_activations_from_voltage(voltage, neuromuscular_params)
    projection = build_muscle_projection(state.body.joint_angles.shape[0])
    dorsal, ventral = project_muscles_to_joints(muscles, projection)
    body = muscle_body_step(state.body, body_params, dorsal, ventral, dt)
    next_state, observation = _finish_step(
        state,
        body,
        command,
        muscles,
        concentration,
        relative,
        dish_params,
        body_params,
        gait_params,
        dt,
    )
    return next_state, (observation, voltage)


@partial(jax.jit, static_argnames=("steps",))
def simulate_petri_dish(
    raw_policy: Array,
    dish_params: PetriDishParams,
    body_params: MuscleBodyParams,
    gait_params: CommandedControllerParams,
    *,
    heading: Array = jnp.asarray(0.0),
    steps: int = 500,
    dt: float = 0.02,
) -> tuple[PetriWormState, PetriWormState, PetriObservation]:
    """Roll out head sensing, a compact policy, muscles, and body physics."""

    policy = decode_sensory_policy(raw_policy)
    initial = initialize_petri_worm(
        dish_params, body_params, gait_params, heading=heading
    )

    def advance(state: PetriWormState, _: Array):
        next_state, observation = _direct_petri_step(
            state, dish_params, body_params, gait_params, policy, dt
        )
        return next_state, (next_state, observation)

    final, (trajectory, observations) = jax.lax.scan(
        advance, initial, xs=jnp.arange(steps)
    )
    return final, trajectory, observations


@partial(jax.jit, static_argnames=("steps",))
def simulate_neural_petri_dish(
    raw_policy: Array,
    dish_params: PetriDishParams,
    body_params: MuscleBodyParams,
    gait_params: CommandedControllerParams,
    neural_coefficients: Array,
    neuromuscular_params: NeuromuscularParams,
    *,
    heading: Array = jnp.asarray(0.0),
    steps: int = 500,
    dt: float = 0.02,
) -> tuple[PetriWormState, PetriWormState, PetriObservation, Array]:
    """Validate the same sensory policy through 302 neural outputs and NMJs."""

    policy = decode_sensory_policy(raw_policy)
    initial = initialize_petri_worm(
        dish_params, body_params, gait_params, heading=heading
    )

    def advance(state: PetriWormState, _: Array):
        next_state, (observation, voltage) = _neural_petri_step(
            state,
            dish_params,
            body_params,
            gait_params,
            policy,
            neural_coefficients,
            neuromuscular_params,
            dt,
        )
        return next_state, (next_state, observation, voltage)

    final, (trajectory, observations, voltage) = jax.lax.scan(
        advance, initial, xs=jnp.arange(steps)
    )
    return final, trajectory, observations, voltage


def petri_navigation_loss(
    raw_policy: Array,
    source_positions: Array,
    headings: Array,
    body_params: MuscleBodyParams,
    gait_params: CommandedControllerParams,
    *,
    steps: int = 500,
    dt: float = 0.02,
) -> Array:
    """Train chemotaxis across source locations without revealing them to policy."""

    source_positions = jnp.asarray(source_positions)
    headings = jnp.asarray(headings)
    if source_positions.ndim != 2 or source_positions.shape[1] != 2:
        raise ValueError("Source positions must have shape [episodes, 2]")
    if headings.shape != (source_positions.shape[0],):
        raise ValueError("Headings must have one value per episode")

    def run(source_position: Array, heading: Array):
        dish = default_petri_dish_params(source_position)
        _, _, observations = simulate_petri_dish(
            raw_policy,
            dish,
            body_params,
            gait_params,
            heading=heading,
            steps=steps,
            dt=dt,
        )
        distances = observations.distance_to_source
        return (
            jnp.min(distances)
            + 0.25 * distances[-1]
            + 1e-3 * jnp.mean(observations.command[:, 1] ** 2)
        )

    return jnp.mean(jax.vmap(run)(source_positions, headings))
