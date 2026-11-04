"""Direct 95-muscle actions; no fitted gait or oscillator in the control path."""

import jax.numpy as jnp
from gymnax.environments import spaces

from .gymnax_petri_dish import PetriDishGymnaxEnv


class FullNetworkPetriDishEnv(PetriDishGymnaxEnv):
    """Reuse normalized dish mechanics, sensing and rewards, without gait control.

    Observations are adapted log concentration, its derivative and relative
    concentration. These are engineered chemical sensor proxies, not a fitted
    receptor model. The inherited phase state is unused and stays constant.
    """

    def __init__(self, body_params, *, num_segments=12):
        # The base constructor stores gait_params; this environment never uses it.
        super().__init__(body_params, None, num_segments=num_segments)
        from r_elegans.body import build_muscle_projection
        self.muscle_projection = build_muscle_projection(num_segments - 1, head_at_end=True)
        self.obs_shape = (3,)

    def _state_at(self, source_position, heading, params):
        # Base initialization only needs a phase offset. Avoid constructing a gait.
        from r_elegans.body import initialize_muscle_body
        from .gymnax_petri_dish import PetriGymnaxState
        from .petri_dish import food_concentration, head_position
        body = initialize_muscle_body(self.num_segments, heading=heading)
        dish = self._dish_params(params, source_position)
        concentration, _ = food_concentration(head_position(body, self.body_params), body.time, dish)
        log_c = jnp.log(concentration + dish.concentration_floor)
        return PetriGymnaxState(body=body, phase=jnp.asarray(0.0),
                               adapted_log_concentration=log_c, previous_log_concentration=log_c,
                               source_position=jnp.asarray(source_position), time=0)

    def get_obs(self, state, params=None, key=None):
        params = self.default_params if params is None else params
        _, relative, log_c, _ = self._sense(state, self._dish_params(params, state.source_position))
        return jnp.array([log_c - state.adapted_log_concentration,
                          (log_c - state.previous_log_concentration) / params.dt,
                          relative])

    def motor_command(self, state, action, params):
        muscles = jnp.clip(action, 0.0, 1.0)
        return muscles, state.phase, jnp.mean(muscles ** 2)

    @property
    def name(self):
        return "PetriDishFullNetwork-v0"

    @property
    def num_actions(self):
        return 95

    def action_space(self, params=None):
        return spaces.Box(0.0, 1.0, shape=(95,), dtype=jnp.float32)

    def observation_space(self, params):
        return spaces.Box(-jnp.inf, jnp.inf, shape=(3,), dtype=jnp.float32)
