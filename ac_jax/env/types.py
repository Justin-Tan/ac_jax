import jaxtyping
import jax.numpy as jnp

from chex import dataclass, PRNGKey
from jaxtyping import Array, Float, Int
from typing import Any, Dict, NamedTuple, Optional, Union

import optax

@dataclass  # so tree_utils work correctly
class State:
    """
    presentation: current group presentation. Shape (n_gen * max_relator_length,)
    step_count: number of steps taken so far.
    action_history: previous actions taken in episode.
    score: current score of the game state.
    key: PRNG key for randomness at each step + for auto-reset.
    curriculum_idx: index into initial pool of presentations.
    initial_presentation: starting presentation of episode.
    tier_id: curriculum tier identifier (easy, medium, hard).
    """
    presentation: jaxtyping.Array  # (n_gen * max_relator_length,)
    step_count: int  # ()
    action_history: jaxtyping.Array  # (horizon_length,)
    score: jaxtyping.Float  # ()
    curriculum_idx: int = 0
    key: PRNGKey = None
    initial_presentation: jaxtyping.Array = None  # (n_gen * max_relator_length,)
    tier_id: Optional[jaxtyping.Array] = -1

@dataclass
class Observation:
    presentation: jaxtyping.Array  # (n_gen * max_relator_length,)

class Transition(NamedTuple):
    """Structure for rollout data."""
    observation: jaxtyping.Array
    action: jaxtyping.Array
    value: jaxtyping.Array
    reward: jaxtyping.Array
    discount: jaxtyping.Array
    next_observation: jaxtyping.Array
    log_prob: jaxtyping.Array
    logits: jaxtyping.Array
    done: bool = None
    extras: Optional[Dict] = None

class ActorCriticParams(NamedTuple):
    actor: Dict
    critic: Dict

class ParamsState(NamedTuple):
    """Container for variables used during agent training."""
    params: Dict
    opt_state: optax.OptState
    update_count: float
    
class ActingState(NamedTuple):
    """Container for data used during rollouts."""
    state: Any
    timestep: Any
    key: PRNGKey
    episode_count: float
    env_step_count: float

class AgentState(NamedTuple):
    """Container for data used during agent training on rollout data."""
    params_state: Optional[ParamsState]
    acting_state: ActingState


@dataclass
class CurriculumBatch:
    presentations: Float[Array, "N n_gens * max_relator_length"]
    lengths: Int[Array, "N n_gens"]
    move_histories: Int[Array, "N max_steps"]
    depths: Int[Array, "N"]
    n: Optional[Int[Array, "N"]] = None
