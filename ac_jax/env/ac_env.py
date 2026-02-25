import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

from functools import partial
from dataclasses import field
from typing import Union, NamedTuple, Tuple

import jaxtyping
import jumanji
from jumanji import specs
from chex import dataclass

from ac_jax.env import utils
from ac_jax.env.types import State, Observation

@dataclass
class ACEnvConfig:
    initial_presentation: Union[np.ndarray, list] = field(
        default_factory=lambda: np.array([1, 0, 2, 0])
    )  # trivial state <x, y>
    horizon_length: int = 1000
    clip_rewards: bool = True
    min_reward: float = -5.0
    max_reward: float = 1000.0
    n_actions: int = 12  # for 2 generators
    normalise_rewards: bool = True

    def __post_init__(self):
        # Convert initial_state to a numpy array if it's a list
        if isinstance(self.initial_presentation, list):
            self.initial_presentation = np.array(self.initial_presentation)

        # Assert that initial_state is a 1-dimensional numpy array
        if not isinstance(self.initial_presentation, np.ndarray):
            raise TypeError("initial_presentation must be a numpy array")
        if self.initial_presentation.ndim != 1:
            raise ValueError("initial_presentation must be a 1-dimensional array")
        if len(self.initial_presentation) % 2 != 0:
            raise ValueError("initial_presentation must have even length")
        if not utils.is_array_valid_presentation(self.initial_presentation):
            raise ValueError("initial_presentation must be a valid presentation")
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            initial_presentation=np.array(
                config_dict.get("initial_presentation", cls().initial_presentation)
            ),
            horizon_length=config_dict.get("horizon_length", cls().horizon_length),
        )
    

class ACEnv(jumanji.env.Environment[State, specs.DiscreteArray, Observation]):
    """Base AC environment; methods are functional and pytree-compatible."""
    def __init__(self, config: ACEnvConfig = ACEnvConfig(),
                 initial_presentations: jaxtyping.ArrayLike = None,
                 eval_presentations: jaxtyping.ArrayLike = None):
        """
        If undertaking curriculum training, initial_presentations should be all tier
        presentations stacked together of shape (n_presentations, n_gen * max_relator_length),
        access via some global idx. 
        """
        self.n_gen = 2
        self.horizon_length = config.horizon_length
        self.n_actions = config.n_actions  # for 2 generators
        self.clip_rewards = config.clip_rewards
        self.min_reward, self.max_reward = config.min_reward, config.max_reward
        self.normalise_rewards = config.normalise_rewards

        if initial_presentations is not None:
            self.initial_presentations = jnp.array(initial_presentations, dtype=jnp.int8)
        else:
            self.initial_presentations = jnp.array([config.initial_presentation], dtype=jnp.int8)

        self.obs_dim = self.initial_presentations[0].shape[-1]
        self.max_relator_length = self.obs_dim // self.n_gen
        self.n_presentations = self.initial_presentations.shape[0]
        self.identity = jnp.eye(self.n_gen, dtype=jnp.int32)

        if eval_presentations is not None:
            self.eval_presentations = jnp.array(eval_presentations, dtype=jnp.int8)

    def action_spec(self):
        # indexes possible AC moves
        return specs.DiscreteArray(self.n_actions, name="action", dtype=jnp.uint8)

    def observation_spec(self):
        # array representation of group presentation
        shape = (self.n_gen * self.max_relator_length,)
        low, high = -self.n_gen, self.n_gen
        return specs.BoundedArray(shape, dtype=jnp.int8, minimum=low, maximum=high, 
                                  name="observation")

    def reset(self, key: jaxtyping.Key) -> Tuple[State, Observation]:
        """Resets environment to first presentation in curriculum.
        """
        return self.reset_to_idx(key, 0)
    
    def reset_to_idx(self, key: jaxtyping.Key, idx: int) -> Tuple[State, Observation]:
        """Resets the environment to the initial state.
        Parameters:
            key: PRNG key for randomness.
            idx: index into initial presentations.
        Returns:
            (state, timestep) pair
        """
        key, subkey = jax.random.split(key)
        initial_presentation = self.initial_presentations[idx]  # TODO: randomly sample logic
        initial_action_history = jnp.full((self.horizon_length,), -1, dtype=jnp.int8)

        initial_state = State(
            presentation=initial_presentation,
            step_count=jnp.array(0, jnp.int32),
            action_history=initial_action_history,
            score=jnp.array(0.0, jnp.float32),
            key=subkey,
            curriculum_idx=idx,
            initial_presentation=initial_presentation,
        )
        observation = Observation(presentation=initial_presentation)
        timestep = jumanji.types.restart(observation)
        return initial_state, timestep
    
    def get_reward(self, old_presentation: jaxtyping.Array,
                   new_presentation: jaxtyping.Array) -> jaxtyping.Float:

        rel_lengths = utils.presentation_length(new_presentation, n_gens=self.n_gen)
        presentation_length = jnp.sum(rel_lengths)
        old_rel_lengths = utils.presentation_length(old_presentation, n_gens=self.n_gen)
        old_presentation_length = jnp.sum(old_rel_lengths)

        delta_length = presentation_length - old_presentation_length
        done = presentation_length == 2

        length_increase_reward, length_decrease_reward = -1e-1 * delta_length, -delta_length  # jnp.exp(delta_length)
        reward = self.max_reward * done - (1 - done) * presentation_length
        # exponent_sum_matrix = utils.get_exponent_sum_matrix(new_presentation)
        # reward = self.max_reward * done - (1 - done) * jnp.linalg.norm(jnp.abs(exponent_sum_matrix) - self.identity) / 10.
        #reward = (1 - done) * jnp.where(delta_length > 0, length_increase_reward, length_decrease_reward) \
        #         + done * self.max_reward
        # reward = done.astype(jnp.float32) - jnp.sum(length) * (1 - done) / max_length
        if self.clip_rewards is True:
            reward = jnp.clip(reward, self.min_reward, self.max_reward)
        if self.normalise_rewards is True: 
            # reward = utils.symlog(reward)
            reward /= self.max_reward

        return reward

    def step(self, state: State, action: jaxtyping.Array):
        new_presentation, lengths = move(state.presentation, action)
        presentation_length = jnp.sum(lengths)
        done = presentation_length == 2

        reward = self.get_reward(state.presentation, new_presentation)

        truncated = state.step_count + 1 >= self.horizon_length

        # build new state, generate observation
        key, subkey = jax.random.split(state.key)
        new_state = State(
            presentation=new_presentation,
            step_count=state.step_count + 1,
            action_history=state.action_history.at[state.step_count].set(action),
            score=state.score + reward,
            key=subkey,
            curriculum_idx=state.curriculum_idx,
            initial_presentation=state.initial_presentation,
            tier_id=state.tier_id,
        )
        observation = Observation(presentation=new_presentation)
        timestep = self._get_timestep(observation, reward, done, truncated)
        return new_state, timestep
    

    def _step(self, presentation, action):
        # minimal step
        new_presentation, lengths = move(presentation, action)
        presentation_length = jnp.sum(lengths)
        done = presentation_length == 2

        reward = self.max_reward * done - (1 - done) * presentation_length
        truncated = False # state.step_count + 1 >= self.horizon_length
        observation = Observation(presentation=new_presentation)
        timestep = self._get_timestep(observation, reward, done, truncated)
        return new_presentation, timestep

    def _get_timestep(self, obs, reward, done, truncated):
        status = jax.lax.cond(done,
                              lambda: jumanji.types.termination(reward=reward, observation=obs),
                              lambda: jax.lax.cond(
                                  truncated,
                                  lambda: jumanji.types.truncation(reward=reward, observation=obs),
                                  lambda: jumanji.types.transition(reward=reward, observation=obs)))
        return status

    def __repr__(self) -> str:
        """String representation of environment. Stateless.
        """
        return ("ACEnv(\n"
            f"    n_gen={self.n_gen},\n"
            f"    max_relator_length={self.max_relator_length},\n"
            f"    horizon_length={self.horizon_length},\n"
            f"    initial_presentation_shape={self.initial_presentations.shape}\n"
            ")")
    
    @staticmethod
    def word2str(word: jaxtyping.Array) -> str:
        """Converts a word represented as an array to a string.
        E.g. [1, -2, 1, 2, 0, 0] -> "x y^-1 x y"
        """
        gen_dict = {1: 'x', -1: 'x^-1', 2: 'y', -2: 'y^-1', 0: ''}
        word_str = ' '.join([gen_dict.get(int(g),'?') for g in word if g != 0])
        return word_str

    def render(self, state: State):
        """Renders current state of the game.
        """
        r1 = state.presentation[:self.max_relator_length]
        r2 = state.presentation[self.max_relator_length:]
        print(f"Step: {state.step_count} | Score: {state.score}")
        print(f"Relator 1: {self.word2str(r1)}")
        print(f"Relator 2: {self.word2str(r2)}")
        print(f"Lengths: {utils.presentation_length(state.presentation)}")
        print(f"Curriculum idx: {state.curriculum_idx}")


# @partial(jit, static_argnums=(1,2))
def concatenate(presentation, max_relator_length, i, sign,
                n_gens=2):
    r"""
    Let $r_i$ be a relator: $r_i \mapsto r_i r_j^{sign}$.
    """
    j = 1 - i
    relators = utils.get_relators(presentation, n_gens=n_gens)
    lengths = utils.presentation_length(presentation, n_gens=n_gens)
    r_i, r_j = relators[i].squeeze(), relators[j].squeeze()
    len_r_i, len_r_j = lengths[i], lengths[j]

    _r_j = jax.lax.cond(sign > 0, 
                       lambda *_: r_j,
                       utils.invert_relator,
                       r_j, len_r_j)
    buffer = jnp.zeros_like(presentation, shape=max_relator_length * 2)
    buffer = jax.lax.dynamic_update_slice(buffer, r_i, (0,))
    buffer = jax.lax.dynamic_update_slice(buffer, _r_j, (len_r_i,))
    buffer, new_relator_length = utils.free_reduce(buffer)
    _r_i = buffer[:max_relator_length]

    new_presentation = jax.lax.dynamic_update_slice(presentation, _r_i, (i * max_relator_length,))
    new_presentation = jax.lax.dynamic_update_slice(new_presentation, r_j, (j * max_relator_length,))
    return jnp.where(new_relator_length <= max_relator_length, new_presentation, presentation)

# @partial(jit, static_argnums=(1,2))
def conjugate(presentation, max_relator_length, i, sign, gen):
    r"""
    Let $r_i$ be a relator, $x_j$ a generator:
    $r_i \mapsto x_j^{sign} r_i x_j^{-sign}$.
    """
    relators = utils.get_relators(presentation, n_gens=2)
    lengths = utils.presentation_length(presentation, n_gens=2)
    r_i, len_r_i = relators[i].squeeze(), lengths[i]
    gen = gen * sign
    inv_gen = -gen

    buffer = jnp.zeros_like(presentation, shape=max_relator_length + 2)

    buffer = buffer.at[0].set(gen)
    buffer = jax.lax.dynamic_update_slice(buffer, r_i, (1,))
    buffer = buffer.at[1 + len_r_i].set(inv_gen)
    buffer, new_relator_length = utils.free_reduce(buffer)
    # compress zeros after reduction
    # buffer = utils.collapse_zeros(buffer)
    _r_i = buffer[:max_relator_length]
    new_presentation = jax.lax.dynamic_update_slice(presentation, _r_i, (i * max_relator_length,))

    return jnp.where(new_relator_length <= max_relator_length, new_presentation, presentation)

@partial(jit, static_argnums=(2,))
def move(presentation, action, n_gens=2, cyclic_reduce=True):
    """
    Applies AC move to presentation.
    """
    max_relator_length = presentation.shape[0] // n_gens
    def _concatenate(i, sign):
        return lambda pres: concatenate(pres, max_relator_length,
                                        i, sign)
    def _conjugate(i, sign, gen):
        return lambda pres: conjugate(pres, max_relator_length,
                                      i, sign, gen)
    
    branches = [
        # --- relator 0 ops ---
        _concatenate(0, sign=+1),       # 0: r0 -> r0 * r1
        _concatenate(0, sign=-1),       # 1: r0 -> r0 * r1^-1
        _conjugate(0, sign=+1, gen=1),  # 2: r0 -> x r0 x^-1
        _conjugate(0, sign=-1, gen=1),  # 3: r0 -> x^-1 r0 x
        _conjugate(0, sign=+1, gen=2),  # 4: r0 -> y r0 y^-1
        _conjugate(0, sign=-1, gen=2),  # 5: r0 -> y^-1 r0 y

        # --- relator 1 ops ---
        _concatenate(1, sign=+1),       # 6: r1 -> r1 * r0
        _concatenate(1, sign=-1),       # 7: r1 -> r1 * r0^-1
        _conjugate(1, sign=+1, gen=1),  # 8: r1 -> x r1 x^-1
        _conjugate(1, sign=-1, gen=1),  # 9: r1 -> x^-1 r1 x
        _conjugate(1, sign=+1, gen=2),  # 10: r1 -> y r1 y^-1
        _conjugate(1, sign=-1, gen=2),  # 11: r1 -> y^-1 r1 y
    ]

    _pres = jax.lax.switch(action, branches, presentation)
    relators = _pres.reshape((n_gens, -1))
    if cyclic_reduce is True:
        relators, lengths = vmap(utils.cyclic_reduce)(relators)
    else:
        lengths = vmap(jnp.count_nonzero)(relators)
    _pres = relators.reshape((-1,))
    return _pres, lengths
