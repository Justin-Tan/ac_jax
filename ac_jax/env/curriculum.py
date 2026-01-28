"""
Curriculum learning utilities for AC environment in JAX.

- CurriculumState: tracks solved/unsolved presentations and curriculum progress
- select_next_presentation: chooses next presentation based on curriculum phase
- CurriculumAutoResetWrapper: auto-reset wrapper with curriculum-aware selection
- VmapCurriculumAutoResetWrapper: vectorised version for batched environments, 
    handles heterogenous resets across a vector of states.
- BatchedVmapCurriculumAutoResetWrapper: batched scatter instead of heterogenous
    resets for GPU.

"""

import jax
import jax.numpy as jnp
from jax import lax, vmap, random
import jax.ops
import chex
from chex import dataclass
from typing import Tuple, NamedTuple
import jaxtyping

import jumanji
from jumanji.wrappers import Wrapper

from ac_jax.env import utils

@dataclass
class CurriculumState:
    """
    Global curriculum state - shared across all environments.
    
    Attributes:
        solved_mask: (n_presentations,) bool array - True if presentation has been solved
        states_processed: counter for sequential iteration during initial phase
        initial_phase_complete: flag - attempted each presentation at least once?
        n_presentations: total #presentations in the curriculum
    """
    solved_mask: jaxtyping.Array  # (n_presentations,) bool
    states_processed: int
    initial_phase_complete: int
    n_presentations: int
    best_episode_lengths: jaxtyping.Array  # (n_presentations,) int
    best_action_sequences: jaxtyping.Array  # (n_presentations, horizon_length) int


def create_curriculum_state(n_presentations: int, horizon_length: int) -> CurriculumState:
    """Initialise curriculum state for n presentations."""
    best_episode_lengths = jnp.full(n_presentations, -1, dtype=jnp.int32)
    best_action_sequences = jnp.full((n_presentations, horizon_length), -1, dtype=jnp.int32)
    return CurriculumState(
        solved_mask=jnp.zeros(n_presentations, dtype=jnp.bool),
        states_processed=0,
        initial_phase_complete=False,
        n_presentations=n_presentations,
        best_episode_lengths=best_episode_lengths,
        best_action_sequences=best_action_sequences)


def _sample_from_mask(key: chex.PRNGKey, mask: jaxtyping.Array) -> int:
    """Sample an index uniformly from positions where mask is True."""
    n = mask.shape[0]
    eps = 1e-6
    # Convert mask to probabilities
    probs = mask.astype(jnp.float32)
    probs = probs / jnp.maximum(jnp.sum(probs), eps)
    return jax.random.choice(key, n, p=probs)


def _sample_initial_phase(curriculum_state: CurriculumState) -> Tuple[int, CurriculumState]:
    """
    During initial phase, iterate sequentially through all presentations.
    """
    next_idx = curriculum_state.states_processed
    _states_processed = curriculum_state.states_processed + 1
    _initial_phase_complete = _states_processed >= curriculum_state.n_presentations
    
    new_curriculum_state = curriculum_state.replace(
        states_processed=_states_processed,
        initial_phase_complete=_initial_phase_complete,
    )
    return next_idx, new_curriculum_state


def _sample_by_probability(key: chex.PRNGKey, curriculum_state: CurriculumState, 
                           repeat_solved_prob: float = 0.25) -> Tuple[int, CurriculumState]:
    """
    Post-initial phase: 
    With probability `repeat_solved_prob`, sample from solved presentations.
    Otherwise, sample from unsolved presentations.
    """
    solved_mask = curriculum_state.solved_mask
    unsolved_mask = ~solved_mask
    n_solved = jnp.sum(solved_mask)
    n_unsolved = jnp.sum(unsolved_mask)
    
    key, _key, __key = jax.random.split(key, 3)
    
    # Decide whether to sample from solved
    sample_from_solved = jax.random.uniform(key) < repeat_solved_prob
    
    def sample_solved(): return _sample_from_mask(_key, solved_mask)
    def sample_unsolved(): return _sample_from_mask(__key, unsolved_mask)
    
    idx = lax.cond(sample_from_solved & (n_solved > 0),
                   sample_solved,
                   lambda: lax.cond(n_unsolved > 0,
                                    sample_unsolved,
                                    sample_solved))
    return idx, curriculum_state


def select_next_presentation(key: chex.PRNGKey, curriculum_state: CurriculumState, 
                             repeat_solved_prob: float = 0.25) -> Tuple[int, CurriculumState]:
    """
    Select the next presentation index based on curriculum phase.
    
    initial_phase True: Sequential iteration through all presentations.
    After initial_phase: Sampling from solved/unsolved pools.
    
    Args:
        key: PRNG key for random selection
        curriculum_state: current curriculum tracking state
        repeat_solved_prob: probability of sampling from solved presentations (post-initial_phase)
    
    Returns:
        (idx, new_curriculum_state): selected presentation index and updated state
    """
    return lax.cond(
        curriculum_state.initial_phase_complete,
        lambda: _sample_by_probability(key, curriculum_state, repeat_solved_prob),
        lambda: _sample_initial_phase(curriculum_state),
    )


def update_solved_status(curriculum_state: CurriculumState, state, is_solved: bool) -> CurriculumState:
    """
    Update global solved_mask for completed episode, updates best achieved action sequence/length.
    """
    c_idx = state.curriculum_idx
    episode_length = state.step_count
    action_history = state.action_history

    _solved_mask = curriculum_state.solved_mask.at[c_idx].set(
        curriculum_state.solved_mask[c_idx] | is_solved)
    current_best_length = curriculum_state.best_episode_lengths[c_idx]
    update_length = (current_best_length < 0) | (episode_length < current_best_length)

    best_episode_lengths = lax.cond(update_length, 
                                    lambda: curriculum_state.best_episode_lengths.at[c_idx].set(episode_length),
                                    lambda: curriculum_state.best_episode_lengths)
    best_action_sequences = lax.cond(update_length,
                                     lambda: curriculum_state.best_action_sequences.at[c_idx].set(action_history),
                                     lambda: curriculum_state.best_action_sequences)
    
    return curriculum_state.replace(solved_mask=_solved_mask, best_episode_lengths=best_episode_lengths,
                                    best_action_sequences=best_action_sequences)

def add_obs_to_extras(timestep, score=None):
    """Used when auto-resetting to store observation from terminal TimeStep (useful for truncation).
    """
    extras = timestep.extras
    extras["terminal_obs"] = timestep.observation
    if score is not None:
        extras["return"] = score + timestep.reward
    return timestep.replace(extras=extras)

class CurriculumAutoResetWrapper(Wrapper):
    """
    Auto-reset wrapper with curriculum-aware presentation selection.
    
    On episode termination:
    1. Updates solved_mask if the episode was successful
    2. Selects the next presentation based on curriculum phase
    3. Resets the environment to the selected presentation
    
    Wrapper modifies step signature to include curriculum_state:
        step(state, action, curriculum_state) -> (state, timestep, curriculum_state)
    """
    
    def __init__(self, env, repeat_solved_prob: float = 0.25,
                 retain_terminal_observation: bool = True):
        super().__init__(env)
        self.repeat_solved_prob = repeat_solved_prob
        if retain_terminal_observation:
            self._maybe_add_obs_to_extras = add_obs_to_extras
        else:
            self._maybe_add_obs_to_extras = lambda timestep: timestep
        
    def reset(self, key: chex.PRNGKey, curriculum_state: CurriculumState):
        """
        Reset using curriculum-based selection.
        Returns:
            (state, timestep, curriculum_state)
        """
        key, select_key = jax.random.split(key)
        idx, new_curriculum_state = select_next_presentation(
            select_key, curriculum_state, self.repeat_solved_prob)
        state, timestep = self._env.reset_to_idx(key, idx)
        timestep = self._maybe_add_obs_to_extras(timestep)
        return state, timestep, new_curriculum_state
    
    def _auto_reset(self, state, timestep, curriculum_state):
        """
        Reset env, and update curriculum state on terminal timesteps (solved or truncated).
        """
        key, subkey = jax.random.split(state.key)
        relator_lengths = utils.presentation_length(timestep.observation.presentation)
        is_solved = jnp.sum(relator_lengths) == 2
        # max_reward = self._env.max_reward if self._env.normalise_rewards is False else 1.0
        # is_solved = timestep.reward >= max_reward
        _curriculum_state = lax.cond(is_solved,
                                     lambda cs: update_solved_status(cs, state, jnp.bool_(True)),
                                     lambda cs: cs,
                                     curriculum_state)
        idx, _curriculum_state = select_next_presentation(subkey, _curriculum_state, self.repeat_solved_prob)
        _state, reset_timestep = self._env.reset_to_idx(key, idx)
        # Preserve terminal observation for proper value bootstrap
        timestep = self._maybe_add_obs_to_extras(timestep)
        timestep = timestep.replace(observation=reset_timestep.observation)

        return _state, timestep, _curriculum_state

    def no_op(self, state, timestep, curriculum_state):
        timestep = self._maybe_add_obs_to_extras(timestep)
        return state, timestep, curriculum_state

    def step(self, state, action: jaxtyping.Array, curriculum_state: CurriculumState):
        """
        Environment step with curriculum-aware auto-reset.
        Args:
            state: current environment state
            action: action to take
            curriculum_state: current curriculum state
        Returns:
            (new_state, timestep, new_curriculum_state)
        """
        state, timestep = self._env.step(state, action)

        # overswrite state/timestep if episode terminates
        state, timestep, curriculum_state = jax.lax.cond(timestep.last(),
                                                self._auto_reset,
                                                self.no_op,
                                                state, timestep, curriculum_state)
        return state, timestep, curriculum_state


class VmapCurriculumAutoResetWrapper(CurriculumAutoResetWrapper):
    """
    Vectorised curriculum auto-reset wrapper for batched environments.
    
    Uses jax.vmap for parallel step execution and jax.lax.scan
    for heterogeneous reset handling.
    - Heterogeneous computation: conditional auto-reset (call reset function 
      for some environments within the batch because they have terminated).
    """

    def reset(self, key: chex.PRNGKey, curriculum_state: CurriculumState, num_envs: int):
        """
        Reset all environments using curriculum-based selection. 
        Sequentially resets states to properly update global curriculum state
            
        Returns:
            (batched_state, batched_timestep, curriculum_state)
        """
        keys = jax.random.split(key, num_envs + 1)
        reset_keys = keys[1:]
        
        # Sequential resets to properly update curriculum state
        def _reset_one(carry, key):
            curriculum_state = carry
            key, select_key = jax.random.split(key)
            idx, new_curriculum_state = select_next_presentation(
                select_key, curriculum_state, self.repeat_solved_prob)
            state, timestep = self._env.reset_to_idx(key, idx)
            timestep = self._maybe_add_obs_to_extras(timestep)
            return new_curriculum_state, (state, timestep)
        
        final_curriculum_state, (states, timesteps) = lax.scan(
            _reset_one, curriculum_state, reset_keys)
        
        return states, timesteps, final_curriculum_state

    def _process_single_env(self, curriculum_state: CurriculumState, state_info):
        
        state, timestep = state_info
        new_state, new_timestep, new_curriculum_state = jax.lax.cond(timestep.last(),
            self._auto_reset, self.no_op, 
            state, timestep, curriculum_state)
        return new_curriculum_state, (new_state, new_timestep)

    def step(self, state, action: jaxtyping.Array, curriculum_state: CurriculumState):
        """
        Batched step with curriculum-aware auto-reset.
        1. `vmap` across all environments in parallel.
        2. Scan through envs to handle resets sequentially due to shared
           curriculum state.
        Args:
            state: batched environment state (pytree with leading batch dim)
            action: batched actions (num_envs,)
            curriculum_state: current global curriculum state.            
        Returns:
            (new_state, timestep, new_curriculum_state)
        """
        # parallel step for all environments
        new_state, timestep = jax.vmap(self._env.step)(state, action)
        
        final_curriculum_state, (new_state, new_timestep) = jax.lax.scan(
            self._process_single_env, curriculum_state, (new_state, timestep))
        
        return new_state, new_timestep, final_curriculum_state


# =============================================================================
# Batched Curriculum State (GPU-optimised, no sequential phase tracking)
# =============================================================================

@dataclass
class CurriculumStateBatched:
    """
    Simplified curriculum state for batched/parallel updates.
    
    Drops sequential phase tracking (states_processed, initial_phase_complete)
    and best_action_sequences for GPU-friendly batched scatter operations.
    
    Attributes:
        solved_mask: (n_presentations,) bool array - True if presentation has been solved
        best_episode_lengths: (n_presentations,) int array - shortest solve length (-1 = unsolved)
        n_presentations: total #presentations in the curriculum
    """
    solved_mask: jaxtyping.Array  # (n_presentations,) bool
    best_episode_lengths: jaxtyping.Array  # (n_presentations,) int
    n_presentations: int
    best_action_sequences: jaxtyping.Array  # (n_presentations, horizon_length) int


def create_curriculum_state_batched(n_presentations: int, horizon_length: int) -> CurriculumStateBatched:
    """Initialise batched curriculum state for n presentations."""
    return CurriculumStateBatched(
        solved_mask=jnp.zeros(n_presentations, dtype=jnp.bool),
        best_episode_lengths=jnp.full(n_presentations, -1, dtype=jnp.int32),
        n_presentations=n_presentations,
        best_action_sequences=jnp.full((n_presentations, horizon_length), -1, dtype=jnp.int32))


def _sample_presentation_stateless(key: chex.PRNGKey, solved_mask: jaxtyping.Array,
                                   repeat_solved_prob: float = 0.25) -> int:
    """
    Sample a presentation index based on solved/unsolved status.
    
    Stateless function suitable for vmap. Does not mutate curriculum state.
    
    Args:
        key: PRNG key for random selection
        solved_mask: (n_presentations,) bool array
        repeat_solved_prob: probability of sampling from solved presentations
    
    Returns:
        Selected presentation index
    """
    unsolved_mask = ~solved_mask
    n_solved = jnp.sum(solved_mask)
    n_unsolved = jnp.sum(unsolved_mask)
    n = solved_mask.shape[0]
    eps = 1e-6
    
    key, key_choice, key_solved, key_unsolved = jax.random.split(key, 4)
    
    # Decide whether to sample from solved
    sample_from_solved = jax.random.uniform(key_choice) < repeat_solved_prob
    
    # Compute probabilities for each pool
    solved_probs = solved_mask.astype(jnp.float32)
    solved_probs = solved_probs / jnp.maximum(jnp.sum(solved_probs), eps)
    
    unsolved_probs = unsolved_mask.astype(jnp.float32)
    unsolved_probs = unsolved_probs / jnp.maximum(jnp.sum(unsolved_probs), eps)
    
    # Sample from each pool
    idx_solved = jax.random.choice(key_solved, n, p=solved_probs)
    idx_unsolved = jax.random.choice(key_unsolved, n, p=unsolved_probs)
    
    # Select based on probability and availability
    # Priority: if sample_from_solved and solved exists -> solved
    #           elif unsolved exists -> unsolved
    #           else -> solved (fallback)
    idx = jnp.where(sample_from_solved & (n_solved > 0), idx_solved,
                    jnp.where(n_unsolved > 0, idx_unsolved, idx_solved))
    return idx


def batched_update_curriculum(
    curriculum_state: CurriculumStateBatched,
    curr_idx: jaxtyping.Array,
    solved_per_env: jaxtyping.Array,
    episode_lengths: jaxtyping.Array,
    action_histories: jaxtyping.Array,
    n_presentations: int
) -> CurriculumStateBatched:
    """
    Batch-update curriculum state using scatter operations.
    
    Handles conflicts when multiple envs solve the same presentation by:
    - OR-ing solved_mask updates
    - Taking minimum episode length
    
    Args:
        curriculum_state: current batched curriculum state
        c_indices: (num_envs,) curriculum indices for each env
        solved_per_env: (num_envs,) bool - whether each env solved its presentation
        episode_lengths: (num_envs,) int - step count for each env
    
    Returns:
        Updated curriculum state
    """
    num_envs = curr_idx.shape[0]
    env_idx = jnp.arange(num_envs, dtype=jnp.int32)
    # update solved_mask using max
    # Only update indices where solved_per_env is True
    new_solved_mask = curriculum_state.solved_mask.at[curr_idx].max(solved_per_env)
    
    # update best_episode_lengths
    # For envs that didn't solve, use a large value so they don't affect min
    int_inf = jnp.int32(2**30)
    lengths_to_scatter = jnp.where(solved_per_env, episode_lengths, int_inf)
    
    # segment_min finds minimum length per presentation index
    segment_mins = jax.ops.segment_min(
        lengths_to_scatter, curr_idx,
        num_segments=n_presentations, 
        indices_are_sorted=False)
        
    # Merge with existing best_episode_lengths:
    current_best = curriculum_state.best_episode_lengths
    _new_solve = segment_mins < int_inf
    new_best = jnp.where(
        _new_solve,
        jnp.where(current_best < 0, segment_mins, jnp.minimum(current_best, segment_mins)),
        current_best)  # no solve, keep existing

    segment_mins_per_env = segment_mins[curr_idx]
    env_optimal_mask = jnp.logical_and(solved_per_env, lengths_to_scatter == segment_mins_per_env)
    env_idx_inf = jnp.where(env_optimal_mask, env_idx, int_inf)

    # get env index
    optimal_per_env = jax.ops.segment_min(env_idx_inf, curr_idx, num_segments=n_presentations,
        indices_are_sorted=False)
    optimal_per_env = jnp.where(optimal_per_env < int_inf, optimal_per_env, 0)

    _action_sequences = action_histories[optimal_per_env]
    _update_flag = jnp.logical_and(_new_solve, jnp.logical_or(current_best < 0, (segment_mins < current_best)))

    current_action_sequences = curriculum_state.best_action_sequences
    new_action_sequences = jnp.where(_update_flag[:,None], _action_sequences, current_action_sequences)
    
    return curriculum_state.replace(
        solved_mask=new_solved_mask,
        best_episode_lengths=new_best,
        best_action_sequences=new_action_sequences)


# =============================================================================
# Batched Vmap Curriculum Wrapper (GPU-optimised, fully parallel)
# =============================================================================

class BatchedVmapCurriculumAutoResetWrapper(Wrapper):
    """
    Fully parallelised curriculum auto-reset wrapper for batched environments.
    
    GPU-optimised implementation eliminating sequential lax.scan by:
    1. Using batched scatter operations for curriculum state updates
    2. Using vmap for parallel presentation selection and resets
    3. Merging reset/stepped states with jnp.where
    """
    
    def __init__(self, env, repeat_solved_prob: float = 0.25,
                 retain_terminal_observation: bool = True):
        super().__init__(env)
        self.repeat_solved_prob = repeat_solved_prob
        self.n_presentations = env.n_presentations
        self.horizon_length = env.horizon_length
        if retain_terminal_observation:
            self._maybe_add_obs_to_extras = add_obs_to_extras
        else:
            self._maybe_add_obs_to_extras = lambda timestep: timestep
    
    def reset(self, key: chex.PRNGKey, curriculum_state: CurriculumStateBatched, 
              num_envs: int) -> Tuple:
        """
        Reset all environments using parallel curriculum-based selection.        
        Returns:
            (batched_state, batched_timestep, curriculum_state)
        """
        keys = jax.random.split(key, num_envs + 1)
        key, reset_keys = keys[0], keys[1:]
        
        # Parallel presentation selection for all envs
        select_keys = jax.random.split(key, num_envs)
        indices = vmap(
            lambda k: _sample_presentation_stateless(k, curriculum_state.solved_mask, 
                                                     self.repeat_solved_prob)
        )(select_keys)
        
        # Parallel reset to selected presentations
        states, timesteps = vmap(self._env.reset_to_idx)(reset_keys, indices)
        timesteps = vmap(self._maybe_add_obs_to_extras)(timesteps, states.score)
        
        return states, timesteps, curriculum_state
    
    def step(self, state, action: jaxtyping.Array, 
             curriculum_state: CurriculumStateBatched) -> Tuple:
        """
        Batched step with fully parallel curriculum-aware auto-reset.
        
        1. vmap(env.step) - parallel environment steps
        2. Compute terminal/solved masks
        3. batched_update_curriculum() - single batched scatter op
        4. vmap(sample_presentation) - parallel selection for resets
        5. vmap(env.reset_to_idx) - parallel resets for all envs
        6. jnp.where merge - keep stepped state for non-terminal, reset for terminal
        
        Args:
            state: batched environment state (pytree with leading batch dim)
            action: batched actions (num_envs,)
            curriculum_state: current batched curriculum state
            
        Returns:
            (new_state, new_timestep, new_curriculum_state)
        """
        num_envs = action.shape[0]
        
        # parallel environment step for all envs
        stepped_state, stepped_timestep = vmap(self._env.step)(state, action)
        terminal_mask = vmap(lambda ts: ts.last())(stepped_timestep)  # (num_envs,)
        
        # Check if solved (total relator length == 2)
        obs_lengths = vmap(utils.presentation_length)(
            stepped_timestep.observation.presentation)  # (num_envs, n_gens)
        total_lengths = jnp.sum(obs_lengths, axis=-1)  # (num_envs,)
        solved_per_env = terminal_mask & (total_lengths == 2)  # (num_envs,)
        
        # Gather curriculum indices and episode lengths
        c_indices = stepped_state.curriculum_idx  # (num_envs,)
        episode_lengths = stepped_state.step_count  # (num_envs,)
        
        # batched curriculum state update
        new_curriculum_state = batched_update_curriculum(
            curriculum_state, c_indices, solved_per_env, episode_lengths,
            stepped_state.action_history, self.n_presentations)
        
        # parallel presentation selection for resets
        # Split keys for selection and reset
        key = stepped_state.key[0]
        select_keys, reset_keys = select_keys[:num_envs], select_keys[num_envs:]
        
        reset_indices = vmap(
            lambda k: _sample_presentation_stateless(k, new_curriculum_state.solved_mask,
                                                     self.repeat_solved_prob))(select_keys)
        
        # parallel reset for all envs
        reset_state, reset_timestep = vmap(self._env.reset_to_idx)(reset_keys, reset_indices)
        
        # merge stepped and reset states
        # Store terminal observation in extras before replacing
        stepped_timestep = vmap(self._maybe_add_obs_to_extras)(stepped_timestep, stepped_state.score)
        
        # Helper to broadcast terminal_mask for tree_map
        def _merge(stepped, reset):
            # Expand terminal_mask to match array dimensions
            expand_dims = tuple(range(1, stepped.ndim)) if stepped.ndim > 1 else ()
            mask = jnp.expand_dims(terminal_mask, axis=expand_dims)
            return jnp.where(mask, reset, stepped)
        
        # Merge states - use reset state where terminal, else keep stepped
        final_state = jax.tree_util.tree_map(_merge, stepped_state, reset_state)
        
        # For timestep, replace observation with reset obs but keep other fields from stepped
        # (reward, discount, step_type from the step that caused termination)
        final_timestep = stepped_timestep.replace(
            observation=jax.tree.map(_merge, stepped_timestep.observation, 
                                     reset_timestep.observation))
        
        return final_state, final_timestep, new_curriculum_state

# =============================================================================
# Multi-level curriculum state tracking, batched and GPU-optimised
# =============================================================================

@dataclass
class MultiLevelCurriculumState:
    """
    Container for multiple curriculum tiers (Easy, Med, Hard).
    """
    easy: CurriculumStateBatched
    medium: CurriculumStateBatched
    hard: CurriculumStateBatched
    # sampling probabilities (p_easy, p_med, p_hard)
    mixing_probs: jaxtyping.Array  # (n_tiers,) - default n_tiers = 3
    # per-tier success rates to handle transition logic
    tier_success_rates: jaxtyping.Array  # (n_tiers,)
    curriculum_lambda: 0.  # interpolant

def create_multilevel_curriculum(counts: Tuple[int, int, int], horizon: int):
    return MultiLevelCurriculumState(
        easy=create_curriculum_state_batched(counts[0], horizon),
        medium=create_curriculum_state_batched(counts[1], horizon),
        hard=create_curriculum_state_batched(counts[2], horizon),
        mixing_probs=jnp.array([1.0, 0.0, 0.0]),  # all easy initially
        tier_success_rates=jnp.zeros(3),
        curriculum_lambda=jnp.array(0.0))

def _sample_presentation_stateless(key: chex.PRNGKey, solved_mask: jaxtyping.Array,
                                   repeat_solved_prob: float = 0.25) -> int:
    """
    Sample a presentation index based on solved/unsolved status.
    Does not mutate curriculum state.
    
    Args:
        key: PRNG key for random selection
        solved_mask: (n_presentations,) bool array
        repeat_solved_prob: probability of sampling from solved presentations
    
    Returns:
        Selected presentation index
    """
    unsolved_mask = ~solved_mask
    n_solved = jnp.sum(solved_mask)
    n_unsolved = jnp.sum(unsolved_mask)
    n = solved_mask.shape[0]
    eps = 1e-6
    
    key, key_choice, key_solved, key_unsolved = jax.random.split(key, 4)
    
    # Decide whether to sample from solved
    sample_from_solved = jax.random.uniform(key_choice) < repeat_solved_prob
    
    # Compute probabilities for each pool
    solved_probs = solved_mask.astype(jnp.float32)
    solved_probs = solved_probs / jnp.maximum(jnp.sum(solved_probs), eps)
    
    unsolved_probs = unsolved_mask.astype(jnp.float32)
    unsolved_probs = unsolved_probs / jnp.maximum(jnp.sum(unsolved_probs), eps)
    
    # Sample from each pool
    idx_solved = jax.random.choice(key_solved, n, p=solved_probs)
    idx_unsolved = jax.random.choice(key_unsolved, n, p=unsolved_probs)
    
    # Select based on probability and availability
    # Priority: if sample_from_solved and solved exists -> solved
    #           elif unsolved exists -> unsolved
    #           else -> solved (fallback)
    idx = jnp.where(sample_from_solved & (n_solved > 0), idx_solved,
                    jnp.where(n_unsolved > 0, idx_unsolved, idx_solved))
    return idx


def _sample_multilevel(key: chex.PRNGKey, curriculum_state: MultiLevelCurriculumState, 
                       repeat_solved_prob: float) -> Tuple[int, int]:
    """
    Returns (tier_id, presentation_index_within_tier)
    """
    cs = curriculum_state
    key, subkey = jax.random.split(key)
    
    # sample tier (0=Easy, 1=Med, 2=Hard)
    tier_id = jax.random.choice(key, 3, p=cs.mixing_probs)
    
    def _sample_easy(k):
        return _sample_presentation_stateless(k, cs.easy.solved_mask, repeat_solved_prob)
        
    def _sample_medium(k):
        return _sample_presentation_stateless(k, cs.medium.solved_mask, repeat_solved_prob)
        
    def _sample_hard(k):
        return _sample_presentation_stateless(k, cs.hard.solved_mask, repeat_solved_prob)
    
    local_idx = jax.lax.switch(tier_id, [_sample_easy, _sample_medium, _sample_hard], subkey)
    return tier_id, local_idx
    

def _sample_tier_batch(key, solved_mask, repeat_solved_prob, batch_size):
    """ Sample from unsolved pool without replacement, reverting to solved pool
        if number of requested unsolved items exceeds actual number of unsolved.
        Solved items sampled with replacement.
    """
    boost_score_constant = 1e3
    eps = 1e-6
    n_total = solved_mask.shape[0]
    n_solved = jnp.sum(solved_mask)
    n_unsolved = n_total - n_solved

    key, subkey, _subkey = random.split(key, 3)

    # unsolved queue, no replacement
    scores = ~solved_mask * boost_score_constant + random.uniform(key, (n_total,))
    _, unsolved_idx = jax.lax.top_k(scores, k=batch_size)

    # solved pool with replacement
    p_solved = solved_mask.astype(jnp.float32)
    p_solved = p_solved / jnp.maximum(jnp.sum(p_solved), eps)
    solved_pool = random.choice(subkey, n_total, shape=(batch_size,), p=p_solved)

    # select between pools
    sample_from_solved = random.uniform(_subkey, (batch_size,)) < repeat_solved_prob
    solved_overflow = jnp.arange(batch_size) >= n_unsolved

    idx = jnp.where(jnp.logical_or(sample_from_solved, solved_overflow), solved_pool, unsolved_idx)
    return idx


def _adaptive_resample_prob(base_prob: float, solved_mask: float) -> float:
    n_total = solved_mask.shape[0]
    n_solved = jnp.sum(solved_mask)
    success_rate = n_solved / n_total

    resample_prob = jnp.select(condlist=[success_rate > 0.99, success_rate > 0.9, success_rate > 0.6],
                               choicelist=[0.95, 0.9, 0.8], default=base_prob)
    return resample_prob

def _sample_multilevel_batch(key: chex.PRNGKey, curriculum_state: MultiLevelCurriculumState,
        base_repeat_solved_prob: float, batch_size: int):

    cs = curriculum_state
    key, subkey = jax.random.split(key)
    subkey, *sample_keys = jax.random.split(subkey, 4)
    tier_id = random.choice(subkey, 3, p=cs.mixing_probs, shape=(batch_size,))

    repeat_solved_easy_prob = _adaptive_resample_prob(base_repeat_solved_prob, cs.easy.solved_mask)
    repeat_solved_med_prob = _adaptive_resample_prob(base_repeat_solved_prob, cs.medium.solved_mask)
    repeat_solved_hard_prob = _adaptive_resample_prob(base_repeat_solved_prob, cs.hard.solved_mask)

    candidates_easy = _sample_tier_batch(sample_keys[0], cs.easy.solved_mask, repeat_solved_easy_prob, batch_size)
    candidates_med = _sample_tier_batch(sample_keys[1], cs.medium.solved_mask, repeat_solved_med_prob, batch_size)
    candidates_hard = _sample_tier_batch(sample_keys[2], cs.hard.solved_mask, repeat_solved_hard_prob, batch_size)

    def _select_cands(tier_id, candidates_easy, candidates_med, candidates_hard):
        return jax.lax.switch(tier_id,
                [lambda: candidates_easy, lambda: candidates_med, lambda: candidates_hard])
    local_idx = vmap(_select_cands)(tier_id, candidates_easy, candidates_med, candidates_hard)
    return tier_id, local_idx

class BatchedMultiLevelVmapCurriculumAutoResetWrapper(Wrapper):
    """
    Like BatchedVmapCurriculumAutoResetWrapper but implementing multi-level curriculum.
    Transitions between levels based on per-tier curriculum success rates.
    """
    
    def __init__(self, env, tier_counts: Tuple[int, int, int],
                 repeat_solved_prob: float = 0.25,
                 retain_terminal_observation: bool = True):
        super().__init__(env)
        self.repeat_solved_prob = repeat_solved_prob
        self.n_presentations = env.n_presentations
        self.horizon_length = env.horizon_length
        self.tier_counts = tier_counts
        self.idx_offsets = jnp.array([0, tier_counts[0], tier_counts[0] + tier_counts[1]])

        if retain_terminal_observation:
            self._maybe_add_obs_to_extras = add_obs_to_extras
        else:
            self._maybe_add_obs_to_extras = lambda timestep: timestep

        # admixture of difficulty tiers
        self.curriculum_lambda_stepsize = 1e-5
        self.prob_phases = jnp.array([0., 0.5, 1.0])
        self.probability_mixing_matrix = jnp.array([
            [0.70, 0.20, 0.05], # (lambda=0.0)
            [0.50, 0.25, 0.25], # (lambda=0.5)
            [0.30, 0.35, 0.35]  # (lambda=1.0)
        ])
        self.lambda_increase_boundary = 0.7 # weighted prob success rate condition; increment
        self.lambda_decrease_boundary = 0.5 # weighted prob success rate condition; decrement
    
    def reset(self, key: chex.PRNGKey, curriculum_state: MultiLevelCurriculumState, 
              num_envs: int) -> Tuple:
        """
        Reset all environments using parallel curriculum-based selection. 
                
        Returns:
            (batched_state, batched_timestep, curriculum_state)
        """
        keys = jax.random.split(key, num_envs + 1)
        key, reset_keys = keys[0], keys[1:]
        
        # Parallel presentation selection for all envs
        key, select_key = jax.random.split(key)
        tier_id, local_idx = _sample_multilevel_batch(select_key, curriculum_state,
                self.repeat_solved_prob, batch_size=num_envs)
        
        # Parallel reset to selected presentations
        global_idx = local_idx + self.idx_offsets[tier_id]
        states, timesteps = vmap(self._env.reset_to_idx)(reset_keys, global_idx)
        timesteps = vmap(self._maybe_add_obs_to_extras)(timesteps, states.score)
        states = states.replace(tier_id=tier_id) #, curriculum_idx=local_idx)
        
        return states, timesteps, curriculum_state
    
    def step(self, state, action: jaxtyping.Array, 
             curriculum_state: MultiLevelCurriculumState) -> Tuple:
        """
        Batched step with fully parallel curriculum-aware auto-reset.
        
        Phases:
        1. vmap(env.step) - parallel environment steps
        2. Compute terminal/solved masks
        3. batched_update_curriculum() - single batched scatter op
        4. vmap(sample_presentation) - parallel selection for resets
        5. vmap(env.reset_to_idx) - parallel resets for ALL envs
        6. jnp.where merge - keep stepped state for non-terminal, reset for terminal
        
        Args:
            state: batched environment state (pytree with leading batch dim)
            action: batched actions (num_envs,)
            curriculum_state: current batched curriculum state
            
        Returns:
            (new_state, new_timestep, new_curriculum_state)
        """
        num_envs = action.shape[0]
        
        stepped_state, stepped_timestep = vmap(self._env.step)(state, action)
        terminal_mask = vmap(lambda ts: ts.last())(stepped_timestep)  # (num_envs,)
        
        # Check if solved (total relator length == 2)
        obs_lengths = vmap(utils.presentation_length)(
            stepped_timestep.observation.presentation)  # (num_envs, n_gens)
        total_lengths = jnp.sum(obs_lengths, axis=-1)  # (num_envs,)
        solved_per_env = terminal_mask & (total_lengths == 2)  # (num_envs,)
        
        # Gather curriculum indices and episode lengths
        # state holds global index, so convert to local per-tier index
        current_tier_id = stepped_state.tier_id  # (num_envs,)
        current_global_idx = stepped_state.curriculum_idx  # (num_envs,)
        current_local_idx = current_global_idx - self.idx_offsets[current_tier_id]
        episode_lengths = stepped_state.step_count  # (num_envs,)
        
        # batched curriculum state update
        def _update_tier(tier_state, tier_idx):
            tier_mask = current_tier_id == tier_idx
            tier_solved_per_env = solved_per_env & tier_mask

            n_presentations = self.tier_counts[tier_idx]
            return batched_update_curriculum(tier_state, current_local_idx, 
                tier_solved_per_env, episode_lengths, stepped_state.action_history,
                n_presentations)
        
        new_easy_state = _update_tier(curriculum_state.easy, 0)
        new_medium_state = _update_tier(curriculum_state.medium, 1)
        new_hard_state = _update_tier(curriculum_state.hard, 2)

        # success rates and interpolate betweeen tiers.
        def _get_success_rate(tier_id, current_ema, ema_decay=0.99):
            tier_mask = (tier_id == current_tier_id) & terminal_mask
            tier_count = jnp.sum(tier_mask)
            _rate = jnp.sum(solved_per_env & tier_mask) / jnp.maximum(tier_count, 1.)
            return jnp.where(tier_count > 0,
                             ema_decay * current_ema + (1 - ema_decay) * _rate,
                             current_ema)
        
        tier_range = jnp.arange(3)
        new_success_rates = vmap(_get_success_rate)(tier_range, curriculum_state.tier_success_rates)

        weighted_success_rates = jnp.dot(curriculum_state.mixing_probs, new_success_rates)
        easy_proficient_flag = new_success_rates[0] > self.lambda_increase_boundary
        easy_forgetting_flag = new_success_rates[0] < self.lambda_decrease_boundary 
        curriculum_lam = curriculum_state.curriculum_lambda
        curriculum_lam = jnp.where((weighted_success_rates > self.lambda_increase_boundary) & (easy_proficient_flag), 
                                   curriculum_lam + self.curriculum_lambda_stepsize,
                                   curriculum_lam)
        curriculum_lam = jnp.where((weighted_success_rates < self.lambda_decrease_boundary) & (easy_forgetting_flag), 
                                   curriculum_lam - self.curriculum_lambda_stepsize,
                                   curriculum_lam)
        curriculum_lam = jnp.clip(curriculum_lam, 0.0, 1.0)

        new_mixing_probs = vmap(jnp.interp, in_axes=(None,None,1))(curriculum_lam,
                            self.prob_phases, self.probability_mixing_matrix)
        new_mixing_probs = new_mixing_probs / jnp.sum(new_mixing_probs)

        new_curriculum_state = MultiLevelCurriculumState(easy=new_easy_state,
                                                         medium=new_medium_state,
                                                         hard=new_hard_state,
                                                         mixing_probs=new_mixing_probs,
                                                         tier_success_rates=new_success_rates,
                                                         curriculum_lambda=curriculum_lam)


        # parallel presentation selection for resets
        key = stepped_state.key[0]
        key, select_key = jax.random.split(key)
        reset_keys = jax.random.split(key, num_envs+1)
        key, reset_keys = reset_keys[0], reset_keys[1:]
        new_tier_id, new_local_idx = _sample_multilevel_batch(select_key, curriculum_state,
                self.repeat_solved_prob, batch_size=num_envs)
        new_global_idx = new_local_idx + self.idx_offsets[new_tier_id]


        # parallel reset for all envs
        reset_state, reset_timestep = vmap(self._env.reset_to_idx)(reset_keys, new_global_idx)
        
        # merge stepped and reset states
        # Store terminal observation in extras before replacing for bootstrap
        stepped_timestep = vmap(self._maybe_add_obs_to_extras)(stepped_timestep, stepped_state.score)
        
        # Helper to broadcast terminal_mask for tree_map
        def _merge(stepped, reset):
            # Expand terminal_mask to match array dimensions
            expand_dims = tuple(range(1, stepped.ndim)) if stepped.ndim > 1 else ()
            mask = jnp.expand_dims(terminal_mask, axis=expand_dims)
            return jnp.where(mask, reset, stepped)
        
        # Merge states - use reset state where terminal, else keep stepped
        final_state = jax.tree_util.tree_map(_merge, stepped_state, reset_state)

        # update tier metadata in state
        final_tier_id = jnp.where(terminal_mask, new_tier_id, current_tier_id)
        final_global_idx = jnp.where(terminal_mask, new_global_idx, current_global_idx)
        final_state = final_state.replace(tier_id=final_tier_id,
                                          curriculum_idx=final_global_idx)
        
        # For timestep, replace observation with reset obs but keep other fields from stepped
        # (reward, discount, step_type from the step that caused termination)
        final_timestep = stepped_timestep.replace(
            observation=jax.tree.map(_merge, stepped_timestep.observation, 
                                     reset_timestep.observation))
        
        return final_state, final_timestep, new_curriculum_state
