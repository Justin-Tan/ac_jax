"""
Minimal evaluation function for use in evolutionary search loop.
"""

import jax
import jax.numpy as jnp
from jax import lax, random, vmap, jit

import functools
import numpy as np

import mctx
from ac_jax.env import ac_env, utils    

class HeuristicCallback:
    """
    Mutable holder: same Python object identity so JAX reuses cached compilation.
    Heuristic can be swapped freely without triggering recompilation.
    """
    
    def __init__(self, fn=None, batch_size=None):
        self.fn = fn
        self.batch_size = batch_size
    
    def update(self, new_heuristic_fn):
        """Call this before each evaluation with the new evolved heuristic."""
        self.fn = jax.jit(jax.vmap(new_heuristic_fn))
    
    def __call__(self, presentations):
        """Called by XLA at runtime via pure_callback."""
        return self.fn(presentations)


class SearchEvaluator:

    def __init__(self, initial_pool, horizon_length=256):
        self.initial_pool = initial_pool
        self.horizon_length = horizon_length
        self._heuristic_cb = HeuristicCallback()
        self.init_env()

    def init_env(self):
        initial_presentations = np.load(self.initial_pool)
        ac_env_conf = ac_env.ACEnvConfig(horizon_length=self.horizon_length)
        self.eval_env = ac_env.ACEnv(ac_env_conf, initial_presentations=initial_presentations)
        self.n_presentations, self.obs_dim = initial_presentations.shape
        self.max_relator_length = self.eval_env.max_relator_length
        self.n_eval = self.n_presentations

    def heuristic_fn(self, presentation: jnp.ndarray) -> float:
        """
        Assign an integer to each generator, and assign its negation to the inverse generator. We encode a presentation $\langle x_1, x_2 : r_1, r_2 \rangle$ solely in terms of the relators $r_i$, as the concatenation of two integer arrays denoting the definition of each relator.

        Baseline heuristic: current presentation length
        
        Args:
            presentation: Array representing current group presentation in terms of relators
                [r_1; r_2]. Shape (72,) int32 array. 0 is padding.
        
        Returns:
            Scalar heuristic value in [0,1] estimating likelihood of trivialisation in the
            future (higher is better).
        """
        N_GENERATORS = 2
        MAX_RELATOR_LENGTH = 36
        MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH

        # Example baseline logic: negative normalised presentation length
        is_generator = jnp.abs(presentation) > 0
        total_length = jnp.sum(is_generator)
        return -1. * total_length / MAX_PRESENTATION_LENGTH

    @functools.partial(jit, static_argnums=(0,2,3))
    def _evaluate_batch_mcts_custom_heuristic_callback(self, key,
                             n_simulations=256, max_depth=None):
        # evaluate n_eval environments in parallel
        eval_idx = jnp.arange(self.n_eval)
        key, subkey, _subkey, __subkey = random.split(key, 4)
        reset_keys = random.split(__subkey, self.n_eval)
        states, timesteps = vmap(self.eval_env.reset_to_idx)(reset_keys, eval_idx)
        # could the evolved heuristic output dummy logits to rank actions?
        dummy_logits = jnp.zeros((self.n_eval, self.eval_env.n_actions))

        # Shape template for the callback output
        values_shape = jax.ShapeDtypeStruct((self.n_eval,), jnp.float32)

        def _eval_heuristic(presentations):
            """Calls the mutable holder via pure_callback — no recompile."""
            return jax.pure_callback(
                self._heuristic_cb,  # same object identity every time
                values_shape,
                presentations,
            )

        def _recurrent_fn(params, rng, action, embedding):
            # Simulates one environment step inside MCTS phase.
            presentation = embedding
            next_presentation, next_timestep = vmap(self.eval_env._step)(
                presentation, action)
            obs, next_obs = presentation, next_presentation
            values = _eval_heuristic(next_obs)
            logits = dummy_logits

            reward = next_timestep.reward
            discount = next_timestep.discount

            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward, discount=discount,
                prior_logits=logits, value=values)
            return recurrent_fn_output, next_presentation

        def _eval_step(carry, sentinel):
            state, timestep, key = carry
            key, mcts_key = random.split(key)
            obs = timestep.observation.presentation
            # root_value = vmap(heuristic_fn)(obs)
            root_value = _eval_heuristic(obs)
            root_logits = dummy_logits  # (B, A)

            root = mctx.RootFnOutput(
                prior_logits=root_logits,
                value=root_value,
                embedding=obs)  # state

            #action selection
            policy_output = mctx.gumbel_muzero_policy(
                params=None,#params,
                rng_key=mcts_key,
                root=root,
                recurrent_fn=_recurrent_fn,
                num_simulations=n_simulations,
                max_depth=max_depth,
                max_num_considered_actions=self.eval_env.n_actions,
            )

            actions = policy_output.action
            next_state, next_timestep = vmap(self.eval_env.step)(state, actions)
            return (next_state, next_timestep, key), next_timestep

        init_carry = (states, timesteps, _subkey)
        # run over full horizon and check if solved at any step
        (final_state, final_timestep, _subkey), eval_rollout = jax.lax.scan(
            _eval_step, init_carry, None, self.horizon_length)

        # eval_rollout: (T, B, ...)
        flattened_rollout_data = jax.tree_util.tree_map(lambda x: x.reshape((-1,)+ x.shape[2:]),
                                                        eval_rollout)

        flattened_obs = flattened_rollout_data.observation.presentation
        flattened_lengths = vmap(utils.presentation_length)(flattened_obs)
        flattened_lengths = jnp.sum(flattened_lengths, axis=-1)

        lengths_v_time = flattened_lengths.reshape((self.horizon_length, self.n_eval))
        min_lengths_over_time = jnp.min(lengths_v_time, axis=0)

        solved = min_lengths_over_time == 2
        n_solved = jnp.sum(solved).astype(jnp.int32)

        solved_mask_t = (lengths_v_time == 2)  # (T, B)
        is_solved = jnp.any(solved_mask_t, axis=0)
        _first_solved_idx = jnp.argmax(solved_mask_t, axis=0)  # (B,)
        time_idx = jnp.arange(self.horizon_length)[:, None]

        end_idx = jnp.where(is_solved, _first_solved_idx, self.horizon_length - 1)
        reward_mask = time_idx <= end_idx[None,:]
        episode_returns = jnp.sum(eval_rollout.reward * reward_mask, axis=0)  # (B,)
        mean_terminal_return = jnp.sum(is_solved * episode_returns) / jnp.maximum(n_solved, 1)
        max_reward = jnp.max(eval_rollout.reward)

        return {"solved_rate_eval": jnp.mean(solved), "n_solved_eval": n_solved,
                "avg_min_length_eval": jnp.mean(min_lengths_over_time),
                "max_reward_eval": max_reward,
                "mean_terminal_return_eval": mean_terminal_return}