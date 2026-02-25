import jax
import jax.numpy as jnp
from jax import lax, random, vmap, jit

import functools

import mctx
from ac_jax import agents, logging, ppo_train
from ac_jax.env import ac_env, curriculum, types, utils    

class HeuristicCallback:
    """Mutable holder: same Python object identity so JAX reuses cached compilation.
    Heuristic can be swapped freely without triggering recompilation."""
    
    def __init__(self, fn=None, batch_size=None):
        self.fn = fn
        self.batch_size = batch_size
    
    def update(self, new_heuristic_fn):
        """Call this before each evaluation with the new evolved heuristic."""
        self.fn = jax.jit(jax.vmap(new_heuristic_fn))
    
    def __call__(self, presentations):
        """Called by XLA at runtime via pure_callback."""
        return self.fn(presentations)


class PPOTrainerCurriculumEval(ppo_train.PPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heuristic_cb = HeuristicCallback()

    def _eval_rollout(self, params, key, indices, stochastic):

        batch_size = indices.shape[0]
        keys = random.split(key, batch_size)

        states, timesteps = vmap(self.eval_env.reset_to_idx)(keys, indices)
        _, initial_values = vmap(self.model.apply, in_axes=(None, 0))(
                {"params": params}, timesteps.observation.presentation)

        def _eval_step(carry, sentinel):
            state, timestep = carry
            obs = timestep.observation.presentation
            eval_keys = state.key
            actions, (log_prob, logits, values, entropy) = vmap(self._apply_policy,
                in_axes=(0, None, 0, None, None))(obs, params, eval_keys, stochastic, True)
            next_state, next_timestep = vmap(self.eval_env.step)(state, actions)
            return (next_state, next_timestep), (next_timestep, actions, values, entropy)

        # run over full horizon and check if solved at any step
        init_carry = (states, timesteps)
        (final_state, final_timestep), (eval_rollout, action_history, value_history, entropy_history) = jax.lax.scan(
            _eval_step, init_carry, None, self.config.horizon_length)

        # eval_rollout: (T, B, ...)
        flattened_rollout_data = jax.tree_util.tree_map(lambda x: x.reshape((-1,)+ x.shape[2:]),
                                                        eval_rollout)

        flattened_obs = flattened_rollout_data.observation.presentation
        flattened_lengths = vmap(utils.presentation_length)(flattened_obs)
        flattened_lengths = jnp.sum(flattened_lengths, axis=-1)

        lengths_v_time = flattened_lengths.reshape((self.config.horizon_length, batch_size))
        min_lengths_over_time = jnp.min(lengths_v_time, axis=0)
        max_lengths_over_time = jnp.max(lengths_v_time, axis=0)

        solved_mask = min_lengths_over_time == 2
        n_solved = jnp.sum(solved_mask).astype(jnp.int32)

        solved_mask_t = (lengths_v_time == 2)  # (T, B)
        is_solved = jnp.any(solved_mask_t, axis=0)
        _first_solved_idx = jnp.argmax(solved_mask_t, axis=0)  # (B,)
        time_idx = jnp.arange(self.config.horizon_length)[:, None]

        end_idx = jnp.where(is_solved, _first_solved_idx, self.config.horizon_length - 1)
        reward_mask = time_idx <= end_idx[None,:]
        episode_returns = jnp.sum(eval_rollout.reward * reward_mask, axis=0)  # (B,)
        mean_terminal_return = jnp.sum(is_solved * episode_returns) / jnp.maximum(n_solved, 1)
        max_reward = jnp.max(eval_rollout.reward)

        action_counts = jnp.sum(jax.nn.one_hot(action_history, self.eval_env.n_actions), axis=0)

        return {"solved_mask": solved_mask,  # B
                "min_lengths": min_lengths_over_time,  # B
                "returns": episode_returns,  # B
                "end_idx": end_idx,  # B
                "initial_values": initial_values,  # B
                "action_counts": action_counts, # (B, |A|)
                "max_reward": max_reward,
                "value_history": value_history.T,  # (B, T)
                "entropy_history": entropy_history.T,
                "length_history": lengths_v_time.T}  # (B, T)

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
            #logits, values = vmap(self.model.apply, in_axes=(None,0))(
            #    {'params': params}, obs)
            #values = vmap(heuristic_fn)(next_obs)
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
            #root_logits, root_value = vmap(self.model.apply, in_axes=(None,0))(
            #    {'params': params}, obs)
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
            _eval_step, init_carry, None, self.config.horizon_length)

        # eval_rollout: (T, B, ...)
        flattened_rollout_data = jax.tree_util.tree_map(lambda x: x.reshape((-1,)+ x.shape[2:]),
                                                        eval_rollout)

        flattened_obs = flattened_rollout_data.observation.presentation
        flattened_lengths = vmap(utils.presentation_length)(flattened_obs)
        flattened_lengths = jnp.sum(flattened_lengths, axis=-1)

        lengths_v_time = flattened_lengths.reshape((self.config.horizon_length, self.n_eval))
        min_lengths_over_time = jnp.min(lengths_v_time, axis=0)

        solved = min_lengths_over_time == 2
        n_solved = jnp.sum(solved).astype(jnp.int32)

        solved_mask_t = (lengths_v_time == 2)  # (T, B)
        is_solved = jnp.any(solved_mask_t, axis=0)
        _first_solved_idx = jnp.argmax(solved_mask_t, axis=0)  # (B,)
        time_idx = jnp.arange(self.config.horizon_length)[:, None]

        end_idx = jnp.where(is_solved, _first_solved_idx, self.config.horizon_length - 1)
        reward_mask = time_idx <= end_idx[None,:]
        episode_returns = jnp.sum(eval_rollout.reward * reward_mask, axis=0)  # (B,)
        mean_terminal_return = jnp.sum(is_solved * episode_returns) / jnp.maximum(n_solved, 1)
        max_reward = jnp.max(eval_rollout.reward)

        return {"solved_rate_eval": jnp.mean(solved), "n_solved_eval": n_solved,
                "avg_min_length_eval": jnp.mean(min_lengths_over_time),
                "max_reward_eval": max_reward,
                "mean_terminal_return_eval": mean_terminal_return}

    @functools.partial(jit, static_argnums=(0,2,3,4,6))
    def _evaluate_mcts_custom_heuristic(self, key, heuristic_fn, 
                             n_simulations=256, max_depth=None, batch_idx=None,
                             batch_size=None):
        # evaluate batch_size environments in parallel
        key, subkey, _subkey, __subkey = random.split(key, 4)
        reset_keys = random.split(__subkey, batch_size)
        states, timesteps = vmap(self.eval_env.reset_to_idx)(reset_keys, batch_idx)
        # could the evolved heuristic output dummy logits to rank actions?
        dummy_logits = jnp.zeros((batch_size, self.eval_env.n_actions))

        # mctx expects a fixed signature
        def _recurrent_fn(params, rng, action, embedding):
            # Simulates one environment step inside MCTS phase.
            # minimal step implementation
            presentation = embedding
            next_presentation, next_timestep = vmap(self.eval_env._step)(
                presentation, action)
            values = vmap(heuristic_fn)(next_presentation)
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
            root_value = vmap(heuristic_fn)(obs)
            root_logits = dummy_logits  # (B, A)
            root = mctx.RootFnOutput(
                prior_logits=root_logits,
                value=root_value,
                embedding=obs)
            
            #action selection
            policy_output = mctx.gumbel_muzero_policy(
                params=None,
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
            _eval_step, init_carry, None, self.config.horizon_length)
        
        # eval_rollout: (T, B, ...)
        flattened_rollout_data = jax.tree_util.tree_map(lambda x: x.reshape((-1,)+ x.shape[2:]),
                                                        eval_rollout)
        
        flattened_obs = flattened_rollout_data.observation.presentation
        flattened_lengths = vmap(utils.presentation_length)(flattened_obs)
        flattened_lengths = jnp.sum(flattened_lengths, axis=-1)

        lengths_v_time = flattened_lengths.reshape((self.config.horizon_length, batch_size))
        min_lengths_over_time = jnp.min(lengths_v_time, axis=0)

        solved = min_lengths_over_time == 2
        n_solved = jnp.sum(solved).astype(jnp.int32)

        solved_mask_t = (lengths_v_time == 2)  # (T, B)
        is_solved = jnp.any(solved_mask_t, axis=0)
        _first_solved_idx = jnp.argmax(solved_mask_t, axis=0)  # (B,)
        time_idx = jnp.arange(self.config.horizon_length)[:, None]

        end_idx = jnp.where(is_solved, _first_solved_idx, self.config.horizon_length - 1)
        reward_mask = time_idx <= end_idx[None,:]
        episode_returns = jnp.sum(eval_rollout.reward * reward_mask, axis=0)  # (B,)
        mean_terminal_return = jnp.sum(is_solved * episode_returns) / jnp.maximum(n_solved, 1)
        max_reward = jnp.max(eval_rollout.reward)

        return {"solved_rate_eval": jnp.mean(solved), "n_solved_eval": n_solved,
                "avg_min_length_eval": jnp.mean(min_lengths_over_time),
                "max_reward_eval": max_reward,
                "mean_terminal_return_eval": mean_terminal_return,
                "min_lengths": min_lengths_over_time}

    @functools.partial(jit, static_argnums=(0,2,3,4,5))
    def _evaluate_dataset_mcts_batched(self, key, heuristic_fn, batch_size=256,
                             n_simulations=256, max_depth=None):
        import tqdx
        """
        Batchwise evaluation over entire eval dataset using MCTS.
        """
        n_total = self.n_eval
        n_padded = (n_total + batch_size - 1) // batch_size * batch_size
        indices = jnp.arange(n_padded) % n_total
        valid_mask = jnp.arange(n_padded) < n_total

        n_batches = n_padded // batch_size
        batched_indices = indices.reshape(n_batches, batch_size)
        batched_mask = valid_mask.reshape(n_batches, batch_size)
        keys = random.split(key, n_batches)

        def _eval_batch(carry, inputs):
            batch_idx, mask, batch_key = inputs

            batch_metrics = self._evaluate_mcts_custom_heuristic(batch_key, heuristic_fn,
                n_simulations=n_simulations, max_depth=max_depth, batch_idx=batch_idx, batch_size=batch_size)
            
            solved = (batch_metrics["min_lengths"] == 2) * mask
            n_solved = jnp.sum(solved).astype(jnp.int32)
            stats = {
                "n_solved": n_solved,
                "max_reward": jnp.max(batch_metrics["max_reward_eval"] * mask),  # 
            }
            batch_metrics["n_solved_eval"] = n_solved
            new_carry = jax.tree.map(lambda c, s: c + s, carry, stats)
            new_carry["max_reward"] = jnp.maximum(carry["max_reward"], stats["max_reward"])

            _subtotal_metrics = {"idx": batch_idx, "mask": mask}
            _subtotal_metrics.update(batch_metrics)
            return new_carry, _subtotal_metrics

        init_stats = {
            "n_solved": 0, "max_reward": -1e9
        }

        final_stats, total_metrics = tqdx.scan(_eval_batch, init_stats,
            (batched_indices, batched_mask, keys))
        n_solved = final_stats["n_solved"].astype(jnp.int32)

        return final_stats, total_metrics
    
    @functools.partial(jit, static_argnums=(0, 3, 4, 5))
    def evaluate_dataset_batched(self, params, key, batch_size=512, k=16, stochastic=True):
        import tqdx
        """
        Batchwise evaluation over entire eval dataset using top-k sampling.
        """
        n_total = self.n_eval
        n_padded = (n_total + batch_size - 1) // batch_size * batch_size
        indices = jnp.arange(n_padded) % n_total
        valid_mask = jnp.arange(n_padded) < n_total

        n_batches = n_padded // batch_size
        batched_indices = indices.reshape(n_batches, batch_size)
        batched_mask = valid_mask.reshape(n_batches, batch_size)
        keys = random.split(key, n_batches)

        def _eval_batch(carry, inputs):
            batch_idx, mask, batch_key = inputs

            top_k_keys = random.split(batch_key, k)
            metrics_k = vmap(self._eval_rollout, in_axes=(None, 0, None, None))(
                params, top_k_keys, batch_idx, stochastic)  # (K, B,...)

            any_solved = jnp.any(metrics_k["solved_mask"], axis=0) # (B,)
            best_min_len = jnp.min(metrics_k["min_lengths"], axis=0) # (B,)
            best_return = jnp.max(metrics_k["returns"], axis=0) # (B,)
            batch_max_r = jnp.max(metrics_k["max_reward"])

            stats = {
                "n_solved": jnp.sum(any_solved * mask),
                "sum_min_len": jnp.sum(best_min_len * mask),
                "sum_return": jnp.sum(best_return * mask), # Mean over ALL
                "sum_term_return": jnp.sum(best_return * any_solved * mask), # Mean over Solved
                "max_reward": batch_max_r
            }

            new_carry = jax.tree.map(lambda c, s: c + s, carry, stats)
            new_carry["max_reward"] = jnp.maximum(carry["max_reward"], stats["max_reward"])

            _subtotal_metrics = {"idx": batch_idx}
            _subtotal_metrics.update(metrics_k)
            return new_carry, _subtotal_metrics

        init_stats = {
            "n_solved": 0, "sum_min_len": 0,
            "sum_return": 0.0, "sum_term_return": 0.0,
            "max_reward": -1e9
        }

        # final_stats, total_metrics = jax.lax.scan(_eval_batch, init_stats, 
        #     (batched_indices, batched_mask, keys))

        final_stats, total_metrics = tqdx.scan(_eval_batch, init_stats,
            (batched_indices, batched_mask, keys))

        n_solved = final_stats["n_solved"].astype(jnp.int32)

        aggregate_metrics = {
            "solved_rate_eval": final_stats["n_solved"] / n_total,
            "n_solved_eval": n_solved,
            "avg_min_length_eval": final_stats["sum_min_len"] / n_total,
            "max_reward_eval": final_stats["max_reward"],
            "mean_terminal_return_eval": final_stats["sum_term_return"] / jnp.maximum(n_solved, 1)}

        # return aggregate_metrics, total_metrics
        def _reshape(x):
            if len(x.shape) == 3:
                x = jnp.permute_dims(x, (0,2,1))
                x = x.reshape(-1, x.shape[-1])
            elif len(x.shape) == 4:
                x = jnp.permute_dims(x, (0,2,1,3))
                x = x.reshape(-1, x.shape[-2], x.shape[-1])
            elif len(x.shape) == 2:
                x = x.reshape(-1)
            return x[:self.n_eval]

        total_metrics = jax.tree.map(_reshape, total_metrics)

        return aggregate_metrics, total_metrics
