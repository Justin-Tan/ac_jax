import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
from jax import lax, random, vmap, jit

import optax
import distrax
import numpy as np

import mctx
import chex
from chex import dataclass
from typing import Tuple, NamedTuple, Dict, Optional, List
import jaxtyping

import functools
from tqdm import tqdm
from collections import defaultdict

from ac_jax import agents, logging
from ac_jax.env import ac_env, curriculum, types, utils    

class PPOTrainer:

    def __init__(self, config):
        self.config = config
        self.init_env()

        self.value_classification_loss = False
        self.n_value_bins = 1
        if self.value_classification_loss is True:
            self.n_value_bins = self.config.n_value_bins
            self.logit_transform = utils.HLGauss(min_value=0., max_value=1., n_bins=self.n_value_bins)

        self.init_network_and_optimiser()
        self.num_updates = int(config.total_timesteps // (config.rollout_steps * config.num_envs))
        self.minibatch_size = (config.rollout_steps * config.num_envs) // config.num_minibatches
        self.batch_size = self.minibatch_size * config.num_minibatches

        print(f'Total updates: {self.num_updates}, minibatch_size: {self.minibatch_size}, '
              f'batch size: {self.batch_size}')
        
        self.top_k_eval = True
        if self.top_k_eval is True:
            self.eval_fn = functools.partial(self.evaluate_batch_top_k, k=self.config.top_k)
        else:
            self.eval_fn = self._evaluate_batch


    def init_env(self):
        initial_presentations = np.load(self.config.initial_pool)
        ac_env_conf = ac_env.ACEnvConfig(horizon_length=self.config.horizon_length)
        _env = ac_env.ACEnv(ac_env_conf, initial_presentations=initial_presentations)
        self.env = curriculum.BatchedVmapCurriculumAutoResetWrapper(_env)
        self.n_presentations, self.obs_dim = initial_presentations.shape
        self.max_relator_length = _env.max_relator_length
        self.evaluate_agent = True
        self.n_eval = self.n_presentations
        self.eval_env = _env

    def init_network_and_optimiser(self) -> None:
        if self.config.model_type == 'mlp':
            self.model = agents.ActorCriticIndependent(
                n_actions= self.env.n_actions,
                actor_layers=self.config.actor_layers,
                critic_layers=self.config.critic_layers)
        elif self.config.model_type == 'conv':
            self.model = agents.ActorCritic(n_actions=self.env.n_actions, 
                                            encoder_type="conv")
        elif self.config.model_type == 'transformer':
            self.model = agents.ActorCriticTransformer(n_actions=self.env.n_actions, 
                                                       max_rel_length=self.max_relator_length,
                                                       n_value_bins=self.n_value_bins)
        else:
            raise ValueError(f'Unknown model type: {self.config.model_type}')
        
        if self.config.anneal_lr is True:
            self.optimiser = optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(learning_rate=self._linear_schedule))
        else:
            self.optimiser = optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(learning_rate=self.config.learning_rate))
        
    def init_params(self, key) -> types.ParamsState:
        key, _key, __key = random.split(key, 3)
        dummy_obs = jnp.ones((1, self.env.obs_dim))
        print(self.model.tabulate(__key, dummy_obs))

        model_params = self.model.init(_key, dummy_obs)['params']
        opt_state = self.optimiser.init(model_params)
        params_state = types.ParamsState(params=model_params, opt_state=opt_state, 
                                         update_count=jnp.array(0,int))
        return params_state
    
    def _linear_schedule(self, count: int) -> float:
        """Learning rate schedule."""
        frac = (1.0 - (count // (self.config.num_minibatches * self.config.update_epochs))
            / self.num_updates)
        return self.config.learning_rate * jnp.maximum(frac,1e-1)
    
    def _create_curriculum_state(self):
        curriculum_state = curriculum.create_curriculum_state_batched(self.n_presentations, 
                                                                      self.env.horizon_length)
        return curriculum_state

    def _env_step(self, params, carry, key):
        """Single parallelised environment step. Handles auto-reset + bootstrapping."""
        acting_state, curriculum_state = carry
        timestep = acting_state.timestep
        key, subkey = random.split(key)
        key_envs = random.split(subkey, self.config.num_envs)

        obs = timestep.observation.presentation
        actions, (log_prob, logits, values) = vmap(self._apply_policy, 
            in_axes=(0, None, 0))(obs, params, key_envs)
        next_state, next_timestep, next_curriculum_state = self.env.step(
            acting_state.state, actions, curriculum_state)

        num_episodes_done = next_timestep.last().sum().astype(jnp.int32)
        num_env_steps = actions.shape[0]
        episode_end_flag, next_discount = next_timestep.last(), next_timestep.discount.astype(jnp.int32)
        terminal_flag = jnp.logical_and(episode_end_flag, ~next_discount)
        truncation_flag = jnp.logical_and(episode_end_flag, next_discount)

        # bootstrap values if truncated for rollout mid-stream
        next_observation = next_timestep.observation.presentation
        bootstrap_observation = next_timestep.extras["terminal_obs"].presentation
        next_observation = jnp.where(jnp.expand_dims(truncation_flag, axis=-1), bootstrap_observation, next_observation)

        _, bootstrap_values = vmap(self.model.apply, in_axes=(None, 0))({'params': params}, next_observation)
        if self.value_classification_loss is True:
            bootstrap_values = vmap(self.logit_transform.to_scalar)(bootstrap_values)
        bootstrap_values = jnp.squeeze(bootstrap_values)

        episode_return = jnp.where(episode_end_flag, next_timestep.extras["return"], -100.)
        extras: Dict = {}
        extras.update({"episode_return": episode_return, "episode_done": episode_end_flag})
        extras.update({"terminal": terminal_flag, "truncation": truncation_flag,
                       "bootstrap_values": bootstrap_values})

        acting_state = types.ActingState(
            state=next_state, timestep=next_timestep, key=key,
            episode_count=acting_state.episode_count + num_episodes_done,
            env_step_count=acting_state.env_step_count + num_env_steps)
        
        transition = types.Transition(
            observation=obs, action=actions, value=values,
            reward=next_timestep.reward,
            discount=next_timestep.discount,
            next_observation=next_timestep.observation.presentation,
            log_prob=log_prob, logits=logits, done=episode_end_flag, 
            extras=extras)
        
        return (acting_state, next_curriculum_state), transition
    
    def rollout(self, acting_state, curriculum_state, params):
        """Collect config.rollout_steps * config.num_envs transition data.
           Data has shape (T,B,obs_dim) where T = rollout_steps, B = num_envs."""
        env_step_fn = functools.partial(self._env_step, params)
        acting_keys = jax.random.split(acting_state.key, self.config.rollout_steps)
        carry = (acting_state, curriculum_state)
        (acting_state, curriculum_state), data = jax.lax.scan(env_step_fn, carry, acting_keys)
        return (acting_state, curriculum_state), data  # time major (T, B, ...)

    def _apply_policy(self, observation, params, key, stochastic=True, return_entropy=False):
        # vmap this
        logits, values = self.model.apply({'params': params}, observation)
        if self.value_classification_loss is True:
            values = self.logit_transform.to_scalar(values)
        pi = distrax.Categorical(logits=logits, dtype=int)
        if stochastic:
            action = pi.sample(seed=key)
            log_prob = pi.log_prob(action)
        else:  # greedy
            action = pi.mode()
            log_prob = jnp.zeros_like(pi.log_prob(action))
        if return_entropy is True:
            entropy = pi.entropy()
            return action, (log_prob, logits, values, entropy)
        return action, (log_prob, logits, values)
    
    def _compute_gae(self, rollout_data, final_value):
        """Compute GAE-Lambda advantage by scanning over rollout data."""
        gamma_lambda = self.config.gamma * self.config.gae_lambda

        def _advantage_comp(carry, transition):
            gae, next_value = carry
            next_value = jnp.where(transition.extras["truncation"], transition.extras["bootstrap_values"],
                                   next_value)
            delta = (transition.reward + self.config.gamma * next_value * (1 - transition.extras["terminal"])
                     - transition.value)

            gae = delta + gamma_lambda * gae * (1 - transition.done)
            return (gae, transition.value), gae
        init_carry = (jnp.zeros_like(final_value), final_value)
        _, advantages = lax.scan(_advantage_comp, init_carry, rollout_data, 
                                 unroll=self.config.gae_unrolls, reverse=True)
        targets = advantages + rollout_data.value
        return advantages, targets
    
    def _get_value_loss_mse(self, values, targets, previous_values):
        # critic loss
        value_loss = jnp.square(values - targets)
        residual_t = values - previous_values
        values_clipped = previous_values + jnp.clip(residual_t, a_min=-self.config.clip_eps, 
                                                      a_max=self.config.clip_eps)
        value_loss_clipped = jnp.square(values_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped)
        value_loss = jnp.mean(value_loss)
        return value_loss

    def _get_value_loss_cross_entropy(self, value_logits, targets):
        target_probs = jax.lax.stop_gradient(vmap(self.logit_transform.to_probabilities)(
            targets))
        log_value_probs = jax.nn.log_softmax(value_logits, axis=-1)
        xentropy = -jnp.sum(target_probs * log_value_probs, axis=-1)
        return jnp.mean(xentropy)

    def _loss_fn(self, params, rollout_data, advantages, targets):
        # network predictions on shuffled rollout data
        # rerun to get gradients
        observations, actions = rollout_data.observation, rollout_data.action
        logits, values = vmap(self.model.apply, in_axes=(None, 0))(
            {'params': params}, observations)
        values = jnp.squeeze(values)
        pi = distrax.Categorical(logits=logits, dtype=int)
        log_prob, entropy = pi.log_prob(actions), jnp.mean(pi.entropy())

        if self.value_classification_loss is True:
            value_loss = self._get_value_loss_cross_entropy(values, targets)
        else:
            value_loss = self._get_value_loss_mse(values, targets, rollout_data.value)

        if self.value_classification_loss is True:
            values = vmap(self.logit_transform.to_scalar)(values)

        # actor loss
        log_ratio = log_prob - rollout_data.log_prob
        is_ratio = jnp.exp(log_ratio)
        gae_norm = jax.nn.standardize(advantages)
        raw_actor_loss = is_ratio * gae_norm
        is_ratio_clipped = jnp.clip(is_ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
        actor_loss_clipped = is_ratio_clipped * gae_norm
        actor_loss = -jnp.minimum(raw_actor_loss, actor_loss_clipped)
        actor_loss = jnp.mean(actor_loss)

        # KL(\pi_{\theta} \Vert \pi_{\old})
        kl_estimator = (is_ratio - 1.) - log_ratio
        kl_estimate = jnp.mean(kl_estimator)

        total_loss = (actor_loss 
                      + self.config.value_coeff * value_loss
                      - self.config.entropy_coeff * entropy)

        episode_returns = rollout_data.extras['episode_return'] 
        episode_done = rollout_data.extras['episode_done']
        completed_returns = jnp.where(episode_done, episode_returns, jnp.nan)
        episodes_completed = jnp.sum(episode_done)
        mean_episode_return = jnp.where(episodes_completed > 0, 
                                        jnp.nanmean(completed_returns), -jnp.inf)


        metrics: Dict = {}
        metrics.update({"total_loss": total_loss, "actor_loss": actor_loss,
                        "value_loss": value_loss, "entropy": entropy, "value": values,
                        "log_prob": log_prob, "advantage": advantages, "targets": targets,
                        "advantage_normalised": gae_norm, "kl_policy": kl_estimate})
                        # "mean_episode_return": mean_episode_return, "episodes_completed": episodes_completed})

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return total_loss, metrics
    
    def _update_minibatch(self, params_state, batch_data):
        rollout_data, advantages, targets = batch_data
        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        (total_loss, metrics), grads = grad_fn(
            params_state.params, rollout_data, advantages, targets)
        updates, opt_state = self.optimiser.update(grads, params_state.opt_state)
        params = optax.apply_updates(params_state.params, updates)

        params_state = types.ParamsState(
            params=params, opt_state=opt_state, update_count=params_state.update_count + 1)
        return params_state, metrics
    
    def _update_epoch(self, epoch_data, key):
        key, subkey = random.split(key)
        params_state, batch_data = epoch_data
        
        # shuffle collected data
        perm = jax.random.permutation(subkey, self.batch_size)
        batch = jax.tree_util.tree_map(lambda x: x.reshape((self.batch_size,)+ x.shape[2:]),
                                        batch_data)
        batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((self.config.num_minibatches, -1) + x.shape[1:]), batch)
        params_state, metrics = jax.lax.scan(
            self._update_minibatch, params_state, minibatches)
        epoch_data = (params_state, batch_data)
        return epoch_data, metrics

    @functools.partial(jit, static_argnums=0)
    def _update_step(self, carry, sentinel):
        """
        Returns `agent_state` and `curriculum_state` after `self.update_epochs` passes over 
        collected rollout data holding `self.num_envs * self.rollout_steps` transitions.
        """
        agent_state, curriculum_state = carry

        # collect rollout trajectories
        (acting_state, curriculum_state), rollout_data = self.rollout(
            agent_state.acting_state, curriculum_state, agent_state.params_state.params)
        
        # compute GAE advantages
        # bootstrapping for boundaries of rollout stream
        last_observation = jax.tree_util.tree_map(lambda x: x[-1], rollout_data.next_observation)
        end_state_flag = acting_state.timestep.last()  # if false, is transition. If true, either terminal or trunc
        truncation_flag = jnp.logical_and(end_state_flag, acting_state.timestep.discount)
        bootstrap_observation = acting_state.timestep.extras["terminal_obs"].presentation
        # terminal state GAE computation is masked; this only applies to truncated states
        last_observation = jnp.where(jnp.expand_dims(truncation_flag, axis=-1), bootstrap_observation, last_observation)

        last_logits, last_values = vmap(self.model.apply, in_axes=(None, 0))(
            {'params': agent_state.params_state.params}, last_observation)
        if self.value_classification_loss is True:
            last_values = vmap(self.logit_transform.to_scalar)(last_values)
        last_values = jnp.squeeze(last_values)
        advantages, targets = self._compute_gae(rollout_data, last_values)

        # shuffle, split into minibatches and update agent network
        key, *epoch_keys = jax.random.split(acting_state.key, self.config.update_epochs + 1)
        epoch_keys = jnp.stack(epoch_keys)
        batch_data = (rollout_data, advantages, targets)
        epoch_data = (agent_state.params_state, batch_data)
        (params_state, _), metrics = jax.lax.scan(self._update_epoch, epoch_data, epoch_keys)
        flattened_rollout_data = jax.tree_util.tree_map(lambda x: x.reshape((self.batch_size,)+ x.shape[2:]),
                                        batch_data[0])
        # metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=-1), metrics)  # (n_epochs, ...)

        mean_reward = jnp.mean(flattened_rollout_data.reward)
        min_reward, max_reward = jnp.min(flattened_rollout_data.reward), jnp.max(flattened_rollout_data.reward)
        n_terminal = jnp.sum(flattened_rollout_data.done)
        terminal_rewards = flattened_rollout_data.reward * flattened_rollout_data.done
        mean_terminal_reward = jnp.where(n_terminal > 0, jnp.sum(terminal_rewards) / n_terminal, 0.0)
        relator_lengths = vmap(utils.presentation_length)(flattened_rollout_data.observation)
        lengths = jnp.sum(relator_lengths, axis=-1)

        extras = flattened_rollout_data.extras
        episode_returns = extras['episode_return'] 
        episode_done = extras['episode_done'].astype(jnp.int32)
        completed_returns = jnp.where(episode_done, episode_returns, jnp.nan)
        episodes_completed = jnp.sum(episode_done)

        mean_episode_return = jnp.where(episodes_completed > 0,
                                        jnp.sum(episode_returns * episode_done) / episodes_completed,
                                        0.)

        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
        metrics.update({"episodes": acting_state.episode_count.astype(jnp.int32),
                        "env_steps": acting_state.env_step_count.astype(jnp.int32),
                        "solved": jnp.sum(curriculum_state.solved_mask).astype(jnp.int32),
                        "mean_reward": mean_reward, "min_reward": min_reward, "max_reward": max_reward,
                        "mean_terminal_reward": mean_terminal_reward, "mean_length": jnp.mean(lengths),
                        "min_length": jnp.min(lengths), "max_length": jnp.max(lengths),
                        "mean_episode_return": mean_episode_return,
                        "episodes_completed": episodes_completed.astype(jnp.int32)})

        acting_state = acting_state._replace(key=key)
        agent_state = types.AgentState(params_state=params_state, acting_state=acting_state)
        return (agent_state, curriculum_state), metrics

    def recurrent_fn(self, params, rng, action, embedding):
        # Simulates one environment step inside MCTS phase.
        state = embedding
        next_state, next_timestep = vmap(self.eval_env.step)(
            state, action)
        obs, next_obs = state.presentation, next_state.presentation
        logits, values = vmap(self.model.apply, in_axes=(None,0))(
            {'params': params}, next_obs)
        if self.value_classification_loss is True:
            values = vmap(self.logit_transform.to_scalar)(values)

        reward = next_timestep.reward
        discount = next_timestep.discount

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward, discount=discount,
            prior_logits=logits, value=values)
        return recurrent_fn_output, next_state

    @functools.partial(jit, static_argnums=(0, 3, 4))
    def _evaluate_batch_mcts(self, params, key, n_simulations=256,
                             max_depth=None):
        # evaluate n_eval environments in parallel
        eval_idx = jnp.arange(self.n_eval)
        key, subkey, _subkey, __subkey = random.split(key, 4)
        reset_keys = random.split(__subkey, self.n_eval)
        states, timesteps = vmap(self.eval_env.reset_to_idx)(reset_keys, eval_idx)

        def _eval_step(carry, sentinel):
            state, timestep, key = carry
            key, mcts_key = random.split(key)
            obs = timestep.observation.presentation
            root_logits, root_value = vmap(self.model.apply, in_axes=(None,0))(
                {'params': params}, obs)
            if self.value_classification_loss is True:
                root_value = vmap(self.logit_transform.to_scalar)(root_value)

            root = mctx.RootFnOutput(
                prior_logits=root_logits,
                value=root_value,
                embedding=state)
            
            #action selection
            policy_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=mcts_key,
                root=root,
                recurrent_fn=self.recurrent_fn,
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

    @functools.partial(jit, static_argnums=(0,3,4))
    def _evaluate_batch(self, params, key, stochastic=False,
                        evaluate_top_k=False):
        # deterministic eval for held-out examples
        eval_idx = jnp.arange(self.n_eval)
        keys = random.split(key, self.n_eval)
        states, timesteps = vmap(self.eval_env.reset_to_idx)(keys, eval_idx)

        def _eval_step(carry, sentinel):
            state, timestep = carry
            obs = timestep.observation.presentation
            eval_keys = state.key
            actions, _ = vmap(self._apply_policy, 
                in_axes=(0, None, 0, None))(obs, params, eval_keys, stochastic)
            next_state, next_timestep = vmap(self.eval_env.step)(state, actions)
            return (next_state, next_timestep), next_timestep
        
        # run over full horizon and check if solved at any step
        init_carry = (states, timesteps)
        (final_state, final_timestep), eval_rollout = jax.lax.scan(
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

        if evaluate_top_k is True:
            return {"is_solved": is_solved, "min_lengths": min_lengths_over_time,
                    "episode_returns": episode_returns}

        return {"solved_rate_eval": jnp.mean(solved), "n_solved_eval": n_solved,
                "avg_min_length_eval": jnp.mean(min_lengths_over_time),
                "max_reward_eval": max_reward,
                "mean_terminal_return_eval": mean_terminal_return}            

    @functools.partial(jit, static_argnums=(0,3))
    def evaluate_batch_top_k(self, params, key, k=8):

        keys = random.split(key, k)
        metrics = vmap(self._evaluate_batch, in_axes=(None, 0, None, None))(
            params, keys, True, True)
        
        is_solved = metrics["is_solved"]  # (k, B)
        min_lengths = metrics["min_lengths"]  # (k, B)
        episode_returns = metrics["episode_returns"]  # (k, B)

        any_solved = jnp.any(is_solved, axis=0)  # (B,)
        min_lengths = jnp.min(min_lengths, axis=0)
        max_returns = jnp.max(episode_returns, axis=0)
        max_reward = jnp.max(episode_returns)  # max given on completion

        solved_at_k = jnp.sum(any_solved).astype(jnp.int32)
        solved_at_k_rate = jnp.mean(any_solved)
        mean_at_k_min_length = jnp.mean(min_lengths)
        mean_at_k_max_return = jnp.mean(max_returns)

        return {"solved_rate_eval": solved_at_k_rate, "n_solved_eval": solved_at_k,
                "avg_min_length_eval": mean_at_k_min_length,
                "max_reward_eval": max_reward, 
                "mean_terminal_return_eval": mean_at_k_max_return}

    def train(self, key):
        key, init_key, env_key, acting_key = jax.random.split(key, 4)
        params_state = self.init_params(init_key)
        curriculum_state = self._create_curriculum_state()
        state, timestep, curriculum_state = self.env.reset(env_key, curriculum_state,
                                                           self.config.num_envs)

        acting_state = types.ActingState(state, timestep, acting_key,
                                         episode_count=jnp.array(0, int),
                                         env_step_count=jnp.array(0, int))
        
        agent_state = types.AgentState(params_state=params_state, acting_state=acting_state)
        carry = (agent_state, curriculum_state)
        carry, metrics = jax.lax.scan(self._update_step, carry, None, self.num_updates)
        agent_state, curriculum_state = carry
        return (agent_state, curriculum_state), metrics

    def train_and_log(self, key, checkpoint_path=None):
        key, init_key, env_key, acting_key = jax.random.split(key, 4)
        
        params_state = self.init_params(init_key)
        curriculum_state = self._create_curriculum_state()
        start_epoch, env_step_count = 0, 0
        if checkpoint_path is not None:
            params, opt_state, curriculum_state, start_epoch, env_step_count, _ = logging.load_checkpoint(
                checkpoint_path, params_state.params, params_state.opt_state)
            params_state = types.ParamsState(
                params=params,
                opt_state=opt_state,
                update_count=jnp.array(start_epoch * self.config.update_epochs * self.config.num_minibatches, int))
            start_epoch = start_epoch + 1  # Resume from next epoch
        
        # Initialise environment
        state, timestep, curriculum_state = self.env.reset(
            env_key, curriculum_state, self.config.num_envs)

        acting_state = types.ActingState(state, timestep, acting_key,
                                         episode_count=jnp.array(0, int),
                                         env_step_count=jnp.array(env_step_count, int))
        
        agent_state = types.AgentState(params_state=params_state, acting_state=acting_state)
        carry = (agent_state, curriculum_state)
        
        # Training loop with logging and checkpointing
        all_metrics: List[Dict] = []
        storage = defaultdict(list)
        
        with logging.TerminalLogger(name="ppo-train") as logger:
            pbar = tqdm(range(start_epoch, self.num_updates), desc="train", 
                        initial=start_epoch, total=self.num_updates, colour="blue", mininterval=0.2)
            
            for epoch in pbar:
                # Single update step
                carry, metrics = self._update_step(carry, None)
                agent_state, curriculum_state = carry
                
                if epoch % self.config.log_interval == 0:
                    metrics_to_log = logging.first_from_device(metrics)
                    log_metrics = metrics_to_log
                    logger.write(log_metrics, label="train", step=epoch, storage=storage)
                    print(curriculum_state)
                    pbar.set_postfix({
                        'loss': f"{log_metrics['total_loss']:.4f}",
                        'solved': int(log_metrics['solved'])}, refresh=False)
                    
                    if self.evaluate_agent is True:
                        eval_metrics = self.eval_fn(agent_state.params_state.params, 
                                        agent_state.acting_state.key)
                        eval_metrics = logging.first_from_device(eval_metrics)
                        logger.write(eval_metrics, label="eval", step=epoch, storage=storage,
                                     save_flag=(epoch % self.config.save_interval == 0))
            
                        log_metrics = log_metrics.update(eval_metrics)

                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    logging.save_checkpoint(
                        path=self.config.checkpoint_dir,
                        params=agent_state.params_state.params,
                        opt_state=agent_state.params_state.opt_state,
                        curriculum_state=curriculum_state,
                        epoch=epoch,
                        env_step_count=int(agent_state.acting_state.env_step_count),
                        metrics=log_metrics,
                        storage=storage,
                        config=self.config())
                
                all_metrics.append(metrics)
        
        # Stack metrics for consistent return type
        stacked_metrics = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *all_metrics)
        logging.save_checkpoint(
            path=self.config.checkpoint_dir,
            params=agent_state.params_state.params,
            opt_state=agent_state.params_state.opt_state,
            curriculum_state=curriculum_state,
            epoch=epoch,
            env_step_count=int(agent_state.acting_state.env_step_count),
            metrics=log_metrics,
            storage=storage,
            config=self.config(),
            name='FIN')
        
        return (agent_state, curriculum_state), stacked_metrics
    
class PPOTrainerCurriculum(PPOTrainer):

    def init_env(self):
        curriculum_batch = utils.load_dataclass_dict(self.config.curriculum_data_path, types.CurriculumBatch,
                                                    ['easy', 'medium', 'hard'])
        ac_env_conf = ac_env.ACEnvConfig(horizon_length=self.config.horizon_length)
        initial_presentations =  np.vstack([curriculum_batch[k].presentations for k in curriculum_batch.keys()])
        initial_presentations_eval = np.load(self.config.initial_pool)
        _env = ac_env.ACEnv(ac_env_conf, initial_presentations=initial_presentations)
        self.eval_env = ac_env.ACEnv(ac_env_conf, initial_presentations=initial_presentations_eval)

        self.tier_counts = [curriculum_batch[k].n.shape[0] for k in curriculum_batch.keys()]
        self.env = curriculum.BatchedMultiLevelVmapCurriculumAutoResetWrapper(_env, self.tier_counts)
        self.n_presentations, self.obs_dim = initial_presentations.shape
        self.max_relator_length = _env.max_relator_length
        self.n_eval = initial_presentations_eval.shape[0]
        print(f"Curriculum data counts: easy: {self.tier_counts[0]}, medium: {self.tier_counts[1]}, hard: {self.tier_counts[2]}")
        print(f"Evaluation pool size: {self.n_eval}")
        self.evaluate_agent = True
        

    def _create_curriculum_state(self):
        curriculum_state = curriculum.create_multilevel_curriculum(self.tier_counts,
                                                                   self.env.horizon_length)
        return curriculum_state
    
    @functools.partial(jit, static_argnums=0)
    def _update_step(self, carry, sentinel):
        """
        Returns `agent_state` and `curriculum_state` after `self.update_epochs` passes over 
        collected rollout data holding `self.num_envs * self.rollout_steps` transitions.
        """
        agent_state, curriculum_state = carry

        # collect rollout trajectories
        (acting_state, curriculum_state), rollout_data = self.rollout(
            agent_state.acting_state, curriculum_state, agent_state.params_state.params)
        
        # compute GAE advantages
        # bootstrapping for boundaries of rollout stream
        last_observation = jax.tree_util.tree_map(lambda x: x[-1], rollout_data.next_observation)
        end_state_flag = acting_state.timestep.last()  # if false, is transition. If true, either terminal or trunc
        truncation_flag = jnp.logical_and(end_state_flag, acting_state.timestep.discount)
        bootstrap_observation = acting_state.timestep.extras["terminal_obs"].presentation
        # terminal state GAE computation is masked; this only applies to truncated states
        last_observation = jnp.where(jnp.expand_dims(truncation_flag, axis=-1), bootstrap_observation, last_observation)

        last_logits, last_values = vmap(self.model.apply, in_axes=(None, 0))(
            {'params': agent_state.params_state.params}, last_observation)
        if self.value_classification_loss is True:
            last_values = vmap(self.logit_transform.to_scalar)(last_values)
        last_values = jnp.squeeze(last_values)
        advantages, targets = self._compute_gae(rollout_data, last_values)

        # shuffle, split into minibatches and update agent network
        key, *epoch_keys = jax.random.split(acting_state.key, self.config.update_epochs + 1)
        epoch_keys = jnp.stack(epoch_keys)
        batch_data = (rollout_data, advantages, targets)
        epoch_data = (agent_state.params_state, batch_data)
        (params_state, _), metrics = jax.lax.scan(self._update_epoch, epoch_data, epoch_keys)
        flattened_rollout_data = jax.tree_util.tree_map(lambda x: x.reshape((self.batch_size,)+ x.shape[2:]),
                                        batch_data[0])
        # metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=-1), metrics)  # (n_epochs, ...)

        mean_reward = jnp.mean(flattened_rollout_data.reward)
        min_reward, max_reward = jnp.min(flattened_rollout_data.reward), jnp.max(flattened_rollout_data.reward)
        n_terminal = jnp.sum(flattened_rollout_data.done)
        terminal_rewards = flattened_rollout_data.reward * flattened_rollout_data.done
        mean_terminal_reward = jnp.where(n_terminal > 0, jnp.sum(terminal_rewards) / n_terminal, 0.0)
        relator_lengths = vmap(utils.presentation_length)(flattened_rollout_data.observation)
        lengths = jnp.sum(relator_lengths, axis=-1)

        extras = flattened_rollout_data.extras
        episode_returns = extras['episode_return'] 
        episode_done = extras['episode_done'].astype(jnp.int32)
        episodes_completed = jnp.sum(episode_done)

        mean_episode_return = jnp.where(episodes_completed > 0,
                                        jnp.sum(episode_returns * episode_done) / episodes_completed, 0.)

        solved_easy = jnp.sum(curriculum_state.easy.solved_mask).astype(jnp.int32)
        solved_med = jnp.sum(curriculum_state.medium.solved_mask).astype(jnp.int32)
        solved_hard = jnp.sum(curriculum_state.hard.solved_mask).astype(jnp.int32)
        total_solved = solved_easy + solved_med + solved_hard

        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
        metrics.update({"episodes": acting_state.episode_count.astype(jnp.int32),
                        "env_steps": acting_state.env_step_count.astype(jnp.int32),
                        "solved": total_solved,
                        "solved_easy": solved_easy, "solved_medium": solved_med, "solved_hard": solved_hard,
                        "solved_easy_rate": curriculum_state.tier_success_rates[0],
                        "solved_medium_rate": curriculum_state.tier_success_rates[1],
                        "solved_hard_rate": curriculum_state.tier_success_rates[2],
                        "curriculum_lambda": curriculum_state.curriculum_lambda,
                        "mean_reward": mean_reward, "min_reward": min_reward, "max_reward": max_reward,
                        "mean_terminal_reward": mean_terminal_reward, "mean_length": jnp.mean(lengths),
                        "min_length": jnp.min(lengths), "max_length": jnp.max(lengths),
                        "mean_episode_return": mean_episode_return,
                        "episodes_completed": episodes_completed.astype(jnp.int32)})

        acting_state = acting_state._replace(key=key)
        agent_state = types.AgentState(params_state=params_state, acting_state=acting_state)
        return (agent_state, curriculum_state), metrics    

if __name__ == "__main__":
    import argparse
    MODEL_TYPES = ['transformer', 'mlp', 'conv']
    parser = argparse.ArgumentParser(description='RL agent training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--initial-pool', type=str, 
                       default="data/initial_presentations.npy",
                       help='Path to initial pool data file')
    parser.add_argument('--curriculum-data-path', type=str,
                       help='Path to curriculum dataset', required=True)
    parser.add_argument('--curriculum-train', action='store_true',
                       help='Enable curriculum training')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                       help='Learning rate')
    parser.add_argument('--num-envs', type=int, default=1024,
                       help='Number of parallel environments')
    parser.add_argument('--model-type', type=str,
                    choices=MODEL_TYPES,
                    default='mlp',
                    help=f'Model architecture type. Choices: {", ".join(MODEL_TYPES)}')
    parser.set_defaults(curriculum_train=True)
    args = parser.parse_args()

    @dataclass
    class config(object):
        learning_rate: float = args.learning_rate
        num_envs: int = args.num_envs  # parallel envs - increase this to be 1024 at least!
        rollout_steps: int = 256  # steps per env, no. transitions = num_envs * rollout_steps
        total_timesteps: float = 5e8  # stopping criterion; total no. of env steps
        update_epochs: int = 2  # 1 epoch - full pass through rollout data
        num_minibatches: int = 128  # rollout data splits
        horizon_length: int = 256  # max length of an episode
        gamma: float = 0.999
        gae_lambda: float = 0.99
        gae_unrolls: int = 64
        clip_eps: float = 0.2
        entropy_coeff: float = 0.01
        value_coeff: float = 0.5
        target_kl: float = 0.01
        max_grad_norm: float = 0.5
        activation: str = "tanh"
        anneal_lr: bool = True
        n_value_bins: int = 128

        actor_layers = (512, 512)
        critic_layers = (512, 512)

        log_interval: int = 8  # log every n epochs
        save_interval: int = 256  # save logs every n epochs
        checkpoint_interval: int = 1024  # save checkpoint every n epochs
        checkpoint_dir: str = "checkpoints"
        top_k = 32

        initial_pool = args.initial_pool
        curriculum_data_path = args.curriculum_data_path
        curriculum_train: bool = args.curriculum_train
        model_type = args.model_type

    rng = jax.random.PRNGKey(42)
    if config.curriculum_train is True:
        trainer = PPOTrainerCurriculum(config)
    else:
        trainer = PPOTrainer(config)

    (agent_state, curriculum_state), metrics = trainer.train_and_log(rng)
