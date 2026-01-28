"""
Logging utilities for JAX PPO training.

Provides TerminalLogger for formatted console output during training.
"""

import time
from typing import Dict, Any, Optional
from contextlib import AbstractContextManager

import jax
import jax.numpy as jnp

import os
import pickle
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

import numpy as np
from flax import serialization

from ac_jax.env.curriculum import CurriculumState

def first_from_device(tree: Any) -> Any:
    # returns pytree with same structure, arrays converted to Python floats
    def convert(x):
        if hasattr(x, 'shape'):
            val = jax.device_get(x)
            if jnp.issubdtype(x.dtype, jnp.integer):
                return int(val)
            else:
                return float(val)
        return x
    return jax.tree_util.tree_map(convert, tree)


def save_logs(storage, name, step, logdir):
    path = os.path.join(logdir, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 
        '{}_{}_{:%Y_%m_%d_%H:%M}_LOG.pkl'.format(name, step, datetime.now())), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TerminalLogger(AbstractContextManager):
    """
    Simple terminal logger for training metrics.
    
    Usage:
        with TerminalLogger(name="PPO") as logger:
            for epoch in range(num_epochs):
                metrics = train_step(...)
                logger.write(metrics, label="train", step=epoch)
    """
    
    def __init__(self, name: str = "Training", time_delta: float = 1.0):
        """
        Args:
            name: Name to display in log header
            time_delta: Minimum seconds between log outputs (rate limiting)
        """
        self.name = name
        self.time_delta = time_delta
        self._start_time: Optional[float] = None
        self._last_log_time: float = 0.0
        
    def __enter__(self) -> "TerminalLogger":
        self._start_time = time.time()
        self._last_log_time = 0.0
        print(f"\n{'='*60}")
        print(f" {self.name} Started")
        print(f"{'='*60}\n")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        print(f"\n{'='*60}")
        print(f" {self.name} Completed in {elapsed:.1f}s")
        print(f"{'='*60}\n")
        return False  # Don't suppress exceptions
    
    def write(
        self, 
        data: Dict[str, Any], 
        label: str, 
        step: int, 
        storage: Dict[str, Any],
        save_flag: bool = False,
        force: bool = False,
        logdir='logs',
    ) -> None:
        """
        Write metrics to terminal, and to logfile.
        
        Args:
            data: Dictionary of metric names to values
            label: Label for this log entry (e.g., "train", "eval")
            step: Current step/epoch number
            force: If True, bypass rate limiting
            save_flag: If True, save logs to disk
        """
        current_time = time.time()
        os.makedirs(logdir, exist_ok=True)
        
        # Rate limiting - skip if too soon since last log
        if not force and (current_time - self._last_log_time) < self.time_delta:
            return
            
        self._last_log_time = current_time
        elapsed = current_time - self._start_time if self._start_time else 0.0
        storage['wallclock_time'].append(elapsed)
        
        # Format metrics string
        metrics_str = self._format_values(data)
        
        print(f"[{label}] step: {step:>6} | time: {elapsed:>7.1f}s | {metrics_str}")
        try:
            summary = jax.tree_util.tree_map(lambda x: x.item(), data)
        except AttributeError:
            summary = data
        [storage[k].append(v) for (k,v) in summary.items()]
        if save_flag is True: save_logs(storage, label, step, logdir)

    
    def _format_values(self, data: Dict[str, Any]) -> str:
        """
        Returns:
            Formatted string like "loss: 0.023 | entropy: 1.46 | episodes: 128"
        """
        parts = []
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, float):
                if abs(value) < 0.001 or abs(value) > 1000:
                    parts.append(f"{key}: {value:.3e}")
                else:
                    parts.append(f"{key}: {value:.4f}")
            elif isinstance(value, int):
                parts.append(f"{key}: {value}")
            else:
                parts.append(f"{key}: {value}")
        return " | ".join(parts)


def save_checkpoint(
    path: str,
    params: Dict,
    opt_state: Any,
    curriculum_state: CurriculumState,
    epoch: int,
    env_step_count: int,
    metrics: Dict,
    storage: Dict,
    config: Optional[Any] = None,
    name: Optional[Any] = None,
) -> str:
    """
    Save training checkpoint to disk.
    
    Args:
        path: Directory to save checkpoint
        params: Model parameters (PyTree)
        opt_state: Optimizer state (PyTree)
        curriculum_state: Current curriculum learning state
        epoch: Current training epoch
        env_step_count: Total environment steps taken
        config: Optional training configuration
        
    Returns:
        Full path to saved checkpoint file
    """
    os.makedirs(path, exist_ok=True)
    
    _params = serialization.to_state_dict(params)
    _opt_state = serialization.to_state_dict(opt_state)
    
    # Convert curriculum state JAX arrays to numpy for pickling
    try: 
        curriculum_dict = {
            'solved_mask': np.array(curriculum_state.solved_mask),
            'n_presentations': int(curriculum_state.n_presentations),
            'best_episode_lengths': np.array(curriculum_state.best_episode_lengths),
            'best_action_sequences': np.array(curriculum_state.best_action_sequences)}
    except AttributeError:
        if hasattr(curriculum_state, "easy"):  # MultiLevelCurriculumState
            curriculum_dict = {
                "type": "multilevel",
            "easy": {
                "solved_mask": np.array(curriculum_state.easy.solved_mask),
                "n_presentations": int(curriculum_state.easy.n_presentations),
                "best_episode_lengths": np.array(curriculum_state.easy.best_episode_lengths),
                "best_action_sequences": np.array(curriculum_state.easy.best_action_sequences)},
            "medium": {
                "solved_mask": np.array(curriculum_state.medium.solved_mask),
                "n_presentations": int(curriculum_state.medium.n_presentations),
                "best_episode_lengths": np.array(curriculum_state.medium.best_episode_lengths),
                "best_action_sequences": np.array(curriculum_state.medium.best_action_sequences)},
            "hard": {
                "solved_mask": np.array(curriculum_state.hard.solved_mask),
                "n_presentations": int(curriculum_state.hard.n_presentations),
                "best_episode_lengths": np.array(curriculum_state.hard.best_episode_lengths),
                "best_action_sequences": np.array(curriculum_state.hard.best_action_sequences)},
            "mixing_probs": np.array(curriculum_state.mixing_probs),
            "tier_success_rates": np.array(curriculum_state.tier_success_rates),
            "curriculum_lambda": float(curriculum_state.curriculum_lambda)}

    
    checkpoint = {
        'params': _params,
        'opt_state': _opt_state,
        'curriculum_dict': curriculum_dict,
        'epoch': epoch,
        'env_step_count': int(env_step_count),
        'metrics': metrics,
        'config': dict(config),
        'storage': storage,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = (name.replace(" ", "_") if name else "run")

    filepath = os.path.join(path, f"{name}_checkpoint_epoch_{epoch}_{timestamp}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved: {filepath}")
    return filepath


def load_checkpoint(
    filepath: str,
    params_template: Dict,
    opt_state_template: Any,
) -> Tuple[Dict, Any, CurriculumState, int, int, Optional[Any]]:
    """
    Load training checkpoint from disk.
    
    Args:
        filepath: Path to checkpoint file
        params_template: Template PyTree for params (e.g., from init)
        opt_state_template: Template PyTree for optimizer state
        
    Returns:
        Tuple of (params, opt_state, curriculum_state, epoch, env_step_count, config)
    """
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    params = serialization.from_state_dict(params_template, checkpoint['params'])
    opt_state = serialization.from_state_dict(opt_state_template, checkpoint['opt_state'])
    
    # Reconstruct curriculum state from dict
    curriculum_dict = checkpoint['curriculum_dict']

    def _make_simple_cs(d):
        return CurriculumState(
            solved_mask=jnp.array(d['solved_mask']),
            states_processed=int(d.get('states_processed', 0)),
            initial_phase_complete=bool(d.get('initial_phase_complete', False)),
            n_presentations=int(d['n_presentations']),
            best_episode_lengths=jnp.array(d['best_episode_lengths']),
            best_action_sequences=jnp.array(d['best_action_sequences']),
        )

    if curriculum_dict["type"] == "multilevel":
        easy_cs = _make_simple_cs(curriculum_dict['easy'])
        medium_cs = _make_simple_cs(curriculum_dict['medium'])
        hard_cs = _make_simple_cs(curriculum_dict['hard'])

        from ac_jax.env.curriculum import MultiLevelCurriculumState
        curriculum_state = MultiLevelCurriculumState(
            easy=easy_cs,
            medium=medium_cs,
            hard=hard_cs,
            mixing_probs=jnp.array(curriculum_dict['mixing_probs']),
            tier_success_rates=jnp.array(curriculum_dict['tier_success_rates']),
            curriculum_lambda=float(curriculum_dict['curriculum_lambda']),
        )
    else:
        curriculum_state = _make_simple_cs(curriculum_dict)

    epoch = checkpoint['epoch']
    env_step_count = checkpoint['env_step_count']
    config = checkpoint.get('config', None)
    
    print(f"Checkpoint loaded: {filepath} (epoch {epoch}, {env_step_count} env steps)")
    return params, opt_state, curriculum_state, epoch, env_step_count, config
