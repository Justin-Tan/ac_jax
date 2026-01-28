"""
Utility functions for JAX PPO training.

Provides checkpoint save/load functionality using pickle and flax serialization.
"""

import os
import pickle
from typing import Any, Dict, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from ac_jax.env.curriculum import CurriculumState


def save_checkpoint(
    path: str,
    params: Dict,
    opt_state: Any,
    curriculum_state: CurriculumState,
    epoch: int,
    env_step_count: int,
    config: Optional[Any] = None,
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
    
    # Serialize PyTrees to bytes
    params_bytes = serialization.to_bytes(params)
    opt_state_bytes = serialization.to_bytes(opt_state)
    
    # Convert curriculum state JAX arrays to numpy for pickling
    curriculum_dict = {
        'solved_mask': np.array(curriculum_state.solved_mask),
        'states_processed': int(curriculum_state.states_processed),
        'initial_phase_complete': int(curriculum_state.initial_phase_complete),
        'n_presentations': int(curriculum_state.n_presentations),
        'best_episode_lengths': np.array(curriculum_state.best_episode_lengths),
        'best_action_sequences': np.array(curriculum_state.best_action_sequences),
    }
    
    checkpoint = {
        'params_bytes': params_bytes,
        'opt_state_bytes': opt_state_bytes,
        'curriculum_dict': curriculum_dict,
        'epoch': epoch,
        'env_step_count': int(env_step_count),
        'config': config,
    }
    
    filepath = os.path.join(path, f"checkpoint_epoch_{epoch}.pkl")
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
    
    # Deserialize PyTrees from bytes
    params = serialization.from_bytes(params_template, checkpoint['params_bytes'])
    opt_state = serialization.from_bytes(opt_state_template, checkpoint['opt_state_bytes'])
    
    # Reconstruct curriculum state from dict
    curriculum_dict = checkpoint['curriculum_dict']
    curriculum_state = CurriculumState(
        solved_mask=jnp.array(curriculum_dict['solved_mask']),
        states_processed=curriculum_dict['states_processed'],
        initial_phase_complete=curriculum_dict['initial_phase_complete'],
        n_presentations=curriculum_dict['n_presentations'],
        best_episode_lengths=jnp.array(curriculum_dict['best_episode_lengths']),
        best_action_sequences=jnp.array(curriculum_dict['best_action_sequences']),
    )
    
    epoch = checkpoint['epoch']
    env_step_count = checkpoint['env_step_count']
    config = checkpoint.get('config', None)
    
    print(f"Checkpoint loaded: {filepath} (epoch {epoch}, {env_step_count} env steps)")
    return params, opt_state, curriculum_state, epoch, env_step_count, config
