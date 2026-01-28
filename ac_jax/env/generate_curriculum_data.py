import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np

from functools import partial
from dataclasses import field
from typing import Optional, Union, NamedTuple, Tuple

import jumanji
from jumanji import specs
from chex import dataclass, PRNGKey
from jaxtyping import Array, Float, Int

from ac_jax.env import utils, ac_env
from ac_jax.env.types import State, Observation, CurriculumBatch

idx_2_inverse_action = jnp.array([
    1, 0, 3, 2, 5, 4,       # Relator 0 ops
    7, 6, 9, 8, 11, 10      # Relator 1 ops
], dtype=jnp.int32)

@dataclass
class ScrambledState:
    presentation: Float[Array, "n_gens * max_relator_length"]
    lengths: Int[Array, "n_gens"]
    last_action: int
    depth: int
    key: PRNGKey

UNROLL_DEPTH = 4

class CurriculumGen:
    def __init__(self, max_relator_length: int, n_gens: int, 
                 n_actions: int = 12, max_depth: int = 18,
                 seed_presentation=None):
        self.max_relator_length = max_relator_length
        self.max_depth = max_depth
        # self.max_depth = self.max_relator_length // 3  # (heuristic; conj adds 2 chars, nontrivial concat adds 2 or more)
        self.n_gens = n_gens
        self.n_actions = n_actions
        if seed_presentation is None:
            initial_presentation = jnp.zeros((n_gens * max_relator_length,), dtype=jnp.int32)
            for i in range(n_gens):
                initial_presentation = initial_presentation.at[i * max_relator_length].set(i + 1)
        self.initial_presentation = initial_presentation
        self.initial_lengths = utils.presentation_length(self.initial_presentation)

        self.initial_state = ScrambledState(presentation=self.initial_presentation,
                                            lengths=self.initial_lengths,
                                            last_action=-1,
                                            depth=0,
                                            key=None)
        self.width = n_actions // 2
        self.action_idx = jnp.arange(self.n_actions)
        self.mixing_frac = 0.25  # default fraction of series presentations in mixed batches

        # series-specific parameters
        self.ms_trivial_frac = 0.6
        self.ms_special_frac = 0.35
        self.ak_frac = 1 - self.ms_trivial_frac - self.ms_special_frac
        self.p_series = jnp.array([self.ak_frac, self.ms_special_frac, self.ms_trivial_frac])

        self.n_perturb = 2  # perturbations applied to series presentations
        self.max_n_AK = (self.max_relator_length - 1) // 2  # 2n + 1
        self.AK_word_1 = np.array([1,2,1,-2,-1,-2], dtype=np.int32)
        self.AK_r1 = jnp.zeros((self.max_relator_length,), dtype=jnp.int32)
        self.AK_r1 = jax.lax.dynamic_update_slice(self.AK_r1, self.AK_word_1, (0,))

        self.max_n_MS = (self.max_relator_length - 3) // 2  # 2n + 3
        self.MS_word_1 = np.array([1,1,2,-1,-2], dtype=np.int32)
        self.MS_r1 = jnp.zeros_like(self.AK_r1)
        self.MS_r1 = jax.lax.dynamic_update_slice(self.MS_r1, self.MS_word_1, (0,))

        self.max_w_length = 10
        print("Using initial presentation:", self.initial_presentation)

    @partial(jit, static_argnums=(0,2,3))
    def generate_presentations(self, key: PRNGKey, depth: int, batch_size: int):
        keys = random.split(key, batch_size)
        random_presentations = vmap(self.scramble_presentation, in_axes=(0, None))(keys,  depth)
        return random_presentations

    def _step(self, carry: ScrambledState, sentinel):

        # gen `width` candidates
        carry.key, subkey = random.split(carry.key)
        candidate_actions = random.choice(subkey, self.action_idx, (self.width,), replace=False)
        candidates, lengths = vmap(ac_env.move, in_axes=(None, 0, None, None))(carry.presentation,
                                                                         candidate_actions, self.n_gens, False)
        candidate_presentation_lengths = jnp.sum(lengths, axis=-1)
        current_presentation_length = jnp.sum(carry.lengths, axis=-1)

        # complexity bias
        delta_lengths = candidate_presentation_lengths - current_presentation_length
        length_mask = delta_lengths > 0

        last_action_idx = jnp.maximum(carry.last_action, 0)
        inverse_action = jnp.where(carry.last_action >= 0, 
                                    idx_2_inverse_action[last_action_idx], -1)
        inverse_mask = candidate_actions != inverse_action

        total_mask = jnp.logical_and(length_mask, inverse_mask)

        score = delta_lengths - 100 * (~total_mask)
        best_idx = jnp.argmax(score)
        valid_step = score[best_idx] > -10
        last_action = jnp.where(valid_step, candidate_actions[best_idx], -1)

        new_state = ScrambledState(
            presentation=jnp.where(valid_step, candidates[best_idx], carry.presentation),
            lengths=jnp.where(valid_step, lengths[best_idx], carry.lengths),
            last_action=jnp.where(valid_step, last_action, carry.last_action),
            key=carry.key,
            depth=carry.depth + valid_step * 1)
        
        return new_state, last_action

    def _step_masked(self, carry, x):
        i, target_depth = x
        candidate_state, candidate_action = self._step(carry, None)
        mask = i < target_depth
        new_state = jax.tree_util.tree_map(
            lambda x, y: jnp.where(mask, x, y), candidate_state, carry)
        last_action = jnp.where(mask, candidate_action, -1)
        return new_state, last_action

    def scramble_presentation(self, key: PRNGKey, depth: int):

        _initial_state = self.initial_state.replace(key=key)
        final_state, action_history = jax.lax.scan(self._step, _initial_state, None, 
                                                   length=depth, unroll=UNROLL_DEPTH)

        action_history = jnp.pad(action_history, (0, self.max_depth - depth), constant_values=-1)
        return CurriculumBatch(
            presentations=final_state.presentation,
            lengths=final_state.lengths,
            move_histories=action_history,
            depths=final_state.depth)
    
    def scramble_presentation_fixed_depth(self, key: PRNGKey, depth: int):

        _initial_state = self.initial_state.replace(key=key)
        loop_idx = jnp.arange(self.max_depth)
        def _step_fixed_wrap(carry, i):
            return self._step_masked(carry, (i, depth))
        final_state, action_history = jax.lax.scan(_step_fixed_wrap, _initial_state, 
                                                   loop_idx, 
                                                   length=self.max_depth, 
                                                   unroll=UNROLL_DEPTH)

        # action_history = jnp.pad(action_history, (0, self.max_depth - depth), constant_values=-1)
        return CurriculumBatch(
            presentations=final_state.presentation,
            lengths=final_state.lengths,
            move_histories=action_history,
            depths=final_state.depth)

    def generate_AK(self, n: int):
        # Return presentations drawn from AK series.
        # AK(n): \langle x^n y^{-(n+1)}, xyx = yxy \rangle
        idx = jnp.arange(self.max_relator_length)
        r0 = jnp.zeros((self.max_relator_length,), dtype=jnp.int32)
        r0 = jnp.where(idx < n, 1, r0)
        r0 = jnp.where((n <= idx) & (idx  < 2*n+1), -2, r0)
        return jnp.concatenate([r0, self.AK_r1], axis=0)

    def generate_MS_special(self, n: int):
        # Return presentations drawn from a special case of the MS series, 
        # which are AC-trivial.
        # MS(n): \langle x^-1 y^n x = y^{n+1}, x^2 y = yx \rangle
        idx = jnp.arange(self.max_relator_length)
        r0 = jnp.zeros((self.max_relator_length,), dtype=jnp.int32)
        r0 = r0.at[0].set(-1)
        r0 = r0.at[n+1].set(1)
        r0 = jnp.where((1 <= idx) & (idx <= n), 2, r0)
        r0 = jnp.where((idx >= n+2) & (idx < 2*n+3), -2, r0)
        return jnp.concatenate([r0, self.MS_r1], axis=0)
    
    @partial(jit, static_argnums=(0,2))
    def _sample_w(self, key, length):
        key, subkey, _subkey = random.split(key, 3)
        idx = jnp.arange(self.max_relator_length)

        max_x_pairs = length // 2
        n_x_pairs = random.randint(subkey, (), 0, max_x_pairs + 1)

        y_part = random.choice(_subkey, jnp.array([2, -2], dtype=jnp.int32), 
                          (self.max_relator_length,))
        w = jnp.zeros((self.max_relator_length,), dtype=jnp.int32)
        w = jnp.where(idx < n_x_pairs, 1, w)
        w = jnp.where((n_x_pairs <= idx) & (idx < 2*n_x_pairs), -1, w)
        w = jnp.where((idx >= 2*n_x_pairs) & (idx < length), y_part, w)

        w_idx = jnp.arange(length)
        perm_idx = random.permutation(key, w_idx)
        buffer = jnp.zeros(self.max_relator_length-1, dtype=jnp.int32)
        return jax.lax.dynamic_update_slice(buffer, w[perm_idx], (0,))
    
    def generate_MS_general(self, key, n: int):
        # Return presentations drawn from the general MS series.
        # MS(n): \langle x^-1 y^n x = y^{n+1}, x = w \rangle,
        # where w is a word in x,y with zero exponent sum in x.

        key, subkey = random.split(key)
        idx = jnp.arange(self.max_relator_length)
        r0 = jnp.zeros((self.max_relator_length,), dtype=jnp.int32)
        r0 = r0.at[0].set(-1)
        r0 = r0.at[n+1].set(1)
        r0 = jnp.where((1 <= idx) & (idx <= n), 2, r0)
        r0 = jnp.where((idx >= n+2) & (idx < 2*n+3), -2, r0)

        length_w = random.randint(subkey, (), 2, self.max_w_length - 1)
        branches = [lambda k, l=l: self._sample_w(k, l)
                    for l in range(2, self.max_w_length)]

        w = jax.lax.switch(length_w - 2, branches, key)

        r1 = jnp.zeros_like(r0)
        r1 = r1.at[0].set(-1)
        r1 = jax.lax.dynamic_update_slice(r1, w, (1,))

        return jnp.concatenate([r0, r1], axis=0)
    
    def _sample_series(self, key, depth: int):
        key, subkey, _subkey = random.split(key, 3)
        type_idx = random.multinomial(subkey, n=1, p=self.p_series).argmax()

        # n should be roughly correlated with depth
        n_avg = (depth - 1) // 2
        n_min, n_max = jnp.maximum(1,n_avg - 1), jnp.minimum(n_avg + 1, self.max_n_MS - 1)
        n = random.randint(_subkey, (), n_min, n_max + 1)
        branches = [lambda _,n: self.generate_AK(n),
                    lambda _,n: self.generate_MS_special(n),
                    lambda k,n: self.generate_MS_general(k, n)]
        presentation = jax.lax.switch(type_idx, branches, key, n)
        lengths = utils.presentation_length(presentation, self.n_gens)
        return presentation, lengths, n

    @partial(jit, static_argnums=(0,3))
    def generate_mixed_batch(self, key, depth: int, batch_size: int, 
                                  mixing_frac=0.1):
        """
        Generates a batch of presentations from the scrambled
        trivial set with a small percentage of scrambled AK/MS presentations.
        """
        keys = random.split(key, 4)
        key_scramble = random.split(keys[0], batch_size)

        # scramble trivial presentations
        scrambled_batch = vmap(self.scramble_presentation_fixed_depth, in_axes=(0, None))(
            key_scramble, depth)

        # sample series presentations
        key_series = jax.random.split(keys[1], batch_size)
        series_pres, series_lengths, series_n = vmap(self._sample_series, in_axes=(0, None))(
            key_series, depth)
        
        # lightly perturb series presentations to prevent overfitting
        key_perturb = jax.random.split(keys[2], batch_size)

        def _perturb_fn(key, presentation, length, n_perturb=2):
            starting_state = ScrambledState(
                presentation=presentation,
                lengths=length,
                last_action=-1,
                depth=0,
                key=key)
            
            final_state, _ = jax.lax.scan(self._step, starting_state, None, 
                                          length=n_perturb, unroll=UNROLL_DEPTH)
            return final_state.presentation, final_state.lengths
        

        series_pres, series_lengths = vmap(_perturb_fn, in_axes=(0,0,0,None))(
            key_perturb, series_pres, series_lengths, self.n_perturb)
        
        # mixing
        mask = random.bernoulli(keys[3], p=mixing_frac, shape=(batch_size,))
        mask_2d = jnp.expand_dims(mask, axis=-1)
        mixed_pres = jnp.where(mask_2d, series_pres, scrambled_batch.presentations)
        mixed_lengths = jnp.where(mask_2d, series_lengths, scrambled_batch.lengths)

        dummy_series_hist = jnp.full_like(scrambled_batch.move_histories, -1)
        mixed_hist = jnp.where(mask_2d, dummy_series_hist, scrambled_batch.move_histories)
        mixed_depths = jnp.where(mask, -1, scrambled_batch.depths)
        mixed_n = jnp.where(mask, series_n, -1)

        return CurriculumBatch(
            presentations=mixed_pres,
            lengths=mixed_lengths,
            move_histories=mixed_hist,
            depths=mixed_depths,
            n=mixed_n)
    
def stack_batches(batches):
    return CurriculumBatch(
        presentations=np.concatenate([b.presentations for b in batches], axis=0),
        lengths=np.concatenate([b.lengths for b in batches], axis=0),
        move_histories=np.concatenate([b.move_histories for b in batches], axis=0),
        depths=np.concatenate([b.depths for b in batches], axis=0),
        n=np.concatenate([b.n for b in batches], axis=0))


def generate_static_dataset(generator: CurriculumGen, key: PRNGKey, 
                            total_easy=2e5, total_med=4e5, total_hard=2e5, 
                            batch_size=8192, max_depth=18):
    from tqdm import tqdm
    from collections import defaultdict
    
    # (count, min_depth, max_depth, mixing_prob)
    # configs = {"easy":   (total_easy, 2, 4,  0.05),
    #            "medium": (total_med,  5, 8,  0.2),
    #            "hard":   (total_hard, 9, 12, 0.3)}
    
    configs = {"easy":   (total_easy, 2, 6,  0.05),
               "medium": (total_med,  7, 12,  0.2),
               "hard":   (total_hard, 13, max_depth, 0.3)}

    all_batches = defaultdict(list)
    
    for label, cfg in configs.items():
        total_count, min_d, max_d, mix_prob = cfg
        total_count = int(total_count)
        print(f"--- Generating {label} ({total_count} examples) ---")
        
        # distribute count evenly across depths
        n_depths = max_d - min_d + 1
        per_depth_count = total_count // n_depths
        
        for depth in range(min_d, max_d + 1):
            if depth == max_d:
                current_target = total_count - (per_depth_count * (n_depths - 1))
            else:
                current_target = per_depth_count
            
            print(f">>> Depth {depth}: Generating {current_target}...")
            
            num_chunks = (current_target + batch_size - 1) // batch_size
            
            for _ in tqdm(range(num_chunks), leave=False):
                key, subkey = jax.random.split(key)
                
                current_bs = min(batch_size, current_target)
                current_target -= batch_size
                
                # jit compilation expensive, depth and batch size static
                batch = generator.generate_mixed_batch(
                    subkey, 
                    depth=depth, 
                    batch_size=batch_size, 
                    mixing_frac=mix_prob)
                
                cpu_batch = jax.tree_util.tree_map(lambda x: np.array(x)[:current_bs], 
                                                   batch)
                all_batches[label].append(cpu_batch)
    
    dataset = {k: stack_batches(all_batches[k]) for k in all_batches.keys()}
    return dataset



if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Generate AC curriculum dataset", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-gens", type=int, default=2, help="Number of group generators")
    parser.add_argument("--max-relator-length", type=int, default=64, help="Maximum relator length")
    parser.add_argument("--max-depth", type=int, default=18, help="Maximum scramble depth")
    parser.add_argument("--total-easy", type=int, default=int(2e5), help="Total easy examples")
    parser.add_argument("--total-med", type=int, default=int(4e5), help="Total medium examples")
    parser.add_argument("--total-hard", type=int, default=int(2e5), help="Total hard examples")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fname", type=str, default="data/ac_dataset_800k_64", help="Output filename")
    args = parser.parse_args()

    cg = CurriculumGen(max_relator_length=args.max_relator_length, n_gens=args.n_gens)

    master_key = jax.random.PRNGKey(args.seed)
    total_easy, total_med, total_hard = args.total_easy, args.total_med, args.total_hard

    # slow compile
    dataset = generate_static_dataset(cg, master_key, total_easy, total_med, total_hard,
                                      batch_size=args.batch_size)
    
    print(f"Saving dataset to {args.fname}...")
    utils.save_dataclass_dict(dataset, args.fname)