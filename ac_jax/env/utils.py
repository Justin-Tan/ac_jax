import numpy as np

import functools

import jax
from jax import vmap
import jax.numpy as jnp

def symlog(x):
    # Symmetric log transform: sign(x) * log(|x| + 1)
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def symexp(x):
    # Inverse of symlog.
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

def presentation_length(presentation, n_gens=2):
    # compute relator lengths
    presentation = jnp.reshape(presentation, (n_gens, -1))
    lengths = vmap(jnp.count_nonzero)(presentation)
    return lengths

def get_relators(presentation, n_gens=2):
    pres = presentation.reshape((n_gens, -1))
    rels = jnp.split(pres, n_gens, axis=0)
    return [jnp.squeeze(rel) for rel in rels]

def invert_relator(relator, length):
    idx = jnp.arange(relator.shape[0])
    target_idx = length - 1 - idx
    mask = idx < length

    values = relator[jnp.where(mask, target_idx, 0)]
    return jnp.where(mask, -values, 0)

def collapse_zeros(relator):
    return jnp.compress(relator != 0, relator, size=relator.shape[0])

def cyclic_reduce(relator):
    r"""
    Simplify relator by cyclic permutation: $x r x^{-1} \mapsto r$.
    """
    length = jnp.count_nonzero(relator)
    max_relator_length = relator.shape[0]

    l, r = 0, length - 1
    def step(carry, unused):
        l, r, n_remove = carry
        can_remove = (l < r) & (relator[l] == -relator[r])
        l += can_remove
        r -= can_remove
        n_remove += can_remove
        return (l, r, n_remove), None
    (_, _, n_remove), _ = jax.lax.scan(step, (l, r, 0), None, length=max_relator_length//2, unroll=16)

    # get middle part - jit doesn't support dynamic slicing
    new_length = length - 2 * n_remove
    idx = jnp.arange(max_relator_length)
    buffer = jnp.where(idx < new_length, relator[idx + n_remove], 0)

    return buffer, new_length

def free_reduce(relator):
    r"""
    Simplify relator, remove adjacent inverses: $x x^{-1} \mapsto 1$.
    """
    stack = jnp.zeros_like(relator)
    def step(carry, x):
        stack, ptr = carry
        is_nonzero = x != 0
        top = stack[ptr - 1]

        _pop =  is_nonzero & (ptr > 0) & (top == -x)
        _push = is_nonzero & ~_pop

        new_ptr = ptr + _push - _pop
        new_stack = jax.lax.cond(_push, lambda: stack.at[ptr].set(x),
                                 lambda: jax.lax.cond(_pop, lambda: stack.at[ptr - 1].set(0), lambda: stack))
        return (new_stack, new_ptr), None
    (reduced, length), _ = jax.lax.scan(step, (stack, 0), relator, unroll=16)
    return reduced, length

def is_array_valid_presentation(array):
    """
    Checks whether a given Numpy Array is a valid presentation or not.
    An array is a valid presentation with two words if each half has all zeros padded to the right.
    That is, [1, 2, 0, 0, -2, -1, 0, 0] is a valid presentation, but [1, 0, 2, 0, -2, -1, 0, 0] is not.
    And if each word has nonzero length, i.e. [0, 0, 0, 0] and [1, 2, 0, 0] are invalid.

    Parameters:
    presentation: A Numpy Array

    Returns: True / False
    """

    # for two generators and relators, the length of the presentation should be even.
    assert isinstance(
        array, (list, np.ndarray)
    ), f"array must be a list or a numpy array, got {type(array)}"
    if isinstance(array, list):
        array = np.array(array)

    length_valid = len(array) % 2 == 0

    max_relator_length = len(array) // 2

    first_word_length = np.count_nonzero(array[:max_relator_length])
    second_word_length = np.count_nonzero(array[max_relator_length:])

    first_word_valid = (array[first_word_length:max_relator_length] == 0).all()
    second_word_valid = (array[max_relator_length + second_word_length :] == 0).all()

    # for a presentation to be valid, each word should have length >= 1 and it should have all the zeros padded to the right.
    is_valid = all([length_valid, first_word_length > 0, second_word_length > 0, first_word_valid, second_word_valid])

    return is_valid

def variable_length_presentations(path):
    from pathlib import Path
    import ast
    """Load presentations with variable lengths."""
    path = Path(path)
    if not path.exists():
        print(f"Presentations file not found: {path}")
    
    presentations = []
    max_len = 0
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Parse Python list syntax: [-1, 2, 1, -2, ...]
                try:
                    pres = ast.literal_eval(line)
                    if isinstance(pres, list):
                        presentations.append(pres)
                        max_len = max(max_len, len(pres))
                except (ValueError, SyntaxError):
                    continue
    
    # Pad all presentations to max length
    padded = []
    for pres in presentations:
        rel_len = len(pres) // 2
        max_rel_len = max_len // 2
        r1 = pres[:rel_len] + [0] * (max_rel_len - rel_len)
        r2 = pres[rel_len:] + [0] * (max_rel_len - (len(pres) - rel_len))
        padded.append(r1 + r2)
    
    return np.array(padded, dtype=np.int8)

def save_dataclass_dict(data_dict, filepath):
    arrays_to_save = {}
    for key, dataclass_obj in data_dict.items():
        for field_name, field_value in dataclass_obj.__dict__.items():
            arrays_to_save[f"{key}_{field_name}"] = field_value
    np.savez_compressed(filepath, **arrays_to_save)

def load_dataclass_dict(filepath, dataclass_type, keys):
    data = np.load(filepath)
    result = {}
    for key in keys:
        field_dict = {}
        for field_name in dataclass_type.__dataclass_fields__.keys():
            field_dict[field_name] = data[f"{key}_{field_name}"]
        result[key] = dataclass_type(**field_dict)
    return result


class HLGauss:
    """
    Based on https://arxiv.org/abs/2403.03950. 
    
    Transformation of continuous values into
    Gaussian histogram for categorical value function loss.
    """
    def __init__(self, min_value, max_value, n_bins=32, sigma_ratio=0.75):
        import jax.scipy.special
        self.min_value = min_value
        self.max_value = max_value
        self.n_bins = n_bins
        self.support = jnp.linspace(min_value, max_value, n_bins, dtype=jnp.float32)

        self.bin_width = (max_value - min_value) / (n_bins - 1)
        self.sigma = sigma_ratio * self.bin_width

        self.edges = jnp.concatenate([self.support[:1] - self.bin_width / 2,
                                      self.support + self.bin_width / 2])

    def to_probabilities(self, targets):
        targets = jnp.clip(targets, self.min_value, self.max_value)
        # CDF values at edges
        cdf_values = 0.5 * jax.scipy.special.erf(
            (self.edges - targets) / (self.sigma * jnp.sqrt(2)))
        probs = cdf_values[1:] - cdf_values[:-1]
        probs = probs / jnp.sum(probs, keepdims=True)
        return probs  # [n_bins]
        
    def to_scalar(self, logits):
        probs = jax.nn.softmax(logits, axis=0)  # [n_bins]
        scalar_values = jnp.sum(probs * self.support)
        return scalar_values

@functools.partial(jax.jit, static_argnums=(1,))
def get_exponent_sum_matrix(presentation, n_gens=2):
    def get_gen_counts(relator):
        n_x = jnp.sum(relator == 1)
        n_y = jnp.sum(relator == 2)
        n_x_inv = jnp.sum(relator == -1)
        n_y_inv = jnp.sum(relator == -2)
        return n_x - n_x_inv, n_y - n_y_inv
    relators = presentation.reshape(n_gens, -1)
    x_counts, y_counts = vmap(get_gen_counts)(relators)
    return jnp.hstack((x_counts[:,None], y_counts[:,None]))
