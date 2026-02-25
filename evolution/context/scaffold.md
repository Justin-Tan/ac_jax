```python
import jax
import jax.numpy as jnp

def heuristic_fn_v0(presentation: jnp.ndarray) -> float:
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
    # [EVOLVE-BLOCK-START]

    # Example baseline logic: negative total length
    is_generator = jnp.abs(presentation) > 0
    total_length = jnp.sum(is_generator)
    return -1. * total_length

    # [EVOLVE-BLOCK-END]
```