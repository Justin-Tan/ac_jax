def heuristic_fn_v24(presentation: jnp.ndarray) -> float:
    """
    Enhanced heuristic that captures potential for cancellation by:
    1. Counting immediate cancellation pairs (adjacent generator-inverse pairs)
    2. Penalizing long runs of same generator (indicating reducibility via relator moves)
    3. Measuring symmetry (palindromic structure often indicates simplicity)
    4. Rewarding short presentation length but with diminishing returns
    5. Accounting for potential to reduce via AC moves by measuring balancedness of generators

    Args:
        presentation: Array representing current group presentation in terms of relators
            [r_1; r_2]. Shape (72,) int32 array. 0 is padding.

    Returns:
        Scalar heuristic value in [0,1] estimating likelihood of trivialisation in the
        future (higher is better).
    """
    MAX_RELATOR_LENGTH = 36

    # Split into two relators
    relator1 = presentation[:MAX_RELATOR_LENGTH]
    relator2 = presentation[MAX_RELATOR_LENGTH:]

    # Helper function to compute cancellation pairs and symmetry for a relator
    def relator_features(r):
        # Count non-zero (non-padding) elements
        length = jnp.sum(jnp.abs(r) > 0)

        # Immediate cancellation pairs (r[i] = -r[i+1])
        shifted = jnp.roll(r, -1)
        pairs = r[:-1] + shifted[:-1]
        cancel_pairs = jnp.sum(jnp.abs(pairs) < jnp.abs(r[:-1]), where=r[:-1] != 0)  # r[i] == -r[i+1] => r[i] + r[i+1] == 0

        # Check for palindromic structure: r[i] == -r[-i-1] for inverses or r[i] == r[-i-1] for symmetry
        reversed_r = jnp.flip(r)
        inverse_reversed = -reversed_r
        palindromic_invs = jnp.sum((r == inverse_reversed) & (r != 0))
        palindromic_same = jnp.sum((r == reversed_r) & (r != 0))
        symmetry_score = jnp.maximum(palindromic_invs, palindromic_same) / jnp.maximum(length, 1.0)

        # Penalty for long runs of same generator (reducible via Tietze moves)
        # Count transitions where generator stays same
        same_gen = jnp.sum((r[:-1] != 0) & (r[:-1] == r[1:]))
        run_penalty = same_gen / jnp.maximum(length - 1.0, 1.0)

        return length, cancel_pairs, symmetry_score, run_penalty

    len1, cp1, sym1, rp1 = relator_features(relator1)
    len2, cp2, sym2, rp2 = relator_features(relator2)

    total_length = len1 + len2
    total_cancel_pairs = cp1 + cp2
    avg_symmetry = (sym1 + sym2) / 2.0
    avg_run_penalty = (rp1 + rp2) / 2.0

    # Length normalization (prefer shorter presentations but with diminishing returns)
    # Scale so that length=2 (trivial case) gets 1.0, length=72 gets near 0
    length_score = jnp.clip(1.0 - (total_length - 2) / 70.0, 0.0, 1.0)

    # Cancellation potential: more immediate pairs => easier to reduce
    max_possible_pairs = jnp.maximum(total_length - 2.0, 1.0)
    cancel_score = jnp.clip(total_cancel_pairs / max_possible_pairs, 0.0, 1.0)

    # Symmetry score (palindromic structure often indicates simpler group)
    # Already normalized to [0,1]

    # Combined score: prefer short length, high cancellation, high symmetry, low run penalty
    # Normalize each component to [0,1] and combine with weights
    w_len = 0.3
    w_cancel = 0.25
    w_sym = 0.3
    w_run = 0.15

    combined_score = (
        w_len * length_score +
        w_cancel * cancel_score +
        w_sym * avg_symmetry +
        w_run * (1.0 - avg_run_penalty)
    )

    return combined_score
