## Request 1
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v0`.

**Latency:** 16.789 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v0(presentation: jnp.ndarray) -> float:
    r"""
    Modified heuristic focusing on cancellation potential and structural simplicity.
    
    Key improvements over baseline length-based heuristic:
    1. Counts maximal cancellation pairs (adjacent inverse pairs within each relator)
    2. Penalizes non-trivial relator structure (measured by run lengths and patterns)
    3. Rewards symmetry (palindromic structure) which often indicates potential for simplification
    4. Tracks identity elements (0 padding) to estimate current simplification level
    
    This heuristic better captures the "structure" needed for successful trivialization
    even when presentations temporarily grow in length during the trivialization process.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Mask non-padding elements
    r1_mask = r1 != 0
    r2_mask = r2 != 0
    
    # Count actual elements in each relator
    len_r1 = jnp.sum(r1_mask).astype(jnp.float32)
    len_r2 = jnp.sum(r2_mask).astype(jnp.float32)
    
    # Calculate cancellation pairs (adjacent inverse pairs)
    def count_cancellations(relator, mask):
        # Get non-zero positions and their values
        indices = jnp.where(mask, jnp.arange(MAX_RELATOR_LENGTH), -1)
        # For each position i, check if r[i] and r[i+1] are inverses
        # Shifted version (next element)
        shifted = jnp.roll(relator, -1)
        # Check if consecutive elements are inverses (a and -a)
        is_inverse = (relator + shifted == 0) & (relator != 0) & (shifted != 0)
        # Mask out padding positions and the last position that wraps around
        valid = mask & jnp.roll(mask, -1)
        return jnp.sum(is_inverse & valid)
    
    cancels_r1 = count_cancellations(r1, r1_mask)
    cancels_r2 = count_cancellations(r2, r2_mask)
    
    # Count palindromic structure - how much of the relator reads the same forwards/backwards
    def palindromic_score(relator, mask):
        # Reverse the relator
        reversed_rel = jnp.flip(relator)
        # Check matching pairs, weighted by distance from center
        matches = (relator == reversed_rel) & (relator != 0) & (reversed_rel != 0)
        return jnp.sum(matches & mask) / (jnp.sum(mask) + 1e-6)
    
    pal_score_r1 = palindromic_score(r1, r1_mask)
    pal_score_r2 = palindromic_score(r2, r2_mask)
    
    # Compute total cancellation potential and structural score
    total_len = len_r1 + len_r2
    total_cancels = cancels_r1 + cancels_r2
    avg_pal = (pal_score_r1 + pal_score_r2) / 2.0
    
    # Normalize to [0,1] with emphasis on cancellation and structure
    # Cancellation potential contributes 0.4, palindrome score 0.4, and inverse length 0.2
    cancellation_score = total_cancels / (total_len + 1e-6)
    length_score = 1.0 - (total_len / MAX_PRESENTATION_LENGTH)
    
    # Combine components with weights tuned for exploration balance
    score = 0.4 * cancellation_score + 0.4 * avg_pal + 0.2 * length_score
    
    # Ensure bounded [0,1]
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 2
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v1`.

**Latency:** 15.219 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that captures structural properties indicating potential for
    trivialization via AC moves. Focuses on:
    1. Cancellation potential: pairs of adjacent inverse generators
    2. Substring symmetry: detection of palindrome-like structures that may simplify
    3. Length management: penalizes excessive presentation length but with diminishing returns
    4. Generator diversity:鼓励 balanced representation of both generators
    
    Key modification: Uses normalized cancellation pairs count (scaled by max possible)
    plus symmetry measure, with a weak penalty for total length to avoid getting stuck
    in local maxima where length is artificially minimized.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Split presentation into two relators (first 36, next 36 entries)
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Compute length of each relator (non-zero entries)
    len1 = jnp.sum(r1 != 0)
    len2 = jnp.sum(r2 != 0)
    total_length = len1 + len2
    
    # Normalize length term: better for smaller presentations
    # Using sigmoid-like scaling to avoid extreme penalties for moderately long presentations
    length_score = 1.0 / (1.0 + 0.02 * total_length)
    
    # Compute cancellation pairs within each relator
    # Adjacent inverse pairs (e.g., [1, -1] or [2, -2])
    def calc_cancellation_pairs(relator):
        # Create shifted arrays: current and next positions
        cur = relator[:-1]
        next_pos = relator[1:]
        
        # Check for inverse pairs: a * b == -1 when b == -a (and a != 0, b != 0)
        is_inverse_pair = (cur != 0) & (next_pos != 0) & (cur + next_pos == 0)
        return jnp.sum(is_inverse_pair)
    
    # Also check for palindrome patterns: relator == reverse(relator)
    def palindrome_score(relator):
        # Only consider non-zero prefix
        length = jnp.sum(relator != 0)
        if length == 0:
            return 0.0
        
        # Create padded reverse for comparison
        prefix = relator[:length]
        rev_prefix = prefix[::-1]
        
        # Compare prefix with its reverse
        matches = jnp.sum(prefix == rev_prefix)
        return matches / length  # Normalized to [0,1]
    
    # Calculate scores
    cancel1 = calc_cancellation_pairs(r1)
    cancel2 = calc_cancellation_pairs(r2)
    total_cancel = cancel1 + cancel2
    
    # Normalize cancellation by max possible (length - 1 per relator)
    max_cancel = MAX_RELATOR_LENGTH - 1  # Upper bound per relator
    cancel_score = jnp.minimum(total_cancel / (2 * max_cancel), 1.0)
    
    # Palindrome symmetry score
    pali1 = palindrome_score(r1)
    pali2 = palindrome_score(r2)
    symmetry_score = (pali1 + pali2) / 2.0
    
    # Combined score: weighted combination favoring cancellations and symmetry
    # Give higher weight to cancellations as they directly enable AC moves
    combined_score = 0.6 * cancel_score + 0.3 * symmetry_score + 0.1 * length_score
    
    # Return in [0,1]
    return jnp.clip(combined_score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 3
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v2`.

**Latency:** 37.317 s

**Output:**

Looking at the problem, I need to create a more sophisticated heuristic that captures structural properties indicating potential for cancellation and closeness to triviality, rather than just using presentation length.

The key insight is that we need to detect patterns that suggest a presentation can be simplified, even if the current length is large. Important features include:
1. **Cancellation potential**: Adjacent inverse pairs within relators
2. **Symmetry**: Palindromic structure in relators
3. **Subword repetition**: Repeated substrings that could be eliminated via conjugation
4. **Balance**: Equal numbers of generators and inverses in each relator (necessary for trivial group)
5. **Shared structure**: Similar substrings between relators

Let me implement a heuristic that combines these features with appropriate weights.

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn(presentation: jnp.ndarray) -> float:
    r"""
    Heuristic focusing on structural properties indicating trivialisation potential:
    - Balance of generators/inverses within each relator
    - Cancellation potential (adjacent inverse pairs)
    - Palindromic structure (suggests conjugacy potential)
    - Subword symmetry and repetition
    
    Args:
        presentation: Array representing current group presentation [r_1; r_2], shape (72,)
    
    Returns:
        Scalar heuristic value in [0,1] (higher = better trivialisation potential)
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Count non-zero (actual generator) elements in each relator
    r1_active = jnp.sum(jnp.abs(r1) > 0)
    r2_active = jnp.sum(jnp.abs(r2) > 0)
    
    # Compute balance score: how equal are generators and inverses in each relator
    def compute_balance(relator):
        # For each generator type, count positive vs negative occurrences
        gen1_pos = jnp.sum(relator == 1)
        gen1_neg = jnp.sum(relator == -1)
        gen2_pos = jnp.sum(relator == 2)
        gen2_neg = jnp.sum(relator == -2)
        
        # Balance is maximized when pos == neg for each generator
        balance_gen1 = 1.0 - jnp.abs(gen1_pos - gen1_neg) / (gen1_pos + gen1_neg + 1e-6)
        balance_gen2 = 1.0 - jnp.abs(gen2_pos - gen2_neg) / (gen2_pos + gen2_neg + 1e-6)
        return (balance_gen1 + balance_gen2) / 2.0
    
    balance1 = compute_balance(r1)
    balance2 = compute_balance(r2)
    avg_balance = (balance1 + balance2) / 2.0
    
    # Compute cancellation potential: count adjacent inverse pairs
    def count_cancellations(relator):
        # Check for adjacent inverse pairs: [x, -x]
        shifted = jnp.roll(relator, 1)
        inv_pairs = jnp.sum((relator * shifted) == -1)  # x * (-x) = -1
        # Also check for self-inverse patterns that could lead to simplification
        return inv_pairs
    
    canc1 = count_cancellations(r1)
    canc2 = count_cancellations(r2)
    total_canc = (canc1 + canc2)
    
    # Compute palindromic score for each relator
    def compute_palindrome_score(relator):
        # Only consider active part (non-zero elements)
        active_mask = (relator != 0).astype(jnp.float32)
        reversed_rel = jnp.flip(relator * active_mask)
        
        # Compare original with reversed (only where both have active elements)
        match = jnp.sum((relator == reversed_rel) * active_mask)
        total_active = jnp.sum(active_mask)
        return match / (total_active + 1e-6)
    
    pal1 = compute_palindrome_score(r1)
    pal2 = compute_palindrome_score(r2)
    avg_pal = (pal1 + pal2) / 2.0
    
    # Compute shared structure score: similarity between relators
    def compute_shared_similarity(rel1, rel2):
        # Normalize relators for comparison (only non-zero elements)
        active1 = (rel1 != 0).astype(jnp.float32)
        active2 = (rel2 != 0).astype(jnp.float32)
        
        # Count common generator values (positive and negative)
        common_gen = jnp.sum((jnp.abs(rel1) == jnp.abs(rel2)) * active1 * active2)
        total_active = jnp.maximum(jnp.sum(active1), jnp.sum(active2))
        return common_gen / (total_active + 1e-6)
    
    shared = compute_shared_similarity(r1, r2)
    
    # Normalize presentation length factor (shorter is better, but not decisive)
    total_active = r1_active + r2_active
    max_possible = 2 * MAX_RELATOR_LENGTH
    length_norm = 1.0 - total_active / max_possible  # 1 for minimal, 0 for max length
    
    # Combine scores with weights - balance is most important, then cancellation potential
    # Length normalization has diminishing returns for large presentations
    weights = jnp.array([0.35, 0.25, 0.20, 0.10, 0.10])  # [balance, cancellations, palindrome, shared, length]
    features = jnp.array([avg_balance, total_canc / 4.0, avg_pal, shared, length_norm])
    
    # Clip to [0,1] range and return weighted sum
    score = jnp.clip(jnp.sum(weights * features), 0.0, 1.0)
    
    return score
# [EVOLVE-BLOCK-END]
```

Key improvements over a simple length-based heuristic:
1. **Balance scoring** - Checks if each relator has equal numbers of generators and their inverses, which is necessary for triviality
2. **Cancellation potential** - Counts adjacent inverse pairs that could be eliminated via conjugation moves
3. **Palindromic structure** - Detects relators that are palindromes, suggesting potential for conjugacy-based simplification
4. **Shared structure** - Measures similarity between relators, indicating potential for AC1 moves (concatenation)
5. **Length normalization** - Still considers length but as a secondary factor with diminishing returns

The weighting scheme prioritizes structural balance and cancellation potential, which are more indicative of trivialisation potential than length alone.


---

## Request 4
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v3`.

**Latency:** 23.773 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v3(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that combines multiple structural features to better detect
    potential for cancellation and progression toward triviality.
    
    Key improvements over v0:
    1. Counts cancellation pairs (adjacent inverse pairs like [1, -1])
    2. Detects palindromic structure in relators (common in trivial group presentations)
    3. Measures imbalance between relator lengths (favors balanced presentations)
    4. Penalizes complex repeated patterns that indicate non-trivial structure
    5. Combines features into a weighted score with sigmoid normalization
    
    These features help identify presentations that may require expansion before
    contraction, and detect hidden structure even in long intermediate presentations.
    """
    # Constants
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Extract relators (first 36 elements = r1, last 36 = r2)
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Helper functions
    def get_active_length(relator):
        """Count non-zero elements in relator"""
        return jnp.sum(jnp.abs(relator) > 0)
    
    def count_cancellation_pairs(relator):
        """Count adjacent inverse pairs (like [1, -1] or [-2, 2])"""
        # Shifted versions to compare adjacent elements
        shifted = jnp.concatenate([relator[1:], jnp.array([0])])
        # Check if adjacent pairs are inverses: a * b == -1 (in terms of absolute values)
        abs_pairs = jnp.abs(relator) * jnp.abs(shifted)
        sign_products = jnp.sign(relator) * jnp.sign(shifted)
        return jnp.sum((abs_pairs > 0) & (sign_products == -1))
    
    def count_palindromic_matches(relator):
        """Count matching elements from start and end moving inward"""
        active_len = get_active_length(relator)
        # Only consider active part for palindrome detection
        active_relator = relator[:active_len]
        # Compare i-th element from start with i-th from end
        if active_len == 0:
            return 0.0
        reversed_active = active_relator[::-1]
        matches = jnp.sum(active_relator == reversed_active)
        return matches / active_len  # Normalize by length
    
    def count_repeated_patterns(relator):
        """Count repeated substrings of length 2 (simpler pattern detection)"""
        # Get all length-2 substrings
        substrings = jnp.stack([relator[:-1], relator[1:]], axis=1)
        # Only consider active substrings (no zeros)
        active_mask = jnp.all(jnp.abs(substrings) > 0, axis=1)
        active_substrings = substrings[active_mask]
        # Count unique substrings
        if active_substrings.size == 0:
            return 0.0
        # Flatten and convert to hashable-ish representation
        pattern_ids = jnp.abs(active_substrings[:, 0]) * 3 + jnp.sign(active_substrings[:, 0]) * jnp.abs(active_substrings[:, 1])
        unique_patterns = jnp.unique(pattern_ids)
        return jnp.sum(pattern_ids[:, None] == unique_patterns[None, :], axis=0).max()
    
    # Calculate features for each relator
    len_r1 = get_active_length(r1)
    len_r2 = get_active_length(r2)
    total_len = len_r1 + len_r2
    
    # Features
    cancel_r1 = count_cancellation_pairs(r1)
    cancel_r2 = count_cancellation_pairs(r2)
    cancel_total = cancel_r1 + cancel_r2
    
    palin_r1 = count_palindromic_matches(r1)
    palin_r2 = count_palindromic_matches(r2)
    palin_total = (palin_r1 + palin_r2) / 2.0  # Average palindrome score
    
    pattern_complexity_r1 = count_repeated_patterns(r1)
    pattern_complexity_r2 = count_repeated_patterns(r2)
    pattern_avg = (pattern_complexity_r1 + pattern_complexity_r2) / 2.0
    
    # Length ratio (favor balanced lengths)
    length_imbalance = jnp.abs(len_r1 - len_r2) / (jnp.maximum(len_r1, len_r2) + 1e-6)
    
    # Normalized presentation length (0=minimum, 1=maximum)
    norm_length = total_len / MAX_PRESENTATION_LENGTH
    
    # Combine features into final score with reasonable weights
    # Higher weight on cancellation potential (indicates immediate simplification possible)
    # Lower weight on palindrome detection (indicates potential symmetry)
    # Penalty for length and pattern complexity (complexity often indicates non-triviality)
    
    base_score = (
        0.35 * (1 - norm_length) +  # Length component (0 to 1)
        0.30 * (cancel_total / MAX_RELATOR_LENGTH) +  # Cancellation potential
        0.20 * palin_total +  # Palindromic structure
        0.10 * (1 - length_imbalance) -  # Length balance
        0.10 * (jnp.log(pattern_avg + 1e-6) / jnp.log(10))  # Pattern complexity penalty
    )
    
    # Sigmoid normalization to [0, 1]
    # Map reasonable range [-2, 2] to [0, 1]
    score = 1 / (1 + jnp.exp(-5 * (base_score + 0.5)))
    return score
# [EVOLVE-BLOCK-END]
```


---

## Request 5
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v4`.

**Latency:** 20.651 s

**Output:**

Looking at the problem, I need to create a heuristic that goes beyond simple length minimization. The key challenges are:

1. Intermediate steps may need to increase length to enable future cancellations
2. The heuristic should detect "cancellation potential" - patterns that suggest the relators can be transformed to trivial forms
3. Need to identify structural properties like:
   - Self-cancellation patterns (palindromes with inverses)
   - Common substrings between relators that could be eliminated via conjugation
   - Balanced generator/inverse counts (each relator should ideally have equal counts of generators and their inverses for triviality)

My approach for `heuristic_fn_v4`:
1. Continue to track presentation length (shorter is generally better)
2. Add measures for:
   - "Cancellation pairs" within each relator (adjacent generator+inverse pairs)
   - Balance of generators vs inverses in each relator
   - Potential for cancellation via substring matching between relators
3. Use these to create a bounded [0,1] score

Here's the improved heuristic:

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that captures cancellation potential beyond simple length.
    
    Key improvements over baseline:
    - Counts adjacent cancellation pairs (x, x^-1) within relators
    - Measures generator/inverse balance in each relator (should be balanced for trivial group)
    - Penalizes unbalanced generator counts across the presentation
    
    Args:
        presentation: Array representing current group presentation [r_1; r_2]. Shape (72,) int32.
            0 is padding, 1/-1 for x1, 2/-2 for x2.
    
    Returns:
        Scalar heuristic in [0,1] (higher = better potential for trivialization).
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Split presentation into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Create mask for non-padding positions
    def get_valid_mask(relator):
        return jnp.abs(relator) > 0
    
    mask1 = get_valid_mask(r1)
    mask2 = get_valid_mask(r2)
    
    # Count lengths
    len1 = jnp.sum(mask1)
    len2 = jnp.sum(mask2)
    
    # Basic length normalization (shorter is generally better, but not absolute)
    total_len = len1 + len2
    length_score = 1.0 - jnp.clip(total_len / MAX_PRESENTATION_LENGTH, 0.0, 1.0)
    
    # Count cancellation pairs (adjacent x, x^-1) within each relator
    def count_cancellation_pairs(relator, mask):
        # Shifted versions to compare adjacent elements
        shifted = jnp.concatenate([relator[1:], jnp.array([0])])
        # Check if adjacent pairs are inverses: a + b == 0 where a,b != 0
        is_inverse = (relator + shifted == 0) & (mask & jnp.concatenate([mask[1:], jnp.array([False])]))
        return jnp.sum(is_inverse)
    
    cancel1 = count_cancellation_pairs(r1, mask1)
    cancel2 = count_cancellation_pairs(r2, mask2)
    
    # Cancellation pair score (reward for adjacent inverse pairs)
    cancel_score = (cancel1 + cancel2) / MAX_RELATOR_LENGTH
    
    # Generator balance score: each relator should have balanced counts of each generator type
    def generator_balance(relator, mask):
        # For each generator type (1,2), count positive vs negative
        balance = 0.0
        for gen in [1, 2]:
            pos_count = jnp.sum((relator == gen) & mask)
            neg_count = jnp.sum((relator == -gen) & mask)
            # Perfect balance is ideal; maximum imbalance is length of relator
            total = pos_count + neg_count
            # Add 1 to avoid division by zero, normalize
            if total > 0:
                balance += 1.0 - jnp.abs(pos_count - neg_count) / total
            else:
                balance += 1.0  # No occurrences = perfectly balanced
        return balance / 2.0
    
    balance1 = generator_balance(r1, mask1)
    balance2 = generator_balance(r2, mask2)
    balance_score = (balance1 + balance2) / 2.0
    
    # Combined score with weights emphasizing cancellation and balance over pure length
    # This reflects the insight that intermediate steps may be longer but have better structure
    total_score = 0.3 * length_score + 0.35 * cancel_score + 0.35 * balance_score
    
    return jnp.clip(total_score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 6
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v5`.

**Latency:** 21.808 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v5(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that captures structural potential for trivialisation by:
    1. Prioritising reduced presentations (cancellation pairs)
    2. Penalising long non-cancelling segments
    3. Detecting palindromic/rotationally symmetric relators (signs of potential reducibility)
    4. Rewarding balance between relators (similar lengths suggest symmetry)
    5. Using a smooth sigmoid scaling to bound output in [0,1]
    
    These features better capture "hidden structure" even in long intermediate presentations
    that may be part of a valid trivialisation path.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Split presentation into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Helper to compute reduction: count adjacent cancellation pairs
    def count_cancellations(relator):
        # For positions i and i+1, check if they are inverses
        pair_product = relator[:-1] * relator[1:]
        # Inverses multiply to -1 or -4 (1*-1=-1, 2*-2=-4)
        cancellation_pairs = (pair_product == -1) | (pair_product == -4)
        return jnp.sum(cancellation_pairs)
    
    r1_cancels = count_cancellations(r1)
    r2_cancels = count_cancellations(r2)
    
    # Total effective cancellations (upper bound on reducible length)
    total_cancellations = r1_cancels + r2_cancels
    
    # Compute actual lengths (non-zero elements)
    def count_nonzero(relator):
        return jnp.sum(relator != 0)
    
    len_r1 = count_nonzero(r1)
    len_r2 = count_nonzero(r2)
    total_length = len_r1 + len_r2
    
    # Estimate of reducible length (upper bound)
    reducible_length = total_cancellations
    
    # Penalty for long presentations relative to reducible potential
    # Using log to smooth growth
    length_penalty = jnp.log1p(total_length) / jnp.log1p(MAX_PRESENTATION_LENGTH)
    
    # Reward for reducible potential (ratio of cancellations to length)
    potential_ratio = reducible_length / jnp.maximum(total_length, 1.0)
    
    # Palindrome detection: check if relator reads same forwards and backwards (with inverse)
    def palindrome_score(relator):
        # Find first and last non-zero indices
        indices = jnp.arange(MAX_RELATOR_LENGTH)
        non_zero_mask = relator != 0
        first_idx = jnp.argmax(non_zero_mask * indices) if jnp.any(non_zero_mask) else 0
        last_idx = jnp.argmax(non_zero_mask[::-1] * (MAX_RELATOR_LENGTH - 1 - indices)) if jnp.any(non_zero_mask) else MAX_RELATOR_LENGTH - 1
        
        # Extract active segment
        active_length = last_idx - first_idx + 1
        segment = relator[first_idx:first_idx + active_length]
        
        # Check for palindrome with inverse (relator[i] == -relator[-i-1])
        reversed_segment = segment[::-1]
        inverse_segment = -reversed_segment
        matches = segment == inverse_segment
        return jnp.mean(matches.astype(jnp.float32)) if active_length > 0 else 0.0
    
    pal1 = palindrome_score(r1)
    pal2 = palindrome_score(r2)
    avg_palindrome = (pal1 + pal2) / 2.0
    
    # Balance penalty: difference in lengths normalized
    length_balance = 1.0 - jnp.abs(len_r1 - len_r2) / jnp.maximum(jnp.maximum(len_r1, len_r2), 1.0)
    
    # Combine features with learned weights (heuristic constants)
    c1, c2, c3, c4 = 0.4, 0.3, 0.2, 0.1  # weights for each component
    
    # Base score from reducibility potential and length penalty
    base_score = c1 * potential_ratio + c2 * (1.0 - length_penalty)
    
    # Add bonus for structural symmetry features
    structural_bonus = c3 * avg_palindrome + c4 * length_balance
    
    # Final score: combination with smooth sigmoid normalization
    raw_score = base_score + structural_bonus
    # Clamp and scale to [0,1]
    normalized_score = jnp.clip(raw_score, 0.0, 1.0)
    
    # Special bonus for presentations that are already trivial (two relators of length 0 or 2 with x1 and x2)
    # Check if presentation is trivial: two relators each of length 1 with values 1 and 2
    is_trivial = (len_r1 <= 1) & (len_r2 <= 1) & (total_length <= 2)
    return jnp.where(is_trivial, 1.0, normalized_score)
# [EVOLVE-BLOCK-END]
```


---

## Request 7
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v6`.

**Latency:** 20.461 s

**Output:**

Looking at the problem, I need to create a more sophisticated heuristic that can detect "structure" or "potential for cancellation" even in long presentations. The key insight is that intermediate states might have long relators but still be on a path to trivialization, so we need measures beyond just length.

Let me analyze what makes a presentation "closer to trivial":
1. Shorter total length is good (but not the only factor)
2. More opportunities for cancellation (like adjacent inverse pairs)
3. Palindromic or symmetric structure
4. Repeated patterns that might simplify via AC moves
5. Relators that are close to being powers of generators

I'll create `heuristic_fn_v6` that combines multiple signals with careful weighting:

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that combines multiple signals for detecting trivialization potential:
    1. Normalized presentation length (shorter is better)
    2. Cancellation potential (adjacent inverse pairs and palindromic structure)
    3. Relator structure analysis (repetitions, symmetry)
    4. Balance between relators (similar lengths often indicates progress)
    
    Args:
        presentation: Array representing current group presentation in terms of relators
            [r_1; r_2]. Shape (72,) int32 array. 0 is padding.
    
    Returns:
        Scalar heuristic value in [0,1] estimating likelihood of trivialisation in the
        future (higher is better).
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    
    # Split presentation into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Create masks for non-padding positions
    mask1 = r1 != 0
    mask2 = r2 != 0
    
    # Calculate actual lengths
    len1 = jnp.sum(mask1)
    len2 = jnp.sum(mask2)
    total_len = len1 + len2
    max_total_len = 2 * MAX_RELATOR_LENGTH
    
    # Normalized length score (shorter is better, capped at 0.8)
    length_score = jnp.maximum(0.0, 1.0 - total_len / max_total_len * 0.8)
    
    # Cancellation potential: count adjacent inverse pairs within each relator
    def count_inverses(relator, mask):
        # Get absolute values for inverse detection
        abs_rel = jnp.abs(relator)
        
        # Check for inverse pairs at positions i and i+1
        shift_forward = jnp.roll(relator, -1)
        # For last position, wrap to first (periodic boundary for structure detection)
        # But ensure we only compare non-padding positions
        valid_pairs = mask & jnp.roll(mask, -1)
        inverse_pairs = (relator + shift_forward == 0) & valid_pairs
        return jnp.sum(inverse_pairs)
    
    inv1 = count_inverses(r1, mask1)
    inv2 = count_inverses(r2, mask2)
    
    # Palindromic symmetry score: count matching positions from ends
    def palindromic_score(relator, mask):
        # Reverse the relator (with masking consideration)
        reversed_r = jnp.where(mask, relator, 0)[::-1]
        # Count matching non-zero positions
        matches = (relator == reversed_r) & mask
        return jnp.sum(matches)
    
    pal1 = palindromic_score(r1, mask1)
    pal2 = palindromic_score(r2, mask2)
    
    # Symmetry score (normalized by length)
    avg_len = (len1 + len2) / 2 + 1e-6
    symmetry_score = (pal1 + pal2) / (2 * avg_len)
    
    # Balance between relator lengths (prefer similar lengths)
    length_balance = 1.0 - jnp.abs(len1 - len2) / (max_total_len / 2 + 1e-6)
    
    # Generators used (prefer using fewer distinct generators effectively)
    used_gens = jnp.sum((jnp.abs(r1) > 0) | (jnp.abs(r2) > 0))
    gen_score = 1.0 - used_gens / (2 * max_total_len)
    
    # Combined heuristic with reasonable weights
    # Emphasize cancellation potential and symmetry as these are key for AC moves
    score = (
        0.3 * length_score +
        0.25 * symmetry_score +
        0.2 * length_balance +
        0.15 * (inv1 + inv2) / (avg_len + 1) +
        0.1 * gen_score
    )
    
    # Ensure bounded to [0,1]
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 8
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v7`.

**Latency:** 20.287 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v7(presentation: jnp.ndarray) -> float:
    r"""
    Improved heuristic that captures structural properties for trivialisation potential:
    1. Penalizes long presentations (baseline)
    2. Rewards high cancellation potential: counts matching adjacent generator-inverse pairs
    3. Detects palindromic structure in relators (common in trivial group presentations)
    4. Rewards presence of trivial subwords (x x^{-1} patterns)
    5. Encourages symmetry between relators
    
    The heuristic uses a weighted combination that balances length with structural
    indicators of reducibility, helping navigate through intermediate states that
    temporarily increase length but enable later cancellation.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    
    # Extract the two relators
    relator1 = presentation[:MAX_RELATOR_LENGTH]
    relator2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Count non-zero (actual) elements in each relator
    len1 = jnp.sum(relator1 != 0)
    len2 = jnp.sum(relator2 != 0)
    total_length = len1 + len2
    
    # Normalize length penalty to [0, 1] range, with shorter being better
    max_possible_len = 2 * MAX_RELATOR_LENGTH
    length_score = 1.0 - total_length / max_possible_len
    
    # Compute cancellation potential (count adjacent x x^{-1} or x^{-1} x pairs)
    def calc_cancellation(relator):
        # Check adjacent pairs: if relator[i] * relator[i+1] == -1, they're inverses
        # We consider pairs (1,-1), (-1,1), (2,-2), (-2,2)
        pairs = relator[:-1] * relator[1:]
        # Count positions where product is -1 or -4 (for generator pairs)
        inv1_count = jnp.sum(pairs == -1)
        inv2_count = jnp.sum(pairs == -4)
        return inv1_count + inv2_count
    
    cancel1 = calc_cancellation(relator1)
    cancel2 = calc_cancellation(relator2)
    cancellation_score = (cancel1 + cancel2) / (MAX_RELATOR_LENGTH - 1)  # normalize by max possible pairs
    
    # Check for palindrome structure (good for trivial group presentations)
    def palindrome_score(relator):
        # Get non-zero elements only
        mask = relator != 0
        non_zero = relator * mask
        # Reverse the non-zero portion
        reversed_rel = jnp.where(mask, relator[::-1], 0)
        
        # Compare non-zero positions
        match_count = jnp.sum((relator != 0) & (reversed_rel != 0) & (relator == reversed_rel))
        total_nonzero = jnp.sum(mask)
        return match_count / total_nonzero if total_nonzero > 0 else 0.0
    
    pal1 = palindrome_score(relator1)
    pal2 = palindrome_score(relator2)
    palindrome_score_val = (pal1 + pal2) / 2.0
    
    # Check for symmetric relators (r2 ≈ r1 or r2 ≈ r1^{-1})
    def symmetry_score():
        # Create inverse of relator1
        inv1 = -relator1
        # Pad both to same length for comparison
        len1_pad = jnp.sum(relator1 != 0)
        len2_pad = jnp.sum(relator2 != 0)
        min_len = jnp.minimum(len1_pad, len2_pad)
        
        if min_len == 0:
            return 0.0
        
        # Compare first min_len elements of relator2 with relator1 and inv1
        r1_match = jnp.sum(relator1[:min_len] == relator2[:min_len])
        r1_inv_match = jnp.sum(inv1[:min_len] == relator2[:min_len])
        
        return jnp.maximum(r1_match, r1_inv_match) / min_len
    
    sym_score = symmetry_score()
    
    # Combine scores with weights
    # Weight cancellation and symmetry heavily as they indicate reducibility
    weight_length = 0.2
    weight_cancellation = 0.3
    weight_palindrome = 0.2
    weight_symmetry = 0.3
    
    final_score = (weight_length * jnp.clip(length_score, 0.0, 1.0) +
                   weight_cancellation * jnp.clip(cancellation_score, 0.0, 1.0) +
                   weight_palindrome * jnp.clip(palindrome_score_val, 0.0, 1.0) +
                   weight_symmetry * jnp.clip(sym_score, 0.0, 1.0))
    
    return jnp.clip(final_score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 9
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v8`.

**Latency:** 35.054 s

**Output:**

[EVOLVE-BLOCK-START]
def heuristic_fn_v8(presentation: jnp.ndarray) -> float:
    r"""
    Modified heuristic that captures structural properties conducive to trivialisation:
    1. Encourages short presentations (primary driver)
    2. Rewards high degree of local cancellation (adjacent inverse pairs like [1,-1], [2,-2])
    3. Rewards palindromic symmetry in relators (potential for conjugation to simplify)
    4. Rewards presence of sub-palindromes (fragments that could be removed via AC moves)
    5. Penalises long runs of same generator (harder to cancel)
    
    The scoring is normalised to [0,1] using sigmoid-like scaling to avoid extreme values.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = 2 * MAX_RELATOR_LENGTH

    # Extract relators (assume first 36 entries = relator 1, last 36 = relator 2)
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Mask out padding (0s)
    mask1 = r1 != 0
    mask2 = r2 != 0
    
    # Lengths (non-zero entries)
    len1 = jnp.sum(mask1)
    len2 = jnp.sum(mask2)
    total_len = len1 + len2
    
    # Normalised length component (0=bad, 1=good)
    # Use exponential decay so short presentations get high scores
    # Normalise against max length 72, but use len=2 as ideal (trivial case)
    ideal_len = 2.0
    length_score = jnp.exp(-0.1 * jnp.maximum(0.0, total_len - ideal_len))
    
    # Local cancellation pairs: count [g, -g] adjacent pairs in each relator
    def count_cancellations(relator, mask):
        # Create shifted version: [1,2,3,...] -> [0,1,2,...]
        shifted = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), relator[:-1]])
        # Identify where current = -shifted and neither is zero
        cancel_pairs = (relator == -shifted) & (relator != 0) & (shifted != 0)
        # Mask out pairs involving padding
        return jnp.sum(cancel_pairs[:-1])  # avoid counting last element
        # Actually, we need to be careful with indexing: pairs at positions (i,i+1)
    
    # Better: compute adjacent cancellation pairs properly
    def adj_cancellation_count(r, m):
        # Consider positions where both r[i] and r[i+1] are non-zero
        # Only count if r[i] == -r[i+1]
        non_zero_mask = (r[:-1] != 0) & (r[1:] != 0)
        cancel_mask = (r[:-1] == -r[1:]) & non_zero_mask
        return jnp.sum(cancel_mask.astype(jnp.float32))
    
    cancel_count = adj_cancellation_count(r1, mask1) + adj_cancellation_count(r2, mask2)
    
    # Normalise cancellation score: max possible is roughly length-1 per relator
    # Use log scaling to avoid saturation
    max_cancel = MAX_RELATOR_LENGTH - 1
    cancel_score = jnp.log1p(cancel_count) / jnp.log1p(max_cancel)
    
    # Palindromic symmetry: check how many prefix/suffix matches (within tolerance)
    # For relator r, compare r[i] with r[-i-1] for non-zero entries
    def pal_score(relator):
        # Truncate at actual length
        actual_len = jnp.sum(relator != 0)
        # Only check if length >= 2
        def safe_pal():
            # Get actual elements (trimming 0s)
            # Use jnp.where to get indices of non-zero elements
            indices = jnp.nonzero(relator != 0, size=MAX_RELATOR_LENGTH, fill_value=0)[0]
            actual = jnp.take(relator, indices)
            # Compare first half with reversed second half
            half = actual.shape[0] // 2
            if half < 1:
                return 1.0
            # Compute matching prefix/suffix
            rev = jnp.flip(actual[:half])
            forward = actual[-half:]
            matches = jnp.sum((actual[:half] == jnp.flip(actual[-half:])).astype(jnp.float32))
            return matches / half
        return jax.lax.cond(actual_len >= 2, safe_pal, lambda: 0.5)
    
    pal_score_r1 = pal_score(r1)
    pal_score_r2 = pal_score(r2)
    pal_score_mean = (pal_score_r1 + pal_score_r2) / 2.0
    
    # Combined score: weighted sum
    w_len = 0.5
    w_cancel = 0.3
    w_pal = 0.2
    
    raw_score = w_len * length_score + w_cancel * cancel_score + w_pal * pal_score_mean
    
    # Sigmoid-like mapping to [0,1]
    # Ensure strict bounds
    score = 0.95 * (1.0 - jnp.exp(-raw_score * 3.0)) + 0.05
    
    # Clamp to [0,1]
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]


---

## Request 10
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v9`.

**Latency:** 25.893 s

**Output:**

[EVOLVE-BLOCK-START]
def heuristic_fn_v9(presentation: jnp.ndarray) -> float:
    r"""
    Improved heuristic based on:
    1. *Cancellation potential*: counts maximal cancellation pairs (e.g., 1 followed by -1) across relator boundaries and within each relator.
    2. *Palindrome symmetry*: rewards relators that are symmetric (potential self-cancellation under conjugation).
    3. *Balanced generator counts*: rewards relators where each generator appears equally often as its inverse (necessary for triviality).
    4. *Relator similarity*: encourages structural similarity between r1 and r2 (a pattern observed in known trivial presentations).
    All components are normalized and combined into a bounded [0,1] score.
    """
    # Parse relators: first 36 entries = r1, next 36 = r2
    r1 = presentation[:36]
    r2 = presentation[36:]

    # Mask padding (0s)
    def get_active_portion(relator):
        mask = relator != 0
        return relator[mask]

    r1_active = get_active_portion(r1)
    r2_active = get_active_portion(r2)

    # Length normalization (penalize long presentations)
    total_len = r1_active.shape[0] + r2_active.shape[0]
    MAX_LEN = 72.0
    len_norm = 1.0 - (total_len / MAX_LEN)
    len_norm = jnp.maximum(0.0, len_norm)  # floor at 0

    # Cancellation pairs within each relator: count (g, -g) adjacent pairs
    def count_cancellation_pairs(relator):
        pairs = relator[:-1] * relator[1:]  # adjacent products: g * (-g) = -g^2 -> negative if cancellation
        # A cancellation pair is when product < 0 and abs product is perfect square of generator (1 or 4)
        abs_pairs = jnp.abs(pairs)
        valid_cancellation = (pairs < 0) & ((abs_pairs == 1) | (abs_pairs == 4))
        return jnp.sum(valid_cancellation.astype(jnp.int32))

    cancel_r1 = count_cancellation_pairs(r1_active)
    cancel_r2 = count_cancellation_pairs(r2_active)

    # Cancellation across the concatenation boundary r1[-1] + r2[0]
    def boundary_cancellation(r1_arr, r2_arr):
        if (r1_arr.shape[0] == 0) or (r2_arr.shape[0] == 0):
            return 0
        product = r1_arr[-1] * r2_arr[0]
        return ((product < 0) & ((jnp.abs(product) == 1) | (jnp.abs(product) == 4))).astype(jnp.int32)

    boundary = boundary_cancellation(r1_active, r2_active)
    total_cancel = cancel_r1 + cancel_r2 + boundary

    # Normalise cancellation by expected max: length-1 per relator + 1 boundary
    max_cancel = max(r1_active.shape[0] - 1, 0) + max(r2_active.shape[0] - 1, 0) + 1
    cancel_norm = total_cancel / max(max_cancel, 1)

    # Balanced generator counts: |count(g) - count(-g)| should be small
    def generator_balance(relator):
        if relator.shape[0] == 0:
            return 0.0
        counts = jnp.array([
            jnp.sum(relator == 1) - jnp.sum(relator == -1),
            jnp.sum(relator == 2) - jnp.sum(relator == -2)
        ])
        return jnp.sum(jnp.abs(counts)) / relator.shape[0]  # deviation per symbol

    balance_r1 = generator_balance(r1_active)
    balance_r2 = generator_balance(r2_active)
    balance_norm = 1.0 - jnp.minimum(1.0, (balance_r1 + balance_r2))

    # Palindrome symmetry: inner product of relator with its reversed sign-flipped version
    def palindrome_score(relator):
        if relator.shape[0] == 0:
            return 0.0
        rev = relator[::-1]
        # reversed ideal for trivial relator: r == rev * (-1) (i.e., r_i = -rev_i)
        sign_rev = -rev
        matches = jnp.sum((relator == sign_rev).astype(jnp.int32))
        return matches / relator.shape[0]

    pal_r1 = palindrome_score(r1_active)
    pal_r2 = palindrome_score(r2_active)
    pal_norm = (pal_r1 + pal_r2) / 2.0

    # Similarity between r1 and r2: longest common subsequence approximation via dot product
    # Use normalized convolution-like similarity
    def relator_similarity(a, b):
        # Zero-pad to equal length
        len_a = a.shape[0]
        len_b = b.shape[0]
        max_len = max(len_a, len_b)
        a_pad = jnp.concatenate([a, jnp.zeros(max_len - len_a, dtype=jnp.int32)])
        b_pad = jnp.concatenate([b, jnp.zeros(max_len - len_b, dtype=jnp.int32)])
        norm = jnp.linalg.norm(a_pad.astype(jnp.float32)) * jnp.linalg.norm(b_pad.astype(jnp.float32))
        if norm == 0:
            return 0.0
        return jnp.abs(jnp.dot(a_pad.astype(jnp.float32), b_pad.astype(jnp.float32))) / norm

    sim = relator_similarity(r1_active, r2_active)

    # Combine components (weights chosen heuristically based on empirical properties)
    # Weight summary:
    #   - len_norm: strong negative effect (penalize long)
    #   - cancel_norm: strong positive (direct cancellation potential)
    #   - balance_norm: moderate (trivial group requires balanced relators)
    #   - pal_norm: moderate (self-inverse structure)
    #   - sim: strong (similar relators often appear in trivial presentations)
    score = (
        0.2 * len_norm +
        0.35 * cancel_norm +
        0.15 * balance_norm +
        0.15 * pal_norm +
        0.15 * sim
    )

    return jnp.clip(score, 0.0, 1.0).astype(jnp.float32)
# [EVOLVE-BLOCK-END]


---

## Request 11
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v10`.

**Latency:** 25.413 s

**Output:**

[EVOLVE-BLOCK-START]
def heuristic_fn_v10(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that combines multiple structural signals:
    1. Normalized presentation length (lower is better)
    2. Cancellation potential: count of adjacent inverse pairs across relator boundaries
    3. Palindromic symmetry score for relators (suggests potential for simplification)
    4. Generator balance: measures symmetry between x1 and x2 usage
    
    These features are combined with hand-tuned weights based on group-theoretic
    intuition that balanced, symmetric presentations with cancellation potential
    are more likely to be trivialisable.
    
    Args:
        presentation: Array representing relators r1, r2 concatenated. Shape (72,), int32.
            0 = padding/identity, 1/-1 = x1/x1^{-1}, 2/-2 = x2/x2^{-1}
    
    Returns:
        Scalar heuristic in [0,1] (higher = better chance of trivialisation)
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    
    # Extract relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Trim padding to get actual lengths
    r1_len = jnp.sum(jnp.abs(r1) > 0)
    r2_len = jnp.sum(jnp.abs(r2) > 0)
    total_len = r1_len + r2_len
    
    # Normalize length term to [0,1] where 1 = minimal length (2)
    # We use a smooth exponential decay to avoid sharp boundaries
    length_score = jnp.exp(-0.05 * (total_len - 2))
    
    # Compute adjacent inverse pairs (potential cancellation points)
    # Check between r1[-1] and r2[0], and within each relator
    def count_adjacent_inverses(relator):
        # Check adjacent pairs in relator
        rel = relator[:jnp.max(jnp.array([1, jnp.sum(jnp.abs(relator) > 0)])) - 1]
        next_el = relator[1:jnp.sum(jnp.abs(relator) > 0)]
        return jnp.sum(rel + next_el == 0)
    
    # Total adjacent inverse pairs within each relator
    inv_r1 = count_adjacent_inverses(r1)
    inv_r2 = count_adjacent_inverses(r2)
    
    # Check between relators (last non-zero of r1 and first non-zero of r2)
    # Find last non-zero in r1
    r1_mask = jnp.abs(r1) > 0
    r1_indices = jnp.arange(MAX_RELATOR_LENGTH)
    last_r1_idx = jnp.max(jnp.where(r1_mask, r1_indices, -1)) if jnp.any(r1_mask) else -1
    
    # Find first non-zero in r2
    r2_mask = jnp.abs(r2) > 0
    r2_indices = jnp.arange(MAX_RELATOR_LENGTH)
    first_r2_idx = jnp.min(jnp.where(r2_mask, r2_indices, MAX_RELATOR_LENGTH)) if jnp.any(r2_mask) else MAX_RELATOR_LENGTH
    
    # Compute between-relator inverse
    between_inv = 0.0
    if last_r1_idx >= 0 and first_r2_idx < MAX_RELATOR_LENGTH:
        between_inv = jnp.where(r1[last_r1_idx] + r2[first_r2_idx] == 0, 1.0, 0.0)
    
    total_inv = inv_r1 + inv_r2 + between_inv
    
    # Normalize inverse count (logarithmic scaling)
    inv_score = jnp.arctan(0.5 * total_inv) / (jnp.pi/2)
    
    # Palindrome score: check how symmetric each relator is
    def palindrome_score(relator):
        # Get actual relator without padding
        mask = jnp.abs(relator) > 0
        indices = jnp.arange(MAX_RELATOR_LENGTH)
        real_len = jnp.sum(mask)
        
        if real_len == 0:
            return 1.0  # Empty relator is maximally symmetric
            
        rel_nonzero = jnp.where(mask, relator, 0)
        # Compare relator with its inverse-reversed version
        rev_indices = jnp.arange(real_len - 1, -1, -1)
        reversed_rel = rel_nonzero[rev_indices]
        inverse_reversed = -reversed_rel
        # Compare up to real_len elements
        matches = jnp.sum((rel_nonzero[:real_len] == inverse_reversed[:real_len]) * mask[:real_len])
        return matches / real_len
    
    pal_score = (palindrome_score(r1) + palindrome_score(r2)) / 2.0
    
    # Generator balance: measure symmetry between x1 and x2 usage
    # Count x1 and x2 occurrences (including inverses) in each relator
    def generator_counts(relator):
        mask = jnp.abs(relator) > 0
        count_x1 = jnp.sum((jnp.abs(relator) == 1) * mask)
        count_x2 = jnp.sum((jnp.abs(relator) == 2) * mask)
        return count_x1, count_x2
    
    x1_r1, x2_r1 = generator_counts(r1)
    x1_r2, x2_r2 = generator_counts(r2)
    
    # Compute balance as inverse relative difference
    def balance_score(a, b):
        total = a + b
        return jnp.where(total == 0, 1.0, 1.0 - jnp.abs(a - b) / total)
    
    bal_r1 = balance_score(x1_r1, x2_r1)
    bal_r2 = balance_score(x1_r2, x2_r2)
    bal_score = (bal_r1 + bal_r2) / 2.0
    
    # Combine scores with weights reflecting importance
    # Main factors: short length, high cancellation potential, symmetry
    score = (
        0.4 * length_score +
        0.3 * inv_score +
        0.2 * pal_score +
        0.1 * bal_score
    )
    
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]


---

## Request 12
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v11`.

**Latency:** 15.215 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v11(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that captures structural properties conducive to trivialisation.
    
    Key improvements:
    1. Measures cancellation potential by counting matching generator/inverse pairs across relators
    2. Penalizes long runs of repeated generators (indicating inefficient representation)
    3. Rewards symmetry (palindromic patterns) as they often indicate reducible structure
    4. Penalizes imbalance between relator lengths as balanced presentations are easier to manipulate
    
    Args:
        presentation: Array representing current group presentation in terms of relators
            [r_1; r_2]. Shape (72,) int32 array. 0 is padding.
    
    Returns:
        Scalar heuristic value in [0,1] estimating likelihood of trivialisation in the
        future (higher is better).
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Remove padding (0) to get actual relators
    r1_active = r1[r1 != 0]
    r2_active = r2[r2 != 0]
    
    # Total presentation length (normalized)
    total_length = len(r1_active) + len(r2_active)
    max_possible_length = 2 * MAX_RELATOR_LENGTH
    
    # Cancellation potential score: count matching generator/inverse pairs
    # between relators (e.g., 1 and -1 cancel)
    # Use vectorized operations for efficiency
    # Count positive and negative occurrences across both relators
    all_generators = jnp.concatenate([r1_active, r2_active])
    
    # Count how many pairs would cancel (min of counts for each generator type)
    counts = jnp.bincount(jnp.abs(all_generators), length=N_GENERATORS+1)
    # Generator types 1 and 2 (index 0 unused)
    cancel_pairs = jnp.minimum(counts[1], counts[2]) * 2  # each pair contributes 2 to cancellation
    cancel_score = cancel_pairs / max_possible_length
    
    # Palindromic potential: check if relators have symmetric structure
    def palindromic_score(relator):
        if len(relator) == 0:
            return 0.0
        # Check how many positions match when reversed
        reversed_rel = relator[::-1]
        matches = jnp.sum(relator == reversed_rel)
        return matches / len(relator)
    
    p1 = palindromic_score(r1_active) if len(r1_active) > 0 else 0.0
    p2 = palindromic_score(r2_active) if len(r2_active) > 0 else 0.0
    symmetry_score = (p1 + p2) / 2.0
    
    # Length penalty (we want shorter presentations, but not at cost of losing structure)
    length_penalty = 1.0 - (total_length / max_possible_length) * 0.7
    
    # Balance between relator lengths (prefers similar lengths)
    length_balance = 1.0 - jnp.abs(len(r1_active) - len(r2_active)) / max_possible_length
    
    # Combine scores with weights learned from analysis
    # Weights emphasize structural features over raw length
    score = (
        0.4 * cancel_score +
        0.3 * symmetry_score +
        0.2 * length_penalty +
        0.1 * length_balance
    )
    
    # Ensure bounded in [0,1]
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 13
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v12`.

**Latency:** 25.206 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v12(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic focusing on cancellation potential and structural simplicity.
    Key modifications from v0:
    1. Counts "cancelable pairs" (adjacent inverse pairs like [1, -1] or [-2, 2])
    2. Counts symmetric patterns (palindromic substrings) that suggest reducibility
    3. Penalizes long runs of the same generator (which resist cancellation)
    4. Incorporates presentation length but with diminishing returns to avoid local maxima
    5. Detects self-conjugate patterns (r_i ≈ g r_i g^{-1}) indicating group structure
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Remove padding (0) for processing
    def strip_padding(relator):
        mask = relator != 0
        return jnp.where(mask, relator, 0)
    
    r1 = strip_padding(r1)
    r2 = strip_padding(r2)
    
    # Helper to get non-zero length
    def get_length(relator):
        return jnp.sum(relator != 0)
    
    len_r1 = get_length(r1)
    len_r2 = get_length(r2)
    total_len = len_r1 + len_r2
    
    # Helper to count cancelable pairs
    def count_cancel_pairs(relator):
        # Check adjacent pairs: [i, -i]
        shift_forward = jnp.roll(relator, -1)
        # Ensure last element doesn't pair with first (cyclic)
        mask = jnp.arange(MAX_RELATOR_LENGTH) < get_length(relator)
        shift_forward = jnp.where(mask, shift_forward, 0)
        
        is_cancel = (relator * shift_forward == -1) | (relator * shift_forward == -4)
        return jnp.sum(is_cancel)
    
    cancel_pairs_r1 = count_cancel_pairs(r1)
    cancel_pairs_r2 = count_cancel_pairs(r2)
    total_cancel_pairs = cancel_pairs_r1 + cancel_pairs_r2
    
    # Helper to count symmetric patterns (substrings that are palindromic)
    def count_palindromes(relator):
        length = get_length(relator)
        # Only check substrings of length 3 to 5 for efficiency
        def check_palindrome_length(k):
            # Check all substrings of length k
            indices = jnp.arange(MAX_RELATOR_LENGTH - k + 1)
            starts = jnp.where(indices < length - k + 1, indices, MAX_RELATOR_LENGTH)
            
            # For each position, extract substring
            def extract_and_check(i):
                substring = relator[i:i+k]
                # Pad if needed
                padded = jnp.where(jnp.arange(k) < (length - i), substring, 0)
                # Check palindrome: compare with reversed
                reversed_sub = jnp.flip(padded)
                return jnp.all(padded == reversed_sub)
            
            counts = jax.vmap(extract_and_check)(indices)
            return jnp.sum(counts)
        
        pal_count_3 = check_palindrome_length(3)
        pal_count_4 = check_palindrome_length(4)
        pal_count_5 = check_palindrome_length(5)
        return pal_count_3 + pal_count_4 + pal_count_5
    
    pal_r1 = count_palindromes(r1)
    pal_r2 = count_palindromes(r2)
    total_palindromes = pal_r1 + pal_r2
    
    # Helper to count long runs (bad for cancellation)
    def count_long_runs(relator):
        length = get_length(relator)
        if length <= 1:
            return 0
        
        # Count positions where same generator repeats 3+ times
        def check_run_at(i):
            run_length = 1
            # Check for run of at least 3
            for j in range(1, 3):
                if i + j < MAX_RELATOR_LENGTH:
                    if relator[i] != 0 and relator[i] == relator[i+j]:
                        run_length += 1
            return (run_length >= 3).astype(jnp.int32)
        
        runs = jax.vmap(check_run_at)(jnp.arange(MAX_RELATOR_LENGTH))
        return jnp.sum(runs)
    
    long_runs_r1 = count_long_runs(r1)
    long_runs_r2 = count_long_runs(r2)
    total_long_runs = long_runs_r1 + long_runs_r2
    
    # Normalize components to [0,1] range
    # Length score: inverse relationship with diminishing returns
    length_score = jnp.exp(-0.03 * total_len)
    
    # Cancellation potential score
    max_cancel = MAX_RELATOR_LENGTH - 1  # max possible adjacent pairs per relator
    cancel_score = jnp.clip(total_cancel_pairs / (2 * max_cancel), 0, 1)
    
    # Palindrome score (structural hint)
    # Estimate max possible palindromes (rough approximation)
    max_pal = MAX_RELATOR_LENGTH * 2  # generous upper bound
    pal_score = jnp.clip(total_palindromes / max_pal, 0, 1)
    
    # Penalty for long runs (bad)
    max_runs = MAX_RELATOR_LENGTH * 0.5  # rough estimate
    run_penalty = jnp.clip(total_long_runs / max_runs, 0, 1)
    run_score = 1 - run_penalty
    
    # Combine scores with weights reflecting heuristic importance
    # Give highest weight to cancellation potential (most direct reduction path)
    # Then palindrome/structure (suggests hidden simplifications)
    # Then length (but with diminishing returns to avoid local maxima)
    # Penalize long runs as they resist simplification
    score = (0.4 * cancel_score + 
             0.3 * pal_score + 
             0.25 * length_score + 
             0.05 * run_score)
    
    # Ensure bounded to [0,1]
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 14
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v13`.

**Latency:** 18.324 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v13(presentation: jnp.ndarray) -> float:
    r"""
    Heuristic for group presentation trivialisation: extends baseline by incorporating
    cancellation potential, palindrome symmetry, and length growth tolerance.
    
    Key improvements over v0:
    1. Counts *local cancellations* (adjacent inverse pairs) which are primary
       drivers of reduction during AC moves
    2. Measures *global symmetry* via reverse-complement matching (palindromic
       self-similarity) which often appears in trivial presentations
    3. Penalises *asymmetry* between relators to encourage balanced trivialisation paths
    4. Uses soft penalty for excessive length while allowing strategic expansion
    
    Returns a bounded [0,1] score where higher indicates greater trivialisation potential.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Extract relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Identify non-padding positions
    is_pad_r1 = (r1 == 0)
    is_pad_r2 = (r2 == 0)
    
    # Count effective length for each relator
    len_r1 = MAX_RELATOR_LENGTH - jnp.sum(is_pad_r1)
    len_r2 = MAX_RELATOR_LENGTH - jnp.sum(is_pad_r2)
    total_len = len_r1 + len_r2
    
    # --- CANCELLATION POTENTIAL ---
    # Count adjacent inverse pairs in each relator (potential for AC1 reduction)
    def count_inverses(relator):
        # Create shifted versions: relator[i] and relator[i+1]
        r_shifted = jnp.concatenate([relator[1:], jnp.array([0])])
        # Check for inverse pairs: a and -a where a ≠ 0
        is_inverse = (relator * r_shifted < 0) & (relator != 0) & (r_shifted != 0)
        # Avoid double counting at boundaries (handled by padding logic)
        return jnp.sum(is_inverse)
    
    canc_r1 = count_inverses(r1)
    canc_r2 = count_inverses(r2)
    total_cancellation = canc_r1 + canc_r2
    
    # --- GLOBAL SYMMETRY (PALINDROMIC CHECK) ---
    # Check if relators or their concatenation resemble reversible structures
    def palindromic_score(relator):
        # Reverse the relator (ignoring padding)
        mask = ~is_pad_r1 if jnp.array_equal(relator, r1) else ~is_pad_r2
        # Build reversed non-pad sequence
        rev_rel = jnp.flip(relator)
        # Only compare non-pad positions
        rev_rel = jnp.where(mask, rev_rel, 0)
        # Compare element-wise (with padding)
        match = (relator == rev_rel) | (mask & (relator == 0)) | (mask & (rev_rel == 0))
        return jnp.mean(jnp.where(mask, match, True))
    
    pal1 = palindromic_score(r1)
    pal2 = palindromic_score(r2)
    avg_pal = (pal1 + pal2) / 2
    
    # --- RELATOR BALANCE ---
    # Prefer balanced relator lengths (closer to equal = better)
    balance = 1.0 - jnp.abs(len_r1 - len_r2) / MAX_RELATOR_LENGTH
    
    # --- SCORE COMPOSITION ---
    # Normalise components to [0,1] range and combine
    # Max cancellation per relator: length-1 (since adjacent pairs), normalise
    max_canc_per_rel = MAX_RELATOR_LENGTH - 1
    norm_cancellation = total_cancellation / (2 * max_canc_per_rel)
    # Length penalty: use exponential decay for short lengths, mild penalty for long
    len_factor = jnp.exp(-total_len / (MAX_PRESENTATION_LENGTH * 2.0))
    # Combine with weights (heuristic tuning)
    score = (0.3 * norm_cancellation + 
             0.3 * avg_pal + 
             0.2 * balance + 
             0.2 * len_factor)
    
    # Clamp to [0,1]
    score = jnp.clip(score, 0.0, 1.0)
    return score
# [EVOLVE-BLOCK-END]
```


---

## Request 15
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v14`.

**Latency:** 22.471 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v14(presentation: jnp.ndarray) -> float:
    r"""
    Improved heuristic that combines multiple structural signals to better estimate
    potential for trivialisation:
    1. Normalised presentation length (shorter is better)
    2. Count of cancellation-prone pairs (adjacent inverse pairs)
    3. Symmetry score (palindromic structure in relators)
    4. Balance score (equal counts of generators and inverses per relator)
    
    These signals are combined with exponentially weighted contributions to prioritize
    cancellation potential and symmetry — key features in successful trivialisation paths.
    
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
    
    # Extract relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Compute actual lengths (non-zero entries)
    len_r1 = jnp.sum(r1 != 0)
    len_r2 = jnp.sum(r2 != 0)
    total_len = len_r1 + len_r2
    
    # Normalise length score (inverse: shorter is better, scaled to [0,1])
    # Use smooth exponential to avoid sharp transitions
    length_score = jnp.exp(-0.05 * total_len)
    
    # Cancellation potential: count adjacent inverse pairs in each relator
    # For r1: check (r1[i] + r1[i+1]) == 0 for i in 0..len_r1-2
    def count_cancellations(relator):
        rel = relator[:MAX_RELATOR_LENGTH]
        # Create shifted versions (left and right)
        left = rel[:-1]
        right = rel[1:]
        # Check inverse pairs: x + (-x) = 0
        inv_check = (left + right) == 0
        # Only count valid (non-zero) pairs
        valid = (left != 0) & (right != 0)
        return jnp.sum(inv_check & valid)
    
    canc_r1 = count_cancellations(r1)
    canc_r2 = count_cancellations(r2)
    total_canc = canc_r1 + canc_r2
    
    # Normalise cancellation score (logarithmic scaling to prevent saturation)
    canc_score = jnp.arctan(0.3 * total_canc) / (jnp.pi / 2)
    
    # Symmetry score: measure palindromic structure in relators
    def symmetry_score(relator):
        rel = relator[:MAX_RELATOR_LENGTH]
        length = jnp.sum(rel != 0)
        # Consider only up to length//2 elements for comparison
        half = jnp.minimum(18, length // 2)
        # Extract relevant prefix and suffix (reversed)
        prefix = rel[:half]
        suffix = jnp.flip(rel[length - half:length]) if length > 0 else jnp.zeros_like(prefix)
        # Compare prefix with reversed suffix (account for inverses)
        # Check if prefix[i] == -suffix[i] for palindromic relator (r[i] == r[n-1-i]^{-1})
        symmetry = (prefix + suffix) == 0
        valid_sym = (prefix != 0) | (suffix != 0)
        if half == 0:
            return 1.0
        return jnp.sum(symmetry & valid_sym) / half
    
    sym_r1 = symmetry_score(r1)
    sym_r2 = symmetry_score(r2)
    avg_sym = (sym_r1 + sym_r2) / 2.0
    
    # Balance score: ratio of generator count to inverse count per relator (should be ~1)
    def balance_score(relator):
        rel = relator[:MAX_RELATOR_LENGTH].astype(jnp.int32)
        # Count positive and negative generators separately
        pos_count = jnp.sum(rel > 0)
        neg_count = jnp.sum(rel < 0)
        # Use harmonic mean-like balance: 2 / (r + 1/r) where r = max(pos/neg, neg/pos)
        ratio = jnp.where(pos_count == 0, 0.0, 
                         jnp.where(neg_count == 0, 0.0,
                                   jnp.minimum(pos_count, neg_count) / jnp.maximum(pos_count, neg_count)))
        return ratio
    
    bal_r1 = balance_score(r1)
    bal_r2 = balance_score(r2)
    avg_bal = (bal_r1 + bal_r2) / 2.0
    
    # Combine scores with weights (optimized heuristically for trivialisation paths)
    # Emphasise cancellation and symmetry heavily as they indicate structural potential
    total_score = (
        0.15 * length_score +
        0.45 * canc_score +
        0.20 * avg_sym +
        0.20 * avg_bal
    )
    
    # Ensure bounded to [0,1]
    return jnp.clip(total_score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 16
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v15`.

**Latency:** 22.136 s

**Output:**

Looking at the problem, I need to create a more sophisticated heuristic that can detect "structure" or "potential for cancellation" even in long presentations, since the baseline (just length) is insufficient.

The key insight is that we need to measure:
1. How close the presentation is to being trivial (short relators, especially identity)
2. Potential for future cancellation (syntactic structure that suggests simplification is possible)
3. Balance between relators (they should be similar in structure for potential cancellation)

Here's my approach for v15:
- Detect cancellation opportunities by counting adjacent inverse pairs within each relator
- Check for palindromic structure (common in trivial group presentations)
- Measure how close each relator is to being the identity
- Include a term for presentation length but with careful normalization to avoid penalizing necessary growth during search
- Add structural symmetry terms between the two relators

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v15(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that detects cancellation potential, structural symmetry,
    and proximity to trivial presentation. Uses multiple signal components:
    - Cancellation pairs (adjacent inverses)
    - Palindromic structure
    - Relator similarity
    - Normalized length with soft ceiling
    
    Args:
        presentation: Array representing current group presentation in terms of relators
            [r_1; r_2]. Shape (72,) int32 array. 0 is padding.
    
    Returns:
        Scalar heuristic value in [0,1] estimating likelihood of trivialisation in the
        future (higher is better).
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Helper to compute cancellation pairs (adjacent inverse pairs)
    def count_cancellation_pairs(relator):
        # Pad with 0 on both sides to handle edge cases
        padded = jnp.concatenate([jnp.array([0]), relator, jnp.array([0])])
        # Check for inverse pairs: adjacent elements that sum to 0
        return jnp.sum((padded[:-1] + padded[1:]) == 0)
    
    # Helper to check palindromic structure (symmetric around center)
    def palindromic_score(relator):
        # Get non-padded portion
        mask = relator != 0
        length = jnp.sum(mask)
        
        # If empty, perfect palindrome
        def empty_case():
            return 1.0
        
        def nonempty_case():
            # Get actual elements
            actual = relator[:length]
            # Reverse and compare
            reversed_actual = actual[::-1]
            matches = jnp.sum(actual == reversed_actual)
            return matches / length
        
        return jax.lax.cond(length == 0, empty_case, nonempty_case)
    
    # Compute components for both relators
    c1 = count_cancellation_pairs(r1)
    c2 = count_cancellation_pairs(r2)
    p1 = palindromic_score(r1)
    p2 = palindromic_score(r2)
    
    # Length-based components (with soft penalty for very long presentations)
    len1 = jnp.sum(r1 != 0)
    len2 = jnp.sum(r2 != 0)
    
    # Normalized length (with soft ceiling using sigmoid-like function)
    def soft_length_score(total_len):
        # Score decreases with length but with diminishing returns
        # Using: 1 / (1 + (length/10)^2) scaled to [0,1]
        return 1.0 / (1.0 + (total_len / 10.0) ** 2)
    
    length_score = soft_length_score(len1 + len2)
    
    # Check if relators are trivial (length 0)
    trivial_score = 0.0
    if len1 == 0 and len2 == 0:
        trivial_score = 1.0
    
    # Similarity between relators (normalized edit distance approximation)
    def similarity():
        min_len = jnp.minimum(len1, len2)
        max_len = jnp.maximum(len1, len2)
        # Simple character overlap score
        overlap = jnp.sum((r1[:min_len] == r2[:min_len]) & (r1[:min_len] != 0))
        return overlap / max_len if max_len > 0 else 0.0
    
    similarity_score = jax.lax.cond((len1 > 0) | (len2 > 0), similarity, lambda: 0.0)
    
    # Combine components with weights (learned or heuristic-tuned)
    # Emphasize cancellation potential and palindromic structure as signs of 
    # underlying group-theoretic simplicity
    score = (
        0.3 * length_score +
        0.2 * (c1 + c2) / 4.0 +  # Normalize cancellation pairs (max 2 per relator typically)
        0.15 * (p1 + p2) / 2.0 +
        0.15 * similarity_score +
        0.2 * trivial_score
    )
    
    # Clamp to [0,1]
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 17
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v16`.

**Latency:** 20.145 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v16(presentation: jnp.ndarray) -> float:
    r"""
    Modified heuristic incorporating:
    1. Normalized cancellation potential (count of adjacent inverse pairs within each relator)
    2. Palindromic structure detection (symmetry suggests potential for simplification)
    3. Relator balance metric (similarity in length and structure between relators)
    4. Identity bias: strong penalty for non-zero padding (incomplete presentations)
    
    This heuristic better captures structural properties that indicate potential for
    successful trivialization, especially for presentations that must temporarily grow
    before shrinking (addressing the AC conjecture's long traversal paths).
    
    Args:
        presentation: Array representing current group presentation in terms of relators
            [r_1; r_2]. Shape (72,) int32 array. 0 is padding.
    
    Returns:
        Scalar heuristic value in [0,1] estimating likelihood of trivialisation.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # 1. Cancellation potential: count inverse pairs (adjacent x, x^-1)
    def count_cancellations(relator):
        # Create shifted versions for adjacent comparison
        shifted = jnp.roll(relator, -1)
        # Check for inverse pairs (x, -x) ignoring zeros
        non_zero_mask = (relator != 0) & (shifted != 0)
        inverse_pairs = (relator == -shifted)
        return jnp.sum(non_zero_mask & inverse_pairs)
    
    cancels_r1 = count_cancellations(r1)
    cancels_r2 = count_cancellations(r2)
    
    # 2. Palindromic score: symmetry within each relator
    def palindromic_score(relator):
        reversed_r = jnp.flip(relator)
        # Compare only non-zero positions
        valid_mask = (relator != 0) | (reversed_r != 0)
        matches = (relator == reversed_r)
        return jnp.sum(valid_mask & matches) / jnp.maximum(jnp.sum(valid_mask), 1.0)
    
    palin_r1 = palindromic_score(r1)
    palin_r2 = palindromic_score(r2)
    
    # 3. Relator length balance
    r1_len = jnp.sum(r1 != 0)
    r2_len = jnp.sum(r2 != 0)
    total_len = r1_len + r2_len
    max_len = MAX_PRESENTATION_LENGTH
    # Normalize to [0,1] with penalty for imbalance
    avg_len = total_len / 2.0
    imbalance = jnp.abs(r1_len - r2_len) / jnp.maximum(avg_len, 1.0)
    balance_score = 1.0 - imbalance
    
    # 4. Padding integrity (penalize accidental padding inside relators)
    # This detects if a relator is "truncated" in an unnatural way
    def padding_score(relator):
        # Count non-zero elements followed by zero (trailing padding is fine)
        non_zero_to_zero = (relator[:-1] != 0) & (relator[1:] == 0)
        # Count total non-zero elements
        total_non_zero = jnp.sum(relator != 0)
        return jnp.sum(non_zero_to_zero) / jnp.maximum(total_non_zero, 1.0)
    
    pad_penalty_r1 = padding_score(r1)
    pad_penalty_r2 = padding_score(r2)
    
    # Combine components with weights reflecting their importance
    # Prioritize cancellation potential and balance, with palindromic and padding as secondary
    cancellation_score = (cancels_r1 + cancels_r2) / 2.0
    normalized_cancellation = jnp.clip(cancellation_score / 4.0, 0.0, 1.0)  # cap at 8 inverse pairs max
    composite_score = (
        0.4 * normalized_cancellation +
        0.2 * (palin_r1 + palin_r2) / 2.0 +
        0.2 * balance_score +
        0.1 * (1.0 - jnp.mean([pad_penalty_r1, pad_penalty_r2]))
    )
    
    # Bonus for trivial presentations (both relators empty or identity)
    is_trivial = (total_len == 0)
    trivial_bonus = 0.1 * is_trivial
    
    # Normalize to [0, 1] range with bonus
    final_score = jnp.clip(composite_score + trivial_bonus, 0.0, 1.0)
    
    return final_score
# [EVOLVE-BLOCK-END]
```


---

## Request 18
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v17`.

**Latency:** 27.850 s

**Output:**

Looking at the problem, I need to create a more sophisticated heuristic for the AC move search problem. The key challenge is that intermediate presentations can grow very large during the trivialisation process, so naive length-based heuristics are insufficient.

The heuristic needs to detect "structure" or "potential for cancellation" even in long presentations. Key insights from group theory and the AC moves:

1. **Cancellation potential**: We want to identify pairs of generators and their inverses that could cancel out
2. **Symmetry/palindromic structure**: Relators that are close to being palindromes might be easier to simplify
3. **Generator balance**: Balanced numbers of generators and their inverses suggest potential for cancellation
4. **Repetition patterns**: Recurring substrings might indicate simplifiable structure

Let me design a heuristic that combines:
- Normalized presentation length (but not as the primary factor)
- Generator/inverse balance for each relator
- Evidence of potential cancellation pairs
- Detection of repeating patterns that might indicate simplifiability

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic based on multiple structural properties:
    - Generator balance (near-zero net exponent for each generator per relator)
    - Cancellation potential (adjacent inverse pairs)
    - Palindromic structure detection
    - Relative length compared to maximum
    
    This heuristic identifies presentations that have structural properties suggesting
    they are close to being trivializable, even if they appear long due to intermediate
    expansions in the AC move process.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Split presentation into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Function to compute properties for a single relator
    def relator_properties(relator):
        # Get non-padding elements
        is_padding = (relator == 0)
        valid_mask = ~is_padding
        valid_len = jnp.sum(valid_mask)
        
        # For length > 0
        def length_positive():
            # Extract actual elements (non-zero)
            relator_elements = jnp.where(valid_mask, relator, 0)
            
            # Generator balance: sum of exponents for each generator
            # generator 1: +1 for 1, -1 for -1; generator 2: +1 for 2, -1 for -2
            gen1_sum = jnp.sum(jnp.where(relator_elements == 1, 1, 0))
            gen1_sum -= jnp.sum(jnp.where(relator_elements == -1, 1, 0))
            gen2_sum = jnp.sum(jnp.where(relator_elements == 2, 1, 0))
            gen2_sum -= jnp.sum(jnp.where(relator_elements == -2, 1, 0))
            
            # Balance score: preference for balanced exponents (closer to 0)
            balance_score = 1.0 - (jnp.abs(gen1_sum) + jnp.abs(gen2_sum)) / (valid_len + 1e-8)
            balance_score = jnp.clip(balance_score, 0.0, 1.0)
            
            # Cancellation potential: count adjacent inverse pairs
            def count_inversions(arr):
                # Create shifted version for comparison
                shifted = jnp.concatenate([arr[1:], jnp.array([0])])
                # Count where adjacent elements are inverses (x and -x)
                inv1 = jnp.sum(jnp.where((arr[:-1] == 1) & (shifted[:-1] == -1), 1, 0))
                inv2 = jnp.sum(jnp.where((arr[:-1] == -1) & (shifted[:-1] == 1), 1, 0))
                inv3 = jnp.sum(jnp.where((arr[:-1] == 2) & (shifted[:-1] == -2), 1, 0))
                inv4 = jnp.sum(jnp.where((arr[:-1] == -2) & (shifted[:-1] == 2), 1, 0))
                return inv1 + inv2 + inv3 + inv4
            
            # Also check wrap-around for potential cyclic cancellations
            def count_wrap_inversions(arr):
                first = arr[0]
                last = arr[-1]
                wrap_inv = 0
                wrap_inv += jnp.where((first == 1) & (last == -1), 1, 0)
                wrap_inv += jnp.where((first == -1) & (last == 1), 1, 0)
                wrap_inv += jnp.where((first == 2) & (last == -2), 1, 0)
                wrap_inv += jnp.where((first == -2) & (last == 2), 1, 0)
                return wrap_inv
            
            cancel_potential = count_inversions(relator_elements)
            wrap_cancel = count_wrap_inversions(relator_elements)
            cancellation_score = 1.0 - jnp.exp(-0.5 * (cancel_potential + wrap_cancel))
            
            # Palindrome check: compare element[i] with -element[-1-i]
            def is_palindrome_like(arr):
                n = valid_len.astype(jnp.float32)
                rev_arr = arr[::-1]
                neg_rev_arr = -rev_arr
                matches = jnp.sum(jnp.where(arr == neg_rev_arr, 1, 0))
                return matches / (n + 1e-8)
            
            pal_score = jnp.where(valid_len > 0, is_palindrome_like(relator_elements), 0.0)
            
            # Normalized length component
            length_score = 1.0 - (valid_len / MAX_RELATOR_LENGTH)
            length_score = jnp.clip(length_score, 0.0, 1.0)
            
            # Combine scores with weights
            # Balance and cancellation are most important
            score = 0.4 * balance_score + 0.3 * cancellation_score + 0.2 * pal_score + 0.1 * length_score
            return score, valid_len
        
        # For empty relator
        def length_zero():
            return jnp.array(1.0), jnp.array(0.0)
        
        return jax.lax.cond(valid_len > 0, length_positive, length_zero)
    
    # Process both relators
    score1, len1 = relator_properties(r1)
    score2, len2 = relator_properties(r2)
    
    # Combined length component
    total_len = len1 + len2
    length_factor = 1.0 - (total_len / MAX_PRESENTATION_LENGTH)
    length_factor = jnp.clip(length_factor, 0.0, 1.0)
    
    # Final combined score: focus on balance and cancellation more than length
    base_score = 0.35 * score1 + 0.35 * score2 + 0.15 * length_factor
    base_score += 0.15 * (1.0 - jnp.tanh(total_len / 20.0))  # Extra penalty for very long presentations
    
    # Ensure output in [0, 1]
    return jnp.clip(base_score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 19
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v18`.

**Latency:** 22.344 s

**Output:**

[EVOLVE-BLOCK-START]
def heuristic_fn_v18(presentation: jnp.ndarray) -> float:
    r"""
    Heuristic v18: Improved cancellation-aware heuristic combining:
    (1) Normalized presentation length (lower is better)
    (2) Expected cancellation potential via local pair counting (adjacent inverse pairs, including across relator boundary)
    (3) Global symmetry score via reverse-complement matching
    (4) Generator balance (equal numbers of each generator and its inverse per relator)
    
    These features are combined in a way that prioritizes cancellation potential, especially
    for long presentations where raw length can be misleading. The design aims to detect
    hidden cancellable structure even in inflated intermediate states.
    
    Args:
        presentation: Array representing current group presentation [r_1; r_2], shape (72,)
    
    Returns:
        Scalar heuristic in [0,1] (higher = better, closer to trivial presentation)
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    
    # Split presentation into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Mask for valid (non-padding) elements
    valid_r1 = r1 != 0
    valid_r2 = r2 != 0
    
    # Count actual lengths
    len_r1 = jnp.sum(valid_r1).astype(float)
    len_r2 = jnp.sum(valid_r2).astype(float)
    
    # Total length normalization (score: shorter = better, 0 to 1)
    total_len = len_r1 + len_r2
    length_score = 1.0 - jnp.clip(total_len / (2 * MAX_RELATOR_LENGTH), a_min=0.0, a_max=1.0)
    
    # 1. Local cancellation potential: count adjacent inverse pairs
    # Check for adjacent inverse pairs in r1
    r1_inv_pairs = jnp.sum(
        (r1[:-1] * r1[1:] == -1) & valid_r1[:-1] & valid_r1[1:]
    )
    r2_inv_pairs = jnp.sum(
        (r2[:-1] * r2[1:] == -1) & valid_r2[:-1] & valid_r2[1:]
    )
    
    # Cross-boundary inverse pair (last of r1 and first of r2)
    cross_inv_pair = jnp.logical_and(
        valid_r1[-1] & valid_r2[0],
        r1[-1] * r2[0] == -1
    ).astype(float)
    
    # 2. Generator-balance score (prefer relators with balanced counts)
    # Count occurrences of each generator and inverse (1, -1, 2, -2)
    counts_r1 = jnp.array([
        jnp.sum(r1 == g) for g in [1, -1, 2, -2]
    ])
    counts_r2 = jnp.array([
        jnp.sum(r2 == g) for g in [1, -1, 2, -2]
    ])
    
    # Balance: max(0, 1 - |#x - #x^{-1}| / length)
    def balance_score(counts, length):
        diff = jnp.abs(counts[0] - counts[1]) + jnp.abs(counts[2] - counts[3])
        max_diff = 2 * MAX_RELATOR_LENGTH  # worst possible imbalance
        return jnp.clip(1.0 - diff / max_diff, a_min=0.0, a_max=1.0)
    
    balance_r1 = balance_score(counts_r1, len_r1) if len_r1 > 0 else 1.0
    balance_r2 = balance_score(counts_r2, len_r2) if len_r2 > 0 else 1.0
    
    # 3. Palindrome/reverse-complement symmetry score
    # Compare r1 with reverse(-r1) and r2 with reverse(-r2)
    def reverse_complement(word):
        # reverse the word, negate each element, and pad with zeros
        rev = jnp.flip(word)
        return jnp.where(word != 0, -rev, 0)
    
    r1_rc = reverse_complement(r1)
    r2_rc = reverse_complement(r2)
    
    # Overlap with reversed complement (ignoring padding)
    r1_sym = jnp.sum((r1 == r1_rc) & valid_r1) / jnp.maximum(len_r1, 1.0)
    r2_sym = jnp.sum((r2 == r2_rc) & valid_r2) / jnp.maximum(len_r2, 1.0)
    
    # Combine scores with weights
    cancellation_potential = (r1_inv_pairs + r2_inv_pairs + cross_inv_pair) / jnp.maximum(total_len, 1.0)
    
    # Composite score (normalize to [0,1])
    composite = (
        0.3 * length_score +
        0.25 * cancellation_potential +
        0.2 * (balance_r1 + balance_r2) / 2 +
        0.25 * (r1_sym + r2_sym) / 2
    )
    
    # Final clamped output
    return jnp.clip(composite, a_min=0.0, a_max=1.0)
# [EVOLVE-BLOCK-END]


---

## Request 20
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v19`.

**Latency:** 23.853 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v19(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that captures structural properties relevant to trivialisation:
    1. Counts cancellation pairs (adjacent inverse pairs within each relator)
    2. Measures symmetry (palindromic structure indicates potential for conjugation moves)
    3. Penalizes long runs of the same generator (reduces complexity for AC moves)
    4. Rewards balanced presentation length ratio between relators (encourages symmetry)
    
    This heuristic better identifies presentations that, while longer, have structure 
    conducive to future cancellation through AC moves (especially AC2 conjugation moves
    that exploit symmetry).
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Split presentation into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Function to count cancellation pairs in a single relator
    def count_cancellation_pairs(relator):
        # Create shifted versions: current element and next element
        shifted = jnp.roll(relator, -1)
        # Cancellation occurs when element and next are exact inverses (sum to 0)
        cancellations = jnp.sum(jnp.abs(relator + shifted) == 0)
        # Also check wrap-around cancellation (last with first)
        wrap_cancellation = jnp.where(
            (relator[-1] != 0) & (relator[0] != 0) & (relator[-1] + relator[0] == 0),
            1, 0
        )
        return cancellations + wrap_cancellation
    
    # Count cancellations in both relators
    c1 = count_cancellation_pairs(r1)
    c2 = count_cancellation_pairs(r2)
    total_cancellations = c1 + c2
    
    # Function to measure palindromic structure
    def palindromic_score(relator):
        # Get non-zero elements
        mask = relator != 0
        non_zero = jnp.where(mask, relator, 0)
        
        # Get actual length
        length = jnp.sum(mask)
        
        if length == 0:
            return 1.0
        
        # Check if palindromic: element[i] should be inverse of element[-i-1]
        # Reverse the relator and check inverse relationship
        reversed_relator = jnp.flip(non_zero)
        inverse_reversed = -reversed_relator
        
        # Count matches between original and inverse-reversed
        matches = jnp.sum(jnp.where(mask, jnp.equal(non_zero, inverse_reversed), 0))
        return matches / length
    
    # Compute palindromic scores for both relators
    p1 = palindromic_score(r1)
    p2 = palindromic_score(r2)
    avg_palindromic = (p1 + p2) / 2.0
    
    # Function to measure run-length penalties
    def run_penalty(relator):
        # Count runs of same generator type
        mask = relator != 0
        non_zero = jnp.where(mask, relator, 0)
        
        if jnp.sum(mask) == 0:
            return 0.0
        
        # Find where generator changes
        shifted = jnp.roll(non_zero, 1)
        changes = jnp.sum(jnp.where(mask, jnp.abs(non_zero - shifted) > 0, 0))
        # Subtract 1 to get number of runs (first element is always start of a run)
        runs = changes
        
        # Penalize long runs by looking at maximum run length
        max_run = 1
        current_run = 1
        prev = 0
        
        for i in range(MAX_RELATOR_LENGTH):
            if non_zero[i] != 0:
                if non_zero[i] == prev:
                    current_run += 1
                    max_run = jnp.maximum(max_run, current_run)
                else:
                    current_run = 1
                    prev = non_zero[i]
            else:
                current_run = 1
                prev = 0
        
        return max_run / 5.0  # Normalize
    
    run_p1 = run_penalty(r1)
    run_p2 = run_penalty(r2)
    avg_run_penalty = (run_p1 + run_p2) / 2.0
    
    # Compute presentation length ratio
    def get_length(relator):
        return jnp.sum(relator != 0)
    
    len1 = get_length(r1)
    len2 = get_length(r2)
    total_length = len1 + len2
    
    # Penalize extreme imbalance between relator lengths
    length_ratio = jnp.minimum(len1, len2) / (jnp.maximum(len1, len2) + 1e-6)
    
    # Normalize lengths for comparison
    length_score = 1.0 - (total_length / MAX_PRESENTATION_LENGTH)
    
    # Combine components with weights determined by importance for trivialisation
    # Base score: high if short and balanced with cancellations and symmetry
    base_score = (
        0.2 * (total_cancellations / 4.0) +        # Reward cancellations (max ~8 for 2 relators)
        0.3 * avg_palindromic +                     # Reward palindromic structure
        0.1 * length_ratio +                        # Reward balanced lengths
        0.3 * length_score -                        # Reward short presentation
        0.1 * avg_run_penalty                       # Penalize long runs
    )
    
    # Ensure output is in [0,1]
    # Clamp to range [0,1] with soft boundaries
    score = jnp.clip(base_score, 0.0, 1.0)
    
    return score
# [EVOLVE-BLOCK-END]
```


---

## Request 21
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v20`.

**Latency:** 23.191 s

**Output:**

Looking at the problem requirements, I need to create a heuristic that captures "structure" or "potential for cancellation" even in long presentations, since intermediate stages may grow very large before trivialization.

The key insight is that good presentations should have:
1. Short length (but not too aggressively since growth is sometimes needed)
2. High potential for cancellation (balanced generators/inverses, palindromic structure, repeated patterns)
3. Symmetry between the two relators

Let me create an improved heuristic that:
- Penalizes long presentations but less severely than baseline
- Rewards balanced generator/inverse counts (more cancellation potential)
- Detects palindromic structure within relators
- Rewards similarity/correlation between the two relators (since trivial group often has symmetric relators)
- Penalizes "frustrated" generators that appear many times without cancellation

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that balances length minimization with cancellation potential detection.
    Key improvements:
    1. Rewards balanced generator/inverse counts within each relator
    2. Detects palindromic structure (self-cancelling potential)
    3. Rewards correlation between the two relators
    4. Uses soft length penalty to avoid local minima
    
    Args:
        presentation: Array representing current group presentation in terms of relators
            [r_1; r_2]. Shape (72,) int32 array. 0 is padding.
    
    Returns:
        Scalar heuristic value in [0,1] estimating likelihood of trivialisation in the
        future (higher is better).
    """
    MAX_RELATOR_LENGTH = 36
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Create masks for valid (non-padding) positions
    mask1 = r1 != 0
    mask2 = r2 != 0
    
    # Lengths of each relator
    len1 = jnp.sum(mask1)
    len2 = jnp.sum(mask2)
    total_len = len1 + len2
    
    # Normalized total length (0=shortest, 1=longest)
    norm_len = total_len / 72.0
    
    # Balance score: how balanced generator/inverse counts are in each relator
    def relator_balance(relator, mask):
        # Count positive and negative occurrences separately
        pos_count = jnp.sum((relator > 0) * mask)
        neg_count = jnp.sum((relator < 0) * mask)
        total = pos_count + neg_count
        balance = jnp.minimum(pos_count, neg_count) / jnp.maximum(total, 1)
        return balance
    
    balance1 = relator_balance(r1, mask1)
    balance2 = relator_balance(r2, mask2)
    avg_balance = (balance1 + balance2) / 2.0
    
    # Palindrome score: check if relator reads similar forward/backward (for cancellation potential)
    def palindrome_score(relator, mask):
        # Only consider non-padding positions
        len_r = jnp.sum(mask)
        if len_r == 0:
            return 1.0
        # Get non-zero elements
        indices = jnp.where(mask, jnp.arange(len(relator)), len(relator))
        valid_indices = jnp.where(mask)
        # Pad for symmetric comparison
        non_zero = jnp.where(mask, relator, 0)
        # Reverse non-zero sequence (simplified: just compare with reversed version using mask)
        reversed_relator = jnp.where(mask, jnp.flip(relator), 0)
        # Count matches between forward and reversed (ignoring padding)
        matches = jnp.sum((non_zero == reversed_relator) * mask)
        return matches / jnp.maximum(len_r, 1)
    
    pal1 = palindrome_score(r1, mask1)
    pal2 = palindrome_score(r2, mask2)
    avg_palindrome = (pal1 + pal2) / 2.0
    
    # Similarity between relators (high similarity may indicate simpler structure)
    # Use normalized correlation of padded sequences
    r1_norm = r1.astype(jnp.float32)
    r2_norm = r2.astype(jnp.float32)
    r1_zero = jnp.where(mask1, r1_norm, 0.0)
    r2_zero = jnp.where(mask2, r2_norm, 0.0)
    
    # Dot product similarity (normalized)
    dot_product = jnp.sum(r1_zero * r2_zero)
    norm1 = jnp.sqrt(jnp.sum(r1_zero ** 2) + 1e-8)
    norm2 = jnp.sqrt(jnp.sum(r2_zero ** 2) + 1e-8)
    similarity = dot_product / (norm1 * norm2)
    
    # Combined heuristic with weights
    # Shorter is better (1 - normalized_length)
    # High balance, palindrome, and similarity scores are better
    
    # Weights - prioritize length and balance more heavily
    w_len = 0.4
    w_balance = 0.3
    w_palindrome = 0.2
    w_similarity = 0.1
    
    score = w_len * (1.0 - norm_len) + w_balance * avg_balance + w_palindrome * avg_palindrome + w_similarity * (similarity + 1.0) / 2.0
    
    # Ensure bounded in [0,1] with some margin
    score = jnp.clip(score, 0.01, 0.99)
    
    return float(score)
# [EVOLVE-BLOCK-END]
```


---

## Request 22
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v21`.

**Latency:** 18.976 s

**Output:**

[EVOLVE-BLOCK-START]
def heuristic_fn_v21(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that captures structural properties indicating potential for
    cancellation and proximity to triviality. Key improvements over v0:
    
    1. Uses a normalized count of *reducible pairs* (adjacent inverse pairs like [1,-1] or [2,-2])
       which directly contribute to word reduction via relator simplification.
    2. Penalizes long presentations but with a soft logarithmic scaling to avoid
       over-penalizing intermediate expansions (which may be necessary).
    3. Includes a palindrome symmetry score: relators that are palindromic (up to sign)
       often arise in trivial group presentations and may be more easily reduced.
    4. Encourages *balanced* relator lengths (closer to each other), as extreme imbalance
       can hinder MCTS search efficiency.
    
    The final score is a linear combination of normalized contributions, each bounded in [0,1].
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36

    # Split relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Convert to bool mask for non-padding elements
    mask1 = r1 != 0
    mask2 = r2 != 0
    
    # Compute actual lengths
    len1 = jnp.sum(mask1)
    len2 = jnp.sum(mask2)
    total_len = len1 + len2
    max_total = 2 * MAX_RELATOR_LENGTH

    # --- 1. Inverse pair score: count reducible adjacent pairs ---
    # Shifted versions to detect [x, -x]
    r1_shift = jnp.roll(r1, -1)
    r2_shift = jnp.roll(r2, -1)
    # Mask out padding boundaries
    r1_valid = mask1 & jnp.roll(mask1, -1)
    r2_valid = mask2 & jnp.roll(mask2, -1)
    # Detect inverse pairs: x + y == 0 and x != 0
    inv_pairs1 = r1_valid & (r1 + r1_shift == 0)
    inv_pairs2 = r2_valid & (r2 + r2_shift == 0)
    inv_count = jnp.sum(inv_pairs1) + jnp.sum(inv_pairs2)
    
    # Normalize by expected max: each relator can have at most len-1 pairs
    max_pairs = jnp.maximum(0, len1 - 1) + jnp.maximum(0, len2 - 1)
    inv_score = jnp.where(max_pairs > 0, inv_count / max_pairs, 1.0)  # if length <=1, assume high score

    # --- 2. Length penalty: logarithmic soft penalty ---
    # Use log to avoid extreme discounting of long (but useful) intermediate states
    log_penalty = jnp.log1p(total_len) / jnp.log1p(max_total)
    length_score = 1.0 - log_penalty
    
    # --- 3. Palindrome symmetry score ---
    # Check if each relator is approximately palindrome (case for trivial presentations)
    def palindrome_sim(relator, mask):
        # Reverse the relator but keep padding at end
        rev = jnp.where(mask, relator, 0)[::-1]
        # Compare non-padded elements
        n = jnp.sum(mask)
        if n == 0:
            return 1.0
        # Only compare first n elements of original vs reversed
        valid_sim = jnp.sum((relator[:n] == rev[:n]) | (relator[:n] == -rev[:n]))
        return valid_sim / n
    
    pal1 = palindrome_sim(r1, mask1)
    pal2 = palindrome_sim(r2, mask2)
    pal_score = (pal1 + pal2) / 2.0
    
    # --- 4. Length balance score ---
    # Prefer relators of similar lengths
    len_diff = jnp.abs(len1 - len2)
    balance_score = 1.0 - jnp.minimum(1.0, len_diff / MAX_RELATOR_LENGTH)
    
    # Combine with weights (heuristic tuning)
    w_inv = 0.35
    w_len = 0.25
    w_pal = 0.25
    w_bal = 0.15
    
    score = (w_inv * inv_score + 
             w_len * length_score + 
             w_pal * pal_score + 
             w_bal * balance_score)
    
    return score
# [EVOLVE-BLOCK-END]


---

## Request 23
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v22`.

**Latency:** 21.315 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that captures structural properties conducive to trivialisation.
    Key improvements over length-based baselines:
    1. Counts *cancellation-ready pairs* (generator followed by its inverse or vice versa)
       which are prime candidates for AC1/AC2 operations to eliminate.
    2. Penalizes *repetition of structure* (e.g., palindromic substrings, repeated blocks)
       which indicate redundant complexity; such structure can be simplified via conjugation.
    3. Reward *balance* between relators (similarity in length/signature) to encourage
       symmetries exploitable by AC moves.
    4. Incorporate *reducer potential* via a lightweight analysis of how many generators
       can be eliminated in a single pass.
    
    This heuristic better reflects potential for future cancellation even in long presentations
    that may arise during intermediate steps of the trivialisation path.
    
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
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Count non-zero elements per relator (actual lengths)
    len_r1 = jnp.sum(r1 != 0)
    len_r2 = jnp.sum(r2 != 0)
    total_len = len_r1 + len_r2
    
    # Normalize length term (0=ideal, 2*36=worst); scale to [0,1] inverted
    length_score = 1.0 - (total_len / MAX_PRESENTATION_LENGTH)
    length_score = jnp.maximum(length_score, 0.0)
    
    # Cancellation pair count: count adjacent inverse pairs (x, -x)
    def count_cancellation_pairs(relator):
        # Pad with 0 on right to avoid edge effects
        shifted = jnp.concatenate([relator[1:], jnp.array([0])])
        # Compare elementwise: where relator[i] = -shifted[i] and neither is 0
        is_inv = (relator != 0) & (shifted != 0) & (relator + shifted == 0)
        return jnp.sum(is_inv)
    
    canc_pairs_r1 = count_cancellation_pairs(r1)
    canc_pairs_r2 = count_cancellation_pairs(r2)
    canc_pairs_total = canc_pairs_r1 + canc_pairs_r2
    
    # Normalize cancellation score (max possible = total_len/2)
    max_canc_pairs = total_len / 2.0
    canc_score = jnp.where(max_canc_pairs > 0, canc_pairs_total / max_canc_pairs, 0.0)
    
    # Symmetry/balance between relators: prefer similar lengths
    len_diff = jnp.abs(len_r1 - len_r2)
    balance_score = 1.0 - jnp.minimum(len_diff / MAX_RELATOR_LENGTH, 1.0)
    
    # Generator usage balance: prefer both generators used in both relators
    def generator_usage(relator):
        g1_used = jnp.any(relator == 1) | jnp.any(relator == -1)
        g2_used = jnp.any(relator == 2) | jnp.any(relator == -2)
        return g1_used.astype(jnp.int32) + g2_used.astype(jnp.int32)
    
    g1_usage_r1, g2_usage_r1 = (r1 == 1).any() | (r1 == -1).any(), (r1 == 2).any() | (r1 == -2).any()
    g1_usage_r2, g2_usage_r2 = (r2 == 1).any() | (r2 == -1).any(), (r2 == 2).any() | (r2 == -2).any()
    gen_balance_score = (
        (g1_usage_r1.astype(jnp.int32) & g1_usage_r2.astype(jnp.int32)).astype(jnp.float32) +
        (g2_usage_r1.astype(jnp.int32) & g2_usage_r2.astype(jnp.float32))
    ) * 0.5  # 0 to 1
    
    # Weighted combination: emphasize cancellation and length
    w_len, w_canc, w_bal, w_gen = 0.3, 0.35, 0.2, 0.15
    score = (
        w_len * length_score +
        w_canc * canc_score +
        w_bal * balance_score +
        w_gen * gen_balance_score
    )
    
    return score
# [EVOLVE-BLOCK-END]
```


---

## Request 24
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v23`.

**Latency:** 17.741 s

**Output:**

[EVOLVE-BLOCK-START]
def heuristic_fn_v23(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic combining:
    1. Normalized inverse length (shorter is better)
    2. Cancellation potential: counts matching generator/inverse pairs at symmetrical positions (anticipating cancellation during reductions)
    3. Palindromic symmetry score (relators that are self-inverse or palindromic indicate structural simplicity)
    4. Balance score: compares relative lengths of the two relators (closer to equal = better for AC transformations)
    
    This heuristic better captures "hidden structure" in long presentations by focusing on
    cancellation potential and symmetry rather than just raw length, enabling escape from local maxima.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    
    # Extract the two relators (first 36 entries = r1, next 36 = r2)
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Create masks for actual elements (non-zero)
    mask1 = (r1 != 0).astype(jnp.float32)
    mask2 = (r2 != 0).astype(jnp.float32)
    
    # Compute actual lengths
    len1 = jnp.sum(mask1)
    len2 = jnp.sum(mask2)
    total_len = len1 + len2
    
    # Normalized length score (0 = very long, 1 = minimal length of 2)
    # Minimal possible is 2 (one generator per relator), maximum is 72
    length_score = jnp.clip(1.0 - (total_len - 2) / 70.0, 0.0, 1.0)
    
    # Cancellation potential: count pairs where generator[i] == -generator[j]
    # Check within each relator for potential self-cancellation when the word is reduced
    def cancellation_score(relator, mask):
        # Create reversed version for cancellation check (reductions often happen from ends inward)
        rev_relator = relator[::-1]
        rev_mask = mask[::-1]
        
        # Count matching generator-inverse pairs at symmetric positions
        same_sign = (relator * rev_relator > 0)
        inverse_pairs = (relator == -rev_relator) & same_sign
        # We need actual pairs that are inverses and at valid positions
        valid_inverse = (relator == -rev_relator) & (mask.astype(bool)) & (rev_mask.astype(bool))
        return jnp.sum(valid_inverse.astype(jnp.float32)) / jnp.maximum(len1, 1.0)
    
    cancel1 = cancellation_score(r1, mask1)
    cancel2 = cancellation_score(r2, mask2)
    cancel_potential = (cancel1 + cancel2) / 2.0
    
    # Palindromic symmetry: check if relator is close to its own inverse (r[i] == -r[n-1-i])
    def symmetry_score(relator, mask):
        rev_relator = relator[::-1]
        inv_rev_relator = -rev_relator
        # Count matches with inverse-reversed version (palindromic up to sign)
        valid_sym = (relator == inv_rev_relator) & (mask.astype(bool))
        return jnp.sum(valid_sym.astype(jnp.float32)) / jnp.maximum(len1, 1.0)
    
    sym1 = symmetry_score(r1, mask1)
    sym2 = symmetry_score(r2, mask2)
    symmetry = (sym1 + sym2) / 2.0
    
    # Balance score: preference for equal-length relators (helps in AC transformations)
    len_ratio = jnp.minimum(len1, len2) / jnp.maximum(len1, len2)
    balance_score = jnp.where((len1 == 0) | (len2 == 0), 0.0, len_ratio)
    
    # Combine scores with weights (heuristic tuned for group-theoretic structure)
    # Length is most important, then cancellation potential, symmetry and balance
    score = 0.4 * length_score + 0.3 * cancel_potential + 0.2 * symmetry + 0.1 * balance_score
    
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]


---

## Request 25
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v24`.

**Latency:** 23.205 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v24(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that captures structural properties conducive to trivialisation:
    
    1. Measures cancellation potential: counts matching generator/inverse pairs in the same relator
       (which can be eliminated via AC2 conjugations)
    2. Penalises unbalanced generator counts per relator (non-zero net exponent sum suggests non-triviality)
    3. Encourages symmetry by detecting palindromic structure within relators
    4. Accounts for length growth potential by normalising against max length
    
    These features better identify presentations that may require intermediate expansion before
    collapse, escaping local maxima that naive length-based heuristics would discard.
    
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
    
    # Split presentation into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Helper to extract non-padded elements (non-zero values)
    def get_relator_info(relator):
        non_zero = relator != 0
        length = jnp.sum(non_zero)
        # Exponent sum for each generator
        exp_sum_1 = jnp.sum(jnp.where(relator == 1, 1, 0)) - jnp.sum(jnp.where(relator == -1, 1, 0))
        exp_sum_2 = jnp.sum(jnp.where(relator == 2, 1, 0)) - jnp.sum(jnp.where(relator == -2, 1, 0))
        # Cancellation potential: count adjacent inverse pairs (r[i] == -r[i+1])
        adjacent_pairs = relator[:-1] + relator[1:]
        inv_pairs = jnp.sum(jnp.where(adjacent_pairs == 0, 1, 0))
        # Palindrome check: compare relator with its reverse
        reversed_relator = jnp.flip(relator)
        palindrome_score = jnp.sum(jnp.where(relator == reversed_relator, 1, 0)) / (length + 1e-8)
        
        return length, exp_sum_1, exp_sum_2, inv_pairs, palindrome_score
    
    # Process both relators
    r1_len, r1_s1, r1_s2, r1_inv, r1_pal = get_relator_info(r1)
    r2_len, r2_s1, r2_s2, r2_inv, r2_pal = get_relator_info(r2)
    
    # Total length and average length
    total_len = r1_len + r2_len
    avg_len = total_len / 2.0
    
    # Triviality indicators: exponent sums should be zero for trivial relators
    r1_balance = (jnp.abs(r1_s1) + jnp.abs(r1_s2)) / (r1_len + 1e-8)
    r2_balance = (jnp.abs(r2_s1) + jnp.abs(r2_s2)) / (r2_len + 1e-8)
    avg_balance = (r1_balance + r2_balance) / 2.0
    
    # Cancellation potential: more adjacent inverse pairs = easier to reduce via conjugation
    total_inv = r1_inv + r2_inv
    
    # Palindrome scores: symmetric structures often easier to handle via AC moves
    avg_palindrome = (r1_pal + r2_pal) / 2.0
    
    # Normalised components
    len_score = 1.0 - total_len / MAX_PRESENTATION_LENGTH
    inv_score = jnp.clip(total_inv / (MAX_RELATOR_LENGTH - 1), 0.0, 1.0)
    pal_score = avg_palindrome
    balance_score = 1.0 - avg_balance
    
    # Weighted combination with emphasis on cancellation potential (key for AC moves)
    # Higher weight on cancellation potential and balance as these directly relate to trivialisation potential
    weight_inv = 0.35
    weight_pal = 0.15
    weight_balance = 0.30
    weight_len = 0.20
    
    score = (weight_len * len_score + 
             weight_inv * inv_score + 
             weight_pal * pal_score + 
             weight_balance * balance_score)
    
    # Ensure output is in [0,1]
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 26
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v25`.

**Latency:** 23.176 s

**Output:**

[EVOLVE-BLOCK-START]
def heuristic_fn_v25(presentation: jnp.ndarray) -> float:
    r"""
    Improved heuristic that estimates trivialisation potential by:
    1. Normalising presentation length (penalising long presentations)
    2. Counting potential cancellation pairs (adjacent inverse pairs within relators)
    3. Detecting palindromic structure (indicative of conjugates or relations that simplify)
    4. Checking for repeated substrings (suggesting simplification via AC1)
    
    The key improvement is capturing structural features that predict future cancellation
    even when the presentation is currently large (which may be necessary for AC moves).
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH

    # Split into two relators
    relator1 = presentation[:MAX_RELATOR_LENGTH]
    relator2 = presentation[MAX_RELATOR_LENGTH:]

    # Normalize lengths
    len1 = jnp.sum(jnp.abs(relator1) > 0)
    len2 = jnp.sum(jnp.abs(relator2) > 0)
    total_length = len1 + len2
    norm_length = total_length / MAX_PRESENTATION_LENGTH

    # Cancellation pair score: count adjacent inverse pairs in each relator
    def count_cancellation_pairs(relator):
        # Pad with zeros to avoid out-of-bounds
        padded = jnp.concatenate([jnp.array([0]), relator, jnp.array([0])])
        # Check for adjacent inverse pairs: x and -x
        pairs = padded[:-1] + padded[1:]
        return jnp.sum(jnp.abs(pairs[:-1]) == 0)  # pairs where consecutive elements sum to zero

    cancel1 = count_cancellation_pairs(relator1)
    cancel2 = count_cancellation_pairs(relator2)
    total_cancellations = cancel1 + cancel2

    # Palindrome score: count matching pairs from ends toward center
    def palindrome_score(relator):
        # Get non-zero elements
        mask = jnp.abs(relator) > 0
        valid_len = jnp.sum(mask)
        if valid_len == 0:
            return 0.0
        # Truncate to actual length
        rel_valid = relator[:valid_len]
        # Compare mirrored positions
        half_len = valid_len // 2
        if half_len == 0:
            return 0.1  # minimal credit for single element
        rev_rel = rel_valid[::-1]
        matches = jnp.sum(rel_valid[:half_len] == rev_rel[:half_len])
        return matches / half_len

    pal1 = palindrome_score(relator1)
    pal2 = palindrome_score(relator2)
    avg_pal = (pal1 + pal2) / 2.0

    # Repeated substring heuristic: look for duplicated patterns in relators
    # Simplified: check if relators share common subwords (nontrivial overlap)
    def overlap_score(rel1, rel2):
        # Only check up to length 4 substrings for efficiency
        max_sub_len = 4
        subs1 = set()
        subs2 = set()
        # Extract non-zero segments
        mask1 = jnp.abs(rel1) > 0
        mask2 = jnp.abs(rel2) > 0
        len1 = jnp.sum(mask1)
        len2 = jnp.sum(mask2)
        r1 = rel1[:len1]
        r2 = rel2[:len2]

        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Build substrings
        for l in range(2, max_sub_len + 1):
            for i in range(len(r1) - l + 1):
                subs1.add(tuple(r1[i:i+l]))
            for j in range(len(r2) - l + 1):
                subs2.add(tuple(r2[j:j+l]))
        
        overlap = len(subs1 & subs2)
        return min(overlap / max(1, len(subs1) + len(subs2)), 1.0)

    rel_overlap = overlap_score(relator1, relator2)

    # Combine into a bounded [0,1] score
    # Emphasise cancellation potential and structure over raw length
    length_bonus = 1.0 - norm_length
    cancellation_bonus = jnp.clip(total_cancellations / (MAX_RELATOR_LENGTH * 0.5), 0.0, 1.0)
    structure_bonus = (avg_pal + rel_overlap) / 2.0

    # Weighting: prioritize structure and cancellations over raw length
    score = 0.3 * length_bonus + 0.35 * cancellation_bonus + 0.35 * structure_bonus
    return jnp.clip(score, 0.0, 1.0)
[EVOLVE-BLOCK-END]


---

## Request 27
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v26`.

**Latency:** 24.495 s

**Output:**

[EVOLVE-BLOCK-START]
```python
def heuristic_fn_v26(presentation: jnp.ndarray) -> float:
    r"""
    Modified heuristic that combines:
    1. Normalized presentation length (shorter is better)
    2. Count of reducible pairs (adjacent generator-inverse pairs)
    3. Detection of palindromic substructures (potential for conjugation-based simplification)
    4. Symmetry measure between the two relators
    These features together better capture potential for future cancellation during AC moves.
    
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
    
    # Extract the two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # 1. Length score: normalized inverse length (penalize long presentations)
    is_nonzero = jnp.abs(presentation) > 0
    total_length = jnp.sum(is_nonzero)
    length_score = 1.0 - (total_length / MAX_PRESENTATION_LENGTH)
    
    # 2. Reducible pair count: adjacent generator-inverse pairs within each relator
    # For r1: count positions i where presentation[i] == -presentation[i+1] and both non-zero
    def count_reducible_pairs(relator):
        # Pad to enable cyclic check for the last and first elements
        padded = jnp.concatenate([relator, relator[:1]])
        # Check adjacent pairs
        reducible = (padded[:-1] != 0) & (padded[1:] != 0) & (padded[:-1] == -padded[1:])
        return jnp.sum(reducible)
    
    reducible_r1 = count_reducible_pairs(r1)
    reducible_r2 = count_reducible_pairs(r2)
    reducible_score = (reducible_r1 + reducible_r2) / (2 * MAX_RELATOR_LENGTH)
    
    # 3. Palindromic symmetry within each relator (indicates potential for conjugation)
    def palindrome_score(relator):
        # Only consider non-zero prefix
        mask = jnp.cumsum(jnp.where(relator != 0, 1, 0)) <= MAX_RELATOR_LENGTH
        actual_len = jnp.sum(mask)
        actual_len = jnp.maximum(actual_len, 1)  # Avoid division by zero
        relator_trunc = relator[:actual_len]
        # Compare with reverse
        reversed_relator = relator_trunc[::-1]
        matches = (relator_trunc == reversed_relator).astype(jnp.float32)
        return jnp.mean(matches)
    
    pal_score_r1 = palindrome_score(r1)
    pal_score_r2 = palindrome_score(r2)
    palindrome_score_val = (pal_score_r1 + pal_score_r2) / 2.0
    
    # 4. Symmetry between r1 and r2: measure how similar they are (potential for cancellation via AC1)
    # Normalize both relators to same length
    len_r1 = jnp.sum((r1 != 0).astype(jnp.int32))
    len_r2 = jnp.sum((r2 != 0).astype(jnp.int32))
    common_len = jnp.minimum(len_r1, len_r2)
    
    # Compute similarity between first common_len elements (as strings)
    sim_r1_r2 = jnp.mean(jnp.where(
        (jnp.arange(MAX_RELATOR_LENGTH) < common_len) & 
        (r1 != 0) & (r2 != 0),
        jnp.where(r1 == r2, 1.0, 0.0),
        0.0
    ))
    
    # Combine scores with weights
    # Emphasize reducible pairs and palindromes as they indicate internal structure
    # for AC2 (conjugation) and AC1 moves to reduce presentation
    total_score = (
        0.3 * length_score +
        0.3 * reducible_score +
        0.25 * palindrome_score_val +
        0.15 * sim_r1_r2
    )
    
    # Ensure output is in [0, 1]
    return jnp.clip(total_score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 28
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v27`.

**Latency:** 28.232 s

**Output:**

Looking at the problem, I need to create a more sophisticated heuristic that detects structural properties useful for trivialisation, rather than just the baseline length metric.

The key insight is that we need to detect:
1. **Cancellation potential** - how many inverse pairs can be cancelled in each relator
2. **Structural patterns** like palindromes or repeated substrings that suggest simplification potential
3. **Balance** - relators with similar structures may be easier to trivialise
4. **Local complexity** - avoiding too many consecutive same generators (repetitions)

Here's my approach:
- Compute the number of possible cancellations by checking adjacent inverse pairs
- Check for palindromic structure (which often indicates reducibility)
- Measure relator complexity (simpler relators are better)
- Penalize very long presentations but with diminishing returns (log scale)

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v27(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic focusing on cancellation potential and structural simplicity.
    Key improvements over v0:
    1. Counts maximal cancellation pairs within each relator
    2. Measures palindromic symmetry which often indicates reducibility
    3. Penalizes extreme length with logarithmic scaling to avoid local maxima
    4. Considers balance between relator complexities
    
    Args:
        presentation: Array representing current group presentation [r_1; r_2]. Shape (72,) int32.
    
    Returns:
        Scalar heuristic in [0,1] (higher = better trivialisation potential).
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = 72
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Get non-padding masks
    r1_mask = r1 != 0
    r2_mask = r2 != 0
    
    # Effective lengths
    len1 = jnp.sum(r1_mask)
    len2 = jnp.sum(r2_mask)
    total_len = len1 + len2
    
    # Compute cancellation potential for each relator
    def compute_cancellation(relator, mask):
        # Find inverse pairs by checking adjacent positions
        # For each position i, check if relator[i] + relator[i+1] == 0
        shifted = jnp.roll(relator, -1)
        inverse_pairs = jnp.logical_and(
            mask[:-1], 
            jnp.roll(mask, -1)[:-1],
            (relator[:-1] + shifted[:-1] == 0)
        )
        # Count non-overlapping inverse pairs greedily
        cancellable = jnp.zeros_like(inverse_pairs, dtype=jnp.bool_)
        i = 0
        while i < len(inverse_pairs):
            if inverse_pairs[i]:
                cancellable = cancellable.at[i].set(True)
                i += 2  # Skip next position (overlapping not allowed)
            else:
                i += 1
        return jnp.sum(cancellable)
    
    # Use lax.scan for efficient computation (vectorization)
    def cancellation_count(arr):
        # Create pairs array and compute inverses
        padded_arr = jnp.concatenate([arr, jnp.array([0])])
        inv_pairs = (arr[:-1] + arr[1:] == 0) & (arr[:-1] != 0) & (arr[1:] != 0)
        
        # Count non-overlapping pairs using scan
        def scan_fn(state, x):
            i, count, last_cancelled = state
            should_cancel = jnp.logical_and(x, jnp.logical_not(last_cancelled))
            new_state = (
                i + 1,
                count + jnp.where(should_cancel, 1, 0),
                jnp.where(should_cancel, True, False)
            )
            return new_state, None
        
        init = (0, 0, False)
        (final_i, final_count, _), _ = jax.lax.scan(
            scan_fn, init, inv_pairs, unroll=2
        )
        return final_count
    
    # Compute cancellations for each relator
    canc1 = cancellation_count(r1)
    canc2 = cancellation_count(r2)
    total_cancellations = canc1 + canc2
    
    # Compute palindromic symmetry score
    def palindrome_score(relator, mask):
        # For each position, check if it matches its mirror position
        rev_rel = jnp.flip(relator * mask)
        rev_mask = jnp.flip(mask)
        matching = (relator == rev_rel) & mask
        return jnp.sum(matching) / jnp.maximum(jnp.sum(mask), 1.0)
    
    pal1 = palindrome_score(r1, r1_mask)
    pal2 = palindrome_score(r2, r2_mask)
    avg_pal = (pal1 + pal2) / 2.0
    
    # Compute normalized presentation length (lower is better)
    # Use log scaling to avoid extreme penalties for moderate growth
    log_len = jnp.log1p(total_len)
    log_max = jnp.log1p(MAX_PRESENTATION_LENGTH)
    norm_len = 1.0 - (log_len / log_max)
    
    # Combine components with weights
    # Emphasize cancellations as primary indicator
    score = (
        0.5 * (total_cancellations / jnp.maximum(total_len, 1.0)) +
        0.3 * avg_pal +
        0.2 * norm_len
    )
    
    # Ensure bounded [0,1]
    score = jnp.clip(score, 0.0, 1.0)
    
    # Bonus for very short presentations (already near trivial)
    bonus = jnp.where(total_len <= 2, 0.2 * (2.0 - total_len), 0.0)
    score = jnp.clip(score + bonus, 0.0, 1.0)
    
    return score
# [EVOLVE-BLOCK-END]
```


---

## Request 29
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v28`.

**Latency:** 24.550 s

**Output:**

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn_v28(presentation: jnp.ndarray) -> float:
    r"""
    Improved heuristic that captures structural properties conducive to trivialisation:
    - Penalises presentation length (to encourage shortening paths)
    - Rewards cancellation potential: counts matching generator-inverse pairs in the same relator
    - Rewards symmetry: counts reversed relator structure (potential palindrome-like patterns)
    - Rewards balance: penalises extreme imbalance between relator lengths
    
    These features help identify presentations that are "structured" and likely to be 
    reducible via AC moves, even when temporarily long.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    MAX_PRESENTATION_LENGTH = N_GENERATORS * MAX_RELATOR_LENGTH
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Calculate actual lengths (non-zero entries)
    len_r1 = jnp.sum(r1 != 0)
    len_r2 = jnp.sum(r2 != 0)
    
    # Normalize lengths (closer to 0 is better)
    norm_len = (len_r1 + len_r2) / MAX_PRESENTATION_LENGTH
    
    # Cancellation potential: count adjacent generator-inverse pairs within each relator
    # For r1: check pairs (i, i+1) where r1[i] = -r1[i+1]
    def count_cancellations(relator):
        # Shift by one and compare
        shifted = jnp.concatenate([relator[1:], jnp.array([0])])
        # Count where current element is negative of next
        return jnp.sum(relator[:-1] * shifted[:-1] < 0) - jnp.sum(relator[:-1] == 0) + jnp.sum(relator == 0)
    
    # More robust cancellation counting (avoiding padding issues)
    r1_nonzero = (r1 != 0).astype(jnp.float32)
    r2_nonzero = (r2 != 0).astype(jnp.float32)
    
    # Create padded versions for safe adjacency checking
    r1_padded = jnp.concatenate([r1, jnp.array([0])])
    r2_padded = jnp.concatenate([r2, jnp.array([0])])
    
    # Adjacent cancellation count for r1
    r1_cancellation = jnp.sum(
        (r1[:-1] != 0) * (r1[1:] != 0) * (r1[:-1] == -r1[1:])
    )
    r2_cancellation = jnp.sum(
        (r2[:-1] != 0) * (r2[1:] != 0) * (r2[:-1] == -r2[1:])
    )
    
    # Total cancellation potential
    total_cancellation = r1_cancellation + r2_cancellation
    
    # Symmetry score: reverse one relator and compare to the other
    # Reverse r1 (ignoring padding at the end)
    r1_reversed = jnp.flip(r1 * r1_nonzero)
    # Count common elements (ignoring padding)
    r1r_r2_match = jnp.sum(
        (r1_reversed != 0) * (r2 != 0) * (r1_reversed == r2)
    )
    
    # Normalize scores
    cancellation_score = total_cancellation / MAX_RELATOR_LENGTH
    symmetry_score = r1r_r2_match / MAX_RELATOR_LENGTH
    
    # Length penalty: prefer shorter presentations
    length_score = 1.0 - norm_len
    
    # Balance score: penalty for extreme length imbalance
    length_diff = jnp.abs(len_r1 - len_r2)
    max_diff = MAX_RELATOR_LENGTH
    balance_score = 1.0 - length_diff / max_diff
    
    # Combine scores with weights reflecting importance for trivialisation
    # Cancellation is most important (direct path to simplification)
    # Balance helps avoid degenerate states
    # Symmetry may indicate hidden structure exploitable by conjugation
    score = (
        0.5 * cancellation_score +
        0.25 * balance_score +
        0.15 * symmetry_score +
        0.10 * length_score
    )
    
    # Ensure bounded in [0,1]
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 30
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v29`.

**Latency:** 36.720 s

**Output:**

Looking at the problem, I need to create a more sophisticated heuristic that can detect "structure" or "potential for cancellation" in group presentations, especially for cases where intermediate states may grow larger before shrinking.

The key insight from the problem description is that intermediate presentations may need to grow very large to trivialise a presentation, so naive length-based heuristics are insufficient. I need to detect patterns like:
- Cancellation potential (adjacent inverse pairs)
- Palindromic structure (which can be simplified via conjugation)
- Repeated substrings that indicate redundancy
- Symmetry between the two relators

Here's my approach for `heuristic_fn_v29`:

1. **Cancellation pairs**: Count adjacent inverse pairs that could be reduced via group operations
2. **Palindrome detection**: Palindromic relators often have simplification potential
3. **Relator symmetry**: Measure how "balanced" the two relators are
4. **Length penalty**: Still consider length but with diminishing returns for very long presentations (to avoid penalizing necessary growth)

The key structural change from a baseline is incorporating multiple cancellation metrics rather than just length.

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn(presentation: jnp.ndarray) -> float:
    r"""
    Enhanced heuristic that captures cancellation potential, palindrome structure,
    and relator symmetry to better estimate trivialisation likelihood.
    
    Key improvements over length-only heuristics:
    - Counts reducible cancellation pairs (adjacent inverses)
    - Detects palindromic structure in relators (potential for conjugation simplification)
    - Measures symmetry between relators
    - Uses sublinear length penalty to avoid penalizing necessary growth
    
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

    # Extract the two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]

    # Function to get actual length (non-zero elements)
    def get_length(relator):
        return jnp.sum(jnp.abs(relator) > 0)

    # Cancellation pairs: adjacent inverse pairs within each relator
    def count_cancellation_pairs(relator):
        # Create shifted arrays for adjacent pairs
        shifted = jnp.concatenate([relator[1:], jnp.array([0])])
        # Check if adjacent elements are inverses (sum to zero and both non-zero)
        is_inverse_pair = (relator + shifted == 0) & (jnp.abs(relator) > 0)
        return jnp.sum(is_inverse_pair)

    # Count palindrome potential: relator equals its reverse (with sign flip for inverses)
    def palindrome_score(relator):
        # Reverse the relator
        reversed_relator = jnp.flip(relator)
        # For palindrome, we need r[i] = -r[n-1-i]
        # But since we're in group theory, the reverse of a relator that's a palindrome 
        # should be its inverse, meaning r[i] = -r[n-1-i]
        inverted_reversed = -reversed_relator
        # Compare only non-padding positions
        is_padding = (relator == 0) & (inverted_reversed == 0)
        valid_positions = jnp.abs(relator) > 0
        matches = (relator == inverted_reversed) | is_padding
        return jnp.sum(matches & valid_positions) / (jnp.sum(valid_positions) + 1e-6)

    # Symmetry between relators r1 and r2 (including inverse)
    def symmetry_score(rel1, rel2):
        # Check if rel1 == rel2 or rel1 == -reverse(rel2)
        reversed_rel2 = jnp.flip(rel2)
        inverse_reversed_rel2 = -reversed_rel2
        
        valid_mask = (jnp.abs(rel1) > 0) | (jnp.abs(rel2) > 0)
        
        match_direct = (rel1 == rel2) | (rel1 == 0) | (rel2 == 0)
        match_inverse = (rel1 == inverse_reversed_rel2) | (rel1 == 0) | (reversed_rel2 == 0)
        
        matches = jnp.maximum(match_direct, match_inverse)
        return jnp.sum(matches & valid_mask) / (jnp.sum(valid_mask) + 1e-6)

    # Compute metrics
    len1, len2 = get_length(r1), get_length(r2)
    total_len = len1 + len2
    
    # Normalized length score (sublinear to avoid punishing necessary growth)
    max_expected_len = 10.0
    length_score = jnp.where(total_len <= max_expected_len, 
                            1.0 - total_len / MAX_PRESENTATION_LENGTH,
                            jnp.maximum(0.1, 1.0 - jnp.log(total_len + 1) / jnp.log(MAX_PRESENTATION_LENGTH + 1)))

    # Cancellation potential score
    cancel1, cancel2 = count_cancellation_pairs(r1), count_cancellation_pairs(r2)
    cancel_score = jnp.minimum(1.0, (cancel1 + cancel2) / 4.0)

    # Palindrome scores
    pal1 = palindrome_score(r1)
    pal2 = palindrome_score(r2)
    palindrome_score_mean = (pal1 + pal2) / 2.0

    # Symmetry between relators
    symmetry = symmetry_score(r1, r2)

    # Combine scores with weights tuned for group-theoretic structure
    # Palindromes and symmetry are strong indicators of simplifiability
    # Cancellation pairs indicate immediate reduction potential
    # Length is still important but penalized less severely
    
    final_score = (
        0.15 * length_score +
        0.25 * (cancel_score) +
        0.30 * (palindrome_score_mean) +
        0.30 * (symmetry)
    )
    
    # Ensure output in [0,1]
    return jnp.clip(final_score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```


---

## Request 31
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v30`.

**Latency:** 21.298 s

**Output:**

[EVOLVE-BLOCK-START]
def heuristic_fn_v30(presentation: jnp.ndarray) -> float:
    r"""
    Improved heuristic incorporating three key components:
    1. Normalized cancellation potential: counts reducible adjacent pairs (generator * inverse)
    2. Palindrome symmetry score: detects self-similar structures that may simplify via conjugation
    3. Length reduction penalty for intermediate bloat: rewards presentations where reduction is possible without excessive growth
    
    This addresses the problem that intermediate states during trivialization often grow in length, so simple length-based heuristics are misleading.
    """
    N_GENERATORS = 2
    MAX_RELATOR_LENGTH = 36
    
    # Reshape to separate the two relators
    relator1 = presentation[:MAX_RELATOR_LENGTH]
    relator2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Helper to compute reducible pair count (adjacent generator * inverse)
    def reducible_pairs(r):
        # Remove padding
        mask = r != 0
        valid = jnp.where(mask, r, 0)
        # Create shifted version (next element)
        shifted = jnp.concatenate([valid[1:], jnp.zeros(1, dtype=jnp.int32)])
        # Count pairs where r[i] * shifted[i] == -1 (i.e., inverse pair)
        product = valid * shifted
        return jnp.sum(product == -1)
    
    # Calculate reducible pairs in each relator
    pairs1 = reducible_pairs(relator1)
    pairs2 = reducible_pairs(relator2)
    
    # Palindrome symmetry score: count positions where r[i] == -r[-i-1]
    def palindrome_score(r):
        mask = r != 0
        valid_len = jnp.sum(mask.astype(jnp.int32))
        # Only consider non-zero part
        truncated = r[:valid_len] if valid_len > 0 else r[:1]
        rev = truncated[::-1]
        return jnp.sum(truncated == -rev)
    
    pal1 = palindrome_score(relator1)
    pal2 = palindrome_score(relator2)
    
    # Total length (non-zero entries)
    total_len = jnp.sum(presentation != 0)
    
    # Normalize components
    max_pairs_per_relator = MAX_RELATOR_LENGTH - 1
    max_pal_per_relator = MAX_RELATOR_LENGTH
    
    pair_score = (pairs1 + pairs2) / (2 * max_pairs_per_relator)
    pal_score = (pal1 + pal2) / (2 * max_pal_per_relator)
    
    # Length penalty: avoid overly long presentations unless they show high reduction potential
    # Use a smoothed penalty: if length is below threshold, reward; above, penalize exponentially
    length_penalty = jnp.exp(-0.03 * (total_len - 12))  # Optimal around 12 total length
    
    # Combine scores
    score = 0.4 * pair_score + 0.3 * pal_score + 0.3 * length_penalty
    
    # Clamp to [0,1]
    return jnp.clip(score, 0.0, 1.0)
[EVOLVE-BLOCK-END]


---

## Request 32
**Prompt:** Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v31`.

**Latency:** 36.462 s

**Output:**

Looking at the problem, I need to create a more sophisticated heuristic than the baseline length-based approach. The key insight is that for trivial group presentations, we want to detect cancellation potential, palindrome symmetry, and structural patterns that suggest the presentation can be simplified.

Let me design a heuristic that captures:
1. **Cancellation potential**: Count pairs of adjacent inverse elements that can be reduced
2. **Symmetry properties**: Palindromic relators or symmetric patterns often indicate simplifiability
3. **Structure detection**: Repeated substrings or patterns that might be removable via AC moves
4. **Length normalization**: Still consider length but weight structure more heavily

Here's my improved heuristic function:

```python
# [EVOLVE-BLOCK-START]
def heuristic_fn(presentation: jnp.ndarray) -> float:
    r"""
    Heuristic estimating likelihood of trivialisation based on cancellation potential,
    symmetry, and structural patterns in the relators.
    
    Improvements over baseline:
    - Detects cancellation pairs (adjacent inverse elements)
    - Rewards palindromic structure in relators
    - Identifies repeated substrings that may be removable via AC moves
    - Combines multiple structural cues into a bounded [0,1] score
    
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
    
    # Split into two relators
    r1 = presentation[:MAX_RELATOR_LENGTH]
    r2 = presentation[MAX_RELATOR_LENGTH:]
    
    # Count non-zero (actual) elements in each relator
    r1_len = jnp.sum(r1 != 0)
    r2_len = jnp.sum(r2 != 0)
    total_len = r1_len + r2_len
    
    # Normalized length component (shorter is better, scaled to [0,1])
    length_score = jnp.maximum(0.0, 1.0 - total_len / MAX_PRESENTATION_LENGTH)
    
    # Cancellation potential: count adjacent inverse pairs
    def count_cancellations(relator):
        # Create shifted versions: relator[i] and relator[i+1]
        shifted = jnp.concatenate([relator[1:], jnp.array([0])])
        # Check if adjacent elements are inverses (sum to 0) and non-zero
        inverses = (relator + shifted == 0) & (relator != 0)
        return jnp.sum(inverses)
    
    cancel_count = count_cancellations(r1) + count_cancellations(r2)
    # Normalize by max possible (length-1 per relator) and scale
    max_cancellations = (MAX_RELATOR_LENGTH - 1) * 2
    cancel_score = jnp.minimum(1.0, cancel_count / jnp.maximum(1, max_cancellations))
    
    # Palindrome symmetry: count matching symmetric pairs
    def palindrome_score(relator, length):
        # Only consider up to actual length
        effective_len = jnp.minimum(length, MAX_RELATOR_LENGTH)
        half_len = jnp.maximum(1, effective_len // 2)
        
        # Get forward and backward halves
        forward = relator[:half_len]
        backward = jnp.flip(relator[MAX_RELATOR_LENGTH - half_len:])
        
        # Check matches (accounting for padding)
        valid_mask = (forward != 0) | (backward != 0)
        matches = jnp.sum((forward == backward) & valid_mask)
        return matches / jnp.maximum(1, jnp.sum(valid_mask))
    
    pal1 = palindrome_score(r1, r1_len)
    pal2 = palindrome_score(r2, r2_len)
    pal_score = (pal1 + pal2) / 2
    
    # Repeated pattern detection: check for substrings that appear multiple times
    def count_repeats(relator, length):
        # Convert to tuple for processing (JAX-friendly approach via scan)
        relator_int = relator.astype(jnp.int32)
        max_pattern_len = jnp.minimum(4, jnp.maximum(1, length // 2))
        
        def scan_patterns(i, carry):
            pattern_len, count = carry
            # Extract pattern at position i
            pattern = relator_int[i:i+pattern_len]
            # Count occurrences of this pattern
            matches = 0
            for start in range(MAX_RELATOR_LENGTH - pattern_len + 1):
                match_pattern = relator_int[start:start+pattern_len]
                matches += jnp.all(pattern == match_pattern)
            return carry, matches
        
        # Simplified repeat count: check for any substring of length 2+ repeated
        repeated_count = 0
        for pl in range(2, jnp.minimum(5, jnp.maximum(1, length))):
            pattern = relator_int[:pl]
            total_matches = 0
            for start in range(MAX_RELATOR_LENGTH - pl + 1):
                check_pattern = relator_int[start:start+pl]
                total_matches += jnp.sum((check_pattern == pattern).astype(jnp.int32)) // pl
            
            # Count if pattern appears more than once
            repeated_count += (total_matches > pl).astype(jnp.int32)
        
        return repeated_count
    
    repeat1 = count_repeats(r1, r1_len)
    repeat2 = count_repeats(r2, r2_len)
    repeat_score = jnp.minimum(1.0, (repeat1 + repeat2) / 4.0)
    
    # Combined score: weighted combination of components
    w_length = 0.25
    w_cancel = 0.35
    w_pal = 0.25
    w_repeat = 0.15
    
    score = (w_length * length_score + 
             w_cancel * cancel_score + 
             w_pal * pal_score + 
             w_repeat * repeat_score)
    
    # Ensure bounded output [0,1]
    return jnp.clip(score, 0.0, 1.0)
# [EVOLVE-BLOCK-END]
```
