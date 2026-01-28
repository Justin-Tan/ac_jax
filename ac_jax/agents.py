import jax
import jax.numpy as jnp
from jax import jit, vmap, jacfwd

import numpy as np

from flax import linen as nn
from flax.linen.initializers import constant, orthogonal

from functools import partial
from typing import Callable, Sequence, Mapping, Optional
from jaxtyping import Array, Float, ArrayLike

def masked_mean_pool(x, mask):
    x_masked = x * mask
    n_valid = jnp.sum(mask, axis=-2)
    return jnp.sum(x_masked, axis=-2) / jnp.maximum(n_valid, 1)

class MLP(nn.Module):
    """
    Baseline MLP for actor/critic architecture.
    """
    layers: Sequence[int] = (512, 512)
    activation_fn: Callable = nn.tanh

    @nn.compact
    def __call__(self, x: Float[Array, "i"]):
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            if i != len(self.layers) - 1:
                x = self.activation_fn(x)
        return x

class EmbeddingMLP(nn.Module):
    """
    MLP with learned token embeddings.
    """
    layers: Sequence[int] = (128, 128, 128)
    activation_fn: Callable = nn.tanh
    n_vocab: int = 5  # \{0, \pm 1, \pm 2\}
    embed_dim: int = 32

    def setup(self):
        self.shift_ids = self.n_vocab // 2
        self.embedding = nn.Embed(num_embeddings=self.n_vocab, features=self.embed_dim)

    @nn.compact
    def __call__(self, x: Float[Array, "i"]):
        token_ids = (x + self.shift_ids).astype(jnp.int32)
        x = self.embedding(token_ids).reshape(-1)
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer)(x)
            if i != len(self.layers) - 1:
                x = self.activation_fn(x)
        return x

class ConvEncoder(nn.Module):
    """
    1D convolutional encoder; expects learned token 
    embeddings as input. Processes individual relators.
    """
    channels: Sequence[int] = (32, 64, 32)
    kernel_sizes: Sequence[int] = (16, 8, 16)
    layers: Sequence[int] = (64, 64)  # post-conv
    activation_fn: Callable = nn.relu


    @nn.compact
    def __call__(self, x: Float[Array, "i"]):
        for ch, kernel_size in zip(self.channels, self.kernel_sizes):
            x = nn.Conv(features=ch, kernel_size=(kernel_size,), padding="SAME")(x)
            x = self.activation_fn(x)  # (..., T, C)
        # jax.debug.print("conv-output shape {cs}", cs=x.shape)

        # pool over seq dimension
        # mask = (x != 0)
        # x = masked_mean_pool(x, mask)
        x = jnp.max(x, axis=-2)  # (channels,)
        return x

    
class ActorCritic(nn.Module):
    """
    Actor: parameterises policy for action selection.
    Critic/assistant: parameterises value function for reward prediction.
    Common vector embedding and convolutional encoder, independent parameters 
    between subseequent actor/critic heads.
    """
    n_actions: int
    encoder_type: str = "conv"
    encoder_config: Mapping = None
    activation: str = "tanh"
    n_vocab: int = 5  # \{0, \pm 1, \pm 2\}
    embed_dim: int = 16
    actor_layers: Sequence[int] = (256,256)
    critic_layers: Sequence[int] = (256,256)
    n_gens: int = 2

    def setup(self):
        self.activation_dict = {"gelu": nn.gelu, "relu": nn.relu, "tanh": nn.tanh}
        assert self.activation in self.activation_dict.keys(), f"Unsupported activation function: {self.activation}"
        self.activation_fn = self.activation_dict[self.activation]
        self.shift_ids = self.n_vocab // 2
        self.embedding = nn.Embed(num_embeddings=self.n_vocab, features=self.embed_dim)

        arch_config = {"mlp": {"layers": (32, 32)},
                       # "conv": {"channels": (32, 64, 32), "kernel_sizes": (3,5,7), "layers": (64, 64)}}
                       "conv": {"channels": (32, 64, 32), "kernel_sizes": (32, 16, 16), "layers": (64, 64)}}
        config = arch_config.get(self.encoder_type, {})
        if self.encoder_config is not None: config.update(self.encoder_config)
        encoder_map = {"mlp": lambda: MLP(layers=config["layers"], activation_fn=self.activation_fn),
                       "conv": lambda: ConvEncoder(channels=config["channels"], kernel_sizes=config["kernel_sizes"], layers=config["layers"], activation_fn=self.activation_fn)}
        # self.encoder = encoder_map[self.encoder_type]()
        BatchEncoder = nn.vmap(ConvEncoder, variable_axes={'params': None}, split_rngs={'params': False},
                               in_axes=0, out_axes=0)
        self.encoder = BatchEncoder(channels=config["channels"], kernel_sizes=config["kernel_sizes"], layers=config["layers"], activation_fn=self.activation_fn)
        self.actor_mlp = MLP(layers=self.actor_layers, activation_fn=self.activation_fn)
        self.critic_mlp = MLP(layers=self.critic_layers, activation_fn=self.activation_fn)

        assert self.n_actions % self.n_gens == 0, "Number of actions must divide evenly between relators!"
        self.actor_head = nn.Dense(self.n_actions // self.n_gens, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.critic_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def embed(self, x: Float[Array, "i"]):
        token_ids = (x + self.shift_ids).astype(jnp.int32)
        x = self.embedding(token_ids)
        return x

    def actor(self, features: Float[Array,"i"]):
        actor_mean = self.actor_mlp(features)
        logits = self.actor_head(actor_mean)
        return logits
    
    def critic(self, features: Float[Array, "i"]):
        value = self.critic_mlp(features)
        value = self.critic_head(value)
        return jnp.squeeze(value, axis=-1)

    def __call__(self, x: Float[Array, "i"]):
        x = jnp.reshape(x, (self.n_gens, -1))  # (n_rels, max_rel_length)
        embeddings = self.embed(x)
        # jax.debug.print("embed shape {es}", es=embeddings.shape)

        features = self.encoder(embeddings)  # (n_rels, dim)
        # jax.debug.print("feature shape {fs}", fs=features.shape)

        policy_logits = self.actor(features)  # (n_rels, action_dim // n_rels)
        policy_logits = policy_logits.reshape(-1)  # (action_dim)

        # pool over relator dimension for value network
        value = self.critic(features.reshape(-1))
        return policy_logits, value
    
class ActorCriticIndependent(nn.Module):
    """
    Actor: parameterises policy for action selection.
    Critic/assistant: parameterises value function for reward prediction.
    Uses independent parameters for actor and critic networks, common vector embeddings.
    """
    n_actions: int
    actor_layers: Sequence[int]
    critic_layers: Sequence[int]
    activation: str = "tanh"
    n_vocab: int = 5
    embed_dim: int = 16

    def setup(self):
        self.activation_dict = {"gelu": nn.gelu, "relu": nn.relu, "tanh": nn.tanh}
        assert self.activation in self.activation_dict.keys(), f"Unsupported activation function: {self.activation}"
        self.activation_fn = self.activation_dict[self.activation]
        self.shift_ids = self.n_vocab // 2
        self.actor_embedding = nn.Embed(num_embeddings=self.n_vocab, features=self.embed_dim)
        self.critic_embedding = nn.Embed(num_embeddings=self.n_vocab, features=self.embed_dim)

        self.actor_mlp = MLP(layers=self.actor_layers, activation_fn=self.activation_fn)
        self.critic_mlp = MLP(layers=self.critic_layers, activation_fn=self.activation_fn)

        self.actor_head = nn.Dense(self.n_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.critic_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def embed(self, x: Float[Array, "i"], embed_layer):
        token_ids = (x + self.shift_ids).astype(jnp.int32)
        x = embed_layer(token_ids).reshape(-1)
        return x

    def actor(self, x: Float[Array, "i"]):
        # features = x
        features = self.embed(x, self.actor_embedding)
        actor_mean = self.actor_mlp(features)
        logits = self.actor_head(actor_mean)
        return logits
    
    def critic(self, x: Float[Array, "i"]):
        # features = x
        features = self.embed(x, self.critic_embedding)
        value = self.critic_mlp(features)
        value = self.critic_head(value)
        return jnp.squeeze(value, axis=-1)

    def __call__(self, x: Float[Array, "i"]):
        """Given batch of observations, return
        logits of policy and value estimates."""
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

class TransformerBlock(nn.Module):
    """Standard Pre-Norm Transformer Block."""
    embed_dim: int
    num_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # x: (..., seq, embed_dim)
        # mask: (batch, 1, seq, seq)
        
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            dropout_rate=self.dropout_rate
        )(h, mask=mask, deterministic=deterministic)
        x = x + h

        h = nn.LayerNorm()(x)
        h = nn.Dense(self.embed_dim * 4)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.embed_dim)(h)
        x = x + h
        return x


class RelatorTransformer(nn.Module):
    """
    tokenised relator sequence [2, 1, 2, 0, 0 ...] -> vector embedding
    """
    embed_dim: int
    num_heads: int
    num_layers: int
    max_len: int = 36
    vocab_size: int = 5

    def setup(self):
        self.cls_token = self.param('cls_token', nn.initializers.normal(stddev=0.02), (1, self.embed_dim))

    @nn.compact
    def __call__(self, token_ids, mask):
        # need to vmap this, don't include batch dim
        # token_ids: (seq_len,)
        # mask: (seq_len,)
        
        seq_len = token_ids.shape[0]

        # 0: pad value, 1-4: generators
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(token_ids)
        
        # position encoding - zeroth token is CLS
        pos_ids = jnp.arange(seq_len) + 1
        pos_emb = nn.Embed(num_embeddings=self.max_len + 1, features=self.embed_dim)(pos_ids)
        x = x + pos_emb

        x = jnp.concatenate((self.cls_token, x), axis=0)
        # Shape: (1, seq_len, seq_len)
        cls_mask = jnp.array([1], dtype=jnp.int32)
        mask = jnp.concatenate((cls_mask, mask), axis=0)
        attn_mask = nn.make_attention_mask(mask > 0, mask > 0)

        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim, 
                num_heads=self.num_heads
            )(x, mask=attn_mask)

        # pool across sequence dimension
        # pooled = masked_mean_pool(x, jnp.expand_dims(mask, axis=-1))  # (..., embed_dim)
        return x[0]  # relator summary


# --- Main Architecture ---

class ActorCriticTransformer(nn.Module):
    n_actions: int = 12
    embed_dim: int = 64
    inner_layers: int = 2
    outer_layers: int = 2
    num_heads: int = 4
    n_gens: int = 2  # \pm 1, \pm 2
    max_rel_length: int = 36
    n_value_bins: int = 1

    def setup(self):
        # per-relator encoder to mitigate O(L^2) scaling.
        # in_axes=0: Map over 0-th relator dim.
        BatchInnerEncoder = nn.vmap(
            RelatorTransformer,
            variable_axes={'params': None},
            split_rngs={'params': False},
            in_axes=0, 
            out_axes=0)
        
        self.inner_encoder = BatchInnerEncoder(
            embed_dim=self.embed_dim, num_heads=self.num_heads, num_layers=self.inner_layers,
            max_len=self.max_rel_length)
        
        # outer cross-relator, independent actor/critic blocks seem to work better
        """
        self.outer_blocks = [
            TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)
            for _ in range(self.outer_layers)]
        """
        self.outer_blocks_actor = [
            TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)
            for _ in range(self.outer_layers)]
        self.outer_blocks_critic = [
            TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)
            for _ in range(self.outer_layers)]

        self.actor_head = nn.Dense(self.n_actions // self.n_gens, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.critic_head = nn.Dense(self.n_value_bins, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        self.global_token = self.param("global_token", nn.initializers.normal(stddev=0.02), (1, self.embed_dim))

    def _tokenise(self, x):
        # 0 still pad value
        shifted = jnp.select(
            [x == 0, x == -2, x == -1, x == 1, x == 2],
            [0,      1,       2,       3,      4],
            default=0).astype(jnp.int32)
        return shifted

    def __call__(self, x):
        """
        Args:
            x: (..., max_len) <-- Input Integers {-2..2}, 0 padding.
        """

        x = jnp.reshape(x, (self.n_gens, -1))  # (n_rels, max_rel_len)
        # 1. preprocessing
        token_ids = self._tokenise(x)
        token_mask = (x != 0).astype(jnp.int32) # Mask for inner tokens
        
        # 2. vmap across relators
        # Input: (n_rels, L) -> Output: (n_rels, embed_dim)
        relator_features = self.inner_encoder(token_ids, token_mask)

        # 3. outer transformer over relator embeddings`
        # Expand batch dim for TransformerBlock expectation (1, n_rels, embed_dim)
        h = jnp.expand_dims(relator_features, axis=0)
        
        """
        for block in self.outer_blocks:
            h = block(h)
        h = jnp.squeeze(h, axis=0) # Back to (n_rels + 1, embed_dim)
        """
        h_actor = h 
        for block in self.outer_blocks_actor:
            h_actor = block(h_actor)
        h_actor = jnp.squeeze(h_actor, axis=0) # Back to (n_rels + 1, embed_dim)

        h_critic = h 
        for block in self.outer_blocks_critic:
            h_critic = block(h_critic)
        h_critic = jnp.squeeze(h_critic, axis=0) # Back to (n_rels + 1, embed_dim)

        # 4. Actor Head
        # Applied to each relator embedding independently
        # Output: (n_rels, n_actions) -> Flatten to (n_rels * n_actions)
        action_logits = self.actor_head(h_actor) # actor_embedding)
        action_logits_flat = action_logits.reshape(-1)
        
        # pool over relator dimension (global)
        critic_embedding = h_critic.reshape(-1)  # seems to work better than mean pooling
        value = self.critic_head(critic_embedding)
        
        return action_logits_flat, jnp.squeeze(value)
