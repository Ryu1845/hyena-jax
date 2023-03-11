"""
This is a slightly modified version of https://github.com/HazyResearch/safari/blob/main/standalone_hyena.py
that I "cleaned up" to make it easier for me to port it
"""
import math
from typing import Tuple, Optional, Union

import jax.nn
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from jax import device_put, lax
from jaxtyping import Float


def fft_conv(
    u: Float[jnp.ndarray, "b width len"], k: Float[jnp.ndarray, "width len"], D: Float[jnp.ndarray, "width"]
) -> Float[jnp.ndarray, "b width len"]:
    sequence_length = u.shape[-1]
    fft_size = 2 * sequence_length

    k_f = jnp.fft.rfft(k, n=fft_size) / fft_size
    u_f = jnp.fft.rfft(device_put(u.astype(k.dtype)), n=fft_size)

    if len(u.shape) > 3:
        k_f = jnp.expand_dims(k_f, 1)
    y = jnp.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :sequence_length]

    out = y + u * jnp.expand_dims(D, -1)
    return device_put(out.astype(u.dtype))


class Sin(nn.Module):
    dim: int

    @nn.compact
    def __call__(
        self, x: Float[jnp.ndarray, "b len ord"], freq: int = 10, train: bool = True
    ) -> Float[jnp.ndarray, "b len ord"]:
        if train:
            freq = self.param("freq", lambda: freq * jnp.ones(1, self.dim))
        else:
            freq_var = self.variable("fixed", "freq", lambda: freq * jnp.ones(1, self.dim))
            freq = freq_var.value
        return jnp.sin(freq * x)


class PositionalEmbedding(nn.Module):
    emb_dim: int
    seq_len: int
    lr_pos_emb: float = 1e-5

    @nn.compact
    def __call__(self, length: int) -> Tuple[Float[jnp.ndarray, "_ width _"], Float[jnp.ndarray, "_ width _"]]:
        """Complex exponential positional embeddings for Hyena filters."""
        # The time embedding fed to the filters is normalized so that t_f = 1
        time_emb = jnp.linspace(0, 1, self.seq_len)[None, :, None]  # 1, seq_len, 1

        bands = (self.emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = jnp.linspace(0, self.seq_len - 1, self.seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / self.seq_len  # 1, seq_len, 1

        f = jnp.linspace(1e-4, bands - 1, bands)[None, None]
        z = jnp.exp(-1j * f * w)
        z = lax.concatenate([time_emb, z.real, z.imag], dimension=-1)

        z = self.param("z", lambda: z)
        # self.z._optim = {"lr": self.lr_pos_emb}

        time_emb = self.variable("time_emb", "time_emb", lambda: time_emb)
        return z[:, :length], time_emb[:, :length]


class ExponentialModulation(nn.Module):
    width: int
    fast_decay_pct: float = 0.3
    slow_decay_pct: float = 1.5
    target: float = 1e-2
    modulation_lr: float = 0.0
    modulate: bool = True
    shift: float = 0.0

    @nn.compact
    def __call__(
        self, t: Float[jnp.ndarray, "_ width _"], x: Float[jnp.ndarray, "_ width _"]
    ) -> Float[jnp.ndarray, "_ width _"]:
        max_decay = math.log(self.target) / self.fast_decay_pct
        min_decay = math.log(self.target) / self.slow_decay_pct
        deltas = jnp.linspace(min_decay, max_decay, self.width)[None, None]
        self.param("deltas", lambda: deltas)
        # self.deltas._optim = {"lr": modulation_lr}

        if self.modulate:
            decay = jnp.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    """
    Implicit long filter with modulation.

    Args:
        num_channels: number of channels in the input
        emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
        order: width of the FFN
        num_inner_mlps: number of inner linear layers inside filter MLP
    """

    num_channels: int
    emb_dim: int = 3  # dim of input to MLP, augments with positional encoding
    order: int = 16  # width of the implicit MLP
    fused_fft_conv: bool = False
    lr: float = 1e-3
    lr_pos_emb: float = 1e-5
    dropout: float = 0.0
    freq: int = 1  # frequency of periodic activations
    weight_decay: int = 0  # weight decay of kernel parameters
    use_bias: bool = True
    num_inner_mlps: int = 2
    normalized: bool = False

    @nn.compact
    def __call__(
        self,
        x: Float[jnp.ndarray, "b width len"],
        seq_len: int = 1024,
        k: Optional[Union[Tuple, Float[jnp.ndarray, "width len"]]] = None,
        bias: Optional[Float[jnp.ndarray, "width"]] = None,
        **kwargs
    ) -> Float[jnp.ndarray, "b width len"]:
        key = jr.PRNGKey(0)
        bias = self.param("bias", jax.nn.initializers.normal(), key, self.num_channels)
        _dropout = nn.Dropout(self.dropout)

        act = Sin(dim=self.order, freq=self.freq)
        assert (
            self.emb_dim % 2 != 0 and self.emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(self.emb_dim, seq_len, self.lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Dense(self.emb_dim, self.order),
            act,
        )
        for _ in range(self.num_inner_mlps):
            self.implicit_filter.append(nn.Dense(self.order, self.order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Dense(self.order, self.num_channels, bias=False))

        self.modulation = ExponentialModulation(self.num_channels, **kwargs)

        # for c in self.implicit_filter.children():
        #     for name, v in c.state_dict().items():
        #         optim = {"weight_decay": self.weight_decay, "lr": lr}
        #         setattr(getattr(c, name), "_optim", optim)

        if k is None:
            k = self.filter(seq_len)
        # Ensure compatibility with filters that return a tuple
        k = k[0] if isinstance(k, tuple) else k

        y = fft_conv(x, k, bias)
        return y

    def filter(self, seq_len: int, *_, **__) -> Float[jnp.ndarray, "b len width"]:
        z, t = self.pos_emb(seq_len)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h
