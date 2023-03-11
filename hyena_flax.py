"""
This is a slightly modified version of https://github.com/HazyResearch/safari/blob/main/standalone_hyena.py
that I "cleaned up" to make it easier for me to port it
"""
import math
from typing import Tuple

import jax.numpy as jnp
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
