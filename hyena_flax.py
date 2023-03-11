"""
This is a slightly modified version of https://github.com/HazyResearch/safari/blob/main/standalone_hyena.py
that I "cleaned up" to make it easier for me to port it
"""
import jax.numpy as jnp
from jax import device_put
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