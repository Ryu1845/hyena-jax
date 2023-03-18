"""
Port of the standalone hyena from pytorch to flax
"""
import math
from functools import partial
from typing import Tuple, Optional, Union

import jax.nn
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from flax import linen as nn
from jax import device_put, lax
from jaxtyping import Float

KEY = jr.PRNGKey(0)


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
    freq: int = 10

    @nn.compact
    def __call__(self, x: Float[jnp.ndarray, "b len ord"], train: bool = True) -> Float[jnp.ndarray, "b len ord"]:
        if train:
            freq = self.param("freq", lambda _: self.freq * jnp.ones((1, self.dim)))
        else:
            freq_var = self.variable("fixed", "freq", lambda: self.freq * jnp.ones(1, self.dim))
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

        time_emb = self.variable(
            "time_emb", "time_emb", lambda: jnp.linspace(0, 1, self.seq_len)[None, :, None]
        ).value  # 1, seq_len, 1

        def z_func(_):
            bands = (self.emb_dim - 1) // 2
            # To compute the right embeddings we use the "proper" linspace
            t_rescaled = jnp.linspace(0, self.seq_len - 1, self.seq_len)[None, :, None]
            w = 2 * math.pi * t_rescaled / self.seq_len  # 1, seq_len, 1
            f = jnp.linspace(1e-4, bands - 1, bands)[None, None]
            _z = jnp.exp(-1j * f * w)
            _z = lax.concatenate([time_emb, _z.real, _z.imag], dimension=2)
            return _z

        z = self.param("z", z_func)
        # self.z._optim = {"lr": self.lr_pos_emb}
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
        def deltas_func(_):
            max_decay = math.log(self.target) / self.fast_decay_pct
            min_decay = math.log(self.target) / self.slow_decay_pct
            deltas = jnp.linspace(min_decay, max_decay, self.width)[None, None]
            return deltas

        deltas = self.param("deltas", deltas_func)
        # self.deltas._optim = {"lr": modulation_lr}

        if self.modulate:
            decay = jnp.exp(-t * jnp.abs(deltas))
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
    seq_len: int = 1024
    order: int = 16  # width of the implicit MLP
    fused_fft_conv: bool = False
    lr: float = 1e-3
    lr_pos_emb: float = 1e-5
    dropout_p: float = 0.0
    freq: int = 1  # frequency of periodic activations
    weight_decay: int = 0  # weight decay of kernel parameters
    use_bias: bool = True
    num_inner_mlps: int = 2
    normalized: bool = False

    def setup(self):
        self.bias = self.param("bias", jax.nn.initializers.normal(), (self.num_channels,), jnp.float32)
        self.dropout = nn.Dropout(self.dropout_p)
        act = Sin(dim=self.order)
        assert (
            self.emb_dim % 2 != 0 and self.emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(self.emb_dim, self.seq_len, self.lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            [i for _ in range(self.num_inner_mlps + 1) for i in (nn.Dense(self.order), act)]
            + [nn.Dense(self.num_channels, use_bias=False)]
        )

        self.modulation = ExponentialModulation(self.num_channels)

    def __call__(
        self,
        x: Float[jnp.ndarray, "b width len"],
        seq_len: int = 1024,
        k: Optional[Union[Tuple, Float[jnp.ndarray, "width len"]]] = None,
        bias: Optional[Float[jnp.ndarray, "width"]] = None,
        **kwargs,
    ) -> Float[jnp.ndarray, "b width len"]:
        if k is None:
            k = self.filter(seq_len)
        # Ensure compatibility with filters that return a tuple
        k = k[0] if isinstance(k, tuple) else k

        y = fft_conv(x, k, bias)
        return y

    def filter(self, seq_len: int) -> Float[jnp.ndarray, "b len width"]:
        z, t = self.pos_emb(seq_len)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h


class HyenaOperator(nn.Module):
    width: int
    max_len: int
    order: int = 2
    filter_order: int = 64
    dropout: float = 0.0
    filter_dropout: float = 0.0

    @nn.compact
    def __call__(
        self, input_seq: Float[jnp.ndarray, "b len width"], *_, **filter_args
    ) -> Float[jnp.ndarray, "b len width"]:
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            width (int): Dimension of the input and output embeddings (width of the layer)
            max_input_len: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        inner_width = self.width * (self.order + 1)
        dropout = nn.Dropout(self.dropout, deterministic=True)
        in_proj = nn.Dense(inner_width)
        out_proj = nn.Dense(self.width)

        filter_fn = HyenaFilter(
            self.width * (self.order - 1),
            seq_len=self.max_len,
            order=self.filter_order,
            dropout_p=self.filter_dropout,
            **filter_args,
        )

        seq_len = input_seq.shape[-2]
        seq_len = min(seq_len, self.max_len)
        input_seq = in_proj(input_seq)

        short_filter = nn.Conv(inner_width, kernel_size=(3,), padding=2, feature_group_count=inner_width)
        uc = short_filter(input_seq)
        uc = rearrange(uc, "b len width -> b width len")[..., :seq_len]
        *x, v = jnp.split(uc, uc.shape[1] // self.width, axis=1)

        k = filter_fn.filter(seq_len)[0]
        k = rearrange(k, "len (ord width) -> ord width len", ord=self.order - 1)
        bias = rearrange(filter_fn.bias, "(ord width) -> ord width", ord=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = dropout(v * x_i)
            v = filter_fn(v, seq_len, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], "b width len -> b len width")

        y = out_proj(y)
        return y


def main() -> None:
    layer = HyenaOperator(width=1024, max_len=2048, order=2, filter_order=64)
    x = jr.normal(KEY, (4, 2048, 1024))
    variables = layer.init(KEY, x)
    layer.apply(variables, jnp.ones((5, 5)))
    y = layer(x)

    print(x.shape, y.shape)

    # grad = torch.autograd.grad(y[:, 10, :].sum(), x)[0]
    # print('Causality check: gradients should not flow "from future to past"')
    # print(grad[0, 11, :].sum(), grad[0, 9, :].sum())


if __name__ == "__main__":
    main()
