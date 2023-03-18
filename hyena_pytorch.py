"""
This is a slightly modified version of https://github.com/HazyResearch/safari/blob/main/standalone_hyena.py
that I "cleaned up" to make it easier for me to port it
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor

torch.manual_seed(0)


def fft_conv(
    u: Float[Tensor, "batch width len"], k: Float[Tensor, "width len"], D: Float[Tensor, "width"]
) -> Float[Tensor, "batch width len"]:
    sequence_length = u.shape[-1]
    fft_size = 2 * sequence_length

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :sequence_length]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


class Sin(nn.Module):
    def __init__(self, dim: int, freq: int = 10, train_freq: bool = True):
        super().__init__()
        self.freq = nn.Parameter(freq * torch.ones(1, dim)) if train_freq else freq * torch.ones(1, dim)

    def forward(self, x: Float[Tensor, "batch len ord"]) -> Float[Tensor, "batch len ord"]:
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5) -> None:
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filters is normalized so that t_f = 1
        time_emb = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, seq_len, 1

        bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, seq_len, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([time_emb, z.real, z.imag], dim=-1)

        self.register_parameter("z", nn.Parameter(z))
        self.z._optim = {"lr": lr_pos_emb}

        self.register_buffer("time_emb", time_emb)

    def forward(self, length: int) -> Tuple[Float[Tensor, "_ width _"], Float[Tensor, "_ width _"]]:
        return self.z[:, :length], self.time_emb[:, :length]


class ExponentialModulation(nn.Module):
    def __init__(
        self,
        width: int,
        fast_decay_pct: float = 0.3,
        slow_decay_pct: float = 1.5,
        target: float = 1e-2,
        modulation_lr: float = 0.0,
        modulate: bool = True,
        shift: float = 0.0,
        **_,
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, width)[None, None]
        self.register_parameter("deltas", nn.Parameter(deltas))
        self.deltas._optim = {"lr": modulation_lr}

    def forward(self, t: Float[Tensor, "_ width _"], x: Float[Tensor, "_ width _"]) -> Float[Tensor, "_ width _"]:
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    def __init__(
        self,
        num_channels: int,
        emb_dim: int = 3,  # dim of input to MLP, augments with positional encoding
        order: int = 16,  # width of the implicit MLP
        fused_fft_conv: bool = False,
        seq_len: int = 1024,
        lr: float = 1e-3,
        lr_pos_emb: float = 1e-5,
        dropout: float = 0.0,
        freq: int = 1,  # frequency of periodic activations
        weight_decay: int = 0,  # weight decay of kernel parameters
        bias: bool = True,
        num_inner_mlps: int = 2,
        normalized: bool = False,
        **kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            num_channels: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = num_channels
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, freq=freq)
        assert (
            emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.emb_dim = emb_dim
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for _ in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, num_channels, bias=False))

        self.modulation = ExponentialModulation(num_channels, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": weight_decay, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, seq_len: int, *_, **__) -> Float[Tensor, "batch len width"]:
        z, t = self.pos_emb(seq_len)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(
        self,
        x: Float[Tensor, "batch width len"],
        seq_len: int,
        k: Optional[Union[Tuple, Float[Tensor, "width len"]]] = None,
        bias: Optional[Float[Tensor, "width"]] = None,
        *_,
        **__,
    ) -> Float[Tensor, "batch width len"]:
        if k is None:
            k = self.filter(seq_len)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if isinstance(k, tuple) else k

        y = fft_conv(x, k, bias)
        return y


class HyenaOperator(nn.Module):
    def __init__(
        self,
        width: int,
        max_input_len: int,
        order: int = 2,
        filter_order: int = 64,
        dropout: float = 0.0,
        filter_dropout: float = 0.0,
        **filter_args,
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            width (int): Dimension of the input and output embeddings (width of the layer)
            max_input_len: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()
        self.width = width
        self.max_len = max_input_len
        self.order = order
        inner_width = width * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(width, inner_width)
        self.out_proj = nn.Linear(width, width)

        self.short_filter = nn.Conv1d(inner_width, inner_width, 3, padding=2, groups=inner_width)
        self.filter_fn = HyenaFilter(
            width * (order - 1),
            order=filter_order,
            seq_len=max_input_len,
            dropout=filter_dropout,
            **filter_args,
        )

    def forward(self, input_seq: Float[Tensor, "batch len width"], *_, **__) -> Float[Tensor, "batch len width"]:
        seq_len = input_seq.size(-2)
        seq_len = min(seq_len, self.max_len)
        input_seq = self.in_proj(input_seq)
        input_seq = rearrange(input_seq, "batch len width -> batch width len")

        uc = self.short_filter(input_seq)[..., :seq_len]
        *x, v = uc.split(self.width, dim=1)

        k = self.filter_fn.filter(seq_len)[0]
        k = rearrange(k, "len (ord width) -> ord width len", ord=self.order - 1)
        bias = rearrange(self.filter_fn.bias, "(ord width) -> ord width", ord=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, seq_len, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], "batch width len -> batch len width")

        y = self.out_proj(y)
        return y


def main() -> None:
    layer = HyenaOperator(width=1024, max_input_len=2048, order=2, filter_order=64)
    x = torch.randn(4, 2048, 1024, requires_grad=True)
    y = layer(x)

    print(x.shape, y.shape)

    grad = torch.autograd.grad(y[:, 10, :].sum(), x)[0]
    print('Causality check: gradients should not flow "from future to past"')
    print(grad[0, 11, :].sum(), grad[0, 9, :].sum())


if __name__ == "__main__":
    main()
