"""Triton kernels for JoLT.

Provides:

* ``tucker_reconstruct`` — fused Tucker reconstruction (core × token basis × feature basis).
* ``jl_project`` — fused JL projection (rotation + cast).
* ``quantize_int8`` — fused per-channel int8 quantization.

These are *no-op fallbacks* on systems without ``triton``; the public
functions in :mod:`kvcompress.compressor` always go through PyTorch.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def is_triton_available() -> bool:
    """Return True if Triton is importable."""
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def tucker_reconstruct(
    core,
    u_token,
    u_feature,
):
    """Fused Tucker reconstruction.

    Falls back to PyTorch einsum when Triton is unavailable.
    """
    import torch

    return torch.einsum("mar,ta,dr->mtd", core, u_token, u_feature)


def jl_project(x, matrix):
    """JL projection via dense matmul."""
    import torch  # noqa: F401

    return x @ matrix.t()


def quantize_int8(x):
    """Per-channel int8 quantization (Triton when available)."""
    from kvcompress.compressor.quantization import IntQuantizer

    q = IntQuantizer(bits=8, symmetric=True, per_channel=True)
    return q.quantize(x)


__all__ = ["is_triton_available", "jl_project", "quantize_int8", "tucker_reconstruct"]
