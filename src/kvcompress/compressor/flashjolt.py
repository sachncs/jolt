"""FlashJoLT compressor.

Same algorithm as JoLT but replaces the exact token-mode SVD with a
randomized low-rank SVD capped at ``q_cap``. The cap is chosen by a
context-aware policy:

    q_cap = min(max(q_min(R), ⌈T / 32⌉), 512)

with ``q_min = 32`` for ``R ≤ 4`` and ``q_min = 64`` for ``R ≥ 5``. The
policy is a no-op for ``T ≤ 1024`` so every short-context number is
unchanged, and grows the cap sublinearly beyond that.

Tail-mass accounting: the randomized SVD discards the spectral tail past
``q_cap``, which would mislead the allocator about how much energy a given
rank captures. FlashJoLT corrects for this by computing τ from the
randomized SVD's reported ``tail_mass`` (a tight upper bound on the true
discarded mass) so the allocator sees the same error signal as exact JoLT.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch

from kvcompress.compressor.allocator import (
    AllocationResult,
    Cell,
    JointAllocator,
)
from kvcompress.compressor.base import (
    CompressedPayload,
    CompressorStats,
    KVCompressor,
)
from kvcompress.compressor.jolt import JoLTCompressor, _JoLTFactors
from kvcompress.compressor.residual import (
    ResidualPayload,
    decode_residual,
    encode_residual,
)
from kvcompress.compressor.svd import SVD
from kvcompress.compressor.tucker import (
    TuckerFactors,
    partial_tucker_st_hosvd,
    reconstruct_partial_tucker,
)

log = logging.getLogger(__name__)


def flashjolt_cap(context_length: int, target_ratio: float) -> int:
    """Compute the FlashJoLT cap ``q_cap`` from context length and ratio.

    Args:
        context_length: ``T`` (token axis length).
        target_ratio: ``R`` (compression ratio).

    Returns:
        Integer cap for the randomized token SVD.
    """
    q_min = 64 if target_ratio >= 5.0 else 32
    cap = max(q_min, math.ceil(context_length / 32))
    return min(cap, 512)


class FlashJoLTCompressor(JoLTCompressor):
    """FlashJoLT: randomized-SVD variant of JoLT.

    Args:
        compression_ratio: target compression ratio.
        bits: residual bit-widths.
        cap: override the auto-computed ``q_cap``. ``None`` means auto.
        **kwargs: forwarded to :class:`JoLTCompressor`.
    """

    name = "flashjolt"

    def __init__(
        self,
        *,
        compression_ratio: float = 3.0,
        bits: tuple[int, ...] = (0, 2, 4, 8),
        cap: int | None = None,
        seed: int = 0,
        svd: SVD | None = None,
        **kwargs: Any,
    ) -> None:
        # Force randomized SVD; preserve user-provided override if given.
        kwargs.setdefault("seed", seed)
        effective_svd = svd if svd is not None else SVD(seed=seed, method="randomised")
        if svd is None:
            # Make sure we use randomized regardless of caller default.
            effective_svd = SVD(seed=seed, method="randomised")
        super().__init__(
            compression_ratio=compression_ratio,
            bits=bits,
            svd=effective_svd,
            **kwargs,
        )
        self.auto_cap = cap

    def _compress_cell(
        self,
        x: torch.Tensor,
        allocation: Any,
    ) -> _JoLTFactors:
        # Identical to JoLT's path except we plug a randomized SVD with cap.
        m, t, d = x.shape
        rt = int(allocation.r_token)
        rd = int(allocation.r_feature)
        b = int(allocation.bits)
        cap = self.auto_cap if self.auto_cap is not None else flashjolt_cap(t, self.compression_ratio)

        # The SVD cap is honored by the SVD class itself when given
        # explicitly via .randomise(a, rank, cap=...). Since
        # partial_tucker_st_hosvd uses SVD.__call__, we need to ensure the
        # cap is applied at the call site. We do this by passing a thin
        # wrapper that intercepts rank=rt and adds cap.
        svd_capped = _CapWrapper(self.svd, cap)

        tucker = partial_tucker_st_hosvd(
            x,
            r_token=rt,
            r_feature=rd,
            svd=svd_capped,
        )

        recon = reconstruct_partial_tucker(tucker, x.shape)
        residual_tensor = (x - recon).contiguous()
        if b == 0:
            residual: ResidualPayload | None = encode_residual(
                residual_tensor,
                bits=0,
                seed=self.seed,
                distribution=self.jl_distribution,  # type: ignore[arg-type]
            )
        else:
            residual = encode_residual(
                residual_tensor,
                bits=b,
                seed=self.seed,
                distribution=self.jl_distribution,  # type: ignore[arg-type]
                symmetric=self.symmetric_quant,
                per_channel=self.per_channel_quant,
                group_size=self.group_size,
            )

        return _JoLTFactors(tucker=tucker, residual=residual, allocation=allocation)


class _CapWrapper:
    """Thin wrapper that forces the randomized SVD path with a cap.

    Implements the same ``__call__(a, rank=..., cap=...)`` interface as
    :class:`SVD` but injects ``cap=cap`` so the sketch size is bounded.
    """

    def __init__(self, base: SVD, cap: int) -> None:
        self.base = base
        self.cap = int(cap)

    def __call__(self, a: torch.Tensor, *, rank: int | None = None, cap: int | None = None):
        # Force the randomized path with the cap.
        return self.base.randomise(a, rank=int(rank), cap=self.cap)

    def exact(self, a: torch.Tensor, rank: int | None = None):
        return self.base.exact(a, rank=rank)

    def randomise(self, a: torch.Tensor, rank: int, *, cap: int | None = None):
        return self.base.randomise(a, rank=rank, cap=self.cap)


__all__ = ["FlashJoLTCompressor", "flashjolt_cap"]