"""02 — Direct compression without Hugging Face.

Use the JoLTCompressor directly when you want to compress K/V tensors
without monkey-patching a model. Useful for offline analysis, ablations,
or building your own KV cache backend.
"""

from __future__ import annotations

import torch

from kvcompress import JoLTCompressor
from kvcompress.cache.compress import CompressedKVCache


def main() -> None:
    torch.manual_seed(0)
    # Simulate a KV cache at one layer: shape (n_kv, T, dh).
    n_kv, T, dh = 8, 256, 64
    K = torch.randn(n_kv, T, dh)
    V = torch.randn(n_kv, T, dh)

    # Direct API.
    comp = JoLTCompressor(compression_ratio=3.0, bits=(0, 4, 8))
    k_payload, v_payload = comp.compress(K, V)
    K_hat, V_hat = comp.decompress(k_payload, v_payload)
    rel_err_K = torch.linalg.norm(K - K_hat) / torch.linalg.norm(K)
    print(f"JoLT round-trip rel err K: {rel_err_K.item():.4f}")
    print(f"  original bytes:  {K.numel() * K.element_size():>10}")
    print(f"  compressed K+V:  {k_payload.bytes_compressed + v_payload.bytes_compressed:>10}")

    # Cache API.
    cache = CompressedKVCache(compressor=comp)
    cache.store(layer=0, key=K, value=V)
    print(f"  cache bytes:     {cache.memory_used()}")
    print(f"  compression:     {cache.compression_ratio():.2f}x")

    K2, V2 = cache.retrieve(0)
    assert torch.allclose(K, K2, atol=1e-4)  # approximate


if __name__ == "__main__":
    main()
