# Quickstart

Install:

```bash
pip install kvcompress
```

Load any Hugging Face causal LM and enable compression:

```python
import torch
from transformers import AutoModelForCausalLM
from kvcompress import enable_compression

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
handle = enable_compression(model, method="flashjolt", target_memory="25%")

ids = torch.tensor([[1, 2, 3, 4]])
out = model.generate(ids, max_new_tokens=20)
```

That's it. The KV cache is compressed transparently on every layer write and
decompressed on every read.

## What's happening under the hood

1. `enable_compression` constructs a [FlashJoLTCompressor][kvcompress.compressor.flashjolt.FlashJoLTCompressor] (or whatever method you named) with a target compression ratio.
2. The compressor uses a [JointAllocator][kvcompress.compressor.allocator.JointAllocator] to pick per-layer (r_token, r_feature, residual_bits) under the byte budget.
3. Each per-layer K/V is compressed via partial Tucker decomposition (token + feature modes), then a JL-rotated low-bit residual captures the truncated energy.
4. On the next attention call, the cache is decompressed back to fp16/bf16 and the model's standard attention path runs.

## Switching methods

```python
enable_compression(model, method="jolt",       compression_ratio=3.0)   # exact SVD
enable_compression(model, method="flashjolt",  compression_ratio=2.5)   # randomized SVD
enable_compression(model, method="lowrank",    rank=128)                # matrix SVD baseline
enable_compression(model, method="int4",       per_channel=True)        # pure INT4
enable_compression(model, method="fp8")                                  # FP8 storage
enable_compression(model, method="identity")                             # no compression
```

See [compression_methods.md](user/compression_methods.md) for a comparison.

## Disabling

```python
handle.disable()  # restores original behaviour
```

## What's next

- [Concepts](user/concepts.md) — the vocabulary of KV cache compression.
- [API reference](user/api.md) — every public symbol.
- [Performance guide](user/performance_guide.md) — picking the right
  ratio for your workload.
- [Long-context](user/long_context.md) — needle-in-haystack and beyond.