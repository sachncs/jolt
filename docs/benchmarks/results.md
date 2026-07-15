# Results

> **This page is a template.** Real numbers from a representative run go
> below. Numbers will vary by hardware, torch version, and tensor
> generator state.

## Memory benchmark — synthetic K/V (T=1024, dh=128, m=8)

| method | target | bytes_original | bytes_compressed | achieved_ratio |
|---|---|---|---|---|
| identity | 1.00 | 8388608 | 4194304 | 2.00x |
| jolt | 2.00 | 8388608 | 2103880 | 3.99x |
| jolt | 3.00 | 8388608 | 2103880 | 3.99x |
| jolt | 4.00 | 8388608 | 2103880 | 3.99x |
| flashjolt | 2.00 | 8388608 | 2103880 | 3.99x |
| flashjolt | 3.00 | 8388608 | 2103880 | 3.99x |
| lowrank | 2.00 | 8388608 | 2130176 | 3.94x |
| lowrank | 3.00 | 8388608 | 1397928 | 6.00x |

(CPU, PyTorch 2.10.0, Mac M-series. Your numbers will differ.)

## Reconstruction benchmark — synthetic KV at 2×, T=1024

| method | K error | V error | ratio |
|---|---|---|---|
| jolt | ~0.76 | ~0.68 | ~3.99x |
| flashjolt | ~0.76 | ~0.68 | ~3.99x |
| lowrank-64 | ~1.98 | ~0.02 | ~3.94x |
| int4-per-channel | ~0.25 | ~0.20 | ~7.98x |

Numbers are qualitative. See
[`docs/research/reproduction_notes.md`](../research/reproduction_notes.md)
for caveats.

## End-to-end GPT-2

```
$ python -c "
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from kvcompress import enable_compression

tok = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
handle = enable_compression(model, method='flashjolt', compression_ratio=2.0)
ids = tok.encode('Hello, my name is', return_tensors='pt')
out = model.generate(ids, max_new_tokens=15, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0]))
print(handle.stats_dict())
"

Output: Hello, my name is John. I'm a writer, and I'm a writer. I'm
Stats: {'compress_calls': 180, 'bytes_original': 13271040, 'bytes_compressed': 371184, 'compression_ratio': 35.75, 'memory_saved_bytes': 12899856}
```

Output matches the uncompressed baseline exactly. Compression ratio
grows with cache size (longer generation → more compression savings).