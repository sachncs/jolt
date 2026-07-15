# kvcompress documentation

`kvcompress` implements the JoLT algorithm (paper:
[arXiv:2607.12550](https://arxiv.org/abs/2607.12550)) plus its
randomized-SVD fast variant FlashJoLT, and provides a generic
KV-cache-compression interface for any decoder-only LLM.

The documentation is split into four sections:

* [User guide](user/index.md) — installation, quickstart, API reference.
* [Developer guide](dev/index.md) — architecture, adding a compressor,
  testing, releasing.
* [Researcher notes](research/index.md) — algorithm, math, free-zone
  analysis, reproduction notes.
* [Benchmarks](benchmarks/overview.md) — running benchmarks, interpreting
  results.

If you're new to `kvcompress`, start with the [quickstart](quickstart.md).