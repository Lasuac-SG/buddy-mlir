# EmbeddingGemma

[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
sentence embeddings, served through the `buddy-cli` / `.rax` runtime. The build
imports the PyTorch SentenceTransformer pipeline, compiles the fixed-shape
forward graph to MLIR, links `embeddinggemma_model.so`, and packs an
`embeddinggemma.rax` manifest plus an `InferenceRunner` plugin that `buddy-cli`
loads at run time.

The compiled kernel is the full SentenceTransformer pipeline traced as a single
AOT graph: Gemma3TextModel encoder -> mean pooling over the real tokens -> dense
768->3072 -> dense 3072->768 -> L2 normalization. The runner tokenizes the input
text and emits the L2-normalized 768-dim embedding vector.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, `sentence_transformers`, and the Buddy Python frontend
  available. This is only needed at **build time**; the packaged `.rax` has no
  Python dependency at run time.
- A local HuggingFace `google/embeddinggemma-300m` snapshot directory.

## Build

`embeddinggemma` is wired through `buddy_add_model(MODEL_KIND single_forward)`
with its own importer/manifest generator
(`codegen/import-embeddinggemma.py` /
`codegen/gen_embeddinggemma_manifest.py`). It is gated behind the
`BUDDY_BUILD_EMBEDDINGGEMMA_MODEL` CMake option:

```bash
cd buddy-mlir
source buddy-env/bin/activate

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_EMBEDDINGGEMMA_MODEL=ON \
  -DBUDDY_EMBEDDINGGEMMA_MODEL_PATH=/path/to/embeddinggemma-300m
cmake --build build --target embeddinggemma_rax
```

`-DBUDDY_EMBEDDINGGEMMA_MODEL_PATH` must point to a local HuggingFace-format
snapshot (it is staged into the build and forwarded to the importer via the
`EMBEDDINGGEMMA_MODEL_PATH` environment variable).

The build emits these artifacts under `build/models/embeddinggemma/`:

| File | Description |
| --- | --- |
| `embeddinggemma.rax` | Model manifest (embeds weights, `embeddinggemma_model.so`, `tokenizer.json`, runner) |
| `embeddinggemma_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `embeddinggemma_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened model weights (307,581,953 f32 elements, ~1.2 GB) |
| `tokenizer.json` | Gemma byte-level BPE tokenizer, staged from the local snapshot |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/embeddinggemma/embeddinggemma.rax \
  --prompt "The quick brown fox jumps over the lazy dog"
```

- `--prompt "<text>"` selects the text to embed.
- `--no-stats` suppresses runner logs and prints only the embedding vector.

The output is the L2-normalized 768-dimensional embedding vector.

## Baseline measurements

On Linux, `benchmark.py` uses the existing runner and GNU `time` to collect
end-to-end latency, the runner's first-forward latency, and peak process RSS.
It requires NumPy; only generating HF references requires PyTorch and
SentenceTransformers.

```bash
python models/embeddinggemma/validate_accuracy.py --mode reference \
  --output-dir build/models/embeddinggemma/reference
python models/embeddinggemma/benchmark.py \
  --model build/models/embeddinggemma/embeddinggemma.rax \
  --reference-dir build/models/embeddinggemma/reference \
  --threads 4 --warmup 1 --repeat 10 --tolerance 1e-5 \
  --output-dir build/models/embeddinggemma/benchmark
```

Choose a new output directory for each measurement. `--buddy-cli` and `--spec`
override the default `build/bin/buddy-cli` and `specs/f32.json`. The script sets
`OMP_NUM_THREADS` and disables dynamic OpenMP thread counts; CPU affinity is
inherited from the caller.

Warmup rounds prepare the extracted payload cache and are excluded from the
statistics. Each request starts a new process and reloads the model, so the
forward timing includes first-call initialization and is not resident-model
steady-state latency. End-to-end timing includes the GNU time wrapper; peak
RSS covers the runner's entire lifetime, including weights and temporary data.

`report.json` contains the configuration, each measured request, latency
mean/median/min/max/population standard deviation, maximum peak RSS, model
package size, and validation status. Raw logs and embeddings are kept under
`runs/<round>/`; `warmup/` holds the unmeasured runs. The extracted payloads are
cached under `payload/`. After timing, every measured round is checked by
`validate_accuracy.compare_outputs`; a failed request, missing measurement,
timeout, or numerical mismatch returns a nonzero exit status. A timeout also
terminates the runner process group. No Git metadata or hashes are collected.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/embeddinggemma/`), not in the source tree.
- The tokenizer used by the runner is a pure-C++ reimplementation
  (`EmbeddinggemmaTokenizer.h`) of the checkpoint's Gemma byte-level BPE over
  the staged `tokenizer.json`; sequences are wrapped in `<bos>` / `<eos>` and
  padded to `max_seq_len` with `<pad>`.
