# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only.  The expert-server path
  (`ds4_engine_compute_experts`, `ds4_engine_compute_experts_batch`) is a
  production CPU workload for sharded inference on Intel machines.  Optimize
  Q2_K, IQ2_XXS, and Q8_K kernels with AVX2 (and AVX-512 VNNI where
  available) for this path.  The full-model CPU forward pass remains
  reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- The expert-server CPU path is expected to run on Linux/Intel x86_64 with
  AVX2.  Test scalar fallbacks for correctness but do not benchmark them.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Metal smoke tests for build verification.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.
