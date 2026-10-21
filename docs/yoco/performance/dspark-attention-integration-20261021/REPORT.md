# Shared-KV FA4 integration validation

The previously tested Shared-KV FA4 adapter is integrated directly on top of
`ca9e851fb8`. No batch/context dispatch, additional graph manager, or additional
graph specification is introduced. The capability fallback in the original
implementation remains. The two attention source files are byte-identical to
the prototype manifest. Existing FP32 proposal-cache and int64-address fixes
are retained; OPD/profiling hooks from the experiment are not imported.

## Combined native validation

This is a fresh compatibility run with Mega hybrid target MoE **and** FA4
Shared-KV draft attention. It is not a new A/B measurement of either component.
The prior controlled attention A/B/B/A remains archived in [RESULTS.md](RESULTS.md).

Single B200 GPU6, TP1/PP1/DP1/EP1, online FP8 target with original-prefill /
Mega decode above16 rows, BF16 draft/KV, K8 probabilistic proposal and standard
rejection, T1, fixed seeds, APC disabled, 8 GiB KV budget, max length 12288.
Short chat inputs and exactly 8193-token synthetic inputs each use C1/C4/C8,
256 output tokens/request, two warmups and three scored trials per cell:
18 scored trials plus 12 warmups. Every result is retained.

| Input | C | Combined tok/s | Acceptance |
| --- | --- | --- | --- |
| short | 1 | 372.73 | 46.68% |
| short | 4 | 858.59 | 36.39% |
| short | 8 | 1436.97 | 34.40% |
| context8193 | 1 | 212.76 | 39.67% |
| context8193 | 4 | 347.45 | 34.43% |
| context8193 | 8 | 403.04 | 32.81% |

All requests completed without corruption or preemption. The timer includes
prefill, target/draft execution, sampling, scheduling and GPU drain; it excludes
loading, compilation, tokenization and HTTP. Output trajectories can differ
because FA4 changes BF16 reductions. There is no low-load performance guarantee.

The native observer confirms FA version 4, 20 prepared Mega routed layers, one
shared Mega workspace and FP32 proposal logits. Target capture sizes remain
`[9,18,36,72]` (four graphs); Shared-KV sizes remain
`[8,16,24,32,40,64]` (six graphs). Both managers actually replayed FULL graphs.
The worker extension observes graph replay and configuration; it does not
replace model/expert/attention forward methods.

The 2048-token teacher-forced panel is finite. Its logprobs are retained as
numerical observations, without a matched target-batch control or task-quality
claim. See [QUALITY_BATCHING_NOTE.md](QUALITY_BATCHING_NOTE.md).

## Checks and provenance

- Local related checks: 98 passed.
- B200 Shared-KV checks: 17 passed, no skips; FA4 capability asserted first.
- Scoped pre-commit passed.
- GPU coverage includes HND/NHD, 16/256-token pages, empty prefixes, exact
  SWA8192 visibility, read-only KV, changed-page/length graph replay, full
  backbone/Markov replay, and physical addresses above 2**31 BF16 elements.

The isolated test runtime reuses the repository-pinned FA4 CuteDSL component
at flash-attention commit `506341a143fcabd4bb79052a7605ada727d6b3f5`, with the
CMake import rewriting already applied in the prototype. The shared runtime
and the repository dependency pin were not changed. Source/runtime hashes are
in [final-audit.json](final-audit.json), [source-receipt.json](source-receipt.json)
and the raw local artifact directory.

Raw artifacts: `/data/vllm_test/work/yoco-dspark-integration-20261021/` and
`/home/fhb/yoco-dspark-integration-20261021/`. Original A/B artifacts remain in
`/data/vllm_test/work/yoco-dspark-attention-20261021/`. GPU6 is released after
the run; Pod UID/node/restart count and unrelated services remain unchanged.
