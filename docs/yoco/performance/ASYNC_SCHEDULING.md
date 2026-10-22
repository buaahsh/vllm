# YOCO fixed-K8 asynchronous scheduling

This experimental branch validates vLLM's existing asynchronous scheduler with
the single-GPU YOCO Shared-KV DSpark path. It adds independent benchmark controls,
runtime receipts, request-lifecycle checks, and optional trace annotations.
It does not change the production scheduler algorithm or enable async by default.

The implementation starts from `c616521d8c`. See the
[experiment report](async-scheduling-20261022/REPORT.md) for measured behavior.

## Supported experiment scope

- One B200 GPU, V2 runner, TP1/PP1/DP1 without CP or DBO.
- The existing FP8 target, BF16 draft/KV, FP32 proposal cache, fixed DSpark K8.
- Existing hybrid MoE dispatch, fast-prefill/raw-h20 semantics, target four and
  draft six CUDA graphs in the initial C1/C4/C8 panel.
- APC disabled; no grammar or P/D in the initial panel.

Adaptive verification, multi-GPU Shared-KV/Mega, APC, P/D, and grammar combinations
are separate validation tasks. Fast/Mega does not provide batch-invariant output.

## Keep process layout fixed

The old `--synchronous-engine` option disables both engine multiprocessing and
async scheduling. Its behavior is retained for historical experiments. For an
async comparison, select both dimensions explicitly:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
  .venv/bin/python -m tools.yoco_alignment.benchmark_mega_moe \
  --backend hybrid --fast-prefill --engine-mode multiprocess \
  --scheduler-mode sync --max-model-len 12288 \
  --concurrencies 1 4 8 --tokens 64 --trials 3 \
  --extra-prompt-tokens 8193 --output /new/path/sync
```

Run the same command with `--scheduler-mode async` and a new output path.
Keep checkpoints, runtime libraries, GPU/KV budgets and other options fixed.
Both `auto` modes retain the existing environment/default behavior. Combining
the legacy option with either new option is rejected to avoid ambiguous controls.

The worker receipt records the resolved scheduler class, async flag, concurrent
batch capacity, executor, process IDs, MoE dispatch and actual graph replays.
An explicitly requested mode that resolves differently fails validation.
Capacity is a configuration observation; actual CPU/GPU overlap requires a trace.

## Longer requests and larger native batches

The native benchmark accepts independent sequence, KV, batch-token and graph
budgets. For example, the larger experimental panel uses:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
  .venv/bin/python -m tools.yoco_alignment.benchmark_mega_moe \
  --backend hybrid --fast-prefill --engine-mode multiprocess \
  --scheduler-mode sync --max-model-len 32768 --max-num-seqs 32 \
  --max-num-batched-tokens 2048 --kv-cache-gib 48 \
  --cudagraph-capture-sizes 1 2 4 8 9 16 18 32 36 64 72 128 144 256 288 \
  --concurrencies 16 32 --token-counts 512 1024 --trials 3 \
  --extra-prompt-tokens 16384 --dispatch-config /path/to/dispatch.yaml \
  --output /new/path/large-sync
```

The dispatch YAML explicitly fixes the original thresholds while testing larger
graphs outside the initial measured panel:

```yaml
additional-config:
  yoco_moe_dispatch:
    mode: manual
    triton_max_tokens: 16
    mega_min_tokens: 17
    mega_prefill: false
```

Repeat with the asynchronous scheduler and the same budgets. The default C8/KV8
configuration is preserved. K8 requires a target graph capacity of at least nine
tokens per configured request; the widest graph receipt uses `max_num_seqs`.
Above C8, the workload adds distinct prompt variants and verifies the actual
completed request count and each output budget. `--token-counts` sweeps several
output lengths within one engine session; `--tokens` still selects one length.
Every input/output/concurrency cell has two warmups and `--trials` scored seeds.

These are simultaneous-arrival native batches. Their output throughput includes
prefill and GPU drain; it does not establish HTTP serving capacity or sustained
online concurrency. Check preemptions, graph replays, resolved settings and
request latencies before interpreting a difference as scheduler performance.
See the [larger-load report](async-scheduling-20261022/LONG_LOAD_REPORT.md) for
the repeated results and latency tradeoffs.

## Staggered requests and cancellation

Use the same engine config JSON with the native AsyncLLM frontend for both modes:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
  .venv/bin/python -m tools.yoco_alignment.benchmark_async_scheduling \
  --config /path/to/config.json --scheduler-mode sync \
  --requests 16 --arrival-interval 0.02 --long-prompt-tokens 8193 \
  --output /new/path/stream-sync
```

Replay the generated workload in the asynchronous run:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
  .venv/bin/python -m tools.yoco_alignment.benchmark_async_scheduling \
  --config /path/to/config.json --scheduler-mode async \
  --workload /new/path/stream-sync/workload.json \
  --output /new/path/stream-async
```

The workload mixes short and long inputs, different output budgets, requests
allowing EOS, and client cancellation after two nonempty output updates. Closing
the generation stream exercises AsyncLLM's abort/cleanup path. Both modes use the
same AsyncMPClient frontend; the Python async API alone does not select the engine
scheduler.

Completed requests must reach a terminal result. Fixed-output requests must
respect their exact budget; cancellation must close the stream. The final receipt
requires zero unfinished frontend requests and drains GPU work before computing
the throughput denominator. Per-request results include planned/actual arrivals,
output token IDs, TTFT, output-update events, cancellation and finish reasons.

Speculative output updates can contain several tokens. The recorded inter-update
latencies and request-mean TPOT are not per-token ITL percentiles. The config and
workload files are persisted; the requested scheduler value explicitly overrides
the value in the input config.

## Optional profiling

Pass `--profile-dir /new/path/traces` to the streaming benchmark in a separate
run. After warmup it starts the worker's PyTorch profiler, annotates scheduler
planning and output commits, and exports Chrome JSON traces for Perfetto.

Scheduler annotations patch only the profiling process and are restored on stop.
Their capture depends on the scheduler and worker sharing the executor process;
check for the `yoco_scheduler::` events rather than assuming they were captured.
GPU time remains on the CUDA tracks, not the CPU annotation ranges.

Profiled runs are explicitly marked and excluded from normal throughput results.
Formal timing and profile interpretation must remain separate.
The lifecycle runner allows 60 seconds for engine shutdown, including profiler
cleanup, and releases the profiler object after exporting its trace.

## Correctness and adoption

The targeted Shared-KV regression keeps GPU positions and rejection results
fixed while advancing CPU length bounds by one K8 lookahead. It checks that the
draft still reads the actual raw h20 and strict KV prefix. Existing scheduler
tests cover in-flight outputs, cancellation and preemption; they complement the
native YOCO lifecycle panel.

Evaluate output counts, cache/state integrity, acceptance, latency and throughput.
Compare controlled inputs when investigating numerical divergence. Changing the
scheduler can change batch composition and the existing Fast/Mega kernel path,
so fixed seeds do not establish cross-batch bitwise equivalence.

Async remains an explicit experiment until the intended workload demonstrates
acceptable behavior and performance. Restore synchronous scheduling with
`--scheduler-mode sync`; no model or runtime-library replacement is required.
