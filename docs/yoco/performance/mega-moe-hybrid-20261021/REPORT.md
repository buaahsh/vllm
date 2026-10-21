# YOCO hybrid MoE dispatch end-to-end report

The hybrid keeps small verification batches and prefill on the original FP8
path, while larger decode graphs use Mega. Auto selection and the equivalent
manual YAML are tested through native model execution.

## Configuration and measurements

Same B200 GPU6, target and Shared-KV draft as the
[all-Mega report](../mega-moe-integration-20261020/REPORT.md): TP1/PP1/DP1/EP1,
online FP8 target, BF16 draft/KV, DSpark K8, T1, APC disabled, max sequences8,
batch-token budget2048, max length8192, FULL_DECODE_ONLY with exact
9/18/36/72-row target graphs. No expert/model forward monkeypatch is used.
The observer counts graph replays and records actual native capture decisions.

Run order is current A1, hybrid B1 (auto), hybrid B2 (manual YAML), current A2.
Each engine has two warmups per case, five scored short-input trials and three
scored 4096-token synthetic-input trials at C1/C2/C4/C8, 512 output tokens per
request: 128 scored samples total. Every sample is retained in [trials.csv](trials.csv).
Aggregate throughput is total output tokens divided by total elapsed seconds.
Timing includes prefill, target/draft generation, scheduling, sampling and GPU
drain; it excludes model loading, compilation, tokenization and HTTP.

| Input | C | Original tok/s | Hybrid tok/s | Change | Original acceptance | Hybrid acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| short | 1 | 423.90 | 419.86 | -0.95% | 48.04% | 48.04% |
| short | 2 | 607.05 | 673.44 | +10.94% | 45.02% | 46.79% |
| short | 4 | 892.02 | 970.65 | +8.81% | 39.80% | 37.71% |
| short | 8 | 1503.78 | 1625.71 | +8.11% | 37.42% | 36.56% |
| context4096 | 1 | 342.27 | 337.86 | -1.29% | 46.73% | 46.73% |
| context4096 | 2 | 480.53 | 503.90 | +4.86% | 44.06% | 42.07% |
| context4096 | 4 | 634.16 | 662.73 | +4.51% | 39.42% | 35.52% |
| context4096 | 8 | 901.86 | 940.06 | +4.24% | 35.68% | 34.96% |

C1 retains some latency overhead despite using the original path: median TTFT
is 104.03 -> 109.90 ms for short input (+5.64%), and 303.96 -> 318.45 ms for
4K input (+4.77%). The throughput regressions are -0.95%/-1.29%. This change
preserves the measured C1 outputs and acceptance, with a smaller remaining
performance gap; it does not establish zero latency overhead. The cause of
that remaining gap has not been isolated.

## Dispatch and numerical behavior

Resolved auto policy: `{'profile': 'b200-l3-fp8-tp1-k8', 'triton_max_tokens': 16, 'mega_min_tokens': 17, 'mega_prefill': False}`.
Thresholds are local token rows including graph padding. Mega changes
activation quantization from group128 to group32; routed FP8 weights/scales
remain frozen. Prefill stays on the original path. This panel does not establish
task accuracy or batch invariance.

| Captured target graph | Original | Hybrid |
| --- | --- | --- |
| 8x9/72 | deep_gemm | mega |
| 4x9/36 | deep_gemm | mega |
| 2x9/18 | deep_gemm | mega |
| 1x9/9 | triton | triton |

C1 short: 10/10 paired sampled continuations are token-identical to baseline.

C1 context4096: 6/6 paired sampled continuations are token-identical to baseline.

Auto/manual sampled outputs: 32/32 scored cases are identical.

Teacher-forced panel: 2048 fixed continuation tokens per engine; all prefill logprobs identical: True. Mean NLL baseline=0.249942250370, hybrid=0.249942250370.
Mean/max absolute prefill logprob difference: 0/0.
The fixed continuations are numerical observations. Decode at larger batches
still uses a different activation recipe and can change sampled outputs.

All requests completed the fixed token budget without corruption or preemption.
The JSON includes TTFT and P95 of each request's mean post-first-token interval;
the latter is not a per-token ITL P95.

## Reproduction and resource receipt

Use the [auto YAML](../../../../examples/offline_inference/spec_decode/yoco_moe_dispatch/auto.yaml)
or [manual YAML](../../../../examples/offline_inference/spec_decode/yoco_moe_dispatch/manual.yaml).
Unprofiled hardware/parallel/precision/graph configurations retain the original
backend and allocate no Mega layout. Manual settings do not add support for
other Mega parallel layouts. This profile is an offline fixed-workload result.

```bash
CUDA_VISIBLE_DEVICES=<assigned-gpu> VLLM_USE_V2_MODEL_RUNNER=1 \
  .venv/bin/python -m tools.yoco_alignment.benchmark_mega_moe \
  --backend hybrid --tokens 512 --trials 5 --extra-prompt-tokens 4096 \
  --output <new-directory>
```

Repeat with `--backend current` for A1/A2. For B2 append
`--dispatch-config examples/offline_inference/spec_decode/yoco_moe_dispatch/manual.yaml`.

GPU checks: 55 passed (Shared-KV, Mega scale/dispatch boundaries, Triton
configurable thresholds and graph-visible scale-cache refresh). CPU checks:
123 passed; V2 graph memory checks: 13 passed.
Scoped pre-commit logs are retained in the local artifact directory.

[Receipt](receipt.json) binds the runtime and measured source hashes.
[Full summary](summary.json) retains per-run data, paired ratios, acceptance,
latency, memory and graph replays. The source manifest and raw engine logs are
under `/data/vllm_test/work/yoco-mega-hybrid-20261021/`; remote artifacts are under
`/home/fhb/yoco-mega-hybrid-20261021/`. Previous all-Mega evidence is preserved.
An initial candidate that missed V2 runtime-NONE FULL captures is retained
under remote `diagnostics/capture-mode-none` and excluded from this comparison.

Prepared layouts add 20.48 GiB of active allocation in this run. All 20 routed
layers share one 525678464-byte Mega workspace.
The existing development Pod, UID, node and unrelated GPU processes remain
untouched; GPU6 is released after each engine.
