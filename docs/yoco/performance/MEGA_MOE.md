# YOCO FP8 MegaMoE

This branch provides an opt-in routed-expert implementation for YOCO L3 on
SM100. It preserves the existing router, latent transforms, independently gated
shared expert, and runner-managed shared-expert overlap.

Enable it with:

```bash
--quantization fp8_per_block \
--additional-config '{"yoco_execution_mode":"fast","yoco_mega_moe":"fp8_group32"}'
```

`yoco_mega_moe` defaults to `off`. The `fp8_group32` name explicitly selects a
different activation quantization recipe: FP8 weight values and their block128
scales are preserved, while routed input and intermediate activations use
group32 scales. Output logits and speculative acceptance can change. This
mode does not provide Align or batch-invariant guarantees.

## YAML dispatch configuration

The default dispatch is now `auto`. Use the
[auto YAML](../../../examples/offline_inference/spec_decode/yoco_moe_dispatch/auto.yaml)
or edit the
[manual YAML](../../../examples/offline_inference/spec_decode/yoco_moe_dispatch/manual.yaml)
and merge it with the checkpoint, draft and parallel configuration:

```yaml
additional-config:
  yoco_execution_mode: fast
  yoco_mega_moe: fp8_group32
  yoco_moe_dispatch:
    mode: manual
    triton_max_tokens: 16
    mega_min_tokens: 17
    mega_prefill: false
```

These thresholds count **local routed token rows, including graph padding**,
not request concurrency. K8 verification usually contributes nine rows/request.
Mega takes precedence at `M >= mega_min_tokens` in eligible execution phases;
otherwise the original backend uses Triton at `M <= triton_max_tokens` and
DeepGEMM above it, subject to its existing compatibility checks. Both boundaries
are inclusive; overlapping manual boundaries are rejected. Triton's specialized
packed-scale fallback currently supports this TP1 FP8 layout and limits in
`[0,64]`; zero disables it. Null retains its original threshold. A null
`mega_min_tokens` disables Mega layout preparation and allocation.

With `mega_prefill: false`, Mega runs only in FULL graphs under
`FULL_DECODE_ONLY`. Prefill, eager execution, piecewise graphs and profiling use
the original path. This also reserves the original workspace before graph
addresses are locked. Request-dependent Python branches are never baked into
shared piecewise graphs. `mega_prefill: true` opts into shape-only Mega dispatch
for prefill and eager execution as well; it can change C1's prefix KV and
acceptance even when small decode batches fall back. Use
`mode: manual, mega_min_tokens: 1, mega_prefill: true` to reproduce the original
all-Mega experiment. Threshold changes require a new engine and graph capture.

`auto` looks up a measured configuration; it does **not** benchmark a live
service or infer thresholds for an unseen parallel layout. Startup logs report
the selected profile and thresholds. The initial registry is:

| Profile | Matching configuration | Triton | DeepGEMM | Mega |
| --- | --- | --- | --- | --- |
| `b200-l3-fp8-tp1-k8` | B200, L3, online block128 FP8 with E8M0 scales, BF16 input, TP1/PP1/DP1/EP1, V2 DSpark K8, FULL_DECODE_ONLY | M ≤16 | Original prefill/eager path above16 | Decode graph M ≥17 |
| `unprofiled-original` | Any configuration outside that measured contract | Original selection | Original selection | Disabled |

The profile additionally requires all supported-configuration restrictions below.
Unknown auto configurations keep the original backend and allocate no Mega
weights. Manual thresholds do not extend Mega's supported parallel layouts.
Future profiles should record hardware, parallel groups, model/precision,
speculation, graph mode and end-to-end evidence before enabling automatic use.

The [native integration A/B report](mega-moe-integration-20261020/REPORT.md)
records short-input C4/C8 gains of 15.04%/15.76% and 4K-input gains of
8.86%/9.77%, together with C1 regressions, numerical differences, and the
additional layout memory. These are fixed-workload offline generation results.

The [hybrid dispatch report](mega-moe-hybrid-20261021/REPORT.md) compares the
automatic policy and equivalent manual YAML against the original backend,
including C2, actual capture decisions, and paired C1 continuations.

## Supported configuration

- SM100, Fast execution, online block128 FP8 routed weights, BF16 latent input.
- L3 routed dimensions: 128 experts, top8, latent1024, intermediate3840,
  SwiGLU clamp10.
- TP1/PP1/DP1/EP1, without CP, DBO, EPLB, LoRA, or sequence parallelism.
- A DeepGEMM package exposing `get_symm_buffer_for_mega_moe`,
  `transform_weights_for_mega_moe`, and FP8 `fp8_fp4_mega_moe` with
  `recipe=(1,1,32)`. The validated dependency is commit
  `1fc9c66975ee322d628b3288534ca0e9ce12a64d`; a package version alone does not
  establish compatibility. The branch does not install or replace this library.
- Frozen weights. Reloading a prepared model raises before loading new weights;
  create a new engine to change the target weights or execution mode.

Unsupported manual Mega configurations fail during startup. Layout preparation
and workspace allocation occur during weight processing, before profiling and CUDA Graph
capture. Layers share one workspace and execute routed kernels on the runner's
stream. The original routed weights remain allocated alongside the Mega layout;
include the extra layout storage and workspace when sizing KV capacity.

Shared-KV K8 also requires a BF16 draft, BF16 target KV, and BF16 shared embedding
and LM-head storage. This branch includes the constrained online FP8-target
integration, FP32 proposal-distribution cache, and the int64 physical-page address
fix used by the benchmark.

## End-to-end benchmark

The benchmark uses the ordinary LLM generation API and native configured MoE
path. Its worker extension only observes runtime configuration and Graph replay;
it does not replace model or expert forward methods.

Run each backend in a fresh process, on the same GPU and runtime, in
`current -> mega -> mega -> current` order:

```bash
CUDA_VISIBLE_DEVICES=<gpu-uuid> VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_BATCH_INVARIANT=0 PYTHONPATH=. .venv/bin/python \
  -m tools.yoco_alignment.benchmark_mega_moe \
  --model <target-checkpoint> --draft <exported-shared-kv-draft> \
  --backend current --tokens 512 --trials 5 --extra-prompt-tokens 4096 \
  --output <new-result-directory>
```

Use `--backend hybrid` for automatic dispatch, or `--backend mega` for all-Mega.
`--dispatch-config <manual.yaml>` exercises the exact YAML thresholds.
The tool records the resolved profile, prepared layers,
shared workspace count, FP32 draft-cache dtype, target/draft Graph replay,
generated IDs, request metrics, and speculative counters. It measures C1/C2/C4/C8
with fixed seeds, T1, K8, BF16 KV, APC disabled, and a fixed output budget. The
optional 4096-token input uses synthetic text padding. Fixed teacher-forced
continuations provide numerical observations, not a task-quality evaluation.

The timer includes prefill, generation, sampling, scheduling, KV handling, and
GPU drain. Loading, compilation, tokenization, and HTTP transport are outside
the timing window. Retain low-concurrency regressions and precision differences
when deciding whether to enable the mode for a workload.

## Regression checks

```bash
.venv/bin/python -m pytest \
  tests/model_executor/test_yoco_config.py \
  tests/model_executor/test_yoco_moe_composition.py \
  tests/model_executor/test_yoco_conversion.py \
  tests/model_executor/test_yoco_shared_kv.py \
  tests/kernels/moe/test_yoco_mega_moe.py -q
```

The physical-page overflow test intentionally reserves approximately 4 GiB to
exercise real addresses above the signed-int32 element-offset boundary. Run GPU
checks on an assigned device with sufficient available memory.
