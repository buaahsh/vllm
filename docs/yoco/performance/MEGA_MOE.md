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

The [native integration A/B report](mega-moe-integration-20261020/REPORT.md)
records short-input C4/C8 gains of 15.04%/15.76% and 4K-input gains of
8.86%/9.77%, together with C1 regressions, numerical differences, and the
additional layout memory. These are fixed-workload offline generation results.

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

Unsupported configurations fail during startup. Layout preparation and workspace
allocation occur during weight processing, before profiling and CUDA Graph
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

Use `--backend mega` for the candidate. The tool records actual prepared layers,
shared workspace count, FP32 draft-cache dtype, target/draft Graph replay,
generated IDs, request metrics, and speculative counters. It measures C1/C4/C8
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
