# YOCO MTP CUDA Graph and EPLB integration

Code commit: `57b232c4b1015ecb9842e2a6e8c17ac074e8df83`.
Merged `fhb/yoco-eplb-20261102` at `12a6666561` with YOCO VL/MTP
support and the CUDA Graph fix at `9e87aff085`.

## Changes

- Compile the MTP model with a dynamic token dimension instead of requiring
  eager execution.
- Resolve the BF16 draft's quantization independently of the FP8 target.
- Preserve EPLB metadata, Shared-KV/DSpark behavior, and pre-final-norm MTP
  states in both normal and fast prefill paths. The dense draft retains TP
  without inheriting the target's EP/EPLB configuration.
- Read prepared shared FP32 RoPE tables as module buffers. Constant-folding
  the getter produced an unsupported `get_attr` in vLLM's stitching graph.
  Cold initialization remains shared and supports fullgraph compilation.

## Environment and scope

One isolated development node, four B200 GPUs, TP4/EP4. Native image:
`buaahsh/pytorch@sha256:cc4bca1d05d2120d5a9b510088667fba28bdfe332eebdc459b415e6dab665820`.
PyTorch `2.13.0+cu130`, CUDA 13.0; native compiled extensions and DeepGEMM
retained from the image. All staged runtime and test files match the code
commit; twenty historical benchmark documentation manifests were omitted
from the runtime copy. No production service was replaced by this change.

## Automated regressions

| Suite | Result |
| --- | --- |
| YOCO configuration, prefill, conversion, MoE composition and EPLB | 159 passed |
| MTP draft input CUDA replay, rejection and request slot reuse | 1 passed |
| Distributed EPLB expert migration, synchronous and asynchronous | 2 passed |
| YOCO VL image/video processing | 19 passed |
| RoPE cache, kernels, compilation and replay | 32 passed; 1 existing failure |

The new RoPE regression uses vLLM's compiler and CUDA Graph replay with
positions changed to 262143 and 1048575 after capture. Applicable pre-commit
checks pass.

The existing failure is
`test_yoco_fused_weighted_qk_rms_clip_rotary_matches_native_bf16[128]`:
NRMSE `1.3204e-5` exceeds its `1e-5` threshold on this image. The unchanged
EPLB-tip rotary implementation reproduces exactly the same failure
(31 passed, 1 failed). No threshold or kernel arithmetic was changed to
hide it. This is not an all-green result for the entire rotary suite.

## Combined TP4 execution

FP8 target, BF16 MTP=1, EP4, EPLB, fast prefill and
`FULL_AND_PIECEWISE` CUDA Graph were enabled together in a tiny dummy-weight
YOCO model. Both synchronous and asynchronous EPLB completed 30 requests
and 224 per-rank draft checks each. All four ranks showed:

- Actual target and draft replay in both graph modes.
- Changed expert mappings with unchanged expert weight storage addresses.
- Identical draft tokens and zero hidden-state difference versus the same
  inputs on the uncaptured path.
- All 6,291,456 target KV bytes unchanged by each draft invocation.
- Updated target hidden-state contents at a stable buffer address.

These checks use dummy weights; real-weight results are recorded separately.

## Real-weight validation

Ref12000 YOCO VL target with FP8 weights and its BF16 MTP=1 checkpoint:
TP4/EP4, asynchronous EPLB (8 redundant experts, window 32, interval 64),
262144 context limit, 4096 sequence/token scheduling limits, memory 0.80,
FlashInfer attention and Triton MoE. Fourteen configured graph sizes span
1–2048 tokens.

- Five functional cases passed: text, one image, two images, tool call and
  multi-turn recall, with complete streaming responses and usage.
- 128 requests at concurrency 64 passed, generating 32768 tokens; measured
  draft acceptance was 76.91%. This instrumented run is a compatibility smoke
  test, not a paired throughput benchmark.
- Input lengths 8192, 131072 and 262080 each generated exactly 64 tokens.
  These are length/execution checks, not a long-context quality evaluation.
- All four ranks recorded actual replay after readiness: target FULL 746 /
  PIECEWISE 962 calls; draft FULL 746 / PIECEWISE 52 calls. These counts are
  graph replay calls, not batches or coverage of every configured size.
- All four ranks changed expert mappings while preserving expert weight
  addresses. Health/metrics probes had no errors; final running/waiting counts
  were zero.

Real requests used temperature 1, top-k 20, top-p 0.95, no explicit request
seed and no inference retry. The native runtime file check covered 2025 files,
including 13 compiled extensions. Production Ref12000's Pod UID, restart
count and launcher PID/start time remained unchanged. Development inference
was stopped after validation while retaining its four-GPU allocation.

## Regression commands

Run in the repository virtual environment with its test dependencies:

```bash
.venv/bin/python -m pytest -q \
  tests/model_executor/test_yoco_config.py \
  tests/model_executor/test_yoco_prefill.py \
  tests/model_executor/test_yoco_conversion.py \
  tests/model_executor/test_yoco_moe_composition.py \
  tests/model_executor/test_yoco_eplb.py
.venv/bin/python -m pytest -q tests/v1/worker/test_yoco_mtp_graph.py
.venv/bin/python -m pytest -q tests/distributed/test_yoco_eplb.py
.venv/bin/python -m pytest -q tests/models/multimodal/processing/test_yoco_vl.py
.venv/bin/python -m pytest -q tests/model_executor/test_yoco_rotary.py
```

The last command retains the documented baseline numerical failure on the
recorded image. The combined tiny-model instrumentation and raw development
results are archived under the task's `merge-eplb` evidence directory; they
are not installed as production hooks.
