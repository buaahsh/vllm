# YOCO Fast Softmax Mega Gate v2

Base vLLM: `7c3f32c985b5b98f83ee45ce9311d291a90b0348`.
DeepInfer reference: `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`.
DeepGEMM patch base: `1fc9c66975ee322d628b3288534ca0e9ce12a64d`.

The supported Fast configuration now enables the padding-safe v2 backend
automatically. [Whole-model measurements](FULLMODEL_RESULTS.md) show
**1.1–2.8%** throughput improvement at C16–64, with a **11.7% C1 regression**
inside the same maxseq64 engine. This is a throughput-oriented default;
output equivalence and MATH-500 accuracy are not established.

Pure Prefill now uses its token budget for eligibility. The separate
[P/D Prefill validation](PREFILL_RESULTS.md) confirms native fused execution,
but finds no stable P throughput gain: C64 median +0.31%, C8 −1.28%, C1 −1.89%.
At C64, 11/64 first-token IDs differ across arms.

## Implementation

The dependency patch is
[`deep_gemm_yoco_softmax.patch`](../../../../tools/yoco_alignment/deep_gemm_yoco_softmax.patch).
It adds a `scoring_func="yoco_softmax"` specialization to the original SM100
Mega Gate. Tensor Core GEMM, task decomposition, scratch/barriers and DeepSeek's
unchanged `select_warp_topk` / four-element sorting network are retained.
After all split-K partials are summed, logits
are rounded to BF16 before selection. The selected eight logits receive a
stable FP32 softmax. IDs are written directly as INT32 and weights as FP32.
The existing sqrtsoftplus API retains its INT64 outputs.

The implementation specializes H3072/E128/K8 with no scoring bias, expert map,
random/fixed routing or shared-as-routed experts. YOCO's independent shared
expert and gate keep their current behavior. This is a fused launch; the
original Mega Gate global scratch traffic is still present.

vLLM routes through the existing modular custom-routing hook. The router tensor
slot carries the original BF16 H3072 input until that hook, because the expert
runner has already transformed its activation input to latent H1024. The
integration requires Fast BF16 routing, SM100, H3072/E128/K8,
TP/PP/PCP/DCP/DP1, EP and DBO off, batch-invariant mode off, a modular MoE
backend and route dumping off. It reads the existing BF16 weight cache,
including in-place weight refreshes.

`VLLM_YOCO_MEGA_GATE` defaults to **1**. Unsupported configurations and missing
or incompatible patched DeepGEMM fall back to the existing router.
`VLLM_YOCO_MEGA_GATE=0` disables the fused backend before engine startup.
`VLLM_YOCO_MEGA_GATE_MIN_TOKENS` defaults to **128**; below this row count the
hook executes the existing BF16 GEMM and Triton Top-8 implementation.
For a pure Prefill engine (a configured KV connector with `kv_role=kv_producer`),
eligibility uses `max_num_batched_tokens >= 128` under the default threshold.
Decode, `kv_both`, and standalone engines retain the capacity check
`max_num_seqs * (num_speculative_tokens + 1) >= 128`.
The per-call row threshold still applies to every eligible engine. A
low-concurrency request can therefore use Mega Gate during a long Prefill.
The threshold is a conservative policy, not a measured universal crossover.
Both settings and the resolved per-module router backend enter compilation
cache factors. Changing the available backend cannot silently reuse a graph
compiled for another implementation.

v1 could trap on nonfinite CUDA Graph padding when the active batch shrank.
v2 validates whole token rows and writes masked expert IDs `-1` / weights `0`
for nonfinite rows without another kernel launch. Finite rows keep the same
selection and normalization. It uses a separate JIT namespace to exclude v1
cubins, and vLLM rejects the v1 extension marker.

## Build the dependency

Use a separate development copy of the pinned DeepGEMM source with its pinned
submodules initialized. Rebuild the C++ extension and install its matching JIT
headers together; the patch changes `RoutingArgs`. vLLM checks the extension's
`yoco_mega_gate_version=2` marker before enabling the new scoring mode.

From the vLLM root, with `DG_SRC` pointing to that development copy:

```bash
gate_patch="$PWD/tools/yoco_alignment/deep_gemm_yoco_softmax.patch"
git -C "$DG_SRC" apply --check "$gate_patch"
git -C "$DG_SRC" apply "$gate_patch"
DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 \
  uv pip install --python .venv/bin/python --no-build-isolation "$DG_SRC"
```

The upstream DeepGEMM build prerequisites, including CUDA 12.8+ and its host
development headers, still apply. The included MIT license covers patch
context copied from DeepSeek's implementation.

## High-concurrency validation

Measure full GEMM + Top-8 against the fused gate, with CUDA Graph GPU timing and
alternating A/B order. Repeated calls are captured inside each measured graph
to avoid Python replay submission becoming the bottleneck. First use synthetic
inputs for smoke testing, then pass `--weights` and `--activations` files with
the actual BF16 compute cache and router activations.

Request concurrency C and router rows M are different quantities. Ordinary
Decode has nominal M=C; K8 target verification has nominal M=9C when fully
populated. Actual scheduler batches can have fewer rows. Prefill rows are
measured independently; these kernel measurements do not simulate KV transfer,
1K request prefixes, scheduling or a full P/D engine.

```bash
.venv/bin/python -m pytest tests/kernels/test_yoco_fast_routing.py -k mega_gate -q
.venv/bin/python -m benchmarks.kernels.benchmark_yoco_mega_gate \
  --concurrencies 16 24 32 40 48 56 64 128 \
  --speculative-tokens 8 --output gate-k8.json
.venv/bin/python -m benchmarks.kernels.benchmark_yoco_mega_gate \
  --concurrencies 16 24 32 40 48 56 64 128 \
  --speculative-tokens 0 --prefill-rows --output gate-decode.json
```

The benchmark reports mismatched expert-selection rows, probability error,
individual timing samples and speedup. It exits unsuccessfully on a routing
mismatch. BF16 rounding and tie rules are preserved, but changing the GEMM
implementation or split-K schedule can still change accumulated results;
equivalence to cuBLAS is not guaranteed. Whole-model timing, K8 acceptance and
a small teacher-forced probe are in [the v2 report](FULLMODEL_RESULTS.md).
MATH-500 has not been run for this candidate. Historical finite-input v1 router
microbenchmark measurements are recorded in [B200 results](B200_RESULTS.md).

## Local checks

The local host has A6000 GPUs; native execution uses the isolated B200 task.
The source patch passes `git apply --check` against its
pinned base. SM100 cubin compilation succeeds for a YOCO single-CTA variant,
a YOCO two-CTA/split-K variant and the original sqrtsoftplus variant; resource
inspection confirms all three kernel bodies are present, without stack or
local-memory spills in those compiled variants.

The patched C++ API passes a host syntax check. Routing/configuration/MoE
composition regression tests pass: **120 passed, 12 SM100 tests skipped**.
Ruff lint/format checks and `git diff --check` pass. The skipped tests cover the
actual fused kernel's BF16 rounding, ties, negative logits and graph replay;
their results must not be inferred from the offline compilation checks.

Host/device build artifacts and logs are in
`/data/vllm_test/work/yoco-mega-integration-20261020/mega-gate-validation/`.
The original DeepInfer/DeepGEMM checkout and installed runtime were not changed.

## B200 checks

The v2 native routing suite passes **18 tests** on an allocated B200 GPU,
including shrinking CUDA Graph batches with NaN/Inf padded rows.
The initial v1 attempt exposed a test-oracle API error
under PyTorch 2.13; changing `argsort(-1, stable=True)` to use `dim=-1` fixed it.
The corrected test suite passes before performance measurement.

The default Mega Gate schedule lowers K8/C64 router latency from 8.63 to
6.89 microseconds in the measured case. Real router weights with synthetic
activations also expose probability and occasional expert-ID sequence
differences. Disabling split-K reduces these differences and removes all
observed expert-ID sequence differences, but K8/C64 latency rises to 9.28
microseconds and some other shapes still have probability differences.
These v1 microbenchmarks are historical; their numerical differences still
apply to the finite-row v2 computation. See the v2 whole-model report for the
current default policy, concurrency results and numerical limitations.
