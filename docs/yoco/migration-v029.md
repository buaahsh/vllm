# YOCO upstream migration and refactor

Status: migration and refactor complete, with the single-B200 validation scope
recorded below. Controlled numerical checks, graph reload, FP8/FP8-KV inference,
functional P/D, Qwen regression and diagnostic load comparisons are complete.
First-pass shape JIT produces a measured TTFT spike; the prewarmed repeat returns
to the reference range. Exact trace input qualification remains limited by six
text-synthesis round-trip deviations, retained in every strict audit.

## Provenance

- YOCO source snapshot (including tracked/untracked development; generated build
  helpers are archived separately):
  `dd465efd1235376267aee0275019ce1b2fc277d0`.
- Official vLLM base: `ca67438c088d5ada7da901b77f9aeaebc1b9542c`, fetched
  2026-09-15. Contains `vllm/models/deepseek_v41`.
- Working copy: `vllm-yoco-version-0.29`; the original development directory is
  untouched. Local code commits: `f7ab70e649` (migration/refactor) and
  `aab6d69e67` (native FA4 FP8 capability fix). Source and upstream refs remain
  available for comparison and rollback.
- Implementation plan: [YOCO refactor plan](../../YOCO-REFACTOR-PLAN-20260915.md).

## Ownership and contracts

| Module | Responsibility |
| --- | --- |
| `models/yoco.py` | Public model entry, decoder/block compile boundaries and model assembly |
| `config/yoco.py` | Execution mode, immutable MoE policy, runner capability requirements |
| `models/yoco_config.py` | Configuration aliases, device/backend decisions and explicit startup changes |
| `layers/yoco_ops/{norm,rotary,routing,projection}.py` | Existing numerical implementations, fake operators and registrations |
| `layers/yoco_ops/{fp8,fp8_permute,fp8_moe,triton_moe}.py` | YOCO-specific weighted quantization, layouts and expert execution |
| `layers/yoco_attention.py`, `layers/yoco_moe.py` | Attention and shared/routed/latent expert composition |
| `models/yoco_weights.py` | Projection mapping, expert shards, load report and derived-cache refresh |
| `models/yoco_prefill.py` | Full self pass, shared KV write, compact cross pass and buffer ownership |
| `models/yoco_diagnostics.py` | Opt-in route dump configured at model construction |
| `v1/yoco_cache.py` | Shared final-prompt-block transfer boundary |

The model entry remains `vllm.model_executor.models.yoco.YOCOForCausalLM`.
Custom operator names and schemas remain stable. Archived scripts can temporarily
import former private names through `models/yoco_compat.py`; current internal
callers use the owning modules. Remove this compatibility surface only after
archived benchmarks and user scripts have migrated.

Weights still accept historical names including `model.k_proj.weight` and flattened
`.mlp.experts.w13_weight` / `.w2_weight`. Current upstream stores routed expert
parameters under `.experts.routed_experts`; the loader returns these actual loaded
names. The last load report records ignored weights and default-loader fallbacks.
Unknown weights and the historical TypeError fallback retain their prior behavior;
strict rejection is a separate behavior change. Derived weight caches use the
existing in-place refresh mechanism and must remain stable across graph replay.

Ordinary single-device YOCO inference uses upstream Model Runner V2. YOCO P/D,
DP fast prefill and opt-in Fast BF16 sampling retain their Model Runner V1
integration, selected through upstream's capability checks. Explicitly forcing V2
for these features fails validation. Colocated TP1 Align P/D has passed real transfer/inference checks on this base.
Multi-GPU DP/EP/TP/CP execution is not qualified by the one-GPU experiment.

## Upstream reuse decisions

Reuse the upstream Attention/FusedMoE factories, quantization configuration and
loader infrastructure, DeepGEMM provider-based warmup, packed generic FP8
quantizer, and latent/shared expert reduction interfaces. YOCO policies live in
`FusedMoEConfig` and survive backend recreation/fallback without per-forward
attribute copying. Generic MoE behavior remains outside YOCO's policy path.

DeepSeek V4.1's fused Q/KV normalization uses 32-element MXFP8 groups and swizzled
scale storage. YOCO uses its own head-wise RMS clip, 128-element UE8M0 groups and
weighted routed-activation rounding rules, so that kernel is not a drop-in
replacement. V4.1 compression/cache kernels require 512-dimensional compressed
latents and CR1/CR2 ring state; YOCO caches self windows plus full shared cross KV.
Do not substitute these kernels based only on similar names or tensor dtypes.

## FP8 numerical policy

The source YOCO branch changed the shared FP8 group amax floor from upstream's
`1e-10` to `1e-4`. Migrating only its fused kernels left those producers inconsistent
with the new unfused quantizer for tiny inputs. The migration now carries the YOCO
floor explicitly in FP8 linear and MoE quantization configuration. Other models
retain upstream defaults; backend reconstruction and prepare/finalize receive the
same resolved value. Fused YOCO activation paths require the matching floor.
Tiny-input linear/MoE, attention, latent, routing and Fast-precision GPU suites
pass with this repair. The native CUDA scale floor is a different quantity and
remains unchanged.

## Validation

Artifacts live under `work/yoco-migration-20260915` in the parent workspace.
Counts below describe particular runs, not the union of all tests.

| Evidence | Result / limit |
| --- | --- |
| Old source on A6000 / Torch 2.11 | 200 passed, 4 skipped, 6 pre-existing failures; preserved baseline log |
| B200 operator snapshot `operators-r1` | 52 passed; predates some final source/binary changes |
| Weight pipeline and cache tests | 44 passed, 2 skipped |
| Policy/config/prefill tests | 114 passed, 2 skipped |
| Runner selection and general runner config | 63 passed, 319 deselected |
| NIXL alias registration across layouts/backends | 48 passed, 110 deselected |
| DeepSeek V4.1 parser / shared warmup / V4 backend selection | 41 passed; not whole-model evaluation |
| V1 YOCO fused slot mapping | 1 passed, 15 deselected |
| Controlled Fast BF16 pre-extraction/refactored comparison | Eager and graph each have 18 matched input frames and bitwise-identical full logits |
| Real B200 Align, snapshot `model-r5` | Passed after FA2 CUDA 13.0 build; repeated tokens equal, selected logprob delta 0 |
| Real B200 Fast BF16 full forward, snapshot `model-r5` | Repeat-token check passed; 316 weights loaded without ignored inputs/fallbacks |
| New model/components type checking and regressions | mypy clean; 117 passed, 2 skipped |
| Align graph with changed-router reload and restoration | Output changes, then restores exactly; 647 CUDA tensor storage descriptions stable |
| Colocated Align 1P1D on one B200 | 8 lengths through 4097 tokens: token/logprob equality; 5616 external cached tokens, zero local hits |
| Pure upstream / candidate Qwen GSM8K fixed subset | Both 57/64 correct, no invalid answers; 33/64 identical token sequences |
| B200 main operator/runtime test matrix | 248 passed, 2 skipped across r5/r6; later FP8-floor regressions also pass |
| Latest Fast-precision GPU suite | 33 passed, including tiny-input YOCO/non-YOCO isolation |
| Fast FP8 graph / real checkpoint | Passed with 336 loaded weights, no ignored/fallbacks and identical repeated output tokens |
| FA4 FP8 capability, paged scales, GQA/local/split-KV and graph scales | 26 passed on B200, including native FA4 interface tests |
| Real FP8 model + FP8 KV, graph enabled | Passed; 336 weights loaded,20 fused attention modules, fixed K/V scales 1, identical repeated tokens |
| Controlled upstream / candidate Qwen graph | 16 matched input frames; full logits byte-identical |
| Candidate streaming/stop/disconnect | Frontend stop completes; client disconnect drains running/waiting queues to 0; health retained |
| Fixed-shape same-device ABBA | Five pooled median latency deltas: -0.57% to -0.10%; two pairs, no speedup claim |
| Latest interface checks | 47 passed, 1 skipped; scheduler finish contracts 17 passed; V4.1 quantization dispatch 1 passed |
| Final scoped pre-commit and manual Python 3.12 mypy | Passed; broader quantizer errors separately match untouched upstream |

B200 has one allocated GPU. Source overlays preserve the official Docker-native
DeepGEMM directory and verify loaded module/library hashes. Source-built vLLM CUDA
libraries are separate artifacts. FA2's upstream build uses forward-compatible
SM80 PTX even in an SM100 build, so building it with a newer toolkit than the
node's driver supports can fail only when the attention path executes.

The upstream packed KV cache layout differs from the source version. Existing
NIXL compatibility metadata must be honored; old/new workers must not be assumed
interchangeable. Both P/D endpoints must be upgraded together and old transfer metadata drained;
the functional test uses two independent TP1 processes colocated on one B200,
not a two-GPU throughput topology.

The selected runtime, code-review and load checks are complete. This is a
single-GPU acceptance scope with the explicit numerical, topology and trace
limits below. A compact machine-readable index is in
[validation/migration-v029.json](validation/migration-v029.json).

## Reading and extending the implementation

The public model entry is now 1,040 lines (source review baseline: 5,437).
Read `YOCOModel.forward` and `yoco_prefill.py` for execution order. A kernel change
belongs in the corresponding `yoco_ops` file; keep its wrapper, fake function and
registration together. A backend choice belongs in `yoco_config.py`; resolve a
`YocoMoEBackendDecision`, then apply startup changes explicitly. Add MoE options
to `YocoMoEPolicy` and preserve them when constructing/reconstructing backends.
Checkpoint mapping changes belong in `yoco_weights.py`, followed by the existing
in-place derived-cache refresh. Extend the matching test module rather than the
checkpoint-conversion suite.

The six planned implementation boundaries are present: source baseline and
focused tests; operator ownership/registration; typed configuration and explicit
backend decisions; model/weight/cache lifecycle; complete MoE policy propagation;
and shared prefill/connector tail rules with opt-in diagnostics. GPU acceptance
is bounded by the matrix above, including the one-GPU topology limitation.

The shared training imports `layers/yoco_probabilities.py` and
`layers/yoco_align_moe.py` are unchanged from the captured source snapshot.
`models/yoco_compat.py` retains former private operator imports for archived
`tools/yoco_alignment`/benchmark scripts. Current tests and internal callers
patch/import the new owning module; successful legacy import alone is not a
correctness test.

Experimental controls remain explicit. Their existing defaults and precision
eligibility are preserved; absence from the runtime matrix is not qualification.

| Control | Default | Purpose |
| --- | --- | --- |
| `additional_config.yoco_execution_mode` | `fast` | Fast/Align execution semantics |
| `VLLM_YOCO_FP8_ATTENTION_FUSION` | `1` | Eligible FP8 Q/KV fusion |
| `VLLM_YOCO_FP8_LATENT_NORM_FUSION` | `0` | Opt-in latent normalization fusion |
| `VLLM_YOCO_BF16_RESIDUAL`, `VLLM_YOCO_BF16_CHAIN` | `0` | Opt-in BF16 residual/chain experiments |
| `VLLM_YOCO_BF16_REDUCTIONS`, `VLLM_YOCO_BF16_SAMPLING` | `0` | Opt-in reductions/sampler; sampler selects V1 |
| `VLLM_YOCO_FP8_W2_TUNING` | `1` | Existing W2 tuning policy |
| `VLLM_YOCO_ALIGN_MOE_CONFIG` | unset | Existing explicit Align tactic file |
| `VLLM_YOCO_LOGICAL_ROUTE_DUMP` | unset | Opt-in eager diagnostic, selected before forward |

## Scope of the evidence

The pre-extraction comparison substitutes the preserved 5,425-line model into
the same migrated supporting runtime and native libraries. It proves extraction
parity in the controlled cases; it is not an end-to-end original-v0.20 comparison.
The earlier batched/shared-prefix cross-mode frame comparison used unmatched
model calls and is explicitly invalidated in the artifact directory.

A separate Qwen graph run with one fixed 128-token prompt, no cache and synchronous
scheduling has 16 matched frames with byte-identical full logits across pure
upstream and final candidate. This strengthens the shared-framework check without
claiming that the batched evaluation is deterministic.

The Qwen evaluation uses 64 fixed GSM8K questions, 5-shot prompts and a 256-token
cap. Equal accuracy does not imply bitwise equality. Batched/APC scheduling and
small sample size limit the conclusion; incidental evaluation latency is not a
controlled performance result. DeepSeek V4.1 implementation is retained from
pinned upstream and dispatch/parser tests pass; a full DeepSeek checkpoint was
not run on the one-B200 allocation.

The standard scoped mypy hook passes. A separate, broader check of shared FP8
files reports 12 errors also present in untouched upstream, with zero introduced
errors in the normalized comparison. These files are not claimed universally
type-clean. Existing source-baseline test failures are retained in the artifacts.

The full upstream output-processor suite requires the gated Llama-3.2-1B
tokenizer, unavailable to this environment. Four tests passed before fixture
errors stopped that attempt. Eight tokenizer-independent frontend stop/abort
cases pass with the local YOCO tokenizer and unchanged assertions; this does
not substitute for the Llama-specific stop-token tests.

## Build and reproduction

Use `uv` and the repository `.venv/bin/python`, following `AGENTS.md`. Local SM86
and B200 SM100 extension builds passed. The B200 image is the official v0.29.0
amd64 digest `sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`.
Docker-native DeepGEMM is retained at
`/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm`;
its `_C` SHA-256 is
`73824dc1312e98cf277ae6bc017becf7e963d5b0bfab81e0c6d475d9b769c9e7`.
No replacement DeepGEMM was built or installed.

FA2 uses unchanged upstream source commit
`506341a143fcabd4bb79052a7605ada727d6b3f5`, built inside the image with CUDA
13.0.88 for the node's 580.95.05 driver. Library SHA-256:
`46115161b93a369dba371389e2fb4436dae54bd589ce9c538bef8e88426e18ae`.
An earlier CUDA 13.3 build produced incompatible PTX; that failed run is retained.
Mooncake CUDA13 0.3.12 is installed only in the isolated test environment.

Fixed-shape charts and raw samples are in
`work/yoco-migration-20260915/figures/{fixed-shape-abba.svg,fixed-shape-abba.png,fixed-shape-samples.csv}`.
The error bars retain individual timing spikes. Larger fixed batches show
repeat variability already present in the reference; all candidate output
fingerprints were also observed in reference runs. Early shape-specific JIT events
are recorded in the process logs and the first two iterations are warmups.

`work/yoco-migration-20260915` in the parent workspace contains the immutable
source bundles/manifests, `bundle_runtime.py`, GPU controller and named requests,
model/graph/reload/P/D probes, fixed-shape scripts/results, frozen traces, client
commands, logs and audits. The controller's owner plan is private and must not
be copied into version control. Do not release or reuse the holder without
checking its live ownership and the current experiment state.

## Trace diagnostic and client limits

The frozen FAST25 toolagent window retains 1,769 of 23,608 source requests
(7.49%), after the t300–610 second window and total-context ≤81,920 filters.
The timestamp speed is 0.5×, giving 617.998 seconds of arrivals. Trace SHA-256:
`e83639bfa5b6ba87d50eb2a3835ac24e2dcf24c5d7218de6a8bdbbae59f712cd`.
Its source hash, filters and retained fraction are preserved in the manifests.

All three complete cases used the same B200, model, Fast BF16/FA4 graph settings,
max sequences 32, max batched tokens 8192, AIPerf 0.12.0/seed 42, streaming completions,
server token accounting, a 512-request in-flight safety ceiling and a fresh cache
salt. Candidate first/repeat used the same running model-r11 service. The before
reference used the pre-extraction model on supporting model-r10; the only later
production change is FP8 capability/validation, inactive on this BF16-KV path.

| Metric | Reference | Candidate first | Candidate repeat |
| --- | ---: | ---: | ---: |
| Requests/s | 2.80 | 2.80 | 2.80 |
| Actual input tokens/s | 23879.85 | 23885.33 | 23889.08 |
| Output tokens/s | 496.38 | 496.49 | 496.57 |
| TTFT ms P50 | 882.89 | 928.39 | 912.92 |
| TTFT ms P95 | 2395.26 | 2804.11 | 2451.00 |
| TTFT ms P99 | 3386.46 | 6459.64 | 3279.37 |
| ITL ms P50 | 33.80 | 33.99 | 34.14 |
| ITL ms P95 | 145.85 | 144.07 | 147.43 |
| ITL ms P99 | 160.51 | 156.64 | 162.39 |
| E2E ms P50 | 2812.45 | 2923.41 | 2829.30 |
| E2E ms P95 | 17514.00 | 17815.14 | 17486.99 |
| E2E ms P99 | 26554.28 | 28246.39 | 26605.87 |
| Completed/errors | 1769/0 | 1769/0 | 1769/0 |
| Schedule lag P99 ms | 5.73 | 5.57 | 6.19 |
| Max effective concurrency | 48 | 47 | 46 |
| Max running/waiting | 32/19 | 32/20 | 32/21 |
| Overhang after final arrival s | 13.25 | 13.11 | 13.01 |
| Maximum metrics scrape gap s | 1.27 | 1.17 | 2.49 |

Offered load: 2.862 requests/s,
24392.08 input tokens/s and
507.03 output tokens/s. Every complete case
has 0 HTTP errors, cancellations, context skips and output-length mismatches;
no degraded scheduling, counter reset or scrape gap >5 s; final queues 0; reported
GPU uncorrected ECC 0 and no recovery action. Host kernel Xid logs were not
available. Prefix queries/hits are 15,074,263/5,825,984 (38.65%) in all three cases.

Actual per-request input/output counts match across all three cases. Each has
313,342 output tokens and 15,074,263 input tokens. The frozen trace requests
15,074,257 inputs: six generated texts re-encode to one extra token each
(rows 261, 362, 370, 822, 1304, 1329). Trace bytes/ISL/OSL/hash_ids are unchanged.
Every strict input audit therefore remains false. These runs support a
same-workload diagnostic comparison, not exact-length trace qualification,
capacity under an SLO, live-agent behavior or model quality.

The first candidate's TTFT P99 is 6.46 s versus reference 3.39 s. Its worst 22 deltas
cluster near the 60 s burst, during creation of two new Triton `fused_moe_kernel`
source/PTX/cubin groups. In the same-service prewarmed repeat, no new cubin files
were written in the inspected Triton/Inductor caches; TTFT P99 is 3.28 s, output
throughput 496.57 tokens/s, and E2E P99 differs by +0.19% from reference. This
supports first-seen shape compilation as the main spike contributor. The first
run remains included in figures and is not presented as a performance pass.
Warm TTFT P50 is +3.40% and P95 is +2.33%; this small shared-node sample does not
establish a general speedup or strict capacity result.

AIPerf's offline workers needed a five-line existing-directory tokenizer resolver
fix; a 900-token decode comparison passed. Scheduling, synthesis and metrics code
were unchanged. Failed setup/pooled-connection cases and the interrupted BOS
mismatch case are retained. Compared runs explicitly disable connection reuse
and server-added special tokens. The lifecycle probe separately confirmed the
new API's four-stop-string limit, normal stop completion and disconnect drain.

Artifacts: `trace-paired-diagnostic.json` retains the first pair;
`trace-warm-verification.json` records the warm comparison;
`figures/trace-warmup-verification.{svg,png}` and the accompanying CSVs show all
three cases. `gpu/tail-latency-cache-mtimes.jsonl` and
`gpu/warm-repeat-cubin-mtimes.json` preserve the JIT evidence. The static
plotting scripts, raw records, server snapshots and all failed attempts remain
under the artifact root.

## Upgrade and retained test environment

Upgrade both P/D endpoints together, drain old requests/transfers and rebuild
KV caches; old/new packed layouts are not promised interoperable. YOCO FP8 KV
requires Fast mode, native FA4 on the tested SM100 path, BF16 output, head size
aligned to 16 and fixed scales. `calculate_kv_scales` is removed from upstream
EngineArgs; the retained compatibility check still rejects unsafe legacy
recalibration with shared fast prefill.

The FA4 acceptance run exposed a stale import from the ignored `fa4_compat.py`
build helper. Commit `aab6d69e67` uses upstream capability detection and restores
existing output/head constraints. The helper is archived in `source-generated`;
upstream CuTe supplies its MMA-kind and FP8-softmax fixes, verified by 26 tests.
The FP8-KV graph run records 20 fused attention modules and fixed K/V scales 1.

The owned B200 holder remains allocated in `bonete01`, Pod
`lidong1-yoco-migrate-v029-b200-g1-0915-r2-holder-0`, node
`slc01-cl02-hgx-0251`. Candidate service run `trace-server-after-r3` uses
`/workspace/model-r11`; the existing local forward is `127.0.0.1:18888` to the
Pod's loopback 18200. Holder readiness is not a capacity signal; inspect the
bound controller and HTTP health before further experiments. No production
routing or external publication was performed.
