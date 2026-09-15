# Experimental YOCO Fast latent Norm → FP8 projection

The fusion is implemented and can be enabled for isolated evaluation with
`VLLM_YOCO_FP8_LATENT_NORM_FUSION=1`. **It defaults to disabled because full-model
quality has not matched the original path.** The main residual, FA4 output,
training implementation, and normal inference defaults are unchanged.

## Implementation

The producer fuses `fc2_latent_norm` with the activation quantizer for
`fc2_latent_proj` (1024 → 3072). It retains FP32 RMS statistics and the existing
BF16 norm-output rounding in registers, directly writes E4M3 activations and
packed per-token group-128 UE8M0 scales, and calls the existing DeepGEMM projection
without another activation quantizer.

There are 20 physical output transforms and 40 logical executions per decode
step in the L3 model. Latent input/output projection weights were already FP8.
This change targets the output-side norm/projection boundary. `fc1_latent_norm`,
whose consumer is the routed experts, keeps its existing path.

Eligibility requires Fast, B200, BF16 input/affine weights, latent width 1024,
a bias-free ReplicatedLinear, and the actual online block-FP8 DeepGEMM backend
with the CUDA quantizer enabled. Ignored/BF16 projections, Align, identity norms,
other formats and FP32 inputs retain their existing path. ReplicatedLinear has
no TP collective to bypass. The environment flag enters the compilation cache
key and must be selected before rebuilding the engine/graphs.

The producer matches CUDA quantization's exponent-bit scale ceiling, epsilon,
nonfinite-value handling, and initialization of holes in the packed scale
storage. The quantizer's NaN behavior is a compatibility requirement, not a
claim that valid model activations may safely contain NaNs.

## Validation and adoption decision

64 targeted checks pass. They cover FP8 bytes/scales, zero/tail/strided inputs, small/large
magnitudes, scale boundaries, nonfinite rows, scale-storage padding, graph replay
with changed inputs and affine weights, actual 1024 → 3072 GEMM results,
quantizer bypass, and fallback/default behavior. Forty captured eager inputs
from the real checkpoint also match the separate Norm + CUDA quantizer and the
audited compiled RMSNorm kernel. These local checks do not establish full-model
quality equivalence.

Full-model evaluation uses L3 `0000-28000-hf`, Fast FP8 projections including
latent, FA4 FP8 KV, the preceding attention fusion, and the same fixed target
text. Only the latent-norm fusion flag differs. All 20 candidate transforms are
enabled.

| Batch-1 quality measure | Result |
| --- | ---: |
| Fixed target positions | 1,024 |
| Baseline NLL | 1.446602718533466 |
| Candidate NLL | 1.4519632863898924 |
| NLL increase | 0.005360567856426357 |
| Sampled exp(NLL) increase | 0.5375% |
| Changed log-probability positions | 1,001 |
| Maximum absolute log-probability difference | 0.7986746 |

The final candidate reproduces the preceding candidate's per-position scores.
The first score difference appears at position 216; reused prefix state can
propagate an earlier difference into later scores. The remaining whole-model
cause is not established. The unrelated native-quantizer experiments were
removed: compiled YOCO block FP8 automatically enables `+quant_fp8` and rejects
its explicit disabling.

Controlled batch-8 tests use eight equal-length streams, 128 forced continuation
steps, and queue each complete group before resuming the scheduler. Two passes
measure baseline NLL 1.5011943 / 1.5010339 and candidate NLL 1.4955298 / 1.4954526.
These different batching/cache histories do not overturn the batch-1 regression
or establish a general quality gain.

Consequently the implementation is retained as an opt-in candidate. It is not
accepted as a default quality-preserving speedup. A future adoption decision
needs the remaining discrepancy understood and broader task/long-context tests.

## Timing

Same physical B200, sequential variants, atomic batch admission, 512 warm-prefix
tokens plus 128 generated tokens; five warmups and seven timed repetitions.

| Batch | Baseline output tok/s | Candidate output tok/s | Observed change |
| ---: | ---: | ---: | ---: |
| 1 | 158.03 | 159.77 | +1.10% |
| 8 | 810.74 | 818.67 | +0.98% |

Generated sequences differ in both batches. These measured improvements do not
establish a quality-preserving speedup, hence the default-off adoption decision.

The final single-token CUDA Graph operator measurement improves Norm + quantizer
from 2.61 to 1.60 µs, and the full Norm + quantizer + projection from 7.17 to
5.94 µs. All five tested row counts (1, 8, 32, 128, 1024) improve locally.

Machine-readable results and source hashes: [LATENT_NORM_FP8.json](LATENT_NORM_FP8.json).

Evidence directory:
`yoco_results/latent-norm-fp8-20260914-l1lzales` in the development workspace.
See also [residual and attention-output precision](RESIDUAL_ATTENTION_PRECISION.md).
