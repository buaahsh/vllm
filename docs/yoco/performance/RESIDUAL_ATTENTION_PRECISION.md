# YOCO Fast residual and attention-output precision

By default, Fast inference stores the main residual in FP32 and FA4 output in BF16.
The subsequent [BF16 residual A/B](BF16_RESIDUAL.md) implements an opt-in Fast
experiment and records full-model timing and precision results.
FP32 residual storage does not mean the model's large GEMMs use FP32 operands.
The residual receives approximately 80 attention/MoE updates over 40 logical
blocks. Keeping the state in FP32 preserves small updates across those additions.

FP32 storage is a choice to validate, not a universal requirement. A 16-bit
residual with FP32 accumulation and normalization statistics is a reasonable
independent experiment. FP16 offers three more fraction bits than BF16, at the
cost of a narrower range; both use two bytes. FP8 has substantially coarser
precision, even with per-group scaling.

The audited DeepSeek V4 reference also separates stored state from high-precision
mixing: `Block.hc_post` computes the Hyper-Connection combination and returns
`y.type_as(x)`, which is BF16 on its normal inference path. Its HC architecture
and training recipe differ from YOCO, so this is evidence that FP32 storage is
not universal, rather than evidence that YOCO can adopt it without validation.
The local llm-train YOCO reference explicitly uses FP32 residual additions.

## Real-activation local replay

The isolated B200 probe loaded the real L3 checkpoint and ran eager Fast with
FP8 projections, FA4 FP8 KV and the preceding attention fusion. It sampled
residual-update inputs and attention outputs, then replayed lower-precision
rounding locally without feeding those changes back into the model. Requested
contexts were 128, 1024 and 4096 tokens; the longest repeats the fixed text.

Only single-row calls are used below: 160 residual-update rows and 80 attention
rows. Multirow samples are excluded because they may contain padded rows.
These are correlated layer/token observations, not independent task examples.

For each residual update, the reference is `r + x` in FP32. A trial rounds `r`
to the candidate storage format, adds `x` in FP32, and rounds the result again.
An update is counted as swallowed when it is nonzero but the stored state is
unchanged. FP8 uses dynamic group-128 power-of-two scaling.

| Residual storage | Result relative L2 error | Error / branch-update L2 | Nonzero component updates swallowed |
| --- | ---: | ---: | ---: |
| BF16 | 0.2335% | 0.7720% | 0.6869% |
| FP16 | 0.02781% | 0.09194% | 0.08016% |
| FP8 E4M3 | 3.6356% | 12.0177% | 10.5343% |

The sampled residual absolute maximum is 81.925; none of these local conversions
overflowed. This supports evaluating FP16 as well as BF16 residual storage, but
does not bound future outliers or predict errors accumulated through the full
model. BF16/FP16 residual deployment still needs full-model NLL and long-context
validation. The main residual has not been changed by this task.

For hidden width 3072, one residual row is 12 KiB in FP32 and 6 KiB in 16-bit.
Across 80 updates, halving one state read and one state write saves about 960 KiB
per token. Actual fused Norm traffic and kernel layout also matter. Decode
speedup cannot be inferred from this byte reduction alone.

## FA4 output and differential attention

FP16 and BF16 both use two bytes, so switching between them has no inherent
storage-bandwidth benefit. The current FA4 FP8 integration uses BF16 output.

Quantizing each 128-element attention head to E4M3 before differential attention
introduces a new rounding boundary before `sigmoid(g1) * A1 - sigmoid(g2) * A2`.
The local replay uses dynamic per-head scales and retains FP32 gate/subtraction
arithmetic. It compares against the same arithmetic on current BF16 outputs.

| Quantity | Relative L2 difference |
| --- | ---: |
| Attention output after an extra FP8 round | 2.6464% |
| After differential combine | 5.0848% |
| After the existing FP8 quantization for O projection | 6.2948% |

Per-head differential-output relative errors have median 6.13%, P95 15.97% and
P99 26.94%. Cancellation amplifies the initial error. These tensor errors are not
NLL changes or task-error rates. The sampled attention absolute maximum is 9,
and conversion of the current BF16 values to FP16 produced no nonfinite values;
that conversion does not recover precision already rounded away in BF16.

The preferred engineering direction is to fuse FA4's output processing with
high-precision differential combine and quantize afterward. The current change
already fuses differential combine with O-input quantization; FA4 still writes
a BF16 output tensor. Fusing its epilogue would be a separate kernel integration,
not simply changing the output dtype.

Machine-readable measurements: [RESIDUAL_ATTENTION_PRECISION.json](RESIDUAL_ATTENTION_PRECISION.json).

Raw evidence is retained in
`yoco_results/latent-norm-fp8-20260914-l1lzales/PRECISION_PROBE.json` and the
corresponding probe script. DeepSeek sources are the audited snapshot in
`yoco_results/dsv4-fp8-policy-20260914/deepseek-v4-flash/inference/`.
