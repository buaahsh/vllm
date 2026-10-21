# Shared-KV draft paged attention

Shared-KV SWA8192 directly reuses the existing paged FlashAttention wrapper
when its selected version is FA4. There is no batch/context threshold, extra
graph manager, graph bank, or added YAML dispatch option. Backend capability
selection is fixed for the engine; other FA versions retain the original
Triton implementation.

The integration keeps the existing DSpark/DFlash scheduler, four-layer draft,
local causal attention, sequential Markov sampling and standard rejection
sampling. It preserves the trained global-attention interval
`[max(0, anchor-8192), anchor)` for every draft slot, reads target pages directly,
and does not gather/materialize history or modify KV.

The adapter folds draft slots into each KV head's GQA group and uses one query
position with `causal=False`, `seqused_k=anchor`, and `window_size=[8191,-1]`.
It reverses the Q/output permutation after attention. Q/KV/output stay BF16;
trained residuals, gates and probability logits keep their existing precision.

Use the normal Shared-KV launch configuration; no new runtime option is needed.
The runtime must provide the repository's pinned FA4 CuteDSL Python component
(`_vllm_fa4_cutedsl_C` in `cmake/external_projects/vllm_flash_attn.cmake`) and its
compatible dependencies. This branch does not vendor or upgrade those packages.

The [integration report](dspark-attention-integration-20261021/REPORT.md)
records validation with the existing Mega integration and the unchanged graph
specifications. The archived
[A/B/B/A experiment](dspark-attention-integration-20261021/RESULTS.md) and
[native draft timing](dspark-attention-integration-20261021/NATIVE_DRAFT_PROFILE.md)
retain the original positive and negative results: short histories regress,
while the measured 8K draft step improves about 1.23–1.33 times. Those are
separate workload observations, not a universal end-to-end gain.

FA4 can round BF16 reductions differently and change draft proposals and
acceptance. The [fixed-prefix numerical note](dspark-attention-integration-20261021/QUALITY_BATCHING_NOTE.md)
also explains why target logprobs must be compared with matching target batches.
This change does not provide batch invariance or establish task-quality parity.
