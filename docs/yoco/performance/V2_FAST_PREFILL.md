# YOCO V2 and Shared-KV fast prefill

YOCO ordinary inference, P/D KV transfer and Shared-KV drafting now use the V2
GPU model runner on the main configuration. The YOCO-specific V1 fallback
reasons have been removed. Select V2 explicitly when launching an engine:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve TARGET \
  --kv-sharing-fast-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

The [SWA8K launcher](../../../examples/offline_inference/spec_decode/serve_yoco_swa8k.py)
selects V2 and enables fast prefill by default, with bounded FULL decode capture.
Use its `--dry-run` option to inspect the command, or `--no-fast-prefill` for a
full-target comparison. Existing MoE YAML dispatch settings remain available.

## Hidden states for K8

Fast prefill runs the self decoder on all new tokens and writes the complete
shared KV. It gathers selected rows into the cross decoder's stable buffers.
The target now also returns the final **raw residual before final normalization**
in BF16 for Shared-KV drafting. For the current target, this stream is h20.

The returned auxiliary tensor retains the full token-row indexing expected by
the speculator. Only selected logits rows contain final-layer features; skipped
prompt rows must not be used as final-layer features. Shared-KV reads
`query_end - num_rejected - 1`, which is among the retained rows, including in
mixed prefill/verification batches and partial rejection.

The supported compact auxiliary configuration is exactly one final raw stream
for a YOCO Shared-KV DSpark draft. Arbitrary intermediate-layer extraction and
ordinary EAGLE/DSpark drafts retain their previous full-execution requirements.
Requests for prompt logprobs use complete prompt-row computation, so returned
prompt scores are not taken from skipped rows.

Uniform FULL decode reuses the ordinary target forward and its existing auxiliary
output buffers. No additional target/draft graph manager or workload-dependent
backend threshold is introduced. The validation profile with max-seqs=8 and K8
retains target `[9,18,36,72]` and draft `[8,16,24,32,40,64]` FULL graph sizes.

## P/D behavior

| Engine | Behavior |
| --- | --- |
| Dedicated `kv_producer` with fast prefill | Self decoder and full shared-KV production; cross decoder omitted on real and dummy DP ranks; no CUDA graph capture |
| `kv_consumer` | Restores KV through the existing V2 connector and computes the retained prompt tail, including the final raw hidden state required by K8 |
| `kv_both` | Uses the general forward path and transfer lifecycle |

Only real target KV owners are registered for transfer. Shared consumers remain
aliases of the same backing cache. Mooncake/NIXL keep their existing transport
protocols, page geometry and completion/freeing lifecycle.

NIXL's compatibility hash excludes a Shared-KV draft's configuration because it
does not add transferred KV or auxiliary state. Target model, dtype, backend,
transfer mode and runtime layout checks remain enforced. Ordinary hidden-state
drafts retain their existing speculative compatibility factors.

The final prompt block remains separate when prefix caching or P/D transfer is
configured. This preserves the existing local/remote tail replay policy. A cold
standalone engine with prefix caching disabled has no alternate replay shape
to match and does not pay for that extra prefill round.

## Precision and scope

Fast/FP8 execution can produce different KV and continuations from a local
recomputation even when transfer bytes are identical. The validation records
this separately from transport integrity; it does not promise bitwise model
equivalence or task-quality equivalence for Fast/FP8. BF16/Align supplies the
strict computation reference.

The optional `VLLM_YOCO_BF16_SAMPLING=1` experiment also has a V2 implementation.
It retains greedy-only requests, the SM90+ BF16 probability primitive, and the
restriction on custom `logprob_token_ids`. Generated, prompt and speculative
logprob paths use the same score policy. Keep this flag disabled for the usual
temperature-1 K8 configuration. Logit processors use the V2 processing path;
the primitive/ordinary greedy comparison is not a blanket V1 equivalence claim
for every processor combination.

Shared-KV's existing TP1/PP1/DP1/no-CP restriction is unchanged. Ordinary YOCO
DP fast-prefill synchronization includes inactive and dummy ranks; this change's
DP coverage is at the contract/unit level, not a multi-GPU model benchmark.
Other global V2 compatibility checks remain in effect.

Configuration, raw measurements, numerical observations and validation limits:
[integration report](v2-fast-prefill-20261021/REPORT.md).
