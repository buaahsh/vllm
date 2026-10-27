# Pure Prefill enablement and B200 validation

The pure-P eligibility guard now uses `max_num_batched_tokens`. Previously,
maxseq64 without speculation was rejected by the Decode capacity check, even
with a P token budget of 8192. A configured pure `kv_producer` can now use
Mega Gate by default. `kv_consumer`, `kv_both` and standalone engines retain
the existing Decode-capacity guard. All other compatibility checks and the
per-call threshold of 128 rows remain.

## Same-card throughput

Two-GPU Mooncake RDMA P/D, ABBA Gate off/on/on/off, C1/C8/C64, 3 warmups and
10 measured batches per concurrency per engine start. The on arm leaves the
Gate environment variable unset, verifying the default. Median of 20 formal
batches per arm:

| P concurrency | Off median tok/s | On median tok/s | Change |
| ---: | ---: | ---: | ---: |
| 1 | 13,115.3 | 12,867.2 | -1.89% |
| 8 | 87,620.1 | 86,495.4 | -1.28% |
| 64 | 87,751.4 | 88,021.1 | +0.31% |

C64's paired engine-start changes were +2.68% and −1.70%. There is no stable
P throughput gain demonstrated here. These are repeated fixed batches on a
shared node, with only two engine starts per arm. Aggregate tokens/time is:

| P concurrency | Off aggregate | On aggregate | Change |
| ---: | ---: | ---: | ---: |
| 1 | 13,072.3 | 13,064.5 | -0.06% |
| 8 | 87,806.5 | 86,764.0 | -1.19% |
| 64 | 87,628.1 | 88,154.6 | +0.60% |

The timer covers P admission through KV publication and GPU drain. Each frozen
1024-token prompt contributes 1008 actual P tokens; D transfer, the remaining
16 tokens, its one output token and cleanup are excluded. Both engines use
TP1, FP8-block weights, BF16 KV/48 GiB, BF16 router, FP32 residual/Norm,
maxlen16384, maxseq64, block16 and APC off; budgets P8192/D2048. P compilation
is enabled with actual CUDA Graph mode NONE. D has no speculation. CPU
placement is fixed to 28–55, P GPU2 and D GPU3 on the allocated B200 node.

## Execution and correctness checks

All 20 P router objects become eligible; the no-speculation D helper keeps its
20 routers disabled. A separate 128-row P profile records 30 native fused
kernel calls. The fast-prefill path executes 10 self-attention MoE modules
for three universal loops and skips the cross-decoder modules. A 16-row P
profile still executes BF16 GEMM and Triton Top-8.

Both short and threshold probes verify 31 nonzero KV-group SHA256 hashes
P→D. All 156 batches (120 formal) have exact scheduled P token counts, zero
extra P tokens, zero preemptions, complete D KV reuse and complete drain. No
captured JIT event occurs inside formal P timing. Local routing/config/MoE
tests: 121 passed. Native B200 Mega Gate tests: 12 passed. Scoped pre-commit
checks passed. Source, model metadata, native extension and Pod identity
remain unchanged during the run, and both test GPUs are released.

At C64, 53/64 first-token IDs agree across arms; 11 differ. Each arm is
repeatable on this frozen workload. This does not establish task accuracy
or full output equivalence. D steady-state throughput was not remeasured.

The first baseline's untimed 1008-row probe sampled evicted sliding-window
pages. Its zero-page matches are excluded from live-KV evidence. Subsequent
probes use 128 P rows. Only the untimed probe input changed; formal worker
and timing code have identical ASTs. No formal sample was discarded.

Raw reports, configuration snapshots, profiles, checksums, source manifest,
validation logs and retained setup attempts are under:
`/data/vllm_test/work/yoco-mega-gate-prefill-enable-20261026/`.
The candidate changes only `yoco_moe.py` and its eligibility tests relative to
the prior frozen Mega Gate v2 source. The native extension is unchanged.
