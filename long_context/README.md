# YOCO-v2 B200 long-context BF16 results

```text
Git branch: shaohanh/yoco-b200-longctx-multigpu-20260804
Docker image: buaahsh/pytorch:26.02-b200-vllm-yoco-longctx-multigpu-20260804
Precision: BF16 only
```

This report covers the three production-shaped workloads below. All reported
serving runs use BF16, tensor parallelism 1, data-parallel ranks, FlashInfer
attention, Triton MoE, prefix caching, chunked prefill, YOCO
KV-sharing fast prefill, and a 32K batched-prefill-token budget. The observed
local fused-MoE width is N1280 at DP1, N320 at DP4, and N160 at DP8; these
deployments must not be interpreted as isolated full-model replicas. The
cold-start logs corroborate this: DP1 loads about 60.62 GiB on its worker,
whereas each DP4 worker loads about 18.43 GiB.

| Workload | Shape | Primary pressure |
| --- | --- | --- |
| W1 | 8,192 input + 65,536 output | Long decode |
| W2 | 65,536 input + 16,384 output | Long prefill and decode |
| W3 | 40 turns, 117K incremental input + 13K output | Prefix reuse over a 130K final trajectory |

W3 stops at 130K because 131K input plus 13K output would exceed the model's
hard 131,072-token context limit.

## Decision summary

| Probe | Controlled evidence | Decision |
| --- | --- | --- |
| Attention backend | FlashInfer TPOT stayed near 8 ms from 8K through 96K context; FA4 reached 16.65 ms at 96K | Use FlashInfer |
| Prefill budget | 32K reduced 64K/96K cold TTFT by 9.6-10.7% versus 8K | Use 32K for these latency-oriented workloads |
| Agentic prefix reuse | The DP1 trace hit 95.58% of cumulative prompt tokens in prefix cache | Keep prefix caching and YOCO KV-sharing fast prefill |
| Scheduler / DP backend | On DP8 W2 batch 8, async/Gloo delivered 741.81 output tok/s versus 491.66 for sync/NCCL | Keep async scheduling; its 50.9% higher output throughput outweighed NCCL collectives |
| DP1 saturation | W2 batch 24 added 9.6% output tok/s over 16 with 36.5% higher TPOT; W3 gained 7.6% with 44.4% higher ITL | Use batch 16 as the DP1 knee |
| DP4 long-prefill batch | W2 batch 96 added 9.6% output tok/s over 64 while TPOT rose 36.7% | Use batch 64 as the DP4 W2 knee |
| DP4 agentic batch | Batch 96 added 9.6% output tok/s over 64, while mean ITL rose 36.7% | Use batch 64 as the W3 knee; 96 is max-throughput only |
| DP8 MoE N160 tuning | Isolated kernels improved by up to 23.3%, but two clean W2 batch-8 runs fell to 652.13 and 661.85 output tok/s from 741.81 | Reject the generated N160 table; keep the runtime fallback |
| DP1 MoE N1280 tuning | Same-GPU kernels improved by up to 28.5%, including the important 8K/32K prefill sizes | Retain the validated N1280 table for one-GPU serving |
| Nested Docker startup | Triton failed with `undefined symbol: cuModuleGetFunction` because the CUDA compatibility driver was not globally visible | Map the GPU/NVML devices and preload the image's CUDA compatibility `libcuda.so.1` |
| Warmup | Cold, late long-shape compilation created latency outliers | Warm the complete 130K trajectory before production traffic |

The scheduler A/B rows use the same image, node, checkpoint, eight B200s,
workload, prompt length, output length, and batch. The MoE end-to-end A/B uses
unique cache-salt namespaces; a faster repeat that reused cached prompts is
excluded.

## Scheduler and DP8 MoE A/B

W2, batch 8, eight B200s:

| Mode | Total time (s) | Output tok/s | Total tok/s | Mean TTFT (s) | Mean TPOT (ms) | Relative output rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Async scheduler, Gloo, runtime N160 fallback | 176.693 | 741.81 | 3,709.02 | 2.751 | 10.61 | 1.000x |
| Sync scheduler, NCCL, runtime N160 fallback | 266.592 | 491.66 | 2,458.29 | 3.188 | 16.07 | 0.663x |
| Async scheduler, generated N160, clean run 1 | 200.991 | 652.13 | 3,260.64 | 2.474 | 12.11 | 0.879x |
| Async scheduler, generated N160, clean run 2 | 198.038 | 661.85 | 3,309.26 | 2.287 | 11.95 | 0.892x |

The generated N160 table looked promising in isolation:

| MoE token batch | Runtime fallback | Generated N160 | Kernel improvement |
| ---: | ---: | ---: | ---: |
| 1 | 36.80 us | 36.81 us | neutral |
| 2 | 42.83 us | 41.99 us | 2.0% |
| 4 | 56.80 us | 51.16 us | 9.9% |
| 8 | 78.58 us | 70.34 us | 10.5% |
| 16 | 102.25 us | 86.79 us | 15.1% |
| 32 | 133.82 us | 102.63 us | 23.3% |
| 128 | 129.00 us | 113.09 us | 12.3% |
| 1,024 | 209.74 us | 196.85 us | 6.1% |
| 2,843 | 352.39 us | 329.74 us | 6.4% |
| 3,899 | 407.97 us | 405.39 us | 0.6% |
| 8,192 | 765.97 us | 749.42 us | 2.2% |
| 32,768 | 2,701.03 us | 2,584.36 us | 4.3% |

This mismatch is why kernel-only results are not used as the production gate.

## Scaled end-to-end results

Total time is batch wall time. Throughput is aggregate across the selected
GPU count. Every single-turn row uses a unique server-side cache salt, so
prompts from earlier rows cannot make TTFT artificially fast.

Every published DP1, DP4, and DP8 row was also measured with that deployment
alone on the node. An early diagnostic accidentally overlapped DP1 work on
GPU 7 with DP4 work on GPUs 0-3. Although the GPU sets did not overlap, host
and collective contention changed throughput; those rows were moved under
`co-located-invalid/` in the raw evidence tree and are excluded below.

### DP1 / one B200

W1 - 8K input, 64K output:

| Mode | GPUs | Batch | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 1 | 533.969 | 0.0019 | 15.34 | 122.73 | 138.08 | 0.182 | 0.182 | 8.15 |
| BF16 | 1 | 2 | 585.329 | 0.0034 | 27.99 | 223.93 | 251.92 | 0.317 | 0.337 | 8.93 |
| BF16 | 1 | 4 | 754.467 | 0.0053 | 43.43 | 347.46 | 390.89 | 0.663 | 0.705 | 11.50 |
| BF16 | 1 | 8 | 876.785 | 0.0091 | 74.75 | 597.97 | 672.71 | 0.969 | 1.145 | 13.36 |
| BF16 | 1 | 12 | 1,160.621 | 0.0103 | 84.70 | 677.60 | 762.30 | 1.320 | 1.624 | 17.69 |
| BF16 | 1 | 16 | 1,160.444 | 0.0138 | 112.95 | 903.60 | 1,016.55 | 1.653 | 2.125 | 17.68 |

The batch-1/2 rows are prior-allocation references; batches 4-16 are dedicated
new-node measurements. Batch 16 is the W1 knee and maximum tested fresh batch.

W2 - 64K input, 16K output:

| Mode | GPUs | Batch | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 1 | 139.610 | 0.0072 | 469.42 | 117.36 | 586.78 | 1.182 | 1.182 | 8.45 |
| BF16 | 1 | 2 | 165.135 | 0.0121 | 793.73 | 198.43 | 992.16 | 2.154 | 2.386 | 9.95 |
| BF16 | 1 | 4 | 202.558 | 0.0197 | 1,294.17 | 323.54 | 1,617.71 | 2.971 | 3.916 | 12.18 |
| BF16 | 1 | 8 | 245.674 | 0.0326 | 2,134.08 | 533.52 | 2,667.60 | 5.197 | 8.261 | 14.67 |
| BF16 | 1 | 12 | 333.996 | 0.0359 | 2,354.62 | 588.65 | 2,943.27 | 7.656 | 12.351 | 19.90 |
| BF16 | 1 | 16 | 353.965 | 0.0452 | 2,962.37 | 740.59 | 3,702.96 | 9.991 | 16.190 | 20.97 |
| BF16 | 1 | 24 | 484.646 | 0.0495 | 3,245.39 | 811.35 | 4,056.74 | 14.967 | 24.886 | 28.62 |

Batch 16 is the DP1 W2 knee. Batch 24 adds 9.6% output throughput while mean
TPOT rises 36.5% and it exceeds the full-context KV-capacity estimate.

W3 - 40-turn, 130K agentic trajectory. Input and total rates count logical
incremental tokens, not repeatedly submitted cached prefixes.

| Mode | GPUs | Batch | Total time (s) | Traj/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean ITL (ms) | Cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 1 | 114.660 | 0.0087 | 1,020.41 | 113.38 | 1,133.79 | 0.168 | 0.227 | 8.32 | 95.58% |
| BF16 | 1 | 2 | 137.597 | 0.0145 | 1,700.62 | 188.96 | 1,889.58 | 0.285 | 0.376 | 9.73 | 95.58% |
| BF16 | 1 | 4 | 171.704 | 0.0233 | 2,725.62 | 302.85 | 3,028.46 | 0.406 | 0.574 | 11.99 | 95.58% |
| BF16 | 1 | 8 | 231.395 | 0.0346 | 4,045.04 | 449.45 | 4,494.49 | 0.770 | 1.196 | 15.42 | 95.58% |
| BF16 | 1 | 12 | 305.287 | 0.0393 | 4,598.96 | 511.00 | 5,109.95 | 0.986 | 1.416 | 20.46 | 95.58% |
| BF16 | 1 | 16 | 323.922 | 0.0494 | 5,779.17 | 642.13 | 6,421.30 | 1.294 | 2.223 | 20.97 | 95.58% |
| BF16 | 1 | 24 | 451.717 | 0.0531 | 6,216.28 | 690.70 | 6,906.98 | 1.462 | 2.307 | 30.29 | 95.58% |

Batch 16 is the DP1 W3 knee. Batch 24 adds only 7.6% output throughput while
mean ITL rises 44.4%, and it exceeds the full-context KV-capacity estimate.

### DP4 / four B200s

W1 - 8K input, 64K output:

| Mode | GPUs | Batch | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 4 | 4 | 679.565 | 0.0059 | 48.22 | 385.75 | 433.97 | 0.454 | 0.624 | 10.36 |
| BF16 | 4 | 16 | 712.713 | 0.0224 | 183.91 | 1,471.25 | 1,655.15 | 0.769 | 0.958 | 10.86 |
| BF16 | 4 | 32 | 803.700 | 0.0398 | 326.17 | 2,609.37 | 2,935.54 | 1.360 | 1.856 | 12.24 |
| BF16 | 4 | 64 | 1,127.065 | 0.0568 | 465.18 | 3,721.44 | 4,186.62 | 2.069 | 3.010 | 17.16 |
| BF16 | 4 | 96 | 1,300.902 | 0.0738 | 604.53 | 4,836.23 | 5,440.76 | 3.072 | 5.000 | 19.80 |

Use batch 64 as the DP4 W1 latency/throughput knee. Batch 96 adds 29.9%
output throughput, but mean TTFT rises 48.5% and mean TPOT rises 15.4%.

W2 - 64K input, 16K output:

| Mode | GPUs | Batch | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 4 | 4 | 155.801 | 0.0257 | 1,682.56 | 420.64 | 2,103.20 | 1.801 | 1.946 | 9.40 |
| BF16 | 4 | 8 | 170.285 | 0.0470 | 3,078.89 | 769.72 | 3,848.62 | 2.881 | 3.657 | 10.22 |
| BF16 | 4 | 16 | 223.182 | 0.0717 | 4,698.30 | 1,174.57 | 5,872.87 | 4.431 | 6.457 | 13.35 |
| BF16 | 4 | 32 | 230.817 | 0.1386 | 9,085.77 | 2,271.44 | 11,357.21 | 7.447 | 11.924 | 13.63 |
| BF16 | 4 | 48 | 290.934 | 0.1650 | 10,812.50 | 2,703.12 | 13,515.62 | 10.360 | 17.269 | 17.11 |
| BF16 | 4 | 64 | 318.169 | 0.2012 | 13,182.63 | 3,295.66 | 16,478.29 | 13.767 | 23.307 | 18.55 |
| BF16 | 4 | 96 | 435.367 | 0.2205 | 14,450.94 | 3,612.73 | 18,063.67 | 19.092 | 33.685 | 25.37 |

Use batch 64 as the DP4 W2 knee. Batch 96 adds 9.6% output throughput while
mean TPOT rises 36.7%.

W3 - 40-turn, 130K agentic trajectory:

| Mode | GPUs | Batch | Total time (s) | Traj/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean ITL (ms) | Cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 4 | 4 | 131.706 | 0.0304 | 3,553.37 | 394.82 | 3,948.19 | 0.267 | 0.393 | 9.33 | 95.58% |
| BF16 | 4 | 8 | 148.893 | 0.0537 | 6,286.41 | 698.49 | 6,984.90 | 0.374 | 0.573 | 10.32 | 95.58% |
| BF16 | 4 | 16 | 178.317 | 0.0897 | 10,498.16 | 1,166.46 | 11,664.62 | 0.563 | 0.832 | 11.99 | 95.58% |
| BF16 | 4 | 32 | 230.255 | 0.1390 | 16,260.26 | 1,806.70 | 18,066.96 | 0.762 | 1.272 | 15.34 | 95.58% |
| BF16 | 4 | 48 | 286.678 | 0.1674 | 19,589.95 | 2,176.66 | 21,766.61 | 1.283 | 2.190 | 18.11 | 95.58% |
| BF16 | 4 | 64 | 323.914 | 0.1976 | 23,117.22 | 2,568.58 | 25,685.80 | 1.097 | 2.174 | 21.51 | 95.58% |
| BF16 | 4 | 96 | 443.509 | 0.2165 | 25,325.32 | 2,813.92 | 28,139.24 | 1.512 | 2.506 | 29.40 | 95.58% |

Batch 64 is the agentic knee. Batch 96 adds only 9.6% output throughput while
mean ITL rises 36.7%; use it only when aggregate throughput is more important
than inter-token latency.

### DP8 / one eight-B200 node

W1 - 8K input, 64K output:

| Mode | GPUs | Batch | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 8 | 8 | 692.102 | 0.0116 | 94.69 | 757.53 | 852.22 | 0.533 | 0.678 | 10.55 |
| BF16 | 8 | 32 | 897.066 | 0.0357 | 292.22 | 2,337.79 | 2,630.01 | 1.129 | 1.563 | 13.67 |
| BF16 | 8 | 64 | 1,008.090 | 0.0635 | 520.08 | 4,160.64 | 4,680.72 | 2.075 | 2.903 | 15.35 |
| BF16 | 8 | 128 | 1,191.112 | 0.1075 | 880.33 | 7,042.67 | 7,923.00 | 2.977 | 4.507 | 18.13 |
| BF16 | 8 | 192 | 1,512.690 | 0.1269 | 1,039.78 | 8,318.24 | 9,358.02 | 4.601 | 7.034 | 23.01 |

W2 - 64K input, 16K output:

| Mode | GPUs | Batch | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 8 | 8 | 176.693 | 0.0453 | 2,967.22 | 741.81 | 3,709.02 | 2.751 | 3.101 | 10.61 |
| BF16 | 8 | 16 | 196.744 | 0.0813 | 5,329.65 | 1,332.41 | 6,662.07 | 3.931 | 5.567 | 11.75 |
| BF16 | 8 | 32 | 219.654 | 0.1457 | 9,547.51 | 2,386.88 | 11,934.39 | 6.569 | 9.551 | 13.00 |
| BF16 | 8 | 64 | 264.067 | 0.2424 | 15,883.47 | 3,970.87 | 19,854.33 | 10.710 | 17.902 | 15.45 |
| BF16 | 8 | 96 | 319.822 | 0.3002 | 19,671.74 | 4,917.93 | 24,589.67 | 16.345 | 27.150 | 18.50 |
| BF16 | 8 | 128 | 361.332 | 0.3542 | 23,215.77 | 5,803.94 | 29,019.71 | 18.989 | 33.320 | 20.87 |
| BF16 | 8 | 192 | 469.213 | 0.4092 | 26,817.07 | 6,704.27 | 33,521.34 | 27.362 | 49.229 | 26.93 |

W3 - 40-turn, 130K agentic trajectory:

| Mode | GPUs | Batch | Total time (s) | Traj/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean ITL (ms) | Cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 8 | 8 | 152.385 | 0.0525 | 6,142.32 | 682.48 | 6,824.80 | 0.331 | 0.588 | 10.73 | 95.58% |
| BF16 | 8 | 16 | 200.419 | 0.0798 | 9,340.42 | 1,037.82 | 10,378.24 | 0.513 | 0.841 | 13.86 | 95.58% |
| BF16 | 8 | 32 | 243.078 | 0.1316 | 15,402.47 | 1,711.39 | 17,113.85 | 0.878 | 1.598 | 15.99 | 95.58% |
| BF16 | 8 | 64 | 271.996 | 0.2353 | 27,529.77 | 3,058.86 | 30,588.63 | 1.207 | 2.172 | 17.17 | 95.58% |
| BF16 | 8 | 96 | 342.158 | 0.2806 | 32,826.91 | 3,647.43 | 36,474.34 | 1.438 | 2.959 | 21.78 | 95.58% |
| BF16 | 8 | 128 | 385.077 | 0.3324 | 38,890.95 | 4,321.22 | 43,212.16 | 1.881 | 3.983 | 23.67 | 95.58% |
| BF16 | 8 | 192 | 520.844 | 0.3686 | 43,129.99 | 4,792.22 | 47,922.22 | 2.324 | 5.439 | 32.63 | 95.58% |

For W1, batch 192 raises output throughput 18.1% over batch 128, but mean TTFT
rises 54.5% and mean TPOT rises 26.9%. Use 128 as the latency/throughput knee;
reserve 192 for maximum-throughput traffic that can accept the extra latency.
For W2 and W3, use batch 128 as the practical knee; batch 192 is the measured
maximum-throughput point when higher TTFT/TPOT or ITL is acceptable.

## Reproduction

The authoritative launch, warmup, and benchmark commands are in
[`yoco.md`](../yoco.md). The benchmark runner supports selecting workloads,
resuming completed JSON files, and assigning an explicit cache namespace:

```bash
docker exec \
  -e TOKENIZER="$MODEL" \
  -e DP_SIZE=8 \
  -e GPU_INDICES=0,1,2,3,4,5,6,7 \
  -e WORKLOADS="1 2 3" \
  -e BATCHES="8 16 32 64 96 128 192" \
  -e RUN_ID=dp8-production-20260804 \
  -e RESULT_DIR=/results/dp8 \
  yoco-longctx-dp8 \
  bash tools/yoco_serving/benchmark_long_context.sh
```

Set a new `RUN_ID` for a clean repeat. `SKIP_EXISTING=1` resumes only valid
JSON results tagged with the same run identity; set it to 0 to overwrite
intentionally.

The service sets `MAX_NUM_SEQS=128` per DP engine. DP requests are distributed
across engines, so DP8 batch 192 has about 24 requests per rank and can run all
192 concurrently. The KV-block budget remains the effective limit for the
highest-context steps; it can queue work before the sequence-count cap. At the
documented 85% GPU-memory utilization, startup reports capacity for 19.43 full
131,072-token sequences per DP1 engine and 26.69 per DP4 engine. Consequently,
DP1 batch 24 intentionally probes beyond full-context KV capacity, while DP4
batch 96 remains below its roughly 107-sequence deployment-wide capacity.

## Test environment and raw evidence

- Node: `assuring-owl-b200g4-dev-d5aab19e-master-0`
- GPUs: eight NVIDIA B200s
- Precision: BF16 only
- Local checkpoint: `/data/models/yoco-0000-0800-hf`
- Persistent disk: `/dev/md1`, ext4, mounted at `/data`
- Docker data root: `/data/docker`, using `fuse-overlayfs`

Detailed JSON, server logs, tuning logs, and telemetry are retained under
`/data/yoco-longctx-results-20260804` on the node and under
`/data/shaohanh/msrallm/b200/long_context/raw/scaled-new-node` in the benchmark
workspace.
