# YOCO-v2 B200 long-context probing — presentation tables

```text
Node: assuring-owl-b200g4-dev-d5aab19e-master-0
Revision: a9cd5c20724c994be085058ca26187b2353d4ac5
Image: buaahsh/pytorch:26.02-b200-vllm-yoco-20260805
Precision: BF16 only
Production mode: FlashInfer, Triton MoE, async scheduling, 32K prefill budget
```

All serving rows use TP=1, exact forced output lengths, EOS ignored, and unique
server-side cache-salt namespaces. Only one deployment was active during every
published row. `FI-32K` means FlashInfer with
`max_num_batched_tokens=32768`; `32K` is not the request batch size.

## 1. Final one-B200 workload speed

These are fresh, post-full-warmup rows from the final hybrid N1280 image.

| Mode | Workload | Batch | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT/ITL (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 FI-32K | W1: 8K in / 64K out | 1 | 536.267 | 0.0019 | 15.28 | **122.21** | 137.48 | 0.182 | 0.182 | **8.18** |
| BF16 FI-32K | W2: 64K in / 16K out | 1 | 137.978 | 0.0072 | 474.98 | **118.74** | 593.72 | 1.227 | 1.227 | **8.35** |
| BF16 FI-32K | W3: 40-turn 130K agent trace | 1 | 113.266 | 0.353† | 1,032.96‡ | **114.77** | 1,147.74‡ | 0.166 | 0.224 | **8.22** |

† Agentic Req/s counts completed turns. ‡ Agentic input and total throughput
count logical incremental prefill, not repeatedly submitted cumulative
prefixes.

## 2. Scaled throughput knees

This is the compact decision view of the full batch sweeps in `README.md`.
DP4 and DP8 use local MoE widths N320 and N160, so the final DP1/N1280 hybrid
change does not affect them. The DP1 knee rows are the dedicated new-node
scaled sweep; batch 1 above is the final hybrid gate.

| Deployment | Workload | Selected batch | Selected output tok/s | Mean TTFT (s) | Mean TPOT/ITL (ms) | Max-throughput batch | Max-throughput output tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 B200 | W1 | 16 | 1,001.13 | 1.597 | 15.96 | 16 | 1,001.13 |
| 1 B200 | W2 | 16 | 807.51 | 9.148 | 19.23 | 24 | 871.01 |
| 1 B200 | W3 | 16 | 707.80 | 1.031 | 19.47 | 24 | 741.01 |
| 4 B200 | W1 | 64 | 4,504.14 | 1.982 | 14.18 | 96 | 5,076.77 |
| 4 B200 | W2 | 64 | 3,550.13 | 12.209 | 17.26 | 96 | 3,800.12 |
| 4 B200 | W3 | 64 | 2,797.76 | 1.399 | 18.55 | 96 | 3,017.81 |
| 8 B200 | W1 | 192 | 9,908.14 | 4.150 | 19.31 | 192 | 9,908.14 |
| 8 B200 | W2 | 128 | 6,590.00 | 18.015 | 18.30 | 192 | 7,202.03 |
| 8 B200 | W3 | 128 | 4,633.41 | 1.407 | 23.15 | 192 | 4,970.24 |

The Router-cache update improves selected-knee output rate by 10.8%, 9.0%,
and 10.2% on DP1 W1/W2/W3, and by 21.0%, 7.7%, and 8.9% on DP4 W1/W2/W3.
The refreshed max-throughput probes likewise use measured end-to-end results
rather than extrapolation from the Router microbenchmark. At DP8 batch 128,
W1 falls 4.3% while W2 and W3 improve by 13.5% and 7.2%. Batch 192 improves
over the prior maximum by 19.1%, 7.4%, and 3.7% on W1/W2/W3. W1 moves its knee
to batch 192: output rate rises 47.0% over batch 128 for only 2.0% higher TPOT.

## 3. vLLM versus llm-train KL validation

vLLM greedily generated one complete 24-token sentence, retaining all 154,880
log-probabilities at every step. The same prompt plus generated prefix was then
scored by the BF16 llm-train `YOCO-MoE-30B-A3B-v2` implementation after loading
all 266 exported parameters. The checkpoint's `universal_loop=3` is applied;
using the preset dataclass default of one would compare a different model.

> “The calm ocean at sunrise was a serene sight, with the gentle waves
> reflecting the soft hues of the morning sky.”

| Tokens | Mean KL vLLM→train | Max KL vLLM→train | Mean KL train→vLLM | Mean JS | Max generated-token logprob delta | Top-1 match |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | **0.003206** | **0.013188** | **0.003210** | **0.000800** | **0.076111** | **100%** |

The small non-zero divergence is consistent with different optimized BF16
execution paths; greedy token selection agrees at every compared step.

## 4. Attention backend probe

Each row is one request with the listed context and exactly 256 output tokens.
The backend A/B used the same pre-hybrid N1280 table in both modes; the final
hybrid makes the common M=1-8 MoE path fallback-identical and therefore does
not change the relative attention decision.

| Mode | Context | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 FI-32K | 8K | **2.242** | **0.446** | **3,653.23** | **114.16** | **3,767.40** | **0.202** | **0.202** | **8.00** |
| BF16 FA4-32K | 8K | 2.523 | 0.396 | 3,247.54 | 101.49 | 3,349.02 | 0.202 | 0.202 | 9.10 |
| BF16 FI-32K | 32K | **2.664** | **0.375** | **12,300.83** | **96.10** | **12,396.93** | 0.596 | 0.596 | **8.11** |
| BF16 FA4-32K | 32K | 3.493 | 0.286 | 9,380.88 | 73.29 | 9,454.17 | **0.596** | **0.596** | 11.36 |
| BF16 FI-32K | 64K | **3.241** | **0.309** | **20,222.33** | **78.99** | **20,301.32** | **1.138** | **1.138** | **8.25** |
| BF16 FA4-32K | 64K | 4.818 | 0.208 | 13,601.46 | 53.13 | 13,654.59 | 1.150 | 1.150 | 14.38 |
| BF16 FI-32K | 96K | **3.861** | **0.259** | **25,460.12** | **66.30** | **25,526.42** | **1.736** | **1.736** | **8.33** |
| BF16 FA4-32K | 96K | 6.232 | 0.160 | 15,773.42 | 41.08 | 15,814.50 | 1.791 | 1.791 | 17.42 |

FA4 decode TPOT is 13.7%, 40.1%, 74.4%, and 109.1% slower than FlashInfer at
8K, 32K, 64K, and 96K respectively. Use FlashInfer.

## 5. Prefill token-budget probe

Attention is FlashInfer and output length is 256 in every row.

| Mode | Context | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 FI-8K | 64K | 3.349 | 0.299 | 19,567.53 | 76.44 | 19,643.97 | 1.316 | 1.316 | **7.97** |
| BF16 FI-32K | 64K | **3.241** | **0.309** | **20,222.33** | **78.99** | **20,301.32** | **1.138** | **1.138** | 8.25 |
| BF16 FI-8K | 96K | 4.085 | 0.245 | 24,067.05 | 62.67 | 24,129.72 | 2.002 | 2.002 | **8.17** |
| BF16 FI-32K | 96K | **3.861** | **0.259** | **25,460.12** | **66.30** | **25,526.42** | **1.736** | **1.736** | 8.33 |

The 32K budget lowers TTFT by 13.6% at 64K and 13.3% at 96K. Keep 32K for
these latency-oriented workloads; consider 8K only when high-concurrency
prefill fairness matters more than single-request latency.

## 6. Agentic prefix-cache and warmup probe

Trace: 40 turns, average 2,925 new input tokens plus 325 output tokens per
turn, ending at 130K tokens.

| Mode | Submitted prompt | Logical new input | Computed prefill | Cached prefill | Cache hit | Prefill service tok/s | Generation service tok/s | Mean GPU util. | Waiting max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 FI-32K | 2,651,403 | 117,000 | 117,195 | 2,534,208 | **95.58%** | **29,925.57** | **120.26** | **94.85%** | **0** |

| Agent state | Wall time (s) | Output tok/s | Mean TTFT (s) | P95 TTFT (s) | Max TTFT (s) | Mean ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before full-trajectory warmup | 115.413 | 112.64 | 0.205 | 0.225 | 1.991 | 8.27 |
| After full-trajectory warmup | **113.266** | **114.77** | **0.166** | **0.224** | **0.231** | **8.22** |

Full warmup removes the late-JIT tail: maximum TTFT falls by 88.4%.

## 7. DP1 hybrid N1280 MoE probe

Exact BF16 shape: E=128, N=1,280, K=3,072, top-k=8. M=1/2/4/8 use the
runtime fallback configuration verbatim, so they are neutral by construction.

| MoE token batch | Runtime fallback | Hybrid image | Kernel improvement |
| ---: | ---: | ---: | ---: |
| 1 / 2 / 4 / 8 | same config | same config | neutral by construction |
| 16 | 353.74 us | 340.50 us | 3.7% |
| 32 | 475.67 us | 449.06 us | 5.6% |
| 128 | 541.20 us | 503.55 us | 7.0% |
| 1,024 | 683.01 us | 659.41 us | 3.5% |
| 2,843 | 1,166.33 us | 938.16 us | 19.6% |
| 3,899 | 1,373.41 us | 1,124.76 us | 18.1% |
| 8,192 | 2,546.30 us | 2,066.41 us | 18.8% |
| 32,768 | 8,669.71 us | 6,964.82 us | **19.7%** |

The hybrid improves final batch-1 output throughput versus the all-tuned table
by 0.5% on W1, 0.5% on W2, and 1.9% on W3.

## 8. Scheduler and DP8 MoE end-to-end gate

W2, batch 8, eight B200s. These are same-node, same-image controlled rows.

| Mode | Total time (s) | Output tok/s | Total tok/s | Mean TTFT (s) | Mean TPOT (ms) | Relative output rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Async scheduler, Gloo, runtime N160 fallback | **176.693** | **741.81** | **3,709.02** | 2.751 | **10.61** | **1.000x** |
| Sync scheduler, NCCL, runtime N160 fallback | 266.592 | 491.66 | 2,458.29 | 3.188 | 16.07 | 0.663x |
| Async scheduler, generated N160, clean run 1 | 200.991 | 652.13 | 3,260.64 | **2.474** | 12.11 | 0.879x |
| Async scheduler, generated N160, clean run 2 | 198.038 | 661.85 | 3,309.26 | **2.287** | 11.95 | 0.892x |

Keep async scheduling. Reject the generated DP8/N160 table despite its
kernel-only gains; it regressed two clean end-to-end runs.

## 9. DP8 expert-parallel ablation

Same image, node, warmup, cache isolation, and workload definitions as the
DP8 results above. `DP+EP` adds `--enable-expert-parallel` with vLLM's
`allgather_reducescatter` backend. Runtime logs confirm `DP*_EP*` workers and
EP ranks. DeepEP is excluded because its binary cannot load against the
installed NVSHMEM.

| Workload | Batch | DP output tok/s | DP+EP output tok/s | Change | DP TTFT (s) | DP+EP TTFT (s) | DP TPOT/ITL (ms) | DP+EP TPOT/ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 128 | 6,742.24 | 7,429.77 | **+10.2%** | 3.258 | 3.898 | 18.93 | 17.17 |
| W1 | 192 | **9,908.14** | 9,492.48 | **-4.2%** | 4.150 | 5.033 | 19.31 | 20.15 |
| W2 | 128 | **6,590.00** | 6,304.96 | **-4.3%** | 18.015 | 22.494 | 18.30 | 18.91 |
| W3 | 128 | **4,633.41** | 4,276.76 | **-7.7%** | 1.407 | 1.875 | 23.15 | 23.89 |

EP helps W1 at batch 128, but regresses the selected W1 batch-192 knee and
both selected mixed/prefill knees. Keep DP-only for the production profile.

## 10. Decision summary

| Probe | Evidence | Decision |
| --- | --- | --- |
| Attention | FI TPOT stays at 8.00–8.33ms; FA4 reaches 17.42ms at 96K | **Use FlashInfer** |
| Prefill budget | 32K reduces 64K/96K TTFT by 13.3–13.6% | **Use 32K for latency-oriented serving** |
| Prefix reuse | 95.58% hit rate and zero waiting | **Keep prefix caching and KV-sharing fast prefill** |
| Warmup | Full warmup cuts maximum agent TTFT from 1.991s to 0.231s | **Warm the complete trajectory before traffic** |
| DP1 MoE | Hybrid is fallback-identical at M=1–8 and up to 19.7% faster above it | **Package the hybrid N1280 table** |
| Scheduler | Async/Gloo is 50.9% faster than sync/NCCL on the controlled DP8 row | **Keep async scheduling** |
| DP8 MoE | Generated N160 regresses end-to-end output rate by 10.8–12.1% | **Keep the N160 runtime fallback** |
| DP8 EP | EP regresses the selected W1/W2/W3 knees by 4.2%, 4.3%, and 7.7% | **Keep DP-only with the current all-gather/reduce-scatter backend** |
| Batch knees | DP1 uses 16; DP4 uses 64; DP8 uses 192 for W1 and 128 for W2/W3 | **Use DP1 24 and DP4 96 only for max-throughput traffic** |

Raw JSON and logs are retained under
`long_context/raw/presentation-new-node-20260804` and
`long_context/raw/router-cache-20260805`; the complete historical scaled batch
tables remain in `long_context/README.md`.
