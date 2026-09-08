# Fast Mooncake 1.2× diagnostic — 2026-09-08

Repeat Fast standalone (GPU5) then 1P1D (P GPU4, D GPU5), serially, on retained B200 Job yoco-align-fast-vllm-vllm-train. Source 8c74b459542ff6b03cab9f355daabc47e80990b5; all 2303 serving source files match the 1× run. No production code change.

Use the same source 300–900 second window, 3643 requests, 30,518,473 input and 643,375 output tokens, context81920. Only timestamps divided by1.2; arrivals last 499.999166667s, offered 7.286012143 req/s. AIPerf0.12.0 CLI synthesis-speedup-ratio stays1.0. Trace SHA 5317e2301656c7d5441bbd63e58b7e8b9d3d445e0118729cd7fde0b8717b960c.

Historical settings preserved: TP1, BF16, FA4, --fast, CLI triton MoE with role-dependent Fast CUTLASS selection; maxseq256, memory0.85, token budgets32768(P/standalone) and8192(D), prefix caching, chunked prefill, YOCO sharing, graphs[1,2,4,8,16,32,64,128,256]. RDMA Mooncake,16workers, fail-on-load. No external autotune cache or Align profile. Model and dependency snapshots captured before/after.

Four functional probes, identical50-request smoke, full replay, client/token/server/GPU/transport audit, complete drain for each topology. Unique cache_salt for every case. Client concurrency512 safety ceiling,32workers, timeout600s, seed42. Stop escalation after hard failure or non-draining backlog; retain failures. Only identity-checked owned test processes are stopped; preserve Job/Pod/PID7 and allocation.

New1.2× cohort; preserve1× table. No Qwen or Align1.2× result implied. Diagnostic: one repeat,500-second arrivals, shared node, no declared SLO. Evaluate throughput response, scheduling lag, queues, latency percentiles, transfer, cache and overhang.
