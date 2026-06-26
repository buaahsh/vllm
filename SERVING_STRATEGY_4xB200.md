# Serving strategy — YOCO-30B-A3B (4× B200)

How to serve `yoco-30A3B-sft` on this box, and why. Based on profiling run on
2026-06-26 (8× NVIDIA B200 180 GB, 4 cards available: GPUs 0–3).

## TL;DR — recommended command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve <MODEL_PATH> \
  --served-model-name yoco-30A3B-sft --host 0.0.0.0 --port 8000 --trust-remote-code \
  --gpu-memory-utilization 0.9 --max-model-len 65536 \
  --moe-backend triton \
  --data-parallel-size 4 \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
```

- **Parallelism: DP=4** (4 independent replicas, 1 GPU each). **No TP, no EP.**
- **MoE backend: `triton`** — *mandatory*, see below.
- Answer text in `message.content`; chain-of-thought in `message.reasoning`.

## Why DP=4 (data-parallel), not TP or EP

The model is ~30B-total / ~3B-active MoE (128 experts, top_k=8, 20 layers,
`moe_ffn_dim=1280`). Weights are ~60 GB and **fit comfortably on a single
180 GB B200**. When a model fits on one GPU, tensor-parallel sharding only adds
cross-GPU communication overhead for no benefit. Pure data parallelism (one full
replica per GPU) maximizes aggregate throughput.

Profiling confirmed this. Two passes, `vllm bench serve`, random dataset,
`--moe-backend triton` for all runs.

### Stage 1 — moderate load (in 1024 / out 256, 300 prompts, concurrency 100)
All configs concurrency-bound and within noise (~19–20 req/s). Not a useful
discriminator on these fast GPUs.

### Stage 2 — saturating load (in 1024 / out 512, 1600 prompts, concurrency 400)

| Config       | req/s    | output tok/s | total tok/s | mean TPOT | p99 TTFT |
|--------------|----------|--------------|-------------|-----------|----------|
| **DP=4**     | **35.4** | **18 127**   | **54 699**  | **19.4ms**| 3850ms   |
| TP=2 + DP=2  | 30.8     | 15 753       | 47 536      | 23.3ms    | 4070ms   |
| TP=4         | 20.1     | 10 270       | 30 991      | 36.6ms    | 5287ms   |

**DP=4 wins on every axis**: ~1.76× the throughput of TP=4 and ~1.15× of
TP2×DP2, plus the lowest per-token latency (TPOT) and lowest p99 TTFT. The
ordering is monotonic — the more data-parallel (less tensor-parallel) the
config, the better. EP (`--enable-expert-parallel`, DP=4) was also tested and
was slightly *worse* than plain DP=4: the all-to-all expert routing overhead
isn't worth it for 128 small experts that already fit per-GPU.

## Why `--moe-backend triton` is mandatory

With the default `--moe-backend auto`, vLLM selects the **FlashInfer TRTLLM
BF16 MoE** kernel, which hard-requires `intermediate_size_per_partition % 128
== 0`. This model's routed-expert intermediate is **1280**, and under any
multi-GPU launch the kernel **crashes at engine init**:

```
RuntimeError: Check failed: args->intermediate_size % 128 == 0 (64 vs. 0) :
the second dimension of weights must be a multiple of 128.
```

(e.g. TP=4 → 1280/4 = 320, and 320 % 128 = 64 ≠ 0.) Every auto-backend config —
TP=4, DP=4, TP2×DP2, DP4+EP — failed this way. Forcing the **Triton** unquantized
MoE backend handles arbitrary intermediate sizes and runs all configs cleanly.

Bonus (fidelity): the Triton path is also the one that can apply the routed-expert
`swiglu_limit` clamp exactly (see `YOCO_V2_NOTES.md`), which the FlashInfer
TRTLLM path cannot.

## Operational notes

- Use **free** GPUs only. On this shared box GPUs 4/5/7 are often busy with other
  workloads; GPUs 0–3 were free and used here. Adjust `CUDA_VISIBLE_DEVICES`
  to whatever 4 cards are idle (`nvidia-smi`).
- `--data-parallel-size 4` auto-sets `api_server_count=4` (one API server per
  replica) and load-balances across the 4 engines behind the single port 8000.
- `--gpu-memory-utilization 0.9` reserves ~164 GB/GPU; with ~15 GB of weights per
  replica that leaves plenty of KV cache for `--max-model-len 65536`.
- Recommended sampling defaults to avoid runaway repetition (see YOCO notes):
  `temperature=0.7, top_p=0.95, repetition_penalty=1.05`.
- Health check: `curl -sf http://localhost:8000/health`.
