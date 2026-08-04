# YOCO B200 最终验收报告

## 2026-08-04 long-context BF16 serving extension

This section is the current reference for the 131K YOCO-v2 workload. The
original acceptance report remains below for historical comparison.

### Artifacts and decision summary

```text
Git branch: shaohanh/yoco-b200-longctx-20260804
Docker image: buaahsh/pytorch:26.02-b200-vllm-yoco-longctx-20260804
Registry digest: sha256:cdc3ebf990c0539190756c2aaac48c123f4243eefa386ad3eb4fccb1a7ae2d24
Model: /mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf
Precision: BF16 only
```

The production decision is FlashInfer attention, Triton MoE, TP=1, data
parallel replication, 32K maximum batched prefill tokens, prefix caching,
chunked prefill, YOCO KV-sharing fast prefill, and
`FULL_AND_PIECEWISE` CUDA graphs. Each B200 holds one complete model replica;
use DP=1, 4, or 8 for one GPU, four GPUs, or one eight-GPU node. Expert
parallelism is intentionally disabled for this independent-request benchmark.

The three benchmark workloads are:

| Workload | Shape | Notes |
| --- | --- | --- |
| W1 | 8,192 input + 65,536 output | Single turn, decode-heavy |
| W2 | 65,536 input + 16,384 output | Single turn, prefill + decode |
| W3 | 40 turns; 117K incremental input + 13K output | 130K final trajectory with prefix reuse |

W3 uses 130K rather than 131K input plus 13K output because the latter would
exceed the model's hard 131,072-token context limit.

### Exact B200 MoE tuning evidence

The base image did not contain a Triton MoE configuration for this model's
exact `E=128, N=1280, K=3072, top-k=8` shape. The new image includes:

```text
vllm/model_executor/layers/fused_moe/configs/E=128,N=1280,device_name=NVIDIA_B200.json
```

Both columns below use vLLM's `benchmarks/kernels/benchmark_moe.py` on the same
B200 and checkpoint. Lower kernel time is better.

| MoE token batch | Base image | Tuned image | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 61.62 us | 61.63 us | neutral |
| 2 | 95.63 us | 90.05 us | 5.8% |
| 4 | 147.40 us | 137.80 us | 6.5% |
| 8 | 225.97 us | 218.67 us | 3.2% |
| 16 | 326.96 us | 319.60 us | 2.3% |
| 32 | 425.87 us | 421.80 us | 1.0% |
| 128 | 519.57 us | 482.30 us | 7.2% |
| 1,024 | 699.49 us | 666.80 us | 4.7% |
| 2,843 | 1,235.96 us | 997.15 us | 19.3% |
| 3,899 | 1,394.78 us | 1,185.36 us | 15.0% |
| 8,192 | 3,107.94 us | 2,221.93 us | 28.5% |
| 32,768 | 9,071.07 us | 7,354.35 us | 18.9% |

This optimization targets prefill and mixed agentic steps. It should not be
described as a batch-1 decode speedup.

### Start inference

Set the common paths once:

```bash
IMAGE=buaahsh/pytorch:26.02-b200-vllm-yoco-longctx-20260804
MODEL=/mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf
mkdir -p "$PWD/yoco-results"
```

One GPU:

```bash
docker run --rm --name yoco-long-dp1 --network host --ipc host \
  --gpus '"device=0"' \
  -e MODEL="$MODEL" -e DP_SIZE=1 \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  -v "$PWD/yoco-results:/results" \
  "$IMAGE" bash tools/yoco_serving/serve_long_context.sh
```

Four GPUs:

```bash
docker run --rm --name yoco-long-dp4 --network host --ipc host \
  --gpus '"device=0,1,2,3"' \
  -e MODEL="$MODEL" -e DP_SIZE=4 \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  -v "$PWD/yoco-results:/results" \
  "$IMAGE" bash tools/yoco_serving/serve_long_context.sh
```

One eight-GPU B200 node:

```bash
docker run --rm --name yoco-long-dp8 --network host --ipc host \
  --gpus all \
  -e MODEL="$MODEL" -e DP_SIZE=8 \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  -v "$PWD/yoco-results:/results" \
  "$IMAGE" bash tools/yoco_serving/serve_long_context.sh
```

The service exposes the OpenAI-compatible API at `http://127.0.0.1:8001/v1`.
Wait for `GET /health` to return HTTP 200 before warming or benchmarking.

### Warm the full context range

A short two-turn warmup misses kernels that first occur near the end of the
trajectory. Run this once after each cold service start; set `DP_SIZE` and
`GPU_INDICES` to match the launcher.

```bash
docker exec \
  -e TOKENIZER="$MODEL" \
  -e DP_SIZE=1 \
  -e GPU_INDICES=0 \
  -e RESULT_DIR=/results/warmup-dp1 \
  yoco-long-dp1 \
  bash tools/yoco_serving/warmup_long_context.sh
```

Examples for the larger deployments are `DP_SIZE=4 GPU_INDICES=0,1,2,3` and
`DP_SIZE=8 GPU_INDICES=0,1,2,3,4,5,6,7`.

### Reproduce the speed report

The runner forces exact output lengths, uses unique random seeds to avoid
cross-run prompt-cache contamination, and emits detailed JSON. Recommended
batch/concurrency sweeps are 1/2/4/8 for DP1, 4/8/16/32 for DP4, and
8/16/32/64 for DP8.

```bash
docker exec \
  -e TOKENIZER="$MODEL" \
  -e DP_SIZE=1 \
  -e GPU_INDICES=0 \
  -e BATCHES="1 2 4 8" \
  -e RESULT_DIR=/results/dp1 \
  yoco-long-dp1 \
  bash tools/yoco_serving/benchmark_long_context.sh
```

The single-turn JSON reports request throughput, input/output/total token
throughput, mean/P95 TTFT, and mean TPOT. The W3 JSON additionally reports
logical incremental prefill throughput, generation throughput, prefix-cache
hit rate, queue depth, KV use, and GPU telemetry.

### Scaled end-to-end results

The tables in this subsection are generated from the detailed JSON described
above. Total time is wall time for the whole batch, and throughput is aggregate
across the selected GPU count.

<!-- LONG_CONTEXT_RESULTS -->

#### DP1 / one B200

W1 — 8K input, 64K output:

| Mode | GPUs | Batch | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 1 | 533.969 | 0.0019 | 15.34 | 122.73 | 138.08 | 0.182 | 0.182 | 8.15 |
| BF16 | 1 | 2 | 585.329 | 0.0034 | 27.99 | 223.93 | 251.92 | 0.317 | 0.337 | 8.93 |
| BF16 | 1 | 4 | 705.400 | 0.0057 | 46.45 | 371.62 | 418.08 | 0.589 | 0.632 | 10.75 |
| BF16 | 1 | 8 | 826.238 | 0.0097 | 79.32 | 634.55 | 713.87 | 0.962 | 1.091 | 12.59 |

W2 — 64K input, 16K output:

| Mode | GPUs | Batch | Total time (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 1 | 137.405 | 0.0073 | 476.95 | 119.24 | 596.19 | 1.215 | 1.215 | 8.31 |
| BF16 | 1 | 2 | 153.344 | 0.0130 | 854.76 | 213.69 | 1,068.44 | 2.024 | 2.337 | 9.24 |
| BF16 | 1 | 4 | 190.258 | 0.0210 | 1,377.83 | 344.46 | 1,722.29 | 3.008 | 3.961 | 11.43 |
| BF16 | 1 | 8 | 234.583 | 0.0341 | 2,234.98 | 558.74 | 2,793.72 | 5.445 | 8.916 | 13.98 |

W3 — 40-turn, 130K agentic trajectory. Input and total rates count logical
incremental tokens, not repeatedly submitted cached prefixes.

| Mode | GPUs | Batch | Total time (s) | Traj/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT (s) | P95 TTFT (s) | Mean ITL (ms) | Cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 1 | 104.181 | 0.0096 | 1,123.04 | 124.78 | 1,247.82 | 0.156 | 0.213 | 7.55 | 95.58% |
| BF16 | 1 | 2 | 127.083 | 0.0157 | 1,841.32 | 204.59 | 2,045.91 | 0.268 | 0.345 | 8.97 | 95.58% |
| BF16 | 1 | 4 | 162.339 | 0.0246 | 2,882.86 | 320.32 | 3,203.18 | 0.409 | 0.578 | 11.25 | 95.58% |
| BF16 | 1 | 8 | 222.274 | 0.0360 | 4,211.02 | 467.89 | 4,678.91 | 0.734 | 1.088 | 14.82 | 95.58% |

<!-- LONG_CONTEXT_MORE_RESULTS -->

Only the DP1 sweep above completed in this allocation. Kubernetes reclaimed
`settled-foal-b200g4-dev-46d12b9f-master-0` at approximately 12:28 PDT on
2026-08-04 before the DP4 and DP8 sweeps could start. Those rows are omitted,
not estimated from DP1. The DP4/DP8 launch and benchmark commands above are
ready to rerun when an eight-GPU B200 node is available.

The earlier concurrency-1 figures in the historical report below were taken
on GPU 0 with a different prompt seed; this scaled DP1 sweep ran on GPU 7.
W1/W2 differ by about 8% between those executions while W3 differs by about
1%, so that cross-run difference is treated as system/GPU variance rather
than evidence of a decode regression or speedup. The paired MoE table above,
which uses the same GPU and inputs for both configurations, is the controlled
optimization result.

本文只记录本次 YOCO-v2/v3 在 B200 上的最终配置、验收方法和结果，不包含
旧镜像、旧分支或中间调试过程。

## 最终产物

Git 分支：

```text
shaohanh/yoco-serving-final-20260730
```

Docker Hub 镜像：

```text
buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final
registry digest: sha256:08a08f36ab8c6c80ee1c7f09b9e5f8b6ce0b91cc684455982b2e5e286c736f2b
local image id: sha256:bff890479962a9267862f3903113239f48ac9de982d184239de0a41d17f1b0e6
size: 37,667,878,332 bytes
```

镜像是自包含 runtime，不需要挂载 vLLM 源码。镜像内包含：

- YOCO-v2 DP4 compile key `1a1773b3c5`；
- YOCO-v3 DP4 compile key `b9be5626e8`；
- YOCO-v2 DP8+EP compile key `be47add45b`；
- 对应的 Torch AOT/Inductor 和首请求 Triton JIT cache。

## 验证模型

YOCO-v2 agentic serving：

```text
/mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf
```

YOCO-v3/L3：

```text
/mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf
```

FA4 Native/vLLM matched alignment：

```text
HF:
/mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu

Native merged:
/mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-merged
```

Qwen 对照：

```text
/mnt/pvc/lidong1/hf_cache/Qwen3.5-35B-A3B
```

## 完整启动命令

### YOCO-v2 DP4

```bash
docker run --rm \
  --name yoco-v2-dp4 \
  --network host \
  --ipc host \
  --gpus '"device=0,1,6,7"' \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final \
  vllm serve \
  /mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf \
  --served-model-name yoco-v2 \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --attention-backend FLASHINFER \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 81920 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser agens \
  --reasoning-parser agens \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

该命令没有 `--enforce-eager`。普通 decode 使用 FULL CUDA Graph；
prefill/decode 混合 step 使用 PIECEWISE graph。

### Qwen3.5-35B-A3B DP4 对照

```bash
docker run --rm \
  --name qwen35-dp4 \
  --network host \
  --ipc host \
  --gpus '"device=0,1,6,7"' \
  -v /mnt/pvc:/mnt/pvc:ro \
  buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final \
  vllm serve \
  /mnt/pvc/lidong1/hf_cache/Qwen3.5-35B-A3B \
  --served-model-name qwen35 \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --attention-backend FLASHINFER \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 81920 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

Qwen 与 YOCO 的速度对照使用同一个镜像、同一 vLLM commit、BF16、
FlashInfer attention、Triton MoE、DP4、相同 GPU 和相同请求序列。

### YOCO-v3 DP4

```bash
docker run --rm \
  --name yoco-v3-dp4 \
  --network host \
  --ipc host \
  --gpus '"device=0,1,6,7"' \
  -v /mnt/pvc:/mnt/pvc:ro \
  buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final \
  vllm serve \
  /mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf \
  --served-model-name yoco-v3 \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --attention-backend FLASHINFER \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 81920 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser agens \
  --reasoning-parser agens \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

### FA4 BF16 验收服务

下面的命令直接从最终镜像启动 FA4 beta13、非 eager CUDA Graph 服务：

```bash
docker run --rm \
  --name yoco-fa4-bf16 \
  --network host \
  --ipc host \
  --gpus '"device=0"' \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final \
  vllm serve \
  /mnt/msranlphot/shaohanh/exp/sft/30A3B-73k-sft-65k-muon-bsz1M-shaohan_sft_260629/0000-6000-hf-gpu \
  --served-model-name yoco-fa4-bf16 \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --attention-config.backend FLASH_ATTN \
  --attention-config.flash_attn_version 4 \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

MXFP8 使用相同命令并增加：

```text
--quantization fp8_per_block
```

### YOCO-v2 DP8 + EP

健康的 NVIDIA container runtime 上使用：

```bash
docker run --rm \
  --name yoco-v2-dp8-ep \
  --network host \
  --ipc host \
  --gpus all \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  buaahsh/pytorch:26.02-b200-vllm-yoco-v2-v3-0729-final \
  vllm serve \
  /mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf \
  --served-model-name yoco-v2 \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --attention-backend FLASHINFER \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.68 \
  --max-model-len 81920 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser agens \
  --reasoning-parser agens \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

本机 NVIDIA container hook 缺少 GPU2-5 的 `/proc/driver/nvidia/gpus` 条目，
因此本次 DP8 验收使用等价的 `runc` device/driver-library 映射。八个
`/dev/nvidia*`、CUDA 和 NVML 均正常；该问题属于宿主 runtime 元数据。

## Agentic 验证方法

本次性能验收代码已放入：

```text
benchmarks/multi_turn/benchmark_agent_trace.py
```

固定 workload：

- 每条 trajectory 为 40 turns；
- 每轮平均增加 1,800 个 prefill token，并强制生成 200 token；
- 每条 trajectory 合计 72K logical prefill + 8K generation；
- 最终上下文为 80K token；
- prefix 长度按 1,056 token 对齐，稳定触发 prefix cache 和 chunked prefill；
- `min_tokens=max_tokens=200`、`ignore_eos=true`，确保两模型输出 token 数一致；
- 使用 `X-data-parallel-rank` 将同一 trajectory 固定到相同 DP rank；
- 每条 trajectory 使用独立 `cache_salt`；
- 同时采集 vLLM metrics、TTFT、ITL、queue、KV cache、每卡 SM、
  memory bandwidth utilization、显存和功耗。

YOCO c8/c16/c32：

```bash
mkdir -p /tmp/agent-trace/yoco

for concurrency in 8 16 32; do
  python benchmarks/multi_turn/benchmark_agent_trace.py \
    --base-url http://127.0.0.1:8001/v1 \
    --model yoco-v2 \
    --tokenizer \
      /mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf \
    --output /tmp/agent-trace/yoco/c${concurrency}.json \
    --concurrency "${concurrency}" \
    --trajectories "${concurrency}" \
    --turns 40 \
    --prefill-per-turn 1800 \
    --output-per-turn 200 \
    --cache-alignment 1056 \
    --dp-size 4 \
    --gpu-indices 0,1,6,7 \
    --seed 20260729
done
```

Qwen 使用同一命令，只替换：

```text
--model qwen35
--tokenizer /mnt/pvc/lidong1/hf_cache/Qwen3.5-35B-A3B
--output /tmp/agent-trace/qwen/c${concurrency}.json
```

输出文件：

- `<output>`：整场汇总；
- `<output>.turns.jsonl`：逐 trajectory/turn 的 prompt/output token 数、
  TTFT、ITL 和 latency；
- `<output>.runtime.json`：逐秒 vLLM metrics 和每卡 telemetry。

`computed_prefill_tokens_per_service_second` 使用 vLLM
`request_prefill_time_seconds_sum` 作分母；
`generation_tokens_per_service_second` 使用
`request_decode_time_seconds_sum` 作分母。它们用于分开比较 prefill 和 decode，
不能用整场 wall throughput 代替。

## Agentic 性能结果

三档测试分别严格生成 64K、128K、256K token；Qwen 与 YOCO 的 prompt
schedule 和 output token 数完全相同。

| c | wall Q/Y | prefill service tok/s Q/Y | decode service tok/s Q/Y | TTFT p50 Q/Y | ITL p50 Q/Y | avg SM Q/Y |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | `85.59 / 109.66s` | `11,576 / 8,649` | `107.16 / 84.24` | `275 / 323ms` | `9.08 / 11.73ms` | `81.2 / 80.5%` |
| 16 | `94.53 / 123.29s` | `9,684 / 7,038` | `99.29 / 74.17` | `360 / 441ms` | `9.91 / 12.97ms` | `83.4 / 81.3%` |
| 32 | `119.42 / 152.65s` | `10,093 / 5,242` | `77.20 / 62.35` | `345 / 629ms` | `12.29 / 15.59ms` | `87.1 / 82.7%` |

其中 `Q/Y` 分别表示 Qwen/YOCO。YOCO 相对 Qwen：

- wall time 慢 `28.1% / 30.4% / 27.8%`；
- prefill service throughput 为 Qwen 的 `74.7% / 72.7% / 51.9%`；
- decode service throughput 为 Qwen 的 `78.6% / 74.7% / 80.8%`。

YOCO 运行状态：

| c | waiting max | KV max | 结论 |
| ---: | ---: | ---: | --- |
| 8 | `0` | `0.57%` | queue/KV 健康 |
| 16 | `0` | `1.24%` | queue/KV 健康 |
| 32 | `5` | `2.70%` | queue/KV 健康 |

c32 没有 eager fallback。此前未覆盖 local batch 8 的 FULL graph 已补齐；
最终 capture sizes 为 `1,2,4,8,16,32`。

Agentic 生产性能使用 FlashInfer attention，而不是 FA4。FA4 用于下面的
matched alignment 验收；在相同 40-turn workload 中，FA4 c32 明显慢于
FlashInfer，因此不作为最终吞吐配置。

## FA4 数值与 decode 验收

matched alignment 固定：

- PyTorch 26.02；
- FlashAttention `4.0.0b13`；
- FA4 commit `9bad4bec7326ad28edb5516b8878fd283f8991c0`；
- CuTeDSL `4.5.1`；
- vLLM 与 llm-train 使用相同 BF16/MXFP8、FA4 和 batch shape；
- MXFP8 两侧均使用 128-element block；
- batch 16 Native reference 使用与 scheduler 一致的 `1 + 15` forward shape；
- vLLM 使用非 eager CUDA Graph。

| 验证矩阵 | Native -> vLLM mean KL | 结论 |
| --- | ---: | --- |
| BF16，batch 1 | `0.00392071` | 通过 `<1e-2`，但不是 exact zero |
| BF16，batch 16 | `0.00283000` | 通过 `<1e-2` |
| MXFP8，batch 1，mixed5 | `0.0182856` | 未通过 `<1e-2` |
| MXFP8，batch 16 | `0.0000724250` | 通过 `<1e-2` |
| BF16，TP2+EP2，batch 16 | `0.00509094` | 通过 `<1e-2` |
| MXFP8，TP2+EP2，batch 16 | `0.00959670` | 通过 `<1e-2` |

非 eager FA4 eager/FULL/FULL_DECODE_ONLY 的 greedy decode 逐 token 一致，
没有 CUDA Graph replay 导致的乱码、首 token 卡死或单 token collapse。
最终 YOCO-v2/v3 serving smoke 中英文生成正常，没有异常重复。

约 4.7K-token 输入按 1,024 token 分块时，打开 KV-sharing fast prefill 对
同一个 vLLM FA4 chunked 输出的增量 KL 为：

| 精度 | fast prefill off -> on KL |
| --- | ---: |
| BF16 | `0.000121237` |
| MXFP8 | `0.00400451` |

因此 KV-sharing fast prefill 本身没有破坏 vLLM 输出。

尚未通过的 matched alignment：

- 约 4.7K-token 的真实 Native -> vLLM chunked prefill：
  BF16 `0.285164`、MXFP8 `0.0209809`；
- MXFP8 batch 1 mixed5：`0.0182856`；
- BF16 batch 1 满足 `<1e-2`，但不是 exact zero。

不能用“服务能开启 `--enable-chunked-prefill`”代替真实 Native-to-vLLM
chunked KL 验收。

## DP fast prefill 与 CUDA Graph

DP 下任一 rank 进入 fast prefill 时，所有 rank 必须统一进入 split
self/cross path；否则 NaiveDPEP MoE collective 的 token vector 会分叉。

最终实现将 fast-prefill metadata 打包进原 DP coordination flag：

- bit 0：ubatch；
- bit 1：fast-prefill active；
- bit 2 起：fast-prefill padded token count。

inactive rank 在其他 rank 开启 fast prefill 时使用主 batch padded count；
所有 rank 都是普通 decode 时返回 `None`，不传播 fast metadata，继续使用
普通 FULL model graph。最终 DP4 mixed prefill/decode、c32 local batch 8、
DP8+EP 均无 collective hang 或 eager fallback。

## DP8 与 YOCO-v3

DP8+EP 验收：

- 78,001-token prompt + 16-token decode：`3.329s`；
- 八张 B200 峰值 SM utilization 均为 `98-99%`；
- GPU5 包含外部约 44GB 占用时峰值显存 `168,889 MiB`；
- FlashInfer、Triton MoE、chunked prefill、fast prefill、EP 和
  `FULL_AND_PIECEWISE` 同时开启；
- 请求后中文生成正常，日志无 eager fallback。

YOCO-v3 验收：

- `diff_v3`、weighted Q/K RMSClip、latent MoE、`universal_loop=3` 可加载；
- 中文和英文直答正常；
- `<|end|>` 被服务层作为 stop string，不返回给用户；
- `get_weather({"city":"Seattle"})` tool call 可正确解析；
- 默认 `enable_thinking=false`，用户显式 chat-template kwargs 仍可覆盖。

## 启动 cache

CUDA Graph 对象不能跨进程持久化；镜像只持久化 Torch
AOT/Inductor/Triton 编译产物。最终镜像实测：

| 模型 | graph compile | graph capture | ready 时间 |
| --- | ---: | ---: | ---: |
| v2 DP4，无新 cache | `67.8s` | `6s` | 约 `5.3 min` |
| v2 DP4，baked cache | `10.5s` | `6s` | 约 `4.7 min` |
| v3 DP4，无新 cache | 约 `125s` | `5s` | 约 `249s` |
| v3 DP4，baked cache | `14.9s` | `5s` | 约 `221s` |

剩余启动时间来自权重读取、KV memory profile、模型 warmup 和 backend
初始化，不是 CUDA Graph capture。

## FLOPs 与 profiler 结论

| 指标 | YOCO-30B-A3B | Qwen3.5-35B-A3B | YOCO / Qwen |
| --- | ---: | ---: | ---: |
| 总参数 | `32.2207B` | `35.9518B` | `0.896x` |
| decode projection + MoE GEMM | `11.943 GF/token` | `4.873 GF/token` | `2.451x` |
| 2K context core compute | `13.117 GF/token` | `5.321 GF/token` | `2.465x` |
| 40K context core compute | `25.554 GF/token` | `11.539 GF/token` | `2.214x` |
| 80K context core compute | `38.661 GF/token` | `18.093 GF/token` | `2.137x` |

YOCO 参数更少但 active FLOPs 更高，主要因为：

- 10 个 self layers 执行 `universal_loop=3`，再执行 10 个 cross layers，
  共 40 次 block execution；
- hidden size 为 3,072，Qwen 为 2,048；
- routed top-8 expert intermediate 为 1,280，Qwen 为 512；
- differential attention 增加 Q 和 combine 工作；
- cross-attention decode work 随上下文增长。

fast prefill 跳过重复 self-decoder 后，80K 理论 active compute 为 YOCO
`11.16 GF/token`、Qwen `11.54 GF/token`。因此 prefill 理论 FLOPs 接近，
剩余差距主要来自 split path、rank imbalance 和小 kernel 效率。

同镜像、同后端的短 torch profile：

| profiled c8 | Qwen | YOCO | YOCO / Qwen |
| --- | ---: | ---: | ---: |
| wall | `6.314s` | `9.035s` | `1.431x` |
| prefill service throughput | `6,195 tok/s` | `5,370 tok/s` | `0.867x` |
| decode service throughput | `96.58 tok/s` | `75.47 tok/s` | `0.781x` |

代表性 rank 的 decode trace：

| 指标 | Qwen | YOCO | YOCO / Qwen |
| --- | ---: | ---: | ---: |
| kernel launches / scheduler step | `1,547` | `1,859` | `1.202x` |
| summed kernel time | `3,793.7ms` | `3,920.8ms` | `1.034x` |
| union GPU busy time | `3,382.8ms` | `3,872.2ms` | `1.145x` |
| 被 overlap 隐藏的 kernel time | `410.9ms` (`10.8%`) | `48.7ms` (`1.2%`) | - |

去掉 collective 和通用 elementwise 后，MoE、GEMM、router、attention 等
主要计算热点合计为 Qwen `1,808ms`、YOCO `2,600ms`，YOCO 为 `1.44x`。
这说明 decode 差距主要是架构 active compute，同时还叠加了：

- YOCO 每 step 多 `20%` kernel launches；
- YOCO 通信/计算 overlap 明显少于 Qwen；
- RMSNorm/RMSClip、gate、differential combine、router TopK/scatter 等
  小 kernel；
- DP fast-prefill rank 不均衡和 collective 等待。

新增的 B200 Triton MoE 配置：

```text
vllm/model_executor/layers/fused_moe/configs/E=128,N=320,device_name=NVIDIA_B200.json
```

使 decode batch 8-128 的 MoE microbenchmark 提升约 `9-16%`。

后续优化优先级：

1. 合并 YOCO norm/gate/differential/router 小 kernel；
2. overlap NaiveDPEP AllGather/ReduceScatter 与 expert/shared compute；
3. 改善 DP fast-prefill chunk packing 和 rank 均衡；
4. 继续调优 tiny-token cross-decoder MoE。

## 本次相关代码

```text
benchmarks/multi_turn/benchmark_agent_trace.py
docker/Dockerfile.b200
docker/Dockerfile.b200.runtime
tests/model_executor/test_yoco_config.py
tests/parser/test_agens_parser.py
tests/v1/worker/test_dp_utils.py
vllm/entrypoints/openai/chat_completion/serving.py
vllm/forward_context.py
vllm/model_executor/models/config.py
vllm/model_executor/models/yoco.py
vllm/reasoning/agens_reasoning_parser.py
vllm/v1/attention/backend.py
vllm/v1/attention/backends/utils.py
vllm/v1/worker/dp_utils.py
vllm/v1/worker/gpu_model_runner.py
```
