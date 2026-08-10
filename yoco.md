# YOCO B200 最终验收报告

## 2026-08-04 long-context BF16 serving extension

This section is the current reference for the 131K YOCO-v2 workload. The
original acceptance report remains below for historical comparison.

### Artifacts and decision summary

```text
Git branch: shaohanh/yoco-b200-longctx-multigpu-20260804
Docker image: buaahsh/pytorch:26.02-b200-vllm-yoco-longctx-multigpu-20260804
Model: /mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf
Precision: BF16 only
```

The production decision is FlashInfer attention, Triton MoE, TP=1, DP=1/4/8,
32K maximum batched prefill tokens, prefix caching, chunked prefill, YOCO
KV-sharing fast prefill, and `FULL_AND_PIECEWISE` CUDA graphs. Async scheduling
is enabled. The observed local fused-MoE width is N1280 at DP1, N320 at DP4,
and N160 at DP8, so the multigpu modes must not be interpreted as isolated
full-model replicas. In the final cold-start logs, the DP1 worker loads about
60.62 GiB while each DP4 worker loads about 18.43 GiB. The optional DeepEP
backend is not used.

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

The final JSON is hybrid: M=1/2/4/8 entries are byte-for-byte equivalent to
vLLM's runtime fallback, while M=16 and larger retain measured tuned configs.
The table uses `benchmark_moe_defaults.py` on the same B200. Lower kernel time
is better; tiny decode batches are marked identical rather than presenting
measurement-order noise as a speedup.

| MoE token batch | Runtime fallback | Hybrid image | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | same config | same config | neutral by construction |
| 2 | same config | same config | neutral by construction |
| 4 | same config | same config | neutral by construction |
| 8 | same config | same config | neutral by construction |
| 16 | 353.74 us | 340.50 us | 3.7% |
| 32 | 475.67 us | 449.06 us | 5.6% |
| 128 | 541.20 us | 503.55 us | 7.0% |
| 1,024 | 683.01 us | 659.41 us | 3.5% |
| 2,843 | 1,166.33 us | 938.16 us | 19.6% |
| 3,899 | 1,373.41 us | 1,124.76 us | 18.1% |
| 8,192 | 2,546.30 us | 2,066.41 us | 18.8% |
| 32,768 | 8,669.71 us | 6,964.82 us | 19.7% |

This targets DP1 prefill and mixed agentic steps without knowingly regressing
the M=1-8 decode path. A separately generated DP8/N160 table improved isolated
kernels but regressed two clean end-to-end runs, so it is deliberately not
packaged. See `long_context/README.md` for that A/B.

### New-node disk and nested Docker setup

The final multigpu run used
`assuring-owl-b200g4-dev-d5aab19e-master-0`. Verify the block device before
mounting it; these commands intentionally keep Docker's large layers on the
persistent `/data` disk.

```bash
apt-get update
apt-get install -y sudo util-linux fdisk docker.io fuse-overlayfs
mkdir -p /data
findmnt /data || mount -t ext4 /dev/md1 /data
mkdir -p /data/docker

dockerd \
  --data-root=/data/docker \
  --storage-driver=fuse-overlayfs \
  --iptables=false \
  --bridge=none \
  --pidfile=/data/dockerd.pid \
  >/data/dockerd.log 2>&1 &
```

The commands below assume a standard host with NVIDIA Container Toolkit. The
benchmark Kubernetes job used nested Docker and therefore also needed
cluster-specific GPU device and NVML library mappings; those infrastructure
details are intentionally kept out of the minimal inference commands.

### Start inference

Set the common paths once:

```bash
IMAGE=buaahsh/pytorch:26.02-b200-vllm-yoco-longctx-multigpu-20260804
MODEL=/mnt/msranlphot/shaohanh/exp/sft/30A3B-65k-muon-xiangyang_ssb_sp-rl-bx9k-hf-v1_from6ksft/0000-0800-hf
mkdir -p "$PWD/yoco-results"
```

One GPU:

```bash
docker run --rm --name yoco-long-dp1 --network host --ipc host \
  --gpus '"device=0"' \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  -v "$PWD/yoco-results:/results" \
  "$IMAGE" vllm serve "$MODEL" \
  --served-model-name yoco-v2-long --host 0.0.0.0 --port 8001 \
  --trust-remote-code --dtype bfloat16 \
  --attention-backend FLASHINFER --moe-backend triton \
  --tensor-parallel-size 1 --data-parallel-size 1 \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --max-num-batched-tokens 32768 --max-num-seqs 128 \
  --enable-prefix-caching --enable-chunked-prefill --kv-sharing-fast-prefill \
  --enable-auto-tool-choice --tool-call-parser agens --reasoning-parser agens \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

Four GPUs:

```bash
docker run --rm --name yoco-long-dp4 --network host --ipc host \
  --gpus '"device=0,1,2,3"' \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  -v "$PWD/yoco-results:/results" \
  "$IMAGE" vllm serve "$MODEL" \
  --served-model-name yoco-v2-long --host 0.0.0.0 --port 8001 \
  --trust-remote-code --dtype bfloat16 \
  --attention-backend FLASHINFER --moe-backend triton \
  --tensor-parallel-size 1 --data-parallel-size 4 \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --max-num-batched-tokens 32768 --max-num-seqs 128 \
  --enable-prefix-caching --enable-chunked-prefill --kv-sharing-fast-prefill \
  --enable-auto-tool-choice --tool-call-parser agens --reasoning-parser agens \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

One eight-GPU B200 node:

```bash
docker run --rm --name yoco-long-dp8 --network host --ipc host \
  --gpus all \
  -v /mnt/msranlphot:/mnt/msranlphot:ro \
  -v "$PWD/yoco-results:/results" \
  "$IMAGE" vllm serve "$MODEL" \
  --served-model-name yoco-v2-long --host 0.0.0.0 --port 8001 \
  --trust-remote-code --dtype bfloat16 \
  --attention-backend FLASHINFER --moe-backend triton \
  --tensor-parallel-size 1 --data-parallel-size 8 \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --max-num-batched-tokens 32768 --max-num-seqs 128 \
  --enable-prefix-caching --enable-chunked-prefill --kv-sharing-fast-prefill \
  --enable-auto-tool-choice --tool-call-parser agens --reasoning-parser agens \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

The service exposes the OpenAI-compatible API at `http://127.0.0.1:8001/v1`.
Wait for `GET /health` to return HTTP 200 before warming or benchmarking.
The commands explicitly select the validated `FLASHINFER` backend. Replace it
with `FLASH_ATTN` only for a controlled FA4 comparison, keeping every other
setting and the cache-salt namespace fixed. `--max-num-seqs` is a per-engine
cap, not a deployment-wide cap. It is 128 for DP1, DP4, and DP8; for example, a
DP8 batch of 192 is distributed to about 24 active requests per rank rather
than requiring 192 slots on every rank. Change it only after checking both
queue depth and KV-cache use in the exported vLLM metrics.

With the documented 85% memory utilization, the final cold-start logs report
space for 19.43 full 131,072-token sequences on DP1 and 26.69 per DP4 engine.
Thus DP1 batch 24 is an intentional over-capacity saturation probe, while DP4
batch 96 is still under the deployment's roughly 107-sequence full-context
capacity. Shorter live contexts can sustain more requests than these
worst-case figures.

### Warm the full context range

A short two-turn warmup misses kernels that first occur near the end of the
trajectory. Run this once after each cold service start; set `DP_SIZE` and
`GPU_INDICES` to match the running server.

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

The runner forces exact output lengths, gives every measurement a unique
server-side cache namespace, resumes complete JSON files, and emits detailed
results. Recommended saturation sweeps are 1/2/4/8/12/16/24 for DP1,
4/8/16/32/48/64/96 for DP4, and 8/16/32/64/96/128/192 for DP8. W1 is expensive
because every request generates 65,536 tokens; use `WORKLOADS` to run the
three shapes independently.

```bash
docker exec \
  -e TOKENIZER="$MODEL" \
  -e DP_SIZE=1 \
  -e GPU_INDICES=0 \
  -e WORKLOADS="1 2 3" \
  -e BATCHES="1 2 4 8 12 16 24" \
  -e RUN_ID=dp1-production-20260804 \
  -e RESULT_DIR=/results/dp1 \
  yoco-long-dp1 \
  bash tools/yoco_serving/benchmark_long_context.sh
```

Use a fresh `RUN_ID` for an independent repeat. The default
`SKIP_EXISTING=1` skips only nonempty, valid JSON results tagged with that same
run identity, so an interrupted sweep can be resumed safely. Set
`SKIP_EXISTING=0` only when intentionally replacing results.

The single-turn JSON reports request throughput, input/output/total token
throughput, mean/P95 TTFT, and mean TPOT. The W3 JSON additionally reports
logical incremental prefill throughput, generation throughput, prefix-cache
hit rate, queue depth, KV use, and GPU telemetry.

### Scaled end-to-end results

The tables in this subsection are generated from the detailed JSON described
above. Total time is wall time for the whole batch, and throughput is aggregate
across the selected GPU count.

<!-- LONG_CONTEXT_RESULTS -->

The full presentation tables are kept in `long_context/README.md` so this
operational guide does not drift from the raw JSON. The measured batch knees
are:

| Deployment | W1 knee | W2 knee | W3 knee | Max-throughput probes |
| --- | ---: | ---: | ---: | --- |
| One B200 | 16 | 16 | 16 | W2/W3 batch 24 |
| Four B200s | 64 | 64 | 64 | Batch 96 |
| One eight-B200 node | 128 | 128 | 128 | Batch 192 |

These are starting points, not hard request limits. Use the detailed TTFT and
TPOT/ITL columns in the report to choose a lower batch for latency-sensitive
traffic or the max-throughput probe when aggregate rate is the priority.

<!-- LONG_CONTEXT_MORE_RESULTS -->

The complete DP1, DP4, and DP8 tables, controlled scheduler/MoE A/B results,
batch-knee decisions, and raw-evidence locations are maintained in
[`long_context/README.md`](long_context/README.md). Every published scaled row
was measured with only that deployment active on the node. Diagnostic rows
from an accidentally co-located DP1/DP4 run are retained under
`co-located-invalid/` for auditability and are excluded from the report.

Use the same GPU count, `RUN_ID`, workload shape, and batch when comparing a
change. Kernel microbenchmarks are supporting evidence only: the generated
DP8/N160 MoE table improved isolated kernels but failed the clean end-to-end
gate, so the final image intentionally keeps the runtime fallback.

A compact, slide-ready reconstruction of the new-node probes is maintained in
[`long_context/presentation_tables.md`](long_context/presentation_tables.md).

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

## Prefill-only 后十层裁剪

这是最终分支之后的 follow-up 优化，尚未写入上面的最终镜像，避免与已发布
镜像内容混淆：

```text
baseline branch: shaohanh/yoco-serving-final-20260730
baseline commit: c27db1e189973cea3164ba66b1d00359d4122088
candidate branch: review/yoco-01-kv-only-prefill
candidate commit: this PR HEAD
```

### 修改文件

- `vllm/model_executor/layers/attention/attention.py`：允许 YOCO 复用已完成的
  KV cache 写入，并跳过重复写入。
- `vllm/model_executor/models/yoco.py`：直接写入共享 K/V；普通 fast prefill
  的十个 cross layers 只处理 logits token，KV-only P 请求跳过全部十层。
- `vllm/v1/worker/gpu_model_runner.py`：为专用 YOCO `kv_producer` 服务启用
  任意 DP 的 KV-only 路径；DP1 `kv_both` 保留逐请求安全检查。
- `vllm/v1/worker/gpu_worker.py`：让专用 producer 的空闲 DP rank runtime
  dummy 同样走 KV-only 路径，保持 MoE collective 顺序一致。
- `tests/model_executor/test_yoco_conversion.py`：覆盖十层 compact-token 执行和
  KV-only 全跳过行为。
- `tests/v1/worker/test_gpu_model_runner.py`：覆盖 P batch、静态 producer 和
  空闲 rank dummy 判定。
- `yoco.md`：记录功能边界、测试数据、正确性和性能指标。

最终分支的普通 fast prefill 已经让后十个 cross layers 中的后九层只处理
logits token，但第一个 KV-owner cross layer 仍处理完整 prompt。本次改为由
self block 直接把共享 K/V 写入第一个 cross layer 的 cache，再让十个 cross
layers 都只处理 logits token。对于只负责传 KV 的 P 请求，十个 cross layers
全部跳过；D 侧的正常 decode 路径不变。

### P/D token 归属和启用条件

当前约定是 **P token 必须丢弃，D 负责 first token 和之后所有用户可见
token**：

1. Gateway 向 P 发送原始 prompt，并设置 `max_tokens=1`、
   `kv_transfer_params.do_remote_decode=true`；
2. P 为了完成 vLLM 请求仍会产生一个 disposable sampled token，但 Gateway
   必须丢弃 P response 的整个 `choices`，只取 `kv_transfer_params`；
3. Gateway 将原始 prompt 和 P 返回的 KV transfer metadata 一起发送给 D；
4. D 生成第一个以及后续全部用户可见 token。

这不是可选的展示策略，而是本优化的正确性前提。如果 Gateway 拼接或返回
P token，输出不保证正确，因为 P 的 disposable logits 没有经过 cross
layers。

专用 P 服务使用静态角色判断，以下条件是部署契约：

- 模型为 YOCO，并开启 `--kv-sharing-fast-prefill`；
- P 服务配置 `kv_transfer_config.kv_role=kv_producer`；
- 该服务只接收 P 请求，每个请求均为 `max_tokens=1` 且
  `kv_transfer_params.do_remote_decode=true`；
- D 是独立的 `kv_consumer` 服务，负责 first token 和之后全部用户可见 token。

`kv_producer` 是整个服务所有 rank 共享的静态角色，因此支持 DP1、DP2、
DP4、DP8 等任意 DP 数；它会无条件走 KV-only 路径，不再逐 step 检查请求
metadata。代价是 producer 不能混入普通请求，否则其 logits 不保证正确。
DP>1 的空闲 rank 会执行 runtime dummy forward 以参与 active rank 的 MoE
collective，这个 dummy 也必须使用 KV-only 路径，否则两边 collective 次数
不同会 hang。

为了兼容旧部署，`kv_both` 仍在 DP1 下使用 request-level 条件动态判断；
`kv_both` 的 DP>1 不能证明服务是纯 P 节点，因此保守回退。专用 P 节点应
使用：

```text
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer",...}'
```

### B200 正确性与性能

本机定向回归为 `3 passed`；完整的两个受影响测试文件在独立 B200 Pod
`lidong1-yoco-pr01-b200-g4-0804-master-0`（节点
`slc01-cl02-hgx-0017`）内运行，结果为：

```text
42 passed, 24 warnings in 86.64s
```

受影响文件通过 `ruff-check`、`ruff-format` 和 `git diff --check`。本机完整
运行同一组测试时有 `40 passed`，另两个失败分别来自已被 Dashboard 占用的
`localhost:12345` 和已被其他服务占满的四张 A6000；因此在隔离 B200 Pod
重跑，并确认全部 42 项通过。

端到端 P/D 正确性覆盖 `1,356 / 12,096 / 43,704` prompt tokens。三个
case 中，baseline P/D、candidate P/D 和单体 reference 的最终 D 输出均
exact match。另做了三轮连续 D -> P -> D KV 回传；前一轮 D 输出进入下一轮
prompt，三轮最终输出也都与单体 reference exact match。

下面是相同 B200 节点、相同 YOCO-v3 模型和服务参数下的 warmed A/B；每个
实例为 DP1，指标是 P 侧 prompt throughput。Baseline 和 candidate 都叠加
相同的 NIXL alias 测试补丁，该补丁不属于本 PR，保证唯一变量是本次
KV-only prefill 裁剪。每个 shape 跑五轮，并按轮次交替 A/B 执行顺序；表中
报告五轮中位数：

| prompt / concurrency | baseline tok/s | candidate tok/s | 提升 |
| --- | ---: | ---: | ---: |
| 1.4K / c1 | `12,838` | `16,479` | `28.36%` |
| 12.1K / c1 | `39,315` | `40,895` | `4.02%` |
| 43.7K / c1 | `44,910` | `47,899` | `6.66%` |
| 12.1K / c4 | `49,099` | `51,888` | `5.68%` |
| 12.1K / c8 | `48,814` | `51,104` | `4.69%` |

五个 shape 每轮分别发送 `50 / 10 / 5 / 12 / 16` 个请求。各组吞吐的
coefficient of variation 为 `0.30%` 到 `2.07%`；长 prompt 和并发场景
稳定提升约 `4%` 到 `7%`。原因是 baseline 已经把九个 cross layers 压缩到
logits token，本次主要再移除第一个 KV-owner cross layer 对完整 prompt
的计算，而不是从零裁掉完整十层计算，因此该收益量级符合预期。

#### DP4 专用 producer 补充验证（2026-08-04）

用户侧部署是纯 P 节点，因此 producer 不需要限制为 DP1。本轮在 8×B200
节点 `slc01-cl02-hgx-0331` 上验证专用 `kv_producer` 的 DP4 路径：candidate
P 使用 GPU 0-3，独立 D 和 reference 分别使用 GPU 4、5；正确性完成后停止
D/reference，再在 GPU 4-7 启动 baseline P 做同机 DP4 A/B。P、D 和 baseline
都使用同一套 UCX 1.21 CUDA modules。

受影响的两个单测文件在 B200 上结果为：

```text
43 passed, 24 warnings in 80.85s
```

端到端 P/D 正确性覆盖 `1,356 / 12,096 / 43,704` prompt tokens。三个 case
均由 DP4 P 产出 KV，再由独立 D 生成全部用户可见 token，并与单体 reference
做 exact match，结果全部通过。随后又发送 43 个独立 P 请求（2 个 warmup、
41 个计量请求），覆盖 c1、c4、c8 和最长 43.7K prompt；全部完成，无 hang
或 collective 次数不一致。

性能 baseline 是 `c27db1e189`，candidate 是本 PR HEAD；两边都是
`kv_producer`、DP4、相同模型和相同服务参数，并叠加相同的测试专用 NIXL
alias 修复，该修复不属于 PR。c1 shape 每轮分别发送 `50 / 10 / 5` 个请求；
c4 和 c8 为降低小样本波动，每轮各发送 80 个请求。每个 shape 五轮交替执行
顺序，报告 warmed 吞吐中位数：

| prompt / concurrency | baseline tok/s | candidate tok/s | 提升 |
| --- | ---: | ---: | ---: |
| 1.4K / c1 | `10,419` | `12,507` | `20.04%` |
| 12.1K / c1 | `48,828` | `54,069` | `10.74%` |
| 43.7K / c1 | `64,757` | `70,462` | `8.81%` |
| 12.1K / c4 | `119,388` | `121,266` | `1.57%` |
| 12.1K / c8 | `152,172` | `157,103` | `3.24%` |

正式五轮数据的 coefficient of variation 为 `1.17%` 到 `3.20%`。完整 JSON
保存在
`/mnt/pvc/lidong1/vllm_pd/pr01-dp4-20260804-node0331/alternating-benchmark-warmed.json`
和同目录的 `concurrency-benchmark-warmed.json`。

## NIXL KV-sharing alias 注册修复

这是 KV-only producer 优化之后的独立 follow-up，避免把 NIXL 兼容性修复和
算子裁剪放在同一个 PR：

```text
baseline branch: fhb-dev
baseline commit: 85eab7b56ec4cb1208364bb0daa74d04f09f440e
candidate branch: review/yoco-02-nixl-kv-alias
candidate commit: this PR HEAD
```

### 修改文件

- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`：识别缺少独立
  layer spec、但底层 tensor 地址已经注册的 KV-sharing alias，并跳过重复
  NIXL memory registration。
- `tests/v1/kv_connector/unit/test_nixl_connector.py`：让现有 cache registration
  参数化测试覆盖“alias 在 connector 捕获 KVCacheConfig 后才加入”的真实顺序。
- `yoco.md`：记录问题边界、测试和性能影响。

ModelRunner 会在 NIXL worker 捕获 `KVCacheConfig` 后加入 cross-layer
KV-sharing alias。alias 与 owner layer 指向相同 tensor，但没有独立 layer
spec；旧逻辑会先用 alias 名称索引 spec，因此尚未执行地址去重就抛出
`KeyError`。

新逻辑保持 fail-closed：有 spec 的 layer 完全走原路径；缺少 spec 时，只有
所有 tensor 地址都已经注册才能判定为 alias 并跳过；缺少 spec 的唯一 tensor
仍抛出 `KeyError`，不会掩盖错误配置。

### 测试与性能

定向 cache-registration 单测覆盖 FlashAttention、TritonAttention 和
cross-layer blocks 开关，结果为：

```text
4 passed, 2 skipped, 55 deselected, 14 warnings in 9.32s
```

这份完全相同的两文件补丁已作为测试 overlay 在 B200 producer DP4 验证中
运行：受影响测试集合为 `43 passed, 24 warnings in 80.85s`；四个 producer
worker 均成功初始化 NIXL/UCX，随后完成 1,356、12,096、43,704 prompt
tokens 的 P/D exact-match，以及额外 43 个 c1/c4/c8 请求，无 hang 或
collective mismatch。

这是一次启动期 memory registration 正确性修复，不进入请求稳态热路径。
参数化单测验证 block-first layout 仍只生成 2 个 registration entries，分离
K/V layout 仍为 4 个，alias 新增 registration entries 为 0；因此预期稳态
吞吐变化为 0，不单独复用上一条算子优化 PR 的吞吐收益。

## Router Top-K 简化

这是 NIXL alias 修复之后的第三个独立 PR，只优化 YOCO 128-expert、Top-8
Router，不混入 RMSNorm 或其他算子改动：

```text
baseline branch: fhb-dev
baseline commit: ea4f80d1b4882ffbddc9aa7135863bab38ba0fee
candidate branch: review/yoco-03-router-topk
candidate commit: this PR HEAD
```

### 修改文件

- `vllm/model_executor/models/yoco.py`：Top-K 后直接原地归一化并返回
  `[tokens, 8]` weights/ids，删除 dense routing materialization、scatter、
  routing map 和第二次 `torch.topk`。
- `tests/model_executor/test_yoco_conversion.py`：覆盖 1、3、66、110、256
  token 的随机 logits，以及全相等 logits 的 tie order。
- `yoco.md`：记录功能边界、测试数据和独立性能指标。

数值路径保持 FP32 softmax、`torch.topk` 和 post-top-k renormalization 不变；
没有改 expert 选择和 tie-breaking 语义。原实现先生成 `[tokens, 128]` 的
FP32 `routing_probs` 和 bool `routing_map`，再从 dense tensor 做第二次
Top-K；新实现让第一次 Top-K 的结果直接成为最终输出，并用 Triton kernel
原地归一化。每个 token 明确少分配 `128 * (4 + 1) = 640` bytes 的两个
dense 临时 tensor；第二次 Top-K 的额外 workspace 未计入这个保守值。

### B200 正确性与性能

独立 B200 Pod `lidong1-yoco-pr03-router-g1-0804-master-0`（节点
`slc01-cl02-hgx-0346`）上的 Router 定向测试结果为：

```text
6 passed, 9 deselected, 19 warnings in 4.05s
```

随机 case 的 expert ids 与 baseline exact match，归一化 weights 在
`rtol=2e-6, atol=0` 下通过；全相等 logits 的 expert ids 和 weights 均 exact
match，确认保留 `torch.topk` 的 tie order。

性能是同一张 B200 上的纯 Router microbenchmark。两边使用相同 FP32 logits、
相同 softmax kernel 和相同 Top-K 参数；分别捕获为 CUDA Graph，warmup 后每个
shape 计时 1,000 次 graph replay，报告中位延迟。这样仍包含实际 GPU kernel
和显存读写成本，但不把 Python/kernel-launch 开销误算成主要收益：

| tokens | baseline (us) | candidate (us) | 加速 | 少分配 dense 临时内存 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `24.704` | `16.096` | `1.535x` | 640 B |
| 3 | `28.864` | `15.968` | `1.808x` | 1,920 B |
| 66 | `30.784` | `17.824` | `1.727x` | 42,240 B |
| 110 | `32.288` | `18.336` | `1.761x` | 70,400 B |
| 256 | `34.768` | `20.112` | `1.729x` | 160 KiB |
| 1,024 | `38.912` | `22.048` | `1.765x` | 640 KiB |
| 4,096 | `73.248` | `40.320` | `1.817x` | 2.5 MiB |
| 16,384 | `214.944` | `116.480` | `1.845x` | 10 MiB |

这里报告的是 Router 单算子的 CUDA Graph 数据，不复用此前
Router + fused add-RMSNorm 的端到端收益，也不外推为整模型吞吐提升。

## Fused residual add-RMSNorm

这是 Router 简化之后的第四个独立功能，只融合 decoder layer 内的 attention
residual add 与 post-attention RMSNorm：

```text
baseline branch: fhb-dev
baseline commit: d77fed9d2a6e6b84a7a1573b6a0a868667648e95
candidate branch: review/yoco-04-fused-add-rmsnorm
candidate commit: this commit
```

### 修改文件

- `vllm/model_executor/models/yoco.py`：增加 YOCO 专用 mixed-dtype Triton
  add-RMSNorm，并接入每个 decoder layer 的 attention 后融合点。
- `tests/model_executor/test_yoco_conversion.py`：覆盖 CPU fallback、B200
  fast path、FP32 residual、BF16 normalized output 和完整 layer boundary。
- `yoco.md`：记录功能边界、正确性和独立性能。

YOCO 的 sublayer output 是 BF16/FP32，长期 residual 是 FP32，RMSNorm output
必须是 BF16。vLLM 通用 fused add-RMSNorm 要求 input、residual 和 weight 同
dtype，并原地更新同 dtype residual，因此不能直接复用。本次 kernel 保持原顺序：

```text
residual_out = residual_fp32 + attention_output.float()
normalized = RMSNorm(residual_out) -> BF16
```

第一遍同时执行 FP32 add、写出完整 `residual_out` 并累加平方和；第二遍读取
`residual_out`、应用 BF16 weight 并输出 BF16 normalized tensor。非 CUDA、
shape/dtype 不匹配或 hidden size 不是 3072 时走相同公式的 PyTorch fallback。

旧实验曾把 MLP output 和 residual 拆开跨 decoder layer 携带，以争取下一层继续
融合，但这会改变 layer boundary 的 FP32 物化/舍入点。本提交没有带入该实验；
每层末尾仍执行原来的 `residual + mlp_output.float()`，只保留层内一个安全融合点。

### B200 正确性

独立 Pod `lidong1-yoco-pr04-add-rmsnorm-g1-0804-master-0`（节点
`slc01-cl02-hgx-0418`）上的定向测试结果为：

```text
5 passed, 15 deselected, 19 warnings in 2.52s
```

完整 YOCO conversion/model test 文件结果为：

```text
20 passed, 19 warnings in 20.55s
```

CUDA case 覆盖 1、66、128 tokens；`residual_out` 和 normalized output 都与顺序
实现 bitwise match（`rtol=0, atol=0`）。完整 decoder layer 测试连续执行两个
loop，证明每层末尾 residual boundary 未被移动。编译版 microbenchmark 还对
1/3/66/110/128/256/1024/4096 tokens 逐 shape 做 exact-match 后才计时。

### B200 CUDA Graph 性能

baseline 是 torch.compile 后的 FP32 add + 原 YOCO RMSNorm，candidate 是本次
mixed-dtype fused op。两边分别 capture 为 CUDA Graph，warmup 后计时 1,000 次并
报告中位数。

单独 replay 一个微算子 graph 时，约 9 us 的整图固定开销会掩盖小 shape 的 GPU
work 差异；它仍能证明大 prefill shape 的独立收益：

| tokens | baseline (us) | candidate (us) | 加速 |
| ---: | ---: | ---: | ---: |
| 1 | `9.568` | `9.152` | `1.045x` |
| 3 | `9.472` | `9.184` | `1.031x` |
| 66 | `9.536` | `9.472` | `1.007x` |
| 110 | `9.408` | `9.408` | `1.000x` |
| 128 | `9.376` | `9.376` | `1.000x` |
| 256 | `9.472` | `9.472` | `1.000x` |
| 1,024 | `17.184` | `11.136` | `1.543x` |
| 4,096 | `48.032` | `31.136` | `1.543x` |

YOCO 每个 token step 有 40 次 decoder block execution。为了不把单次 graph
replay 固定开销重复算 40 次，第二个口径在同一 graph 中放置 40 个算子节点，再
用总延迟除以 40；输入彼此独立，所以它只代表算子节点摊销，不代表整模型：

| tokens | baseline/op (us) | candidate/op (us) | 加速 |
| ---: | ---: | ---: | ---: |
| 1 | `3.217` | `2.162` | `1.488x` |
| 3 | `3.376` | `2.202` | `1.533x` |
| 66 | `3.898` | `2.403` | `1.622x` |
| 110 | `4.505` | `2.456` | `1.834x` |
| 128 | `4.604` | `2.300` | `2.002x` |
| 256 | `6.444` | `2.864` | `2.250x` |
| 1,024 | `18.273` | `6.629` | `2.756x` |
| 4,096 | `68.862` | `26.533` | `2.595x` |

以上是 fused add-RMSNorm 单功能数据。旧分支 Router + Norm 的端到端吞吐变化没有
作为本提交收益复用；整模型收益仍需由后续独立 A/B 判断。

## Shared Expert 与 Routed MoE 并行

这是 fused add-RMSNorm 之后的第五个独立功能，只调整 YOCO MoE 内部的
shared/routed 调度，不改变 Router、expert 权重、Top-K 或数学顺序：

```text
baseline branch:  fhb-dev
baseline commit:  cd1e22c7ff18883dc2b45b7878c70c370c84cbac
candidate branch: review/yoco-05-shared-expert-overlap
candidate commit: 9988ae737fc717a9ba1846867667e6ecf441299d
```

原路径先在当前 CUDA stream 完整执行 routed experts，再串行调用
shared expert。本次把 YOCO shared expert 作为 `FusedMoE` 的 shared path，
复用现有 auxiliary CUDA stream，使 shared expert GEMM 与 routed dispatch/expert
compute 重叠。

为了保持已验证的 YOCO-v3 语义，并行只改变计算时间线，不合并两条
TP reduction：

```text
routed latent input projection + norm
  -> routed expert compute -------------------+
shared expert compute on auxiliary stream ----+--- synchronize
  -> routed TP reduction
  -> shared TP reduction
  -> routed latent output norm + projection
  -> sigmoid(shared_gate(hidden_states)) * reduced_shared
  -> routed + shared
```

shared sigmoid gate 在 shared TP reduction 后执行。`shared_gate` 是 replicated
weight，因此这与原路径数学等价，同时不会把 gate 加入 auxiliary stream 的
跨 stream 依赖。

### 修改文件

- `vllm/model_executor/layers/fused_moe/layer.py`：为 `FusedMoE` 增加
  `shared_output_transform` 和 `reduce_shared_experts_separately`；要求独立
  shared reduction 时却未配置 shared experts 会立即报错。
- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`：在现有 auxiliary
  CUDA stream 上执行 shared expert GEMM；为 YOCO 保留 routed-first、
  shared-second collective 顺序，并在 reduction 后执行模型专用 shared
  output transform。未启用新选项的其他模型仍走原有合并路径。
- `vllm/model_executor/models/yoco.py`：shared expert down projection 保留
  TP-local output；把 routed latent projection/norm 包装成 `FusedMoE` transform；
  在 shared reduction 后应用 YOCO sigmoid gate。
- `tests/model_executor/test_yoco_conversion.py`：增加 latent transform 顺序、
  shared gate 位置和 routed/shared collective 顺序测试。

功能 commit 只修改上述 4 个文件，总计 `223 insertions, 39 deletions`。

### 正确性验证

本地 NVIDIA RTX A6000 回归：

```text
23 passed, 14 warnings in 12.75s
```

独立 B200 Job：

```text
experiment: fhb-moe7-0804
job:        b200
pod:        fhb-moe7-0804-b200-564c6e49-master-0
commit:     9988ae737f
baseline:   cd1e22c7ff
```

B200 回归结果：

```text
68 passed, 24 warnings in 88.70s
```

baseline 和 candidate 都成功加载 YOCO-v3 BF16，使用 TP1/DP1、
FlashInfer attention、Triton MoE、prefix caching、chunked prefill 和 KV-sharing
fast prefill。两边均未使用 eager，启动日志确认为 `FULL_AND_PIECEWISE`
CUDA Graph。

端到端正确性输入的 target prompt 长度为 1,360、4,096 和 7,000
tokens；每个请求使用 `temperature=0`、固定生成 16 tokens。baseline 与
candidate 的以下字段全部 exact match：

- 生成文本、finish reason 和 usage；
- 每个 token id 与 token logprob；
- 每步 top-5 token 和 top-5 logprob。

服务日志无 CUDA、NCCL、collective mismatch 或 Python traceback。本次实机
端到端是 TP1；TP>1 的 routed-then-shared collective 顺序由单元测试覆盖，
后续扩大 TP 部署前仍应做一次真实多卡回归。

功能文件还通过全部适用的 pre-commit hooks，包括 ruff、mypy、
SPDX header、配置检查和 `git diff --check`。

### B200 独立性能 A/B

性能请求使用同一个 random input，分词后实际为 1,299 input tokens，
固定生成 128 tokens。并发 1/4/8 分别发送 4/16/32 个请求；每个档位
执行 3 轮，报告三轮整体 output throughput 的中位数。每轮输入、
输出长度、服务参数和 B200 硬件保持一致。

| 并发 | baseline tok/s | candidate tok/s | 变化 |
| ---: | ---: | ---: | ---: |
| 1 | `136.20` | `141.99` | `+4.25%` |
| 4 | `375.26` | `381.04` | `+1.54%` |
| 8 | `720.69` | `718.50` | `-0.30%` |

收益集中在低、中并发；并发 8 的 `-0.30%` 在本次三轮测量的噪声范围内，
不应解读为高并发收益。这组 endpoint 数据只切换 commit
`cd1e22c7ff -> 9988ae737f`，没有复用之前 Router 或 Norm 的数据。

原始 B200 产物保存于：

```text
/mnt/pvc/lidong1/vllm_pd/shared-expert-overlap-fhb-dev-0804/
fhb-moe7-0804-b200-564c6e49-master-0
```

### 边界、风险与回滚

- 新选项默认关闭，只有 YOCO 显式请求 separate reduction，其他
  `FusedMoE` 调用者不改变。
- 正确性边界是“并行本地 GEMM，串行 routed/shared collective”；不应
  为进一步 overlap 而合并两次 all-reduce，否则会改变 FP32/BF16 reduction
  舍入边界。
- shared gate 必须在 shared output reduction 后、最终求和前执行；位置
  变化要求重做 token/logprob exact-match。
- revert `9988ae737f` 可恢复原先的串行 YOCO MoE；Router、
  fused add-RMSNorm、PD 和 NIXL 提交不受影响。

## Shared Expert FP32 clamped-SwiGLU 单 kernel

这是 Shared/Routed overlap 之后的第六个独立功能，只融合 YOCO
Shared Expert activation：

```text
baseline branch:  fhb-dev
baseline commit:  2e1b8ecc53705574ec38597dcdae81c939b40158
candidate branch: review/yoco-06-fp32-clamped-swiglu
candidate commit: 8eae22948cd627e924d9583f1946eb2b88e8b9eb
```

原路径为了与训练保持一致，会把 BF16/FP16 `gate_up` 先转成 FP32，再执行
gate clamp、up clamp、SiLU、multiply，最后转回 projection dtype。CUDA eager
profile 中这是 6 个 kernel。已有 `SiluAndMulWithClamp` 会在 BF16/FP16
路径提前舍入 clamp 和 SiLU 中间结果，因此不能直接替换。

新 `_C.silu_and_mul_with_clamp_fp32` 从低精度 input load 后，在 FP32 中一次
完成：

```text
gate = clamp(gate_fp32, max=limit)
up = clamp(up_fp32, min=-limit, max=limit)
output = (silu(gate) * up).to(input_dtype)
```

只在最终 store 时做一次 dtype conversion。新 kernel 支持 128-bit vector
load/store；CUDA 12.9+、SM100+ 且 tokens > 128 时进入 256-bit 路径，非对齐
hidden size 使用 scalar fallback。clamp 使用比较表达式而不是
`fminf/fmaxf`，以保留 PyTorch `torch.clamp` 的 NaN 语义。

### 修改文件

- `csrc/activation_kernels.cu`：增加 scalar/packed FP32 compute、单 kernel 与
  128/256-bit launch dispatch。
- `csrc/ops.h`：声明新 CUDA op。
- `csrc/torch_bindings.cpp`：注册
  `_C.silu_and_mul_with_clamp_fp32`。
- `vllm/model_executor/layers/activation.py`：增加
  `SiluAndMulWithClampFP32` CustomOp；CUDA 走单 kernel，CPU/ROCm/XPU 保留
  FP32 PyTorch reference。
- `vllm/model_executor/models/yoco.py`：用新 op 替换原来的多算子
  `YOCOClampedSwiGLU`；`swiglu_limit <= 0` 时仍走普通 `SiluAndMul`。
- `tests/kernels/core/test_activation.py`：增加 FP16/BF16/FP32、两张 GPU、
  scalar/128-bit/SM100 分支输入的 bitwise 与 opcheck 覆盖。

功能 commit 只修改上述 6 个文件，总计 `253 insertions, 18 deletions`。
原有 `SiluAndMulWithClamp` 没有修改，避免改变其他模型的低精度中间语义。

### 当前分支 A6000 正确性

为当前 clean-history 分支用 CUDA 13.0、PyTorch 2.11 和 SM86 重新编译
`_C`。activation matrix 在 GPU0/1 执行：

```text
30 passed, 468 deselected, 14 warnings in 15.11s
```

覆盖 FP16/BF16/FP32、`d=1279` scalar fallback、`d=1280`、1/7/128/129
tokens。每个 case 都要求与 FP32 多算子 reference bitwise match
(`rtol=0, atol=0`) 并通过 PyTorch opcheck；低精度 case 还确认新结果与
现有低精度中间 kernel 不同，防止接错 op。

YOCO conversion/config 与前五项优化的组合回归：

```text
34 passed, 14 warnings in 18.95s
```

全部功能文件通过 ruff、mypy、clang-format、SPDX、配置检查和
`git diff --check`。

### 当前分支 A6000 性能

BF16、`d=1280`，每个 sample 执行 400 次、7 轮取中位数。baseline 是
FP32 多算子 reference，candidate 是新单 kernel：

| tokens | FP32 多算子 eager | fused eager | 加速 |
| ---: | ---: | ---: | ---: |
| 1 | `107.771 us` | `18.611 us` | `5.79x` |
| 8 | `103.790 us` | `19.855 us` | `5.23x` |
| 128 | `110.897 us` | `18.895 us` | `5.87x` |

CUDA Graph 每个 sample 执行 1,000 次 replay、7 轮取中位数；计时前两个
graph 输出先做 bitwise match：

| tokens | FP32 多算子 graph | fused graph | 加速 |
| ---: | ---: | ---: | ---: |
| 129 | `14.402 us` | `4.833 us` | `2.98x` |
| 4,096 | `488.818 us` | `53.897 us` | `9.07x` |

这些是 activation microbenchmark，不等价于整模型 tok/s。graph 结果也说明
优化不是依赖“没开 CUDA Graph”：收益来自 graph 内部从 6 个 GPU
kernel node 减少到 1 个。

### 已验证原 patch 的 B200 数据

本次移植的 6 个功能文件与旧验证 commit
`f265362d2905216c81b8571db20d19f16023a344` 逐文件完全一致；旧 baseline
`21f2066621` 与当前 baseline 在这 6 个文件上也完全一致。因此可将
以下 B200 数据用作“相同 patch 的历史实机验证”，但不冒充为 commit
`8eae22948c` 的新 B200 重跑。

两张 B200 分别为 SM100 重编译完整 `_C`：30 个 correctness 和 30 个
opcheck 全部通过；覆盖 scalar、128-bit、256-bit、NaN、CustomOp、
16,384-token long batch 和更换 static input 后的 CUDA Graph replay，全部
bitwise match。

旧独立 endpoint A/B 使用 YOCO-v3 BF16、TP1、FlashInfer、Triton MoE 和
`FULL_AND_PIECEWISE` CUDA Graph；两个 GPU 布局交换后各执行 3 轮，共
6 个 sample 取中位数：

| 并发 | baseline completion tok/s | fused completion tok/s | 变化 |
| ---: | ---: | ---: | ---: |
| 1 | `133.521` | `136.044` | `+1.89%` |
| 4 | `351.494` | `354.833` | `+0.95%` |
| 8 | `749.076` | `756.981` | `+1.06%` |

1,360/12,097/43,709-token prompt 各固定生成 64 tokens；文本、token id、
token string、token logprob 和 top-5 logprob 在两个 GPU 布局都 exact match。
c1/c4 在两个布局均为正收益；c8 分别为 `+2.36% / +0.08%`，因此
约 1% 的 pooled c8 结果已接近噪声，不应作为 SLA 承诺。原始数据：

```text
/mnt/pvc/lidong1/vllm_pd/fp32-clamped-swiglu-0804/
```

### 风险与回滚

- 数值契约是 FP32 clamp -> FP32 SiLU -> FP32 multiply -> 一次 output
  conversion；不能与已有 low-precision-intermediate op 合并。
- vector 分支与 128-token/SM100 边界必须保留 bitwise、NaN 和 graph
  replay 测试；改 tile 或 vector width 后不能只跑 allclose。
- YOCO 使用 `enforce_enable=True`，防止 compile 路径展开回多 kernel；后续
  修改 CustomOp 调度时需检查 profiler 中仍只有一个 activation node。
- revert `8eae22948c` 可恢复 YOCO Shared Expert 的 FP32 多算子路径；
  前五个 `fhb-dev` 功能提交不受影响。

## Streaming stop 保留 KV transfer metadata

这是第七个独立功能，是 P/D streaming 正确性修复，不是算子性能
优化：

```text
baseline branch:  fhb-dev
baseline commit:  4b0f7d3d4249b7d0629619188a2dcc16bda103d9
candidate branch: review/yoco-07-streamed-stop-kv-metadata
candidate commit: 1c51cac0d7fc374475c97a932dd3d37b9ab9dfaf
```

之前，如果 stop string 由 frontend detokenizer 检测到，EngineCore 尚未标记请求
finished，frontend 会把请求当作 abort 发回 core。abort 可以释放 KV block，但不会
把 connector `_free_request()` 生成的 `kv_transfer_params` 送回 API 层。在 P/D
路径中，这会让最终 streaming chunk 丢失远程 engine/block metadata。

新路径把 frontend stop 从“abort”改为“正常 STOP”：

```text
frontend detects multi-token stop string
  -> keep RequestState and pending stop reason
  -> send EngineCoreRequestType.STOP
  -> scheduler FINISHED_STOPPED + connector free
  -> EngineCore emits final empty-token output with kv_transfer_params
  -> OutputProcessor restores stop reason and finishes request
  -> chat/completion final stream chunk serializes kv_transfer_params
```

abort 路径仍保留给客户端取消等真正 abort 场景。只有 frontend 因 stop
string 成功结束请求时使用 STOP。

### 修改范围

功能 commit 修改 14 个文件，`207 insertions, 25 deletions`。文件数看似
较多，但只实现一条原子链路：

- scheduler interface/implementation：在 finish 时返回 connector metadata；
- EngineCore request enum/core/client：新增 sync、async、MP 和 DP-LB STOP 传递；
- output processor/AsyncLLM/LLMEngine：暂存 stop reason，等 core 终态后再
  生成用户可见 final output；
- chat/completion protocol/serving：只在 response 有 metadata 时序列化
  `kv_transfer_params`；
- 测试：覆盖 stop-string 两阶段终态和 chat/completion streaming schema。

这些层不能拆成可独立合入的多个 commit：只加 API 字段没有 metadata
来源，只加 core STOP 而不保留 frontend state 会丢 stop reason，只改 output
processor 而不改 client 则 MP/DP 路径无法传递。

### 正确性测试

所有功能文件通过 ruff、mypy、SPDX、配置检查和 `git diff --check`。

当前环境无权访问 gated `meta-llama/Llama-3.2-1B` fixture，所以使用本地
OPT-125M tokenizer 重建相同 dummy output vectors，只运行本修复直接相关的
case，不修改测试断言：

```text
6 passed, 30 deselected, 22 warnings in 8.03s
```

其中包括 4 个 stop-string case（有/无 logprobs，有/无 include stop string）和
2 个 chat/completion streaming serialization case。stop-string case 会模拟 core 返回带
metadata 的最终空 token output，并要求 final `RequestOutput` 保留 stop reason 和
`kv_transfer_params`。

另外直接用 fake scheduler 调用 `EngineCore.stop_requests()`，验证：

- status 是 `FINISHED_STOPPED`，不是 `FINISHED_ABORTED`；
- 输出路由到正确 `client_index`；
- final output 为空 `new_token_ids`、`FinishReason.STOP`；
- connector `remote_block_ids` metadata 不变地返回。

### 性能说明

这是 correctness commit，没有吞吐提升目标，不报告 GPU/tok/s 收益。普通
token generation、token-id stop、length stop 和真正 abort 路径没有新的 steady-state
work。

只有 frontend multi-token stop-string 终态多一次 STOP control message 和一个空 token
final output。这可能增加微小的 terminal latency，但是让 connector 在正常 finish
时产生并返回 KV metadata 的必要正确性成本；不能用 abort 的较短路径换取
错误响应。

### 边界与回滚

- `pending_stop_reason` 只用于非 streaming-input 的 frontend stop-string 等待阶段；
  streaming output 与 streaming input 是两个不同概念。
- final empty-token output 必须把 delta logprobs 设为空列表；否则 Python
  `[-0:]` 会错误返回全部历史 logprobs。
- streaming response 字段默认 `None`，普通 OpenAI-compatible response 在
  `exclude_none` 时不增加额外字段。
- revert `1c51cac0d7` 可恢复旧 abort 行为，但 P/D frontend stop-string 会再次
  丢失 KV transfer metadata；六个 YOCO 优化 commit 不受影响。

## 统一 UCX 1.21 PD runtime

这是第八个独立功能，目标是让 NIXL 和 HPC-X 只加载一套 UCX，并让当前
YOCO Python/native 源码在同一个 B200 runtime 内保持一致：

```text
baseline branch:  fhb-dev
baseline commit:  0f8c9d95b550179b662f67d184b12c0688de8ad0
candidate branch: review/yoco-08-ucx-121-runtime
candidate commit: 2765c22a1b1572e2a9fc8dc16465c04a6dfd47a7
```

最终本地镜像：

```text
tag:      vllm-yoco-pd:ucx121-fhb-dev-0804-r2
image id: sha256:02caed17c8793830bd88748baa8fb203489384b1a00263d3312162afb54c0f47
size:     38,002,453,833 bytes
```

### 修改文件

- `docker/Dockerfile.b200.pd`：固定 base/UCX/NIXL revision；原位替换
  HPC-X UCX；从源码构建无私有 UCX 副本的 NIXL wheels；为 SM100 重编译
  当前 `_C`；在 build 中执行版本、符号和 Python 检查。
- `docker/verify_single_ucx.sh`：fail-closed 检查 UCX 版本、重复 library、
  NIXL plugin RUNPATH/实际链接；可选 PID 参数还会检查进程已加载的 UCX
  路径。

功能 commit 只增加这两个文件，`299 insertions`，没有修改模型、scheduler
或请求协议。

### 固定版本和单 UCX 契约

```text
base image digest:
  sha256:08a08f36ab8c6c80ee1c7f09b9e5f8b6ce0b91cc684455982b2e5e286c736f2b
base native revision:
  f4964c907db7ce2d77c2b0ea39e263375b7eba4f
UCX:
  1.21.0 @ b6a9d47fccce849c28111f05a7fa8f1c930ff17d
NIXL / NIXL-cu13:
  1.3.2 @ de8115ca97d3f8fb63a4988e9b4d4a038b2e0f72
native architecture:
  TORCH_CUDA_ARCH_LIST=10.0
```

镜像不会在 UCX 1.20 旁边再安装 1.21，而是删除 inherited
`/opt/hpcx/ucx`，再把 1.21 安装回相同 prefix。这样 HPC-X 既有路径不变，
文件系统中也不会同时存在两个版本。

NIXL platform wheel 本地构建且不经过 auditwheel repair。UCX plugin 保留
generic `libuc*.so` SONAME，并把 `/opt/hpcx/ucx/lib` 写入 RUNPATH；不会把
hash 命名的私有 UCX library 打进 wheel。runtime 中先卸载旧
`nixl/nixl-cu12/nixl-cu13`，再安装同一源码生成的 `nixl==nixl-cu13==1.3.2`。
本镜像只需要 NIXL KV transfer，所以关闭 NIXL EP，并隐藏 metadata wheel
仍会安装的 `nixl_ep` shim；实测 `has_nixl_ep() == False`。

### 为什么必须重编译 `_C`

首个 UCX/NIXL 镜像 `765b69b65871...` 只用 `COPY vllm` 覆盖 Python tree，
仍沿用 base image 在 `f4964c9` 编译的 `_C.abi3.so`。检查立即得到：

```text
hasattr(torch.ops._C, "silu_and_mul_with_clamp_fp32") == False
```

这意味着第六项新增的 `SiluAndMulWithClampFP32` Python 类存在，但对应 C++/CUDA
operator 不存在；YOCO Shared Expert 启动时会失败。该镜像只用于 UCX/NIXL
诊断，没有作为最终 runtime。

最终 Dockerfile 在 pinned base source 上只覆盖当前分支相对 base 变化的三个
native 文件：

```text
csrc/activation_kernels.cu
csrc/ops.h
csrc/torch_bindings.cpp
```

随后只构建和安装 CMake `_C` target，保留 base 的 `_moe_C`、FlashMLA 等
其他 native extension。构建前检查 base Git revision 和三个文件未被 base
镜像额外修改；构建后通过 CUDA stub import 检查新 operator 已注册。最终
`_C.abi3.so` SHA256 为：

```text
c3faa036746f43a50a10e4013021613b851f84b79e0de5c15bf39c7fdef4b298
```

`cuobjdump` 确认该文件包含 `sm_100` cubin。

### B200 正确性和运行态验证

最终镜像先在本机 A6000 上验证 operator 注册、NIXL UCX backend 和 PID
maps；因为 `_C` 是纯 SM100，该机器不用于执行 fused kernel。

B200 使用独立 Volcano Pod：

```text
pod:  lidong1-yoco-ucx121-native-g1-0804-master-0
node: slc01-cl02-hgx-0346
GPU:  NVIDIA B200, compute capability 10.0
```

集群没有发布本地临时 tag，因此从最终镜像提取 216 MiB runtime delta，覆盖到
同一 pinned base digest；覆盖后再次比较 `_C` SHA256，确认与最终镜像逐字节
相同。验证结果：

- FP16/BF16/FP32，`d=1280`，tokens `1/7/128/129/4096` 共 15 个 case，
  fused output 与 FP32-intermediate native reference 全部 bitwise match；
- BF16、129-token operator `opcheck` 通过；
- YOCO `SiluAndMulWithClampFP32` CustomOp 路径 bitwise match；
- CUDA Graph capture 后替换 static input 并 replay，结果 bitwise match；
- NIXL 1.3.2 成功实例化 UCX backend；
- `verify-single-ucx <pid>` 确认进程 maps 中只有 `/opt/hpcx/ucx/lib`；
- `has_nixl_ep() == False`。

另外，`EXPECTED_UCX_VERSION=9.99` 和向 `/usr/local/lib` 注入额外
`libucs.so.duplicate-test` 都被校验脚本拒绝，证明检查是 fail-closed，不是
只打印版本。

### RDMA device 选择边界

测试节点暴露了 12 个有有效 InfiniBand GID 的 `mlx5_*` HCA，以及一个 active
但没有 GID 的 Ethernet `mlx5_bond_0`。不设置拓扑变量时，UCX 自动选择
`mlx5_bond_0:1`，NIXL 初始化按预期失败并报告：

```text
uct_iface_open(rc_verbs/mlx5_bond_0:1) failed: Address not valid
```

设置 `UCX_NET_DEVICES=mlx5_0:1` 后，RDMA backend 和 PID maps 验证立即通过。
因此镜像故意不硬编码 HCA；P/D launch profile 必须按 GPU/NIC 拓扑给每个进程
选择有效、GPU-local 的 RDMA device。统一 library 版本能消除 ABI/loader
混用，但不能替代部署层的网卡选择。

### 性能、风险与回滚

这是 build/runtime 稳定性功能，不改变请求 steady-state 算法，也不把第六项
activation 的历史性能重复归因到本 commit，所以不报告 tok/s 提升。UCX/NIXL
端到端带宽和 P/D endpoint 吞吐应在真实多节点拓扑下单独验收，不能用单 agent
初始化冒充性能数据。

- 当前 `_C` 只面向 B200/SM100；A6000 可检查注册但不能执行该 cubin。
- 后续若新增其他 native 源文件，必须更新 Dockerfile overlay 列表并重编译；
  只覆盖 Python tree 会再次造成 ABI/operator 不一致。
- `NIXL EP` 被有意关闭；该镜像支持 NIXL KV transfer，不宣称支持 NIXL EP MoE。
- `UCX_NET_DEVICES` 是 deployment contract，必须按节点拓扑配置。
- revert `2765c22a1b` 只移除新 image recipe 和校验脚本，不影响前七个
  `fhb-dev` 功能；已构建镜像也不会被 Git revert 自动删除。

## B200 YOCO BF16 Triton MoE 调优配置

这是第九个独立功能，只为 B200 上形状为 `E=128,N=1280` 的非量化 Triton
FusedMoE 增加经过复测的 tile 配置：

```text
baseline branch:  fhb-dev
baseline commit:  66a87747fafb725f2c3c317adb5acc46e6b928ab
candidate branch: review/yoco-09-b200-moe-config
candidate commit: 7c03e0cb730bb11a960eb82a30e73185a743508a
```

### 修改文件和命中范围

功能 commit 只新增一个文件，`131 insertions`：

```text
vllm/model_executor/layers/fused_moe/configs/
  E=128,N=1280,device_name=NVIDIA_B200.json
```

文件记录 16 个 batch-token 配置点和生成配置时使用的 Triton 版本：

```text
triton_version: 3.7.1
tokens: 1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 84, 96, 128, 256, 512, 1024
```

这里的 `N` 是每个 tensor-parallel partition 的 expert intermediate dimension，
即 `w2.shape[2]`。vLLM 只有在运行设备名规范化为 `NVIDIA_B200`、expert 数为
128、`N=1280` 且 dtype selector 为空时才会加载该文件；FP8、INT8/INT4、其他
GPU、其他 expert 数和其他 intermediate size 不会命中。batch token 不在表中时，
loader 使用最近的配置点。

旧 `shaohanh/yoco-0731` commit 把该 JSON 与 Router、RoPE、differential
attention、benchmark 和 Dockerfile 混在一个 884-line commit 中。本次没有
cherry-pick 该 commit，也没有迁移它的 Router 或 Dockerfile；只把 MoE 配置
作为单一、可回滚的功能重新验证。

### 旧配置复核和 token=1 修正

旧 JSON 标注 Triton 3.7.1；独立 B200 runtime 实测也是 Triton 3.7.1，因此没有
跨版本直接复用。首先逐字移植旧文件并覆盖全部 16 个配置点。结果只有 token=1
稳定回退：当前默认配置中位数 `65.83 us`，旧 tile `66.17 us`，回退约
`0.52%`。

因此最终文件没有原样保留旧 token=1 参数，而是改用当前默认 tile：

```text
BLOCK_SIZE_M=16
BLOCK_SIZE_N=64
BLOCK_SIZE_K=128
GROUP_SIZE_M=1
num_warps=4
num_stages=4
```

修正后 token=1 的 baseline/candidate 配置相同；7 轮中位数为
`65.84/65.92 us`，差异 `-0.13%`，属于同一 kernel 配置的测量噪声。其余 15 个
旧配置点全部为正收益或持平，因此保留。

### B200 正确性

独立资源：

```text
pod:     lidong1-yoco-moe-config-g1-0805-master-0
node:    slc01-cl02-hgx-0380
GPU:     NVIDIA B200, compute capability 10.0
PyTorch: 2.11.0a0+eb65b36914.nv26.02
Triton:  3.7.1
shape:   E=128, N=1280, K=3072, Top-K=8, BF16
```

Pod 使用固定 base image digest
`sha256:08a08f36ab8c6c80ee1c7f09b9e5f8b6ce0b91cc684455982b2e5e286c736f2b`，
覆盖当前 `fhb-dev` Python tree；candidate 加载日志明确指向新增 JSON。baseline
通过当前 `get_default_config` 生成，candidate 通过正常 `get_moe_configs` 文件
查找和最近 batch-key 选择生成。两者复用完全相同的 input、expert weights、
router logits、Top-K ids 和 Top-K weights。

最终修正版覆盖 tokens `1/8/32/128/512/1024`：

| Tokens | Output elements | Mismatches | Max abs diff |
| ---: | ---: | ---: | ---: |
| 1 | 3,072 | 0 | 0 |
| 8 | 24,576 | 0 | 0 |
| 32 | 98,304 | 0 | 0 |
| 128 | 393,216 | 0 | 0 |
| 512 | 1,572,864 | 0 | 0 |
| 1024 | 3,145,728 | 0 | 0 |

总计 `5,237,760` 个 BF16 output element 全部 bitwise exact；不是只用宽松
`rtol/atol` 判定。

### B200 CUDA Graph kernel 性能

性能口径为当前 `benchmarks/kernels/benchmark_moe.py::benchmark_config`：

- 每个配置先 JIT 并完成独立 warmup；
- CUDA Graph 每次 replay 包含 10 次完整 routed FusedMoE 调用；
- 每个 sample 做 100 次 graph replay，即 1,000 次 kernel path；
- baseline/candidate 每轮使用同一随机 seed，轮间交替执行顺序；
- 全 16 点各 5 轮取中位数；修正后的 token=1 另做 7 轮复测。

| Tokens | Baseline us | Candidate us | 提升 |
| ---: | ---: | ---: | ---: |
| 1 | 65.84 | 65.92 | -0.13%（同配置噪声） |
| 2 | 100.10 | 95.30 | 5.03% |
| 4 | 152.13 | 142.73 | 6.58% |
| 8 | 226.38 | 220.06 | 2.87% |
| 16 | 331.77 | 326.06 | 1.75% |
| 24 | 398.91 | 393.70 | 1.32% |
| 32 | 437.48 | 432.03 | 1.26% |
| 48 | 512.25 | 468.02 | 9.45% |
| 64 | 532.17 | 483.84 | 9.99% |
| 80 | 556.01 | 488.58 | 13.80% |
| 84 | 557.09 | 490.96 | 13.47% |
| 96 | 560.50 | 493.19 | 13.65% |
| 128 | 550.70 | 498.51 | 10.47% |
| 256 | 566.21 | 554.66 | 2.08% |
| 512 | 587.74 | 587.49 | 0.04% |
| 1024 | 735.99 | 689.78 | 6.70% |

这些数字是 isolated routed-MoE CUDA Graph latency，不是 endpoint tok/s，也不把
Router、Shared Expert overlap、activation 或通信收益归因到本 commit。该 Pod
没有挂载 `E=128,N=1280` 的完整 YOCO checkpoint，因此本次不报告模型端到端
吞吐；真实收益还取决于线上 batch-token 分布以及该 MoE 形状是否实际命中。

### 静态检查、边界与回滚

通过：

- JSON `jq empty`；
- B200 正常 loader 命中和全部 16 个 key 执行；
- 修正版六种 token 数 bitwise correctness；
- pre-commit 的 typos、filename、Docker dependency graph、configuration
  validation、attention docs 和 suggestion 等所有适用 hook；
- `git diff --check`。

部署观察点：启动日志应出现新增文件的完整路径；如果出现 default MoE config
warning，说明 device name、`E/N` 或 dtype selector 不匹配，不能把本表性能当作
实际收益。`triton_version` 当前只是 provenance，loader 不会据此拒绝不同版本；
升级 Triton/PyTorch 后必须重新 A/B，不能仅凭 JSON 仍可加载就认为性能有效。

revert `7c03e0cb73` 可完全删除该配置并恢复运行时 default config，不影响前八个
`fhb-dev` 功能，也不需要重编译 `_C` 或重做 UCX/NIXL runtime。

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

## YOCO BF16 RoPE 单 kernel

本轮从 `origin/shaohanh/yoco-0731` 独立迁移 RoPE 优化，没有带入该分支的旧
Router、differential-attention、Dockerfile 或其他混合修改。功能 commit：

```text
8abfd6c6d0 perf(yoco): fuse BF16 rotary embedding
```

### 修改文件和适用范围

- `vllm/model_executor/models/yoco.py`：新增 `torch.ops.vllm.yoco_rotary`
  CustomOp 及 Triton kernel，并在 `YOCORotaryEmbedding` 中接入；
- `tests/model_executor/test_yoco_conversion.py`：增加真实 packed-QKV stride、
  两种 head layout、多个 token shape、bitwise 对比和 CustomOp `opcheck`。

kernel 只接管 CUDA、BF16、`head_dim=128`、最后一维连续的 YOCO Q/K；其他
dtype、head size 或设备继续走原 `_yoco_apply_rotary_emb` fallback。Q/K 可以来自
packed QKV split，row stride 不要求连续；输出为连续 tensor。

最终 kernel 按 64 个 rotary pair 维度扁平化，每个线程同时读取 `x1/x2`、查询
FP32 cos/sin 并写回输出两半。`BLOCK_SIZE=256`、`num_warps=8`，Q/K 在同一次
launch 中完成，避免旧 head-per-program 版本的大 batch 重复加载和过多 programs。

### 数值语义和正确性

旧 0731 kernel 不是当前 fallback 的 bitwise 等价实现：在非零 position 上观察到
最多 `19 / 1,048,576` 个 BF16 元素不同，`max_abs=0.0078125`。本轮没有沿用旧
测试的宽松 `atol=4e-3`。当前 Inductor PTX 的运算顺序为：

```text
second_product = mul.rn.f32(...)
output = fma.rn.f32(first_input, first_factor, +/-second_product)
```

新 kernel 用显式 `_yoco_mul_rn` 和 `_yoco_fma_rn` 保持相同舍入与 FMA 顺序。
B200 测试覆盖 `(query_heads,key_heads)=(48,4)/(64,8)`，tokens 为
`1/17/128`，position 使用 `arange(tokens) * 3`；Q/K 均来自 packed QKV split。
结果为 6 组全部 `rtol=0, atol=0`，CustomOp 的真实非连续输入 `opcheck` 通过。
完整 `tests/model_executor/test_yoco_conversion.py` 为 `30 passed`。

### B200 CUDA Graph 性能

环境：NVIDIA B200、PyTorch `2.11.0a0+eb65b36914.nv26.02`、Triton `3.7.1`。
baseline 是原 `index_select + chunk + torch.compile` RoPE，candidate 是新
CustomOp；每个 shape 分别 capture CUDA Graph，warmup 10 次，每个 sample
replay 200 次，7 轮交替执行顺序并取中位数。计时包含 cache position gather 和
Q/K rotation，不包含 graph capture、JIT 或 tensor 初始化。

| Q heads / KV heads | Tokens | Baseline us | Fused us | 提升 |
| --- | ---: | ---: | ---: | ---: |
| 48 / 4 | 1 | 6.1582 | 3.4142 | 80.37% |
| 48 / 4 | 17 | 6.8115 | 3.4042 | 100.09% |
| 48 / 4 | 128 | 8.2082 | 4.1144 | 99.50% |
| 48 / 4 | 512 | 14.3458 | 10.2542 | 39.90% |
| 48 / 4 | 1,024 | 20.5027 | 16.4058 | 24.97% |
| 48 / 4 | 4,096 | 69.6250 | 59.4386 | 17.14% |
| 48 / 4 | 8,192 | 131.0243 | 118.7507 | 10.34% |
| 64 / 8 | 1 | 6.1562 | 3.3333 | 84.69% |
| 64 / 8 | 17 | 8.2056 | 3.1910 | 157.15% |
| 64 / 8 | 128 | 10.2571 | 6.1570 | 66.59% |
| 64 / 8 | 512 | 18.4424 | 12.3091 | 49.83% |
| 64 / 8 | 1,024 | 28.6798 | 22.5331 | 27.28% |
| 64 / 8 | 4,096 | 96.2554 | 81.9664 | 17.43% |
| 64 / 8 | 8,192 | 184.7998 | 163.7840 | 12.83% |

全部 14 个点均为正收益，范围 `10.34%～157.15%`。早期 head-per-program
kernel 在 8K tokens 约回退 40%，第一版 flat-output kernel 在 8K 仍回退
`11.51%～12.42%`；最终 pair kernel 消除了这些大 batch 回退。

### 检查、边界与回滚

通过 `py_compile`、两个修改文件的完整 pre-commit、`git diff --check`、7 项
rotary 专项测试和 30 项 YOCO conversion 回归。microbenchmark 只表示 RoPE
局部延迟，不能直接外推为 endpoint tok/s。

当前 kernel 固定 YOCO 已验证的 128 维 half-rotation 语义，不覆盖 interleaved
RoPE、FP16/FP32 或其他 head size；这些情况保留 fallback。升级 PyTorch、Triton
或 CUDA 后，应重新核对 Inductor 算术顺序和完整 A/B。回滚
`8abfd6c6d0` 即恢复原 compiled fallback，不涉及 UCX、NIXL、Docker 或模型权重。

## Differential-attention CustomOp 尝试（未采用）

本轮复核了 `origin/shaohanh/yoco-0731` 中最后一个尚未拆出的
`yoco_diff_attention` CustomOp。结论是不迁移该实现，生产代码保持不变。

当前 `_yoco_diff_attention_v2/v3` 已使用 `@torch.compile`。在 PyTorch
`2.11.0a0+eb65b36914.nv26.02`、Triton `3.7.1` 上检查 Inductor 输出后确认，
sigmoid、两个乘法和最终减法已经生成单个 Triton kernel；slice/reshape 只是
view，不产生额外 GPU launch。因此再加一个 CustomOp 并不会减少 kernel 数。

实验实现覆盖 BF16、`head_dim=128`、24/32 head-pairs、diff-v2/v3。B200
correctness matrix 使用 tokens `1/17/128`，12 组均与 compiled fallback
`rtol=0, atol=0`，两个 CustomOp `opcheck` 也通过。旧 0731 测试只要求
`rtol=1e-2, atol=1e-2`，本轮没有沿用宽松阈值。实验代码和测试随后全部撤回，
没有进入提交。

性能使用独立 CUDA Graph、warmup 10 次、每个 sample replay 200 次、7 轮交替
顺序取中位数。旧版接近的 `2 pairs/block, 8 warps` 映射在 512–8192 tokens
普遍回退：

| Head-pairs / 版本 | 512 | 1,024 | 4,096 | 8,192 |
| --- | ---: | ---: | ---: | ---: |
| 24 / v2 | -24.88% | -16.64% | -26.35% | -37.21% |
| 24 / v3 | -24.93% | -27.45% | -33.66% | -43.49% |
| 32 / v2 | -25.08% | -42.81% | -53.73% | -59.57% |
| 32 / v3 | -19.94% | -24.94% | -33.61% | -38.26% |

继续测试 flat-256、pair-per-program 和 flat-1024。flat-1024 在部分大 shape
确实有收益，例如 32/v3 的 512–8192 tokens 为 `+33.28%～+54.17%`，但同一
kernel 对 24/v3 的 1/17 tokens 回退 `17.33%/28.14%`，4K/8K 仍回退
`7.21%/7.75%`。收益强依赖 head layout、diff 版本和 token shape，不能作为
无条件通用 fast path。

最终决定：

- 不提交 CustomOp、Triton kernel、shape guard 或新测试；
- 不用少数正收益点掩盖其他生产 shape 的回退；
- 保留现有 compiled 单 kernel，避免重复维护与额外 compile/dispatch 风险；
- 若以后继续优化，应优先研究与 `lambda_proj` 或 `o_proj` 的跨算子融合，或基于
  真实线上 shape histogram 做完整的多配置选择，而不是重复包装现有 pointwise
  fusion。

本节是负结果记录，不改变模型行为、数值、CUDA Graph、UCX/NIXL 或 P/D serving。

## `fhb-dev` 与 B200 multigpu long-context 合并验收

最终合并 commit：

```text
9106abeb3ad6963b95083688538e53040500e9b8
Merge YOCO B200 multigpu long-context support
```

目标分支为 `origin/shaohanh/yoco-b200-longctx-multigpu-20260804@85f7d2ac1b`；
此前单 GPU long-context 分支不再参与。唯一冲突是 N1280 MoE JSON，最终完整保留
multigpu 分支的 hybrid 表，并确认与 second parent 逐字节一致。

### 检查和 B200 环境

合并后通过 Python compile、shell syntax、JSON、changed-file pre-commit，以及
YOCO conversion `30 passed, 14 warnings`。B200 端点使用：

```text
Pod / node: assuring-owl-b200g4-dev-d5aab19e-master-0 / slc01-cl02-hgx-0202
GPU: 8 x NVIDIA B200
Model: /data/models/yoco-0000-0800-hf
TP / DP: 1 / 8
BF16, FlashInfer, Triton MoE, async scheduling, Gloo DP sync
max model len / prefill budget: 131072 / 32768
CUDA Graph: FULL_AND_PIECEWISE
```

候选镜像覆盖完整合并后 Python tree，并为 SM100 重编译 `_C`；启动前验证
`torch.ops._C.silu_and_mul_with_clamp_fp32`，避免 native 仍来自旧 base。嵌套
Docker 的 base 已到父层深度上限，因此验证镜像在编译成功后扁平化为单层；该步骤
只处理验证环境，不改变仓库源码。

### 准确性

8,192 和 65,536 prompt token 各做两次 greedy 256-token 生成。baseline/candidate
都得到精确 256 tokens、`finish_reason=length`，同 shape repeat 一致，且跨镜像
token SHA256 完全相同：

```text
8K:  9751294543df49838834be427e34887ff536c92c9b6b044d6d1011875fa8355a
65K: 345b5a43f2d8ef3f7e208b3027430e96c24853657fc72df6f37796e65ce84983
```

### DP8/batch8 性能

先按 multigpu 分支方法完成 8 trajectory、40-turn/130K warmup，再运行 W1/W2/W3：

| Workload | 公开 multigpu 基线 tok/s | 合并版 tok/s | 变化 |
| --- | ---: | ---: | ---: |
| W1, 8K + 64K | 757.53 | 730.67 | -3.55% |
| W2, 64K + 16K | 741.81 | 704.90 | -4.98% |
| W3, 40 turns / 130K | 682.48 | 751.27 | +10.08% |

W1/W2 分别只命中 7/6 个活跃 DP rank，因为原 single-turn harness 没有逐请求设置
rank header；这两项保留为方法复现结果，但不作为代码回退归因。W3 明确使用
`X-data-parallel-rank`，8 个轨迹固定到 8 个 rank。

同节点冷启动、同 warmup、不同 cache salt 的严格 W3 A/B：

| 指标 | Baseline | 合并版 | 变化 |
| --- | ---: | ---: | ---: |
| Wall time | 168.010 s | 138.432 s | -17.61% |
| Output tok/s | 619.01 | 751.27 | +21.37% |
| Mean TTFT | 0.245 s | 0.325 s | +32.80%（约 +80 ms） |
| Mean ITL | 12.104 ms | 9.624 ms | -20.49% |
| Prefix-cache hit | 95.58% | 95.58% | 持平 |

结论：合并版的绑 rank 长 session 吞吐和 ITL 明显改善，准确性逐 token 一致；TTFT
存在约 80 ms tradeoff，应继续观察。两边均实际 capture CUDA Graph。原始
accuracy、benchmark、runtime、逐 turn 和 service log 位于：

```text
/data/fhb-dev-multigpu-results-20260805
```

本轮是 DP8 长上下文验收，不替代 1P2D 的 UCX/NIXL transport 测试。

### DP1 / DP4 同配置重测（2026-08-05）

为补齐合并版的单卡和四卡数据，本轮重新冷启动同一个 candidate image，并分别按
multigpu 分支原方法执行 DP1/batch1 和 DP4/batch4。两组都先做 8K/65K greedy
正确性，再执行完整 40-turn/130K warmup，最后按 W1、W2、W3 顺序测试；没有复用
旧结果或跳过 warmup。

```text
Candidate runtime source: 9106abeb3ad6963b95083688538e53040500e9b8
Candidate image id: sha256:dbef8b82896fc9257f1eb45acb6b90a2a79eafd440d629a37e162ca3b846738d
Pod / node: lidong1-yoco-fhb-dev-dp14-g4-0805-master-0 / slc01-cl02-hgx-0346
Model: /data/models/yoco-0000-0800-hf
GPU: NVIDIA B200; DP1=physical GPU 2, DP4=physical GPUs 2,3,4,5
TP: 1
BF16, FlashInfer, Triton MoE, async scheduling
max model len / prefill budget: 131072 / 32768
CUDA Graph: FULL_AND_PIECEWISE
```

该节点是 4 卡共租节点；GPU 0、1、6、7 在测试期间由另一作业使用。被测 GPU
2--5 没有重叠进程，测试结束后均回到 `0 MiB / 0%`。因此软件、模型、GPU 型号和
workload 与公开基线一致，但不是同节点同时间 A/B；下表变化可用于回归检查，不能
全部归因给某一个 kernel commit。

#### 正确性

DP1 和 DP4 都对 8,192/65,536 prompt tokens 各做两次 greedy 256-token 生成。
8 个请求全部得到 `finish_reason=length`，同 shape 重复一致，并与此前 baseline
逐 token 哈希一致：

| 部署 | 8K SHA256 | 65K SHA256 | 结果 |
| --- | --- | --- | --- |
| DP1 | `9751294543df49838834be427e34887ff536c92c9b6b044d6d1011875fa8355a` | `345b5a43f2d8ef3f7e208b3027430e96c24853657fc72df6f37796e65ce84983` | 两次一致 |
| DP4 | 同上 | 同上 | 两次一致 |

#### DP1 / batch1 性能

| Workload / 指标 | 公开基线 | 本轮合并版 | 变化 |
| --- | ---: | ---: | ---: |
| W1 wall | 536.267 s | 471.895 s | -12.00% |
| W1 output tok/s | 122.21 | 138.88 | +13.64% |
| W1 mean TTFT | 182 ms | 177.6 ms | -2.40% |
| W1 mean TPOT | 8.18 ms | 7.198 ms | -12.01% |
| W2 wall | 137.978 s | 121.820 s | -11.71% |
| W2 output tok/s | 118.74 | 134.49 | +13.27% |
| W2 mean TTFT | 1.227 s | 1.159 s | -5.51% |
| W2 mean TPOT | 8.35 ms | 7.365 ms | -11.80% |
| W3 wall | 113.266 s | 100.582 s | -11.20% |
| W3 generation tok/s | 114.77 | 129.25 | +12.61% |
| W3 mean TTFT | 166 ms | 163.9 ms | -1.26% |
| W3 mean ITL | 8.22 ms | 7.249 ms | -11.81% |
| W3 prefix-cache hit | 95.58% | 95.58% | 持平 |

DP1 三个请求均成功。W3 的 GPU 2 平均利用率为 `95.99%`，waiting 为 0。

#### DP4 / batch4 性能

| Workload / 指标 | 公开基线 | 本轮合并版 | 变化 |
| --- | ---: | ---: | ---: |
| W1 wall | 679.565 s | 578.962 s | -14.80% |
| W1 output tok/s | 385.75 | 452.78 | +17.38% |
| W1 mean TTFT | 454 ms | 267.9 ms | -41.00% |
| W1 mean TPOT | 10.36 ms | 8.830 ms | -14.77% |
| W2 wall | 155.801 s | 148.746 s | -4.53% |
| W2 output tok/s | 420.64 | 440.59 | +4.74% |
| W2 mean TTFT | 1.801 s | 1.631 s | -9.46% |
| W2 mean TPOT | 9.40 ms | 8.979 ms | -4.48% |
| W3 wall | 131.706 s | 126.314 s | -4.09% |
| W3 generation tok/s | 394.82 | 411.67 | +4.27% |
| W3 mean TTFT | 267 ms | 247.4 ms | -7.35% |
| W3 mean ITL | 9.33 ms | 8.975 ms | -3.80% |
| W3 prefix-cache hit | 95.58% | 95.58% | 持平 |

DP4 的 W1/W2 各 4/4 成功。虽然原 single-turn harness 没有显式 rank header，
本轮 Prometheus 指标确认 engine 0--3 各有一个 running request、waiting 全为 0，
四个 engine 的 generation-token 计数等量推进；因此没有复现 DP8 旧轮次的 rank
分配不均。W3 按 trajectory 显式绑 rank，四卡平均利用率为
`94.83%--95.99%`，waiting 为 0。

服务日志确认两种部署都实际执行 FULL_AND_PIECEWISE graph capture，并在完整
warmup 中覆盖后段 YOCO RMSNorm、fused add-RMSNorm、Router renorm 和 MoE Triton
JIT。DeepEP 仍因基础镜像 NVSHMEM symbol 不匹配不可导入；DP4 与公开方法一样使用
AllGather+ReduceScatter，所以这不是本轮变量。

accuracy、benchmark JSON、逐 turn JSONL、runtime、container inspect 和 service
log 保存在：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/fhb-dev-dp14-20260805/results
```

## DeepEP / NVSHMEM 运行时修复与 DP4 取舍（2026-08-05）

本轮基于 `fhb-dev@a9cd5c2072`，功能提交为 `81df1f21e8`，只修改长上下文
Dockerfile 和两个 serving launcher。目标是消除 DeepEP import/ABI 偶然性、让
nested Docker 真正看到 RDMA device，并把 DeepEP LL 的宿主条件错误从异步 CUDA
崩溃提前为可读的启动错误。

### 根因和修复

基础镜像同时带有 CUDA toolkit NVSHMEM 与 pip `nvidia-nvshmem-cu13`。旧 launcher
覆盖 `LD_LIBRARY_PATH`，使 `deep_ep_cpp` 可能解析到不匹配的 host library。修复后
保留镜像自己的 library 顺序，并在 build 和显式启用 DeepEP 时都检查：

```text
DeepEP:  1.2.1+567632d
NVSHMEM: 3.6.5
resolved libnvshmem_host.so.3:
  /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3
required symbol: nvshmem_selected_device_transport
```

build 使用固定 base digest
`sha256:08a08f36ab8c6c80ee1c7f09b9e5f8b6ce0b91cc684455982b2e5e286c736f2b`，
通过 CUDA driver stub 执行 `import deep_ep`。功能 commit 的验证镜像为：

```text
tag: yoco-pr13-deepep-nvshmem-81df1f21e8
id:  sha256:f3920f514a8a164529a7116660dae7f4ee355c14d56fa5e0c21bc4784277d124
```

nested Docker 现在逐个映射外层 Pod 实际可见的 `/dev/infiniband/*` character
device，并设置 `memlock=-1`；NVML 改挂到镜像已有搜索路径
`/usr/local/nvidia/lib64`，不再通过覆盖 `LD_LIBRARY_PATH` 找它。

DeepEP LL 另外增加三层启动保护：

1. 32K token budget 会产生约 97.5 GiB RDMA buffer，超过 DeepEP int32 index
   上限；LL 的 `auto` budget 因此使用 8192，其他 backend 保持 32768；
2. `NVSHMEM_QP_DEPTH` 按 `(max_tokens + 1) * 2` 计算，并向上取二次幂；8192
   tokens 自动得到 32768，launcher sentinel `auto` 不会泄漏给 NVSHMEM；
3. 启动前要求可见 `uverbs`，并要求 NVIDIA `EnableStreamMemOPs=1` 或
   `/dev/gdrdrv`。条件不满足时返回 code 2，不再运行会触发 device assert 的 LL
   kernel。

当前 B200 节点虽然已加载 `nvidia_peermem`，但
`/proc/driver/nvidia/params` 明确为 `EnableStreamMemOPs: 0`，也没有 `gdrdrv`。
旧路径实际先报告 `init failed for transport: IBGDA`，随后在
`internode_ll.cu:285` 触发 `num_rc_per_pe >= num_local_experts`，最后表现成四卡
`CUDA error: unspecified launch failure`。新路径对同一节点稳定提前失败，提示需改
driver 参数并 reboot，或加载 gdrdrv；本提交没有假装在容器内修复宿主 IBGDA。

### DeepEP HT 正确性

在 `slc01-cl02-hgx-0201` 的物理 GPU 4--7 上，DP4 服务成功进入：

```text
Using DeepEPHTAll2AllManager
Using DeepEPHTPrepareAndFinalize
```

8K/65K prompt 各执行两次 greedy 256-token 生成，四个请求都为
`finish_reason=length`，重复一致且与既有 baseline 逐 token hash 相同：

```text
8K:  9751294543df49838834be427e34887ff536c92c9b6b044d6d1011875fa8355a
65K: 345b5a43f2d8ef3f7e208b3027430e96c24853657fc72df6f37796e65ce84983
```

### 纯 Prefill 同节点 A/B

性能口径针对 pure-P node：同一候选镜像、同一节点和四张 B200，DP4/EP4，20 个
65,536-input + 1-output 请求，并发 4，固定 seed；每边先由 harness 执行一个
不计时 warmup 请求。两边均 20/20 成功，实际输入均为 1,310,720 tokens。

| 指标 | AllGather+ReduceScatter | DeepEP HT | DeepEP 变化 |
| --- | ---: | ---: | ---: |
| benchmark wall | 11.33 s | 16.79 s | +48.19% |
| total token throughput | 115,676.90 tok/s | 78,058.63 tok/s | -32.52% |
| mean TTFT | 2.124 s | 3.215 s | +51.35% |
| median TTFT | 1.920 s | 3.006 s | +56.59% |

DeepEP HT 会按 vLLM 设计关闭 CUDA Graph；AllGather+ReduceScatter 实际 capture
`FULL_AND_PIECEWISE`。因此本轮保留 DeepEP HT 作为显式实验选项，但默认 backend
设为 `allgather_reducescatter`，避免把已测得的回退带入 pure-P 服务。DP>1 默认
启用 EP，DP1 保持不启用；可用 `ENABLE_EXPERT_PARALLEL=0` 回滚到非 EP 路径。

原始 accuracy JSON、两边 benchmark detailed JSON、service log、container inspect
和三轮 LL 失败日志位于：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/pr13-deepep-nvshmem-20260805
```

该结果只证明单节点 DP4/EP4。跨节点 DeepEP、完成宿主 IBGDA 配置后的 LL、以及
Decode node 小 batch CUDA Graph 性能仍需单独验证；UCX 1.21 与 DeepEP 的
NVSHMEM/IBGDA 是两条独立 transport 栈，本提交没有重做 UCX 镜像。

## YOCO fast-prefill 的本地缓存与 PD 数值一致性（2026-08-09）

功能提交 `03a0479b67` 修复了 YOCO fast-prefill 在 standalone fresh、本地
prefix-cache hit 和 NIXL remote-KV hit 三条路径上输出不一致的问题。根因不是
UCX 传输损坏，而是三条路径用不同 token row 数执行 FP32 TF32 Router 和 prompt
尾段，shape 相关的归约差异会改变接近边界的 top-k expert，随后被 MoE 放大。

修复同时统一算子和调度 shape：小于 128 行的 Router 输入补零到 128 行；普通
standalone/本地缓存请求在最后一个 KV block 边界拆开 prompt 尾段；NIXL P 端只
发布到相同边界，D 端只接收这部分 KV 并本地重算尾段。P 端已截断请求跳过二次
边界拆分，避免 `1344` 又变为 `1328 + 16`。不满一个 block 的短 prompt 不会被
截成空请求：P 保持有效请求，D 不拉取 partial block 并完整本地重算。Mamba 原有
N−1 远端 prefill 语义保持不变。

B200 真实 1P1D 对 1,356、7,999、65,809 prompt tokens 的串行 correctness
harness 全部与 standalone exact match；1.3K fresh 和两次本地缓存命中也都输出
`" 00663\n"`，缓存命中 1,344 tokens，首 token logprob 逐位一致：

| Prompt tokens | SHA256 | P time | D time | 串行总时延 |
| ---: | --- | ---: | ---: | ---: |
| 1,356 | `fde361c254896b017d5496f6e8cd4a128d7e258544675fab97574d01c973c033` | 0.0692 s | 0.2609 s | 0.3301 s |
| 7,999 | `8b676d6658af7e5d789c559690a8c763683c433e8e857b4146df5054dfa71c01` | 0.1549 s | 0.4190 s | 0.5739 s |
| 65,809 | `82e7da1423e3c6e149b622b75a9674c5508bf91c1a06099b46e75a22033bfe53` | 1.1278 s | 1.1470 s | 2.2748 s |

这些时间来自先 P 后 D 的正确性脚本，不是并发吞吐或加速数据。Router padding
微基准显示 `<128` 行单次调用增加约 `44--50 us`，1/12/127 行相对直接 GEMM
分别约 `+185%/+150%/+160%`，128 行约 `+3%`；端到端吞吐仍需独立并发 A/B。

最终本地回归为 YOCO 文件 `38 passed`；scheduler 与 NIXL/HMA 为
`137 passed`，另有一个 `google/gemma-3-1b-it` 用例因 gated HuggingFace 权限
返回 HTTP 403。ruff check、ruff format 和 `git diff --check` 均通过。原始服务
日志、JSON、hash 和时延位于：

```text
/mnt/pvc/lidong1/vllm_test_artifacts/pd-current-4a39087d27-20260805/run25-yoco-nixl-shape-aligned-pd-20260809
```

本轮只验证 B200 1P1D、文本 token 请求，不代表多 P/D、高并发或 prompt-embeds
覆盖；未修改 Docker、UCX 1.21、DeepEP/NVSHMEM 或 launcher。
