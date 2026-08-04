# YOCO B200 最终验收报告

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
