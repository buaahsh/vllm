# 单卡 Fast BF16 decode 复现

脚本路径：`tools/yoco_alignment/benchmark_fast_bf16_decode.py`。
它与 `benchmark_fast_fp8_decode.py` 共用计时实现及固定输入，
用于比较两套 Fast 推理精度方案。只进行推理，不涉及训练。

脚本现已适配 `vllm-yoco-version-0.29`。本文的 2026-09-15 结果和附带 JSON
保留旧分支实测；当前版本复测与接口变化见[增量合并报告](../fhb-dev-9-18-merge.md)。

## 运行

在安装了本分支的 B200 GPU 环境中，从 vLLM 仓库根目录执行：

```bash
.venv/bin/python tools/yoco_alignment/benchmark_fast_bf16_decode.py \
  --model /mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf \
  --gpu 0 \
  --output results/fast-bf16-decode.json
```

按环境替换 Python 和 checkpoint 路径。在 `yoco-fp8-dev` 测试镜像中，
Python 为 `/workspace/.venv/bin/python`，需增加 `--container-compat`。
测试应使用原始 BF16 checkpoint；脚本不把已离线量化的权重还原成 BF16。

参数与 FP8 脚本相同：默认 `--batches 1 2 4 8 16`，
`--prompt-tokens 512 --output-tokens 128 --warmups 5 --repeats 7`。
增加 `--profile-dir results/bf16-traces` 可在 B1/B8 完成本档计时后
分别采集 12 个稳态 generation 步骤。已有结果 JSON 不会被覆盖。

首次使用 BF16 专家后端时，FlashInfer 可能编译较多 kernel，启动会更久；
编译与 Graph capture 均在计时前完成，后续运行可复用该环境的 JIT 缓存。
可按 CPU/内存配额设置 `MAX_JOBS` 调整编译并发，默认 4；例如在 16 核测试
Pod 中使用 `MAX_JOBS=12`。它不改变推理的 OMP/MKL 线程数与计时参数。

BF16 和 FP8 入口复用同目录 `benchmark_fast_decode.py`；请使用完整仓库，
保留三个脚本和 `data/fast_fp8_decode_tokens.json`。该数据文件是固定 token ID，
不表示输入采用 FP8 精度。

## 精度与公平比较

| 项目 | Fast BF16 脚本 | Fast FP8 脚本 |
| --- | --- | --- |
| 模式 / 并行 | Fast，TP1/DP1/EP1 | 相同 |
| 主投影/专家权重与 GEMM 输入 | BF16，不启用在线量化 | block-128 FP8 |
| Attention / KV | FA4 BF16；`kv_cache_dtype=auto` 随模型 BF16 | FA4 FP8 / FP8 KV |
| 主残差 / router | BF16，`BF16_CHAIN=1` | 相同 |
| Norm 归约 / 采样 | 保留原有实现，BF16 实验关闭 | 相同 |
| FP8 attention 融合 / FP8 W2 调参 | 关闭，不适用于 BF16 | 开启 |
| latent Norm→FP8 融合实验 | 关闭 | 关闭 |
| CUDA Graph | FULL_AND_PIECEWISE，1/2/4/8/16 | 相同 |
| 显存比例 / 调度上限 | 0.65，maxlen 8192，batch tokens 4096，maxseq 16 | 相同 |

脚本检查实际专家和 latent 权重 dtype，以及分配后的 KV tensor dtype，
确认 BF16 配置生效；JSON 记录真实 MoE 后端。不同精度的默认 kernel 和
可用 KV cache 容量可能不同，因此这是完整精度方案比较，不是单个 GEMM 的实验。
BF16 数据格式也不表示 Tensor Core 累加、Norm 归约和采样全部使用 BF16。

## 计时方式

使用与 FP8 完全相同的提示词：第 `i` 条取固定 token 数组的
`tokens[64*i : 64*i+512]`，greedy 固定生成 128 token，忽略 EOS。
每档预热 5 次建立缓存后，测量 7 次并取耗时中位数。

先暂停调度并将整批请求入队，计时从恢复调度前开始，到所有输出取回为止。
模型加载、编译、Graph capture、预热和入队不计时。

- 总吞吐：`batch × output_tokens / median_seconds`。
- 每输出步平均毫秒：`1000 × median_seconds / output_tokens`。
- `cached_prompt_tokens` 记录实际缓存命中，不能假设全部 512 token 都被缓存。

这是缓存前缀后的离线生成吞吐，包含剩余 prefill、采样和调度，
不是冷启动 TTFT 或 HTTP 服务结果。详细计时边界和输入来源参见
[FP8 测试说明](FAST_FP8_DECODE_BENCHMARK.md)。

## 旧分支发布前验证

2026-09-15，B200、L3 step28000、5 次预热/7 次计时；本次使用 `MAX_JOBS=12`。
所有专家和 latent 权重、KV tensor 的实际 dtype 均通过 BF16 检查。

| Batch | 总输出 tok/s | 每输出步平均 ms |
| ---: | ---: | ---: |
| 1 | 173.65 | 5.759 |
| 2 | 276.01 | 7.246 |
| 4 | 427.25 | 9.362 |
| 8 | 726.93 | 11.005 |
| 16 | 1173.09 | 13.639 |

各请求实际缓存命中均为 496/512 token。结果属于短上下文离线推理，
不能据此推断完整任务集的质量或长上下文性能。原始计时、运行时配置和
FP8 共用实现回归记录见 [验证 JSON](FAST_BF16_DECODE_VALIDATION.json)。

同卡顺序复测的 FP8 B1/B8 分别为 171.28 / 831.29 tok/s。
两种精度的 B1/B8 trace 各确认 12 个 generation 步骤和 12 次完整
`cudaGraphLaunch`。这里只验证脚本与运行配置，不构成模型质量评估或
小幅性能差异的统计显著性结论。
