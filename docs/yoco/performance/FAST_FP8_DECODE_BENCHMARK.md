# 单卡 Fast FP8 decode 复现

`vllm-yoco-version-0.29` 已接入来自 `fhb-dev-9-18` 的单卡 Fast FP8 方案。脚本为
[`benchmark_fast_fp8_decode.py`](../../../tools/yoco_alignment/benchmark_fast_fp8_decode.py)。
只进行推理，不涉及训练。EP2 测试没有改变默认并行配置。

本文的 2026-09-15 测量及附带验证 JSON 对应旧分支原环境。
当前脚本使用新版 `RoutedExperts` 和公开的 `LLM.enqueue/wait_for_completion`，
移除了上游已删除的 `calculate_kv_scales` 参数；固定 scale 由当前上游默认配置提供。
两种精度的公共实现位于 `tools/yoco_alignment/benchmark_fast_decode.py`。
新版本运行记录见[增量合并报告](../fhb-dev-9-18-merge.md)，不将旧环境的速度直接作为新版本结果。

相同输入和计时方式的 BF16 对照入口为
[`benchmark_fast_bf16_decode.py`](../../../tools/yoco_alignment/benchmark_fast_bf16_decode.py)，
配置区别见 [BF16 测试说明](FAST_BF16_DECODE_BENCHMARK.md)。

## 如何运行

在安装了本分支的 B200 GPU 环境中，从 vLLM 仓库根目录运行：

```bash
.venv/bin/python tools/yoco_alignment/benchmark_fast_fp8_decode.py \
  --model /mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf \
  --gpu 0 \
  --batches 1 2 4 8 16 \
  --output results/fast-fp8-decode.json
```

按实际环境替换 Python 与 checkpoint 路径。`yoco-fp8-dev` 测试镜像的 Python
为 `/workspace/.venv/bin/python`，源码目录为 `/workspace/vllm_current`；该镜像
需额外加 `--container-compat`，复用已有 metadata/torchvision 兼容处理。
使用环境中已有的原生 DeepGEMM，不需要为此脚本另装或替换 DeepGEMM。

如需查看纯 generation 阶段的 GPU kernel，增加：

```bash
--profile-dir results/fast-fp8-traces
```

每个 batch 先完成该档无 profiler 的吞吐测量，再为已选择的 B1/B8 采集 profile。
每次跳过开头 4 次迭代，再记录 12 次迭代。JSON 中保存实际 trace 文件名；
vLLM 可能复用第一份文件前缀，应以 JSON 的 batch→文件映射为准。
可用 Perfetto 打开 `.pt.trace.json.gz`，核对 generation 标记和 `cudaGraphLaunch`。

输出 JSON 不能已存在，以免覆盖原始数据。脚本每完成一个 batch 就保存一次，
只有整轮成功时 `completed` 才为 `true`。

## 约 171 token/s 是怎么测的

1. 一张 B200，TP1 / DP1 / EP1，YOCO L3 step28000。
2. 每条请求输入 **512 token**，固定输出 **128 token**；greedy，忽略 EOS，
   不请求 logprob。B1/B2/B4/B8/B16 表示同时处理的请求数。
3. 使用仓库附带的固定 token ID。第 `i` 条提示词为
   `tokens[64*i : 64*i+512]`，与之前的单卡和 EP2 对照相同。
4. 每个 batch 先完整生成 **5 次**，完成 JIT 预热并建立 prefix cache；
   不在计时轮清空 prefix cache。再生成 **7 次**，取耗时中位数。
5. 暂停 scheduler，先将整批请求全部入队。**计时从恢复 scheduler 前开始，
   到全部请求生成结束、输出取回为止**。模型加载、编译、Graph capture、
   预热、暂停调度和请求渲染/入队不计时。
6. 总吞吐为 `batch × 128 / median_seconds`。B1 中位数约 0.748 秒，
   因此 `128 / 0.748 ≈ 171 token/s`。

这个数是**缓存前缀后的离线生成吞吐**。它包含剩余 prefill、模型执行、
LM head、采样、scheduler 和输出传输。Prefix cache 按块复用，未必复用全部
512 token；脚本记录每条请求的实际 `cached_prompt_tokens` 并检查命中不为零。
发布前复现中，每条提示词实际命中 **496/512 token**，剩余 **16 token** 仍需计算。
它不是冷启动 TTFT、HTTP 服务吞吐，也不是逐 token 流式 ITL。

`mean_output_step_ms = 1000 × median_seconds / 128` 表示整批每输出步的
平均时间。B1 约 5.844 ms，含上述开销。此前单独 profile 的 B1 完整 GPU
Graph span 约 **5.559 ms**，两者口径不同。不同 stream 的 kernel 时间可能
重叠，不能把 kernel 时间逐项相加当作请求延迟。

早先 W2 调参报告的约 171.13 tok/s 在整个 `llm.generate()` 外计时，包含
暂停和入队；最近的单卡/EP2 对照及本脚本将这些控制操作移到计时外。
两轮 B1 均约 171.13 tok/s，但原始采样与计时边界已分别保留，不能混为同一轮。

## 固定推理配置

Shared/latent 的 M1 专用 GEMV 默认启用，见
[整模型接入验证](SHARED_LATENT_M1_INTEGRATION.md)。设置
`VLLM_YOCO_FP8_SMALL_M=0` 后新建 engine 可进行 native 对照；脚本会记录该值。

| 部分 | 设置 |
| --- | --- |
| 模式 / 权重 | `additional_config.yoco_execution_mode=fast`, `quantization=fp8_per_block` |
| Attention / KV | FA4，FP8 Q/K/V 和 FP8 KV，固定 KV scale |
| 主残差 / router | BF16，`VLLM_YOCO_BF16_CHAIN=1` |
| 归约 / scale / 采样 | 保留原有高精度实现；BF16 归约/采样实验关闭 |
| attention 融合 | `VLLM_YOCO_FP8_ATTENTION_FUSION=1` |
| latent Norm 融合 | `VLLM_YOCO_FP8_LATENT_NORM_FUSION=0` |
| 小 batch W2 | `VLLM_YOCO_FP8_W2_TUNING=1`，M1/M2/M4 独立配置 |
| Shared/latent M1 | `VLLM_YOCO_FP8_SMALL_M=1`，实际 M1 用 direct GEMV，其余 M 用 native |
| CUDA Graph | `FULL_AND_PIECEWISE`，捕获 `[1,2,4,8,16]` |
| Scheduler | 最大长度 8192，最大 batch tokens 4096，最多 16 请求 |
| 显存预算 | `gpu_memory_utilization=0.65` |

脚本在导入 vLLM 前设置这些环境变量，明确复现此单卡预设，不继承外部的
DP/EP 配置或 BF16 归约实验开关。其他硬件、checkpoint、提示词或上下文长度
可能得到不同速度。B200 上验证的环境为 Torch `2.11.0a0+eb65b36914.nv26.02`、
CUDA 13.1、原生 DeepGEMM 2.5.0。

附带的 `tools/yoco_alignment/data/fast_fp8_decode_tokens.json` 只保存本次所需的
1,472 个 token ID。来源为公开 WikiText-2 validation/Wikipedia（CC BY-SA 3.0），
文件包含来源链接、源文件哈希及 tokenizer 标识。它们已由本 checkpoint 的
tokenizer 编码；更换 tokenizer 时应提供相匹配的 `--tokens-json`。

## 旧分支发布前脚本验证

2026-09-15 使用本脚本及 `--profile-dir`，每项 5 次预热、7 次计时：

| Batch | 总输出 tok/s | 每输出步平均 ms |
| ---: | ---: | ---: |
| 1 | 171.18 | 5.842 |
| 2 | 275.84 | 7.251 |
| 4 | 480.86 | 8.318 |
| 8 | 829.89 | 9.640 |
| 16 | 1438.71 | 11.121 |

这是短上下文测量，不代表长上下文速度或模型质量。
相关优化依据见 [FP8 W2 调参](FP8_W2_TUNING.md)。
每次耗时、缓存命中和实际后端信息见
[发布前验证 JSON](FAST_FP8_DECODE_VALIDATION.json)。B1/B8 trace 各核对了
12 个 generation 步骤和 12 次完整 `cudaGraphLaunch`；另外 64 项既有
FP8 W2 / scale / 直接量化回归全部通过。
