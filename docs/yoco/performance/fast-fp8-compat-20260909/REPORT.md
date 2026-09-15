# YOCO Fast 在线 FP8 适配

日期：2026-09-09 UTC，工作跨 2026-09-08/09 PDT。分支：`fhb-dev-9-8`，基于 `511fbed3f75f0fd18a4f93194ca832db2c723f98`，保留上一轮 DeepGEMM padding 和启动预热修改。

此前已有在线 FP8 量化和 YOCO DeepGEMM MoE 路径，手动指定 `--fast --quantization fp8_per_block --moe-backend deep_gemm` 能运行。但这不等于 Fast 的默认分派、全部精度条件和回退路径已经适配完毕。上一轮的性能数据只对应当时显式指定 DeepGEMM 的源码快照。本轮补齐以下缺口。

## 实现

### 按实际专家精度选择默认后端

YOCO 配置钩子此前把 `auto` 统一改为 `triton`。现在识别在线 block-128 FP8 的 MoE 配置，保留 `auto`，在构建每个专家层时再选择后端：TP1、实际量化为 block FP8、DeepGEMM 可用时选 `deep_gemm`；被 ignore 规则排除量化的 BF16 专家或不满足上述条件时选 `triton`。显式 `--moe-backend` 不被覆盖。

配置钩子执行时 `VllmConfig.quant_config` 还未构建，所以判断使用已解析的量化参数，同时覆盖 shorthand 和 `--quantization-config`。BF16 原有的 Fast 分派保持原策略。该自动 DeepGEMM 选择限定于在线 block-128 FP8，不能外推到所有带有“FP8”名称的格式。

### 编译图保留平台 FP8 量化算子

只补齐后端和 BF16 Fast kernel 后，完整编译模型的 log-prob 请求出现 NaN；LM head 前的 3072 个 hidden 元素均非有限，改用 cuBLAS head 也一样。337 个浮点参数检查均有限。Eager 对照及其启用新 Fast kernel 的版本均正常，独立的 FP8 quant/linear、CUDA graph、shared/MoE 编译检查也正常。

在相同完整编译与 CUDA graph 配置中，显式 `+quant_fp8` 后，整模型请求恢复有限输出。最终配置钩子为 Fast 在线 block FP8 的 linear 层自动启用该算子；B200 上它直接产生 DeepGEMM 使用的 packed UE8M0 activation scales。显式关闭该算子的编译配置会在启动时给出错误；显式 `--enforce-eager` 配置仍可选择 native quant。

该修改固定已经通过整模型验证的平台量化路径。故障范围已缩小到整模型编译与量化分派的组合，尚未缩成底层编译器的单算子最小复现；本报告不将原因进一步归结为某一个融合 pass。最终服务仍使用模型编译和 CUDA graphs。

### 修复 Triton 的路由加权顺序

YOCO 设置了 `apply_router_weight_before_w2=True`，在线量化方法也将它传给专家实现，但 TritonExperts 此前未消费这一标志，仍在 W2 的输出端乘概率。现在 FP8 Triton 使用加权 clamped SwiGLU：

```text
W13 → clamp + SiLU × up × routing_probability → BF16 store
    → FP8 activation quantization → W2 → sum
```

W2 不再重复乘路由概率。DeepGEMM 已有量化前加权实现；本轮修正的是 Triton 路径。两种后端的融合和舍入边界不同，本轮不声称后端间 bitwise 相同。Fast BF16 仍保留既有 W2 输出端加权策略。

### 复用精度匹配的 Fast kernel

在线 FP8 不量化 `ParallelLMHead`，所以全局 `quant_config is not None` 不应关闭它的 BF16 Fast kernel。现在要求 LM head 实际采用 `UnquantizedEmbeddingMethod`，再由已有运行时检查约束 BF16 dtype、shape、连续性、GPU、TP 和 batch size。真正量化的 LM head 不会进入该路径。

Cross-attention 的 weighted RMSClip 只处理激活和归一化权重。移除全局量化禁用条件，保留原有实际输入检查，允许 FP8 projection 输出的 BF16 激活使用该 kernel。

QKV/Q 与 lambda 的 BF16 合并投影、BF16 shared-expert 转置 kernel 等仍有精度限制；未将 BF16 权重 kernel 直接用于 FP8 权重。

## 验证

B200：Pod `yoco-align-fast-vllm-vllm-train-master-0`，UID `505f46a3-597a-40c3-8260-d213fae60136`，node `slc01-cl02-hgx-0228`。GPU3 用于测试，GPU2 用于完整服务验证，保留原 Job、Pod 和 PID7。

| 检查 | 结果 |
| --- | --- |
| 新增默认后端、量化算子策略、显式覆盖、ignore 与 LM head 精度测试 | 40 passed |
| Triton FP8 CUDA 测试：block/per-tensor，M=1/7/33 | 6 passed |
| 原有 Fast/config/standalone/decode 回归 | 52 passed |
| 显式 CUDA packed quant 的完整模型对照 | 8 个样本请求 + 1 个 profiler 请求通过，log-prob 有限 |
| 最终默认配置完整模型 | 8 个样本请求 + 1 个 profiler 请求通过，log-prob 有限 |

最终测试共 **98 passed**。CUDA 测试读取实际 W13 结果和 W2 量化前输入，以独立 PyTorch FP32 clamp/SiLU/乘法参考检查加权顺序，并检查 W2 不再二次加权、输出有限，以及普通 FP8 MoE 无 YOCO 标志时仍保留原行为。它是数值容差与分派测试，不是跨后端 bitwise 测试。

首次测试中的 6 项失败来自测试观察器误读 GEMM 的参数位置，将 top-k 读作加权标志；修正观察器后 37 项新增测试全部通过。保留首次失败日志。本地环境缺少 CUDA FlashAttention 扩展，完整测试使用 B200 环境。本次 pre-commit 检查通过；全仓 CI 形式的 mypy 有 3 个 `all2all_utils.py` DeepEP 导入类型错误，在相同 HEAD 的干净 worktree 中逐条复现，见 `MYPY_BASELINE_COMPARISON.json`。

### 整模型验证负载与 kernel 证据

功能请求取冻结的 Mooncake FAST’25 toolagent trace 中前 8 个输入长度不超过 8192 的样本，保留输入长度，使用合成 token IDs，将输出长度截断到最多 32。先单请求，再以并发 4 发送余下请求；另做一个 128-input/8-output 的 profiler 请求。检查实际输入/输出 token 计数和生成 token 的 log-prob 有限性。这是功能检查，不是 AIPerf 固定时间回放，也不代表开源任务质量或吞吐结果。

显式 `+quant_fp8` 对照的 worker snapshot 显示 20 个 routed-expert 层为 FP8/DeepGEMM，BF16 Fast LM head 与 10 层 weighted RMSClip 启用。GPU profiler 实际记录了 packed FP8 quant、DeepGEMM dense/grouped GEMM、`_silu_mul_quant_fp8_packed_kernel`、`_yoco_weighted_rms_clip_kernel` 和 `_yoco_lm_head_kernel`。最终默认配置再次通过相同请求与 profiler 检查：20 层 FP8 专家使用 DeepGEMM，82 个在线 FP8 linear 的量化分派为 `forward_cuda`，Fast LM head 和 weighted RMSClip 的 GPU kernel 实际执行。启动 argv 的 `custom_ops` 为空且未传 `--moe-backend`，配置钩子自动启用 `+quant_fp8`；证据单独保存。

## 使用与边界

在本轮验证的 YOCO L3、B200、BF16 hidden、TP1 配置上，启动入口为：

```bash
vllm serve "$YOCO_MODEL" \
  --trust-remote-code --dtype bfloat16 \
  --fast --quantization fp8_per_block
```

模型路径指向原始 BF16 checkpoint，加载阶段在线量化；无需用户额外指定 MoE 后端。完整验证还使用 FA4、YOCO KV sharing、prefix caching、chunked prefill 和 FULL_AND_PIECEWISE CUDA graphs，具体 argv 随验证证据保存。

本轮不增加 Fast 与 BF16/llm-train 的 bitwise 保证；不覆盖 Align+FP8、MXFP8 block-32、FP8 KV cache、FP8 MoE LoRA、多卡或所有离线 FP8 checkpoint。Triton per-tensor 的本轮新增证据限于算子测试。

本轮用于兼容性验证，不发布新增吞吐提升，上一轮 [FP8 性能表](../FP8.md) 保留原始测量值。历史端到端测试只比较文本 hash，没有本轮的 log-prob 有限性门槛，不能据此声称模型数值正确。后续性能比较须使用本轮通过功能验证的源码重新进行同条件 A/B，并先检查 log-prob 有限性。

## 证据位置

- 工作区：`/home/lidong1/vllm_test/yoco_results/fast-fp8-compat-20260909`。
- B200：`/data/fast-fp8-compat-20260909`。
- 新增回归：[模型测试](../../../../tests/model_executor/test_yoco_fp8.py)、[CUDA MoE 测试](../../../../tests/kernels/moe/test_yoco_fp8.py)。

完整验证摘要见 [FINAL_SUMMARY.json](FINAL_SUMMARY.json)，最终源码 hash 见 [SOURCE_MANIFEST_FINAL.json](SOURCE_MANIFEST_FINAL.json)。[测试结果](tests-final.xml)与 [GPU kernel 计数](fp8-final-default-profile-summary.json)一同保存。测试结束后 GPU2/3 空闲，Job/Pod、UID、node、PID7 与 restart count 保留。

归档同时存于上述工作区和 PVC `/mnt/pvc/lidong1/fast-fp8-compat-20260909/fast-fp8-compat-20260909-evidence.tar.gz`；本地已逐文件校验，归档 SHA256 与位置见 [BACKUP.json](BACKUP.json)。模型文件记录了大小、修改时间与 JSON 内容 hash，未做全部权重内容 hash。
