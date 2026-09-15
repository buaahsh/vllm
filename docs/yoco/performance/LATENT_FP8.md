# Latent in/out FP8：fhb-dev-9-18

2026-09-14，仅在 vLLM `fhb-dev-9-18` 的 Fast 推理模式接入 latent 投影 FP8。llm-train 保留同名分支，训练实现和测试保持改动前原样。

## 行为

- vLLM Fast 的 `fc1_latent_proj` / `fc2_latent_proj` 继承配置的 linear 量化方法。使用 `--quantization fp8_per_block` 时，两者采用现有在线 W8A8 block-128 路径：加载期把权重转换为 FP8，前向量化激活，FP32 累加，输出 BF16。显式 ignore 规则仍生效；只量化专家而未配置 linear 的模式不会额外量化 latent。
- vLLM BF16 与 Align 保留原有 latent 精度。Router、lambda、LM head、KV 和 norm 统计没有在本轮改变。

前向顺序保持为：

```text
latent in : FP8 linear → 原 latent norm
latent out: 原 latent norm → FP8 linear
```

HF checkpoint 的参数名和矩阵形状不变，无需重新导出 checkpoint。在线推理加载后新增对应的量化权重/scale。

## L3 范围

实际形状为 `3072→1024` 和 `1024→3072`。20 个物理 MoE 层共 40 个投影，decode 时随 universal loop 共调用 80 次。两类权重原始 payload 从 BF16 的 240 MiB 降为 FP8 的 120 MiB，未计 scale、padding 和 allocator；这不是整模型实测显存或吞吐变化。

## 使用与对照

vLLM 沿用：

```bash
--fast --quantization fp8_per_block
```

只将 latent 回退到 BF16、保留其余 FP8，可使用已有 ignore 配置：

```bash
--fast --quantization fp8_per_block \
--quantization-config '{"ignore":["re:.*\\.mlp\\.fc[12]_latent_proj$"]}'
```

旧的 BF16/FP8 吞吐表和 NLL 数据对应此前 latent 为 BF16 的源码，不能用于本轮候选的性能或质量结论。

## 验证记录

- 本地 vLLM 定向回归：69 passed，覆盖配置传递、BF16/Align、显式 ignore 及相关已有行为。
- B200 定向检查：66 passed，包含 56 项配置检查和 10 项真实 latent FP8 GEMM/CUDA Graph 检查。覆盖 `3072→1024`、`1024→3072` 和 M=1/8/32/129；GEMM 相对独立量化参考的 L2 小于 0.1%。这是算子实现验证，不能替代相对 BF16 的量化误差或模型质量验证。
- vLLM pre-commit 已在最终 GPU 测试 fixture 更新后通过。
- 整模型对照完成：同一 L3 step28000 checkpoint、Fast、FA4、`fp8_per_block`，只通过 ignore 切换 latent 精度。两个方向共 40 个权重的实际 dtype 已通过只读 worker 接口核验。使用已有冻结 WikiText-2 validation 文本，在 128–4220 范围每 4 token 采样，共 1,024 个固定位置；先验证强制目标 token 返回的是 raw log-prob，再计算 NLL。

| 文本质量初筛 | latent BF16 | latent FP8 |
| --- | ---: | ---: |
| 平均 NLL | 1.443545511 | 1.442959144 |
| latent 权重 payload | 240 MiB | 120 MiB |
| 模型加载显存（日志舍入值） | 31.63 GiB | 31.52 GiB |

NLL 差为 -0.000586367，`exp(mean NLL)` 比值为 0.999413805。这批样本未观察到退化；这不是完整数据集 perplexity、代码/数学任务准确率或长上下文质量结论。固定 GPU 利用率配置下，省出的模型显存可被 KV cache 使用，不能将加载显存差解释为进程总显存差。

## 延迟结果

同一张 B200，使用 checkpoint 的全部 40 个 latent 权重，在 BF16 与 FP8 CUDA Graph 之间交错测量。下表每次执行 40 个投影，包含输入量化和 GEMM，不含 norm、专家和 attention；每轮重复执行热权重，不能直接外推端到端延迟。

| M | BF16（µs） | FP8（µs） | FP8 / BF16 延迟 |
| --- | ---: | ---: | ---: |
| 1 | 194.61 | 300.91 | 1.55× |
| 8 | 183.33 | 308.52 | 1.68× |
| 32 | 171.99 | 326.25 | 1.90× |

单个投影的真实权重、随机 BF16 激活测试中，相对 BF16 输出的 L2 误差约 3.8%。该误差与前述“小于 0.1% 的量化参考误差”是不同指标。

端到端初测：两组分别占一张 B200，以 warm prefix、512 输入 token、128 输出 token 做 3 轮离线生成。batch=1 的中位吞吐为 163.98→161.29 token/s，batch=8 为 830.08→813.93 token/s，约下降 1.6%–1.9%。这里有卡间差异与运行波动，只作初测；同卡算子测量也显示当前路径存在延迟回归。

当前实现完成了 FP8 压缩和精度初筛，尚未获得速度收益。保留为开发分支候选；后续速度工作应针对这两个投影的量化融合或小 M GEMM。本轮没有继续扩展 kernel 修改。

GPU 验证使用独立 Job `bonete01/yoco-fp8-dev`，单 Pod、2×B200，node `slc01-cl02-hgx-0331`。镜像原生 DeepGEMM 2.5.0 保持原路径和二进制 hash；没有单独安装或覆盖 DeepGEMM。vLLM 的已有编译扩展来自此前已验证的 YOCO runtime，源码使用本轮候选。

原始快照、命令、XML 和日志位于工作区 `yoco_results/latent-fp8-20260914-xqge4i_6/`。`FAST_ONLY.patch` 记录相对本轮开始状态的差异；`gpu-fast/`、`fast-ab/`、`FAST_AB_SUMMARY.json` 和 `latent-microbench.json` 保存推理验证证据。测试结束后 holder 回到 IDLE，保留 2 张 B200。本轮没有提交、推送或切换现有生产服务。

范围更正：此前误加的训练代码与测试已经撤回；对应历史实验不作为本次 Fast 推理优化的验证结果。
