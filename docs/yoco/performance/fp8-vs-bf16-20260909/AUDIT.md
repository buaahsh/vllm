# 当前 Fast FP8 的性能缺口

本文对应首轮 FP8/BF16 对照的源码快照。后续[有效路由行量化修复](../fast-fp8-sparse-20260909/REPORT.md)已处理第2项及第3项中的路由权重 scatter；其余候选项保留待优化。

再后续的[分派与布局优化 / 2×评测](../fast-fp8-dispatch-f2-20260909/REPORT.md)补齐了已验证配置下的小batch FP8分派、专家前缀和融合和scale直接打包。混合精度投影融合仍未实现；小batch后端切换不保证bitwise。

2026-09-09：检查当前源码并重新分析已有 GPU profile。当前完成的是 FP8 功能适配，尚未完成与 BF16 Fast 等效的性能分派和融合优化。以下路径和额外工作已确认，但没有逐项消融，不能把端到端的 10%–23% 差距按这些项目分摊。

证据来自上一轮功能验证的 128-input/8-output profiler，源码 manifest 与本轮 BF16/FP8 对照完全相同，当前本地 manifest 文件也逐项匹配。它包含 prefill 和 decode，且请求检查 log-prob；不是无 profiler 的吞吐测试。原始 profile 的位置、SHA256 和统计见 [profile-audit.json](profile-audit.json)。本次没有重新压测或修改服务。

## 1. 小 batch 仍被固定到 DeepGEMM

[`TritonOrDeepGemmExperts._select_experts_impl`](../../../../vllm/model_executor/layers/fused_moe/experts/triton_deep_gemm_moe.py) 使用：

```python
if is_deep_gemm_e8m0_used() or _valid_deep_gemm(hidden_states, w1, w2):
    return self.experts
```

B200 当前启用 UE8M0，因此跳过原有的小 M 检查，M=1 也走 DeepGEMM grouped GEMM。已有 profile 确认实际执行了该路径。BF16 standalone 则在小 M 时回退到已经调优的 YOCO Triton W13/W2；大 M 使用 FlashInfer CUTLASS。两者不是仅替换了权重 dtype 的相同算子链。

这是性能分派缺口，不是已证明的数值错误。不能直接删除 UE8M0 条件：切换后端还要匹配权重与 scale 的布局、量化语义和路由加权顺序。应为小 M 适配 FP8 kernel 并实测切换阈值。

## 2. padding 后的激活量化仍做大量无效工作

[`compute_aligned_M`](../../../../vllm/model_executor/layers/fused_moe/deep_gemm_utils.py) 已把 padding 上界从所有专家收紧到可能活跃的专家，但当前每个专家仍按 128 行对齐。M=1、top-8 时只有 8 条有效专家输入，缓冲区为 1024 行。

[`_silu_mul_quant_fp8_packed_kernel`](../../../../vllm/model_executor/layers/quantization/utils/fp8_utils.py) 按整个缓冲区启动，先加载并计算 SiLU，再乘行权重；没有按实际有效专家行跳过 padding。profile 的 launch grid 与该计算一致：

| MoE 输入 M | 有效路由行 M×8 | 量化缓冲区行数 | 观察调用数 | SiLU/量化平均 µs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 1024 | 280 | 8.24 |
| 16 | 128 | 16384 | 40 | 82.64 |
| 128 | 1024 | 17280 | 40 | 86.02 |

M 来自该 profile 中 scatter kernel 的分派形状，不等于客户端并发；缓冲区行数也不能解释成 GEMM 的 FLOPs 按同样倍数增加。该量化 kernel 累计 9.05 ms，占该 profile 的 kernel 时长之和 10.68%。统计仅累加 `cat=kernel` 事件，排除嵌套 graph annotation、CPU、memcpy 和 memset；不是端到端耗时占比，也不是可直接兑现的加速比。

优先方向：让激活量化只处理有效路由行，并保持 CUDA graph 的静态缓冲区与地址；进一步为小 M 避免整套按专家分桶、padding 和还原流程。

## 3. MoE scale 和布局处理没有完全融合

profile 中每次 MoE 可观察到输入量化、专家计数、scatter、FP32 scale 到 UE8M0 的 pack、独立路由权重 scatter 和 gather。仅 `transpose_and_pack_fp32_into_ue8m0` 就执行 360 次，累计 0.62 ms。它不是最大的单项，但证实仍有中间格式转换和额外 kernel。

当前 dense FP8 linear 的平台量化算子已经直接生成 packed scales，MoE 的 permute 则仍建立 FP32 scale 缓冲区。可在 MoE scatter 时直接生成目标 scale 布局，并合并路由权重写入；需要独立检查舍入、padding 和无效专家处理。

## 4. BF16 的部分 Fast 优化尚未移植

[`yoco.py`](../../../../vllm/model_executor/models/yoco.py) 中以下优化仍要求 `quant_config is None`：self-attention 的 QKV+lambda 合并投影、cross-attention 的 Q+lambda 合并投影、模型级 K/V 合并投影，以及 shared-expert 的单行 down-projection 转置缓存。

当前 FP8 因而有更多独立 GEMM/量化调用；profile 已确认这些 FP8 dense GEMM 实际执行。Q/QKV 与 lambda 的精度不同，lambda 保留 BF16，不能直接去掉条件或把 BF16 kernel 用到 FP8 权重上。模型级 K/V 同精度合并、混合精度小投影融合，以及 small-M FP8 dense kernel 都需要分别实现和验证。

LM head 和 cross-attention weighted RMSClip 的 BF16 Fast kernel 已经在上一轮恢复，profile 确认执行，不应再把它们列为本轮未生效项。权重量化发生在加载阶段，也不存在每步重新量化全模型权重的问题。

## 后续顺序与结论边界

1. 优先减少 MoE 激活量化的 padding 工作，并建立小 M FP8 专家分派。
2. 合并 MoE scatter、scale pack、路由权重处理。
3. 补齐可保持精度语义的投影融合和 small-M dense 优化。
4. 每项先检查实际 log-prob 有限性和必要数值回归，再做同卡、同输入、同图模式的单项 A/B，最后复测 Mooncake。

FP8 的张量核算力和权重带宽优势不会自动转化为端到端加速：小 M 的利用率、额外量化、padding 和 kernel 调用都参与耗时，attention/KV/归一化等仍使用原精度。本轮确认了具体性能缺口；尚不能保证修复后的 FP8 一定快于 BF16，或给出每项预期增幅。
