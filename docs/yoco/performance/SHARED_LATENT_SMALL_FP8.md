# Shared / latent 专用小矩阵 FP8 路径：准备与实测

后续接入：[M1 默认分派、回退开关与整模型验证](SHARED_LATENT_M1_INTEGRATION.md)。
下文保留原型阶段的独立算子测量及当时的准备状态。

2026-09-16 PDT，基于 `vllm-yoco-version-0.29` 的分析分支
`investigate/yoco-v029-fp8-decode-20260916`，起点 `5e4e2cf693`。

**原型和实验入口已准备好。当前值得进入整模型试验的是 M1 的直接 GEMV 路径。**
四个投影轮转测量中，M1 纯 GEMM 平均耗时下降 32.6%，包含独立输入量化后下降
21.3%；M2/M4/M8 整体均回退。模型默认分派仍使用原 DeepGEMM，本轮没有接入
自动模型分派，也没有把算子结果表述为整模型吞吐提升。

## 四个目标矩阵

输入 `[M,K]`、权重 `[N,K]`，输出 `[M,N]`。M 是实际 GEMM token 行数，
应以引擎实际执行/Graph padding 后的 M 分派，而非客户端并发数。

| 投影 | K → N | 每个 decode token 的逻辑调用数 |
| --- | --- | ---: |
| Shared gate/up | 3072 → 2560 | 40 |
| Shared down | 1280 → 3072 | 40 |
| Latent in | 3072 → 1024 | 40 |
| Latent out | 1024 → 3072 | 40 |

真实 checkpoint 共 20 个物理层。10 个 self 层执行三轮，再执行 10 个 cross 层，
所以每步共 160 次这四类投影调用。原型只支持这四个 shape、SM100、M1..8、E4M3 输入
及权重、INT32 packed UE8M0 scales、BF16 输出；不匹配时显式报错。

## 两种候选做了什么

[small_fp8.py](../../../vllm/model_executor/layers/yoco_ops/small_fp8.py) 提供
`small_fp8_mm(a, weight, a_scale, weight_scale, config)` 显式接口。

**Tensor Core 小 tile。** 把矩阵观察方向交换，使用输出列 N=16/32/64、M tile=16、
K tile=128。直接读取现有 packed scales，每个 K block 执行 FP8 `tl.dot`，
乘 scale、FP32 累加，最后写 BF16。扫描 stages=2/4、warps=4。它增加小 N 下的
输出 tile 数，但仍有 block-scale 处理和 K 循环开销，实测没有成为 M1 最优。

**直接 GEMV。** 每个 CTA 负责 1/2/4 个输出列，读取该列完整 K 的 FP8 权重，
在寄存器中转换并应用 scale，使用 FP32 乘法和归约；多条 M 行复用已加载的权重。
扫描 warps=4/8、stages=1。它保留 FP8 存储和输入量化，没有另行物化整份
BF16/FP32 权重；矩阵乘本身使用普通 FP32 运算，不使用 FP8 Tensor Core。

M1 最优的四个投影均为：

```python
SmallFP8Config(backend="direct", block_n=1, num_warps=4, num_stages=1)
```

这解释了“FP8 小矩阵为什么不一定适合通用 GEMM”：M1 几乎没有跨 token 的权重复用，
较多独立输出 CTA 和较短的执行流程可以胜过通用 Tensor Core 的固定成本；M 增大后，
直接路径的逐行计算增长，而 DeepGEMM 能更充分复用矩阵乘资源。

## B1 单层热缓存初筛

同一 B200，真实 layer 0 权重、固定随机 BF16 输入，经现有 native quantizer 得到
同一份 FP8 operands。以下只计 GEMM，五轮随机顺序的 CUDA Graph 计时取中位数。

| 投影 | Native DeepGEMM | Direct GEMV | 耗时下降 |
| --- | ---: | ---: | ---: |
| Shared gate/up | 4.922 µs | 4.212 µs | 14.4% |
| Shared down | 3.588 µs | 3.337 µs | 7.0% |
| Latent in | 4.615 µs | 2.564 µs | 44.4% |
| Latent out | 3.332 µs | 2.825 µs | 15.2% |

这些是热缓存数值。不能把 latent in 的 44.4% 当作完整 MoE 或整模型收益。

## 全部 20 层、80 个矩阵轮转

读取 checkpoint 的全部 20 层，每层四个矩阵，总 FP8 权重 payload
**361,758,720 bytes，约 362 MB**。轮流执行 80 个投影，以减少单层热缓存偏差；
没有将其称作严格冷缓存测试。每个投影使用独立随机输入，没有串接成整模型。

每个 M/shape 从 12 个候选选出热缓存最优者，再在全权重池验证。表中的“候选”
始终是两个原型中选出的实现，即使它比 native 慢；这样才能看出替换边界。

| M | Native GEMM / µs | 候选 GEMM / µs | Native 量化+GEMM / µs | 候选量化+GEMM / µs | 后者耗时变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.269 | 3.551 | 7.651 | 6.020 | **−21.32%** |
| 2 | 5.265 | 5.356 | 7.673 | 7.892 | +2.86% |
| 4 | 5.263 | 6.863 | 7.733 | 9.510 | +22.97% |
| 8 | 5.267 | 7.413 | 7.757 | 10.128 | +30.57% |

以上是每个投影的平均时间，七轮随机顺序取中位数。量化阶段使用现有 group-128
native 输入量化，四个投影同样处理。实际 shared down 前还有融合 SwiGLU/量化，
latent 周围还有 Norm；本表没有代替那些链路，更没有覆盖 shared/routed 的重叠。

据此，第一轮整模型接入范围应限定 **M1 的四个投影**，其他 M 保留 native。
M2 的 latent in 单项仍有热缓存收益（4.615 → 3.337 µs），但只替换这一项的
混合分派尚未做轮转和整模型验证，先不扩大推荐范围。

若单次节省 1.63 µs 能保持到模型的 160 次调用，机械外推约 0.26 ms/token。
这只是几个百分点的整步优化潜力，实际还受重叠、依赖和路由变化影响，不能直接
承诺 170 → 179 tok/s，更不能将 21.3% 当作整模型提速。

## 数值与测试

候选直接消费原量化结果和 packed scales，不改变 group-128 量化、权重存储和
最终 BF16 输出。但 GEMM 的归约顺序不同，**当前结果不是逐位等价**。

| M | 80 个投影中相对 native 改变的元素 | 总输出元素 | 最大绝对差 |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 194,560 | 0.015625 |
| 2 | 26 | 389,120 | 0.015625 |
| 4 | 41 | 778,240 | 0.0625 |
| 8 | 73 | 1,556,480 | 0.0625 |

M1 对独立 FP64 反量化 operands 参考的最大相对 L2 误差为 `9.49e-5`，所有已测
输出有限。第一层的 M1 最优候选均与 native 逐位相同，但扩展到全部层后出现上述
差异，因此不能用第一层检查替代整模型精度验收。

[现有 latent FP8 测试](../../../tests/kernels/test_yoco_fp8_latent.py) 扩展到四种
shape，共 **60 passed**。新增覆盖 M1/2/4/8、两种候选、FP64 参考、输入/权重/
scale 不被修改、Graph 回放时输入和 scale 更新、零输入及合法 scale stride 变化。
原有 native online FP8 测试也通过。零输出比较器补上分母下限，避免将正确的全零
结果算成 `0/0`；首轮失败记录保留，不计入最终通过结果。

这些检查验证算子接口和数值误差。本轮没有运行接入候选的 teacher-forced NLL、
greedy 生成或模型 ABBA，不代表模型质量验收。

## 如何使用准备版

显式实验调用如下；生产模型没有自动引用该接口：

```python
from vllm.model_executor.layers.yoco_ops.small_fp8 import (
    SmallFP8Config,
    small_fp8_mm,
)

# a / a_scale 使用当前量化器的输出，不另行换量化规则。
# 已融合 SwiGLU/量化的 shared down 直接复用其 FP8 输出和 scales。
config = SmallFP8Config("direct", block_n=1, num_warps=4, num_stages=1)
output = small_fp8_mm(a, layer.weight, a_scale, layer.weight_scale_inv, config)
```

正式接入时，建议在这四类 YOCO Fast linear 的 quantized-GEMM 边界挂可回退分派：

1. 首轮只允许 SM100、TP1/DP1、原 native DeepGEMM UE8M0 格式及实际 M1。
   其他 M、模型、精度和并行配置保留原实现；开关在编译/Graph 建立前设置，并参与
   编译缓存标识。保持 bias、输出 reshape、Norm 与 shared 融合量化边界。
2. 用同卡独立 engine ABBA 检查 B1，并以 B2/B8 和较大 prefill 验证回退。
   专门核对实际调用次数，防止只设置开关却没有走候选。
3. 使用固定 teacher-forced token 的 raw logprob/NLL、greedy 输出及长上下文检查
   数值传播。当前有少量 BF16 差异，不能仅凭小于 `1e-3` 的算子 L2 就开启默认值。

## 复现与证据

在同样的 B200/native DeepGEMM 环境中运行：

```bash
.venv/bin/python benchmarks/kernels/benchmark_yoco_small_fp8.py \
  --model /mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf \
  --batches 1 2 4 8 --output /path/to/new-result.json
.venv/bin/python -m pytest tests/kernels/test_yoco_fp8_latent.py -q
```

需要 `VLLM_USE_DEEP_GEMM=1`、`VLLM_USE_DEEP_GEMM_E8M0=1`，使用环境已有原生库。
正式计时前完成加载、量化、JIT、正确性检查和 Graph capture。192 个初筛候选均
通过其测试输入上的 FP64 对照；完整轮转验证另覆盖 320 个投影用例。

[结构化结果](SHARED_LATENT_SMALL_FP8.json) 保存四档 M、全部初筛 tactic 的计时
与误差、轮转样本、源码/原始文件哈希及测试统计。工作区原始记录：
`work/yoco-small-fp8-20260916/`，有效测量为 `small-fp8-screen-r2`、
`small-fp8-screen-m24-r1`，最终测试为 `small-fp8-tests-r2`。

同一单卡 B200，Torch `2.13.0+cu130`。DeepGEMM 模块和 `_C` 实际来自 Docker
原生 `/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/`，
`_C` SHA256 保持
`73824dc1312e98cf277ae6bc017becf7e963d5b0bfab81e0c6d475d9b769c9e7`。
本轮使用独立候选源码目录，保留原 Job/Pod/holder 身份，结束后恢复此前调试服务。
恢复后的 18888 健康检查、8-token 真实生成和空队列检查均通过。
