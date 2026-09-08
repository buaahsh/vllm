# Align GEMM B200 数值验收

2026-09-06，在 `yoco-align-prob-b200-20260901-master-0` 的同一张 B200 GPU 5 上完成。
vLLM 与 llm-train 使用同一个 profile 选择器，保持 K=32、split-K=1、原 BF16 舍入位置和
Top-8 顺序累加。新配置默认关闭；不能仅因没有 split-K 就推断所有输入 bitwise。

## 结果与范围

| 检查 | 比较数 | 结果 |
| --- | ---: | --- |
| 基线 vLLM decode、prefill 和实际 batch | 167 | 全部逐字节一致 |
| 基线训练带梯度前向、完整概率和 CE | 116 | 全部逐字节一致 |
| 候选 vLLM，包括与基线直接比较 | 191 | 全部逐字节一致 |
| 候选训练带梯度前向、完整概率和 CE | 116 | 全部逐字节一致 |
| 基线与候选缓存命中、chunked/mixed prefill | 120 | 全部逐字节一致 |
| 实际 16K／32K 行合批 prefill | 15 | 全部逐字节一致 |
| 合计 | **725** | **最大绝对差 0** |

模型为真实 `30A3B-180M-L3/0000-28000` 的对应 HF/native 权重，BF16、单 rank，
Router FP32、FA2、16-token page、固定单 split。完整词表宽度154880。
训练使用 `model.train()` 且开启梯度；不是完整生产 NNScaler 编译训练图。

核心序列长度为33/129/512/1024/2048/4096/8192，包含嵌套生成前缀，不能当作相同数量的
独立样本。实际 decode batch 为1/8/32/64/128/256，目标位于首/中/尾；prefill/训练包含
单请求与 ragged B3。对比 hidden、BF16 logits、完整 FP32 log-prob 和 token CE，使用
字节比较而不是容差；含 signed-zero 负控制。

额外服务检查开启 KV-sharing fast-prefill、prefix cache 和 chunked prefill，确认实际
缓存命中128/496/2032/8176 tokens。16K/32K 合批通过2/4条约8K请求构造，并检查本次
执行新增的配置命中计数，避免把模型预热计数误当作测试命中。

算子层另有598组候选中间值检查、80组两端跨尺寸用例（470次字节比较）及74项回归测试。
前向 kernel 本体、概率归约与训练 backward 未修改；变化为共享 launch 配置选择。

## Profile

文件：`vllm/model_executor/layers/fused_moe/experts/yoco_configs/align_moe_NVIDIA_B200.experimental.json`。
精确匹配行数1/8/32/64/128/256/512/1024/2048/4096/8192/16384/32768；其他行数保留原配置。
W13/W2 共用 M tile，各自选择 N tile/warps/stages。

环境限定为 NVIDIA B200、Torch `2.11.0a0+eb65b36914.nv26.02`、CUDA13.1、Triton3.8.0。
加载器检查版本和形状；A6000 profile 不能替代 B200 profile。

两端进程使用同一个经过验证的文件：

```bash
export VLLM_YOCO_ALIGN_MOE_CONFIG=/absolute/path/to/align_moe_NVIDIA_B200.experimental.json
```

这仍是上述配置和输入范围的验证，不能扩展成任意长度、权重、batch、多卡、量化或
完整生产训练图的无条件保证。

## 验证脚本问题与处理

所有中止记录都保留，没有作为通过项计数：

- 8192-token native 审计先后因循环 `value` 和未使用的 checkpoint CE 辅助输出保留
  计算图而 OOM。释放引用后，每轮结束显存回到约60.34GiB；只补测尚未完成的位置。
- 候选长序列再遇完整词表缓冲的显存碎片，剩余两位置用独立进程和
  `PYTORCH_ALLOC_CONF=expandable_segments:True` 完成；未分块或缩减完整词表比较。
- YOCO 会将最后一个 prompt KV block 单独调度，最初“一次执行完整大批次”的断言错误。
  修正填充请求长度，使第一段实际为16384/32768行，尾段按原调度器执行。
- 比较器使用逐块精确字节比较；只有已确定完全相同的字节才复用 SHA256。47项对照与
  原比较器一致，没有以哈希或容差替代字节检查。

本地完整证据：`yoco_results/align-gemm-b200-20260906/`；大型原始 capture 保留于
Pod 的 `/data/yoco-align-gemm-b200-20260906/`。
性能结果应单独阅读，数值通过不代表一定加速。
