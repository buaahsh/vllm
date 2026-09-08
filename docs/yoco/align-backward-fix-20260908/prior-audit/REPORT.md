# YOCO bitwise 独立审计（2026-09-07）

## 结论

不能笼统地说两个分支默认就是 bitwise 一致。

- **显式 Align 前向：通过。** 在目标分支声明的 B200/SM100、BF16、TP/CP/EP=1、
  FA2、MTP 关闭、128 experts/Top-8 配置下，真实 L3 checkpoint 的 llm-train
  teacher-forcing、llm-train 原生逐 token KV cache、vLLM cached decode 三者逐位相同。
- **默认 Fast/FA4：不通过。** 同一 prompt、同一 8 个生成 token、同一 checkpoint 下，
  vLLM Fast 与 llm-train Align 的 1,239,040 个 logits 中有 909,286 个不同
  （73.386331%），最大绝对差 0.6875。Fast 输出为 FP32，Align 输出为 BF16。
- **完整训练链路：未通过。** 前向和概率测试通过，但 grouped MoE backward 在
  `llm/kernel/align_moe_backward.py::_route_map` 触发 CUDA illegal memory access。
  该分支本来也不承诺 backward/梯度累加/优化器更新逐位一致，但当前结果还是一个
  独立的可用性问题。

因此准确表述应为：**这两个 commit 已实现并通过抽样验证的 Align/FA2 前向 bitwise
合同；Fast/FA4 没有实现 bitwise；不能把结论扩展到 backward、多卡或默认启动参数。**

## 固定源码

- llm-train `fhb-dev`: `cc700436e03fd6592090ff4ede747cac794d6458`
- vLLM `shaohanh/yoco-260906`: `3a0dba165aa0b453b8cb98f0a3fc4a5b92b74735`
- NNScaler dependency: `119636742ed29505ee58ccc440284c7c667f9458`

源码由新的 Git clone 生成完整 bundle 后传入 Pod；bundle SHA256 分别为：

- llm-train: `86d9e10bab1f397f784eb1e47a18b5b3c5ecdeb8bdf5ecaf3e68011703b2af8f`
- vLLM: `c7a4d45cd50045353cbcc429afbc41b43194005439f8721b6e613d42f51fc09c`
- NNScaler: `fa105422121c762d4ea66c87fd2ee6e05c262a53fa6e0f70358e2288203db788`

两个目标仓库的 tracked diff 均为空。开发中的现有工作树未参与执行。

## 独立环境

- Kubernetes: `oidc@msr02 / bonete01`
- Job/Pod: `yoco-bitwise-audit-v2-260907 / yoco-bitwise-audit-v2-260907-audit-0`
- Pod UID: `afc05ce8-5aa9-4242-9c45-8aa26059215a`
- Node/GPU: `slc01-cl02-hgx-0098`, NVIDIA B200 SM100,
  `GPU-3b237c2f-1dd4-eacd-c44c-082d6a8f3355`
- 基础镜像：目标 vLLM `docker/Dockerfile.b200.pd` 固定的
  `buaahsh/pytorch@sha256:08a08f36ab8c6c80ee1c7f09b9e5f8b6ce0b91cc684455982b2e5e286c736f2b`
- Python 3.12.3；PyTorch `2.11.0a0+eb65b36914.nv26.02`；CUDA 13.1；
  Triton 3.7.1；FlashAttention 2.7.4.post1

Runtime 按目标 Dockerfile 的 overlay 语义组装：YOCO Python 来自目标 commit，
FA2 和 `_moe_C` 来自固定基础镜像，`_C` 从目标 commit 针对 SM100 单独编译。

关键 SHA256：

- target `_C.abi3.so`: `a0933ea95ee02af676b55686e4f135e12ebb2119c14160b87312c1f43b45bc3b`
- base `_moe_C.abi3.so`: `77a12ed3d9b762e96fa855b342f28bdbebee154d29342d0ef988ff5ef9a381d9`
- base FA2: `edf85f872b3bb6bfcf344774fb94df9be264bb902f426534137c419428080266`
- target `yoco.py`: `30f32f1a7dfb80a821f2f3ffb59bafc5691fa3b19609540aabc5d53a82ac55dc`

## 测试结果

| 检查 | 结果 |
| --- | --- |
| vLLM Align kernel/probability | 34 passed |
| vLLM 模型级 Align 专项 | 26 passed，127 deselected |
| llm-train probability/CE（独立进程） | 5 passed |
| llm-train Align 前向、独立/合批/ragged | 前向断言通过 |
| 真实 L3 Align：train teacher-forcing vs vLLM decode | equal=true，different=0，max_abs=0 |
| 真实 L3 Align：train teacher-forcing vs train KV cache | equal=true，different=0，max_abs=0 |
| 真实 L3 Fast/FA4 vs train Align | equal=false，different=909286/1239040，max_abs=0.6875 |
| grouped MoE backward | 失败：`_route_map` CUDA illegal memory access |

真实模型为 `30A3B-180M-L3/0000-28000`，hidden=3072、head_dim=128、20 layers、
10 cross layers、universal loop=3、vocab=154880、128 experts/Top-8。测试 prompt 15
tokens，比较 8 个预测位置的完整词表 logits。

## 证据

- Align 真实模型结果：`pod-results-v2/l3/native-comparison.json`
- Align reference logits：`pod-results-v2/l3/reference-logits.pt`
- Fast 负对照：`pod-results-v2/fast/native-comparison.json`
- Fast reference logits：`pod-results-v2/fast/reference-logits.pt`
- Fast wrapper（未修改仓库）：`check_fast_checkpoint.py`

## 边界

本轮没有证明任意输入、长上下文、多卡、量化、MTP、长期训练或不同依赖版本下都逐位
一致。结论只覆盖上述 commit、环境和验证矩阵。FA4 属于 Fast 路径；Align 在两边都固定
使用 FA2。
