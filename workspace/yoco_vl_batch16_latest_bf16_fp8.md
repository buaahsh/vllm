# YOCO-VL batch=16 最新 BF16 / FP8 对齐结果

更新时间：2026-07-23

## 实验环境与共同配置

- Docker：`wjh-b200`
- Python：容器系统路径 `/usr/bin/python`
- GPU：物理 GPU 6，llm_train 与 vLLM 在同一容器、同一张 GPU 上顺序运行
- checkpoint：`/mnt/nvme/wjh/updates_3000`
- vLLM 模型：`/mnt/nvme/wjh/updates_3000-hf-vl`
- batch size：16，由下面 4 个 image/prompt pair 连续复制 4 份
- prompt：no-BOS
- LLM attention：双方均为 FA2
- ViT attention：双方均为 FA2
- sliding window 配置：512；包含当前 token 的实际 attention span 为 513
- projector：双方参数均为 FP32，并按单张图片执行
- prefix caching：关闭
- chunked prefill：开启
- `max_num_seqs=16`
- `max_model_len=2946`
- `max_num_batched_tokens=34308`，等于当前完整 batch 的 token 数
- KL 行匹配：Hungarian；正常同序比较仍使用原始 batch 行顺序

4 个基础 query：

| Query | 图片 | Prompt | image tokens |
|---:|---|---|---:|
| 1 | `dog1.jpeg` | `Describe this dog and the surrounding scene in one concise sentence.` | 1849 |
| 2 | `dog2.jpeg` | `What is the dog doing? Mention its pose and expression.` | 1849 |
| 3 | `dog3.jpeg` | `What animal is shown, and what is visible in the background?` | 2916 |
| 4 | `dog1.jpeg` | `What colors are most prominent in this image?` | 1849 |

## 最新 BF16

llm_train 与 vLLM 的语言模型、ViT 和 KV cache 均按 BF16 运行；projector 保持双方 FP32。

| 指标 | 结果 |
|---|---:|
| Mean KL, llm_train → vLLM | 0.0041321907 |
| Mean reverse KL | 0.0042584087 |
| Mean JS | 0.0010424850 |
| Top-1 agreement | 100% |
| Probability L1 mean | 0.0506683551 |
| Probability L∞ max | 0.0406217873 |
| Logits relative Frobenius | 0.0368839316 |
| Logits global cosine | 0.9993202090 |

每组 4 个 query 的 KL：

| Query | KL | llm_train Top-1 | vLLM Top-1 |
|---:|---:|---:|---:|
| 1 | 0.0083304625 | 32 (`A`) | 32 (`A`) |
| 2 | 0.0007559316 | 785 (`The`) | 785 (`The`) |
| 3 | 0.0065113185 | 785 (`The`) | 785 (`The`) |
| 4 | 0.0009310504 | 785 (`The`) | 785 (`The`) |

上述 4 条重复 4 组，结果一致；第一组 Query 1 的 KL 为
`0.0083304578`，与后三组只相差约 `4.7e-9`。vLLM logits 调用 shape 为
`[1, 3072] + [15, 3072]`，共 16 行。Hungarian 匹配为 identity。

结果文件：

- `/mnt/nvme/wjh/yoco_diag_batch16_hungarian_all_fa2_bf16_effective_window513_dog3_fullbudget.json`
- `/mnt/nvme/wjh/yoco_diag_batch16_hungarian_all_fa2_bf16_effective_window513_dog3_fullbudget.log`

## 最新 FP8 block-128

- llm_train：`mxfp8`
- vLLM：`fp8_per_block`
- 双方量化 block：128
- FP8 格式：E4M3FN；scale：UE8M0/E8M0
- vLLM linear/MoE 后端：DeepGEMM
- ViT 不量化，保持 BF16
- projector 不量化，双方保持 FP32
- logits 输出 dtype：BF16

| 指标 | 结果 |
|---|---:|
| Mean KL, llm_train → vLLM | 0.0069935173 |
| Mean reverse KL | 0.0073053441 |
| Mean JS | 0.0017752955 |
| Top-1 agreement | 100% |
| Probability L1 mean | 0.0731801242 |
| Probability L∞ max | 0.0614914298 |
| Logits relative Frobenius | 0.0578167327 |
| Logits global cosine | 0.9983287454 |

每组 4 个 query 的 KL：

| Query | 第一组 KL | 后三组 KL | llm_train Top-1 | vLLM Top-1 |
|---:|---:|---:|---:|---:|
| 1 | 0.0134724360 | 0.0064455341 | 32 (`A`) | 32 (`A`) |
| 2 | 0.0030171562 | 0.0030171562 | 785 (`The`) | 785 (`The`) |
| 3 | 0.0114382375 | 0.0114382375 | 785 (`The`) | 785 (`The`) |
| 4 | 0.0053164149 | 0.0053164149 | 785 (`The`) | 785 (`The`) |

vLLM logits 调用 shape 同样为 `[1, 3072] + [15, 3072]`。因此第一条
Query 1 单独进入一个 logits 调用，另外 15 条进入第二次调用；FP8 下第一条
Query 1 与其三个副本出现了明显的执行-shape 数值差异。Hungarian 在完全相同的
Query 1 副本之间轮换匹配，这不表示 batch 行顺序错误。

结果文件：

- `/mnt/nvme/wjh/yoco_diag_batch16_hungarian_all_fa2_both_block128_effective_window513_dog3.json`
- `/mnt/nvme/wjh/yoco_diag_batch16_hungarian_all_fa2_both_block128_effective_window513_dog3.log`

## BF16 与 FP8 对比

| 指标 | BF16 | FP8 block-128 |
|---|---:|---:|
| Mean KL | 0.0041321907 | 0.0069935173 |
| Query 3 KL | 0.0065113185 | 0.0114382375 |
| Top-1 agreement | 100% | 100% |
| Relative Frobenius | 0.0368839316 | 0.0578167327 |
| Global cosine | 0.9993202090 | 0.9983287454 |

当前输入下，FP8 的 Mean KL 比 BF16 高约 69.2%，但双方所有 16 条的
Top-1 均一致。
