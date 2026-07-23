# YOCO-VL batch=16 最新 BF16 / FP8 对齐结果

更新时间：2026-07-23

## 基础镜像与 FlashAttention 版本

- Docker 基础镜像：`donglixp/pytorch:26.02-b200`
- 镜像 digest：`donglixp/pytorch@sha256:cdbdc71b773142a98a303d488816d2faf6b9d85d179a4639b92263fca6da4769`
- 实验容器：`wjh-b200`
- Python：容器系统路径 `/usr/bin/python`，版本 `3.12.3`
- PyTorch：`2.11.0a0+eb65b36914.nv26.02`
- FA2：`flash-attn==2.7.4.post1+nv26.2.44259020`
- FA4：`flash-attn-4==4.0.0b13`，其 `quack-kernels` 版本为 `0.4.1`
- 本报告所列最新 BF16 和 FP8 结果中，LLM attention 与 ViT attention 双方均
  实际使用 FA2；FA4 已安装在同一系统环境中，但不是这两组最新结果的执行后端。

## 本分支 Vision 代码改动

本节记录相对 `origin/shaohanh/yoco-0716`（`3ba0b76f4`）新增或修改的
Vision 生产代码，不包含 `workspace` 下用于 KL 对齐、权重转换和单次推理的实验脚本。

### 改动文件

| 文件 | 内容 |
|---|---|
| `vllm/model_executor/models/yoco_vl.py` | 新增 YOCO-VL 多模态模型、图像/视频预处理、prompt replacement、MoonViT 调用、projector 和权重加载 |
| `vllm/transformers_utils/configs/yoco_vl.py` | 新增组合 `MoonViTConfig` 与 `YOCOConfig` 的 `YOCOVLConfig` |
| `vllm/model_executor/models/registry.py` | 注册 `YOCOVLForConditionalGeneration` 多模态模型 |
| `vllm/model_executor/models/config.py` | 让 YOCO-VL 复用 YOCO 的模型配置校验逻辑 |
| `vllm/transformers_utils/config.py`、`configs/__init__.py` | 注册 `model_type=yoco_vl` 及配置类 |
| `vllm/transformers_utils/configs/moonvit.py` | 补充时间位置编码、视频 attention 和时序 pooling 配置字段 |
| `vllm/model_executor/models/kimi_k25_vit.py` | 时间位置编码在相加前转换到当前 ViT activation 的 device/dtype |
| `tests/models/multimodal/processing/test_yoco_vl.py` | 新增视频预处理及时间位置编码 dtype 单测 |

### 模型执行结构

当前实现的 Vision 路径为：

```text
<image> / <video>
  -> resize、patchify、normalize
  -> MoonViT3dPretrainedModel
  -> 按单张图片/单个视频 chunk 分开执行 YOCOVLMultiModalProjector
  -> multimodal embeddings
  -> YOCOForCausalLM
```

- Vision tower 复用 vLLM 已有的 `MoonViT3dPretrainedModel`，checkpoint 中的
  ViT 配置为 patch size 14、hidden size 1152、27 层、16 个 attention heads，
  patch merge kernel 为 `2×2`。
- `YOCOVLForConditionalGeneration` 将 `vision_tower`、`vision_projector` 和
  `language_model` 分开构建，并通过 `AutoWeightsLoader` 按这三个前缀加载 HF
  checkpoint 权重。
- 支持普通 Vision 执行及 `mm_encoder_tp_mode=data` 的 encoder data parallel
  路径；进入 ViT 前会校验 patch 总数是否与所有 `[T,H,W]` grid 一致。
- 输入 patch 在进入 ViT 前转换为 patch embedding convolution 的权重 dtype。
  本报告的 BF16 和 FP8 全量实验中，ViT 都实际运行在 BF16。

### 图像预处理

- 接受 PIL image、NumPy array 和 Torch tensor；array/tensor 同时兼容 HWC 与
  CHW，最终统一为 RGB。
- 使用 bicubic resize，并同时受以下限制约束：单图最多 4096 个合并后 token、
  单边最多 512 个 patch、原始 patch 总数上限 16384。
- 当前 `vision_align_mode=resize`：高宽会调整到 `patch_size × merge_size = 28`
  的整数倍，不额外补黑边；代码也保留了 `pad` 模式。
- resize 后先按 `14×14` patch 展开，再用 mean/std 均为 0.5 做 FP32
  normalization。图片 grid 记录为 `[1,H/14,W/14]`，最终 image token 数为
  `(H/28) × (W/28)`。

### 视频预处理

- 支持 frame sequence、THWC NumPy/Tensor 和 TCHW NumPy/Tensor 输入。
- 同一视频 chunk 的所有帧必须具有相同尺寸，所有帧共用同一个 resize 结果，
  patch grid 为 `[T,H/14,W/14]`。
- 时间位置编码默认最多覆盖 4 帧，因此单个视频 chunk 当前限制为 1～4 帧；
  更长视频需要调用方先采样或切 chunk。
- MoonViT 对视频执行 spatial-temporal attention 和 temporal pooling。相同空间
  grid 下，一个视频 chunk pooling 后的 placeholder token 数与单张图片相同。
- 修正了多帧输入时 temporal position buffer 仍为 FP32 而导致 activation 被
  提升到 FP32 的问题：相加前显式转换到当前 position embedding 的 dtype/device。

### Projector

当前 projector 与 checkpoint/llm_train 的结构一致：

```text
LayerNorm(1152, eps=1e-6)
  -> merge 2×2 patches，reshape 到 4608
  -> Linear(4608, 4608)
  -> GELU
  -> Linear(4608, 3072)
```

- 两个 Linear 使用普通 `torch.nn.Linear`，没有使用 TP shard；权重名称保持为
  `vision_projector.pre_norm`、`vision_projector.proj.0` 和
  `vision_projector.proj.2`。
- projector 模块显式保持 FP32；进入 projector 前，ViT 输出先转换到
  `pre_norm.weight.dtype`。
- 多图 batch 不会把所有图片的 ViT feature 合并成一次 projector GEMM，而是按
  单张图片/单个视频 chunk 分开执行，保证 batch=1 与 batch>1 的 projector
  执行 shape 一致。
- 因此 FP8 block-128 实验只量化语言模型的 linear/MoE；ViT 保持 BF16，
  projector 保持 FP32。

### Prompt 与多模态输入接入

- 同时注册 image 和 video 两种 modality，并分别传递 `pixel_values` /
  `image_grid_thws` 与 `pixel_values_videos` / `video_grid_thws`。
- 每个 `<image>` 或 `<video>` 被展开为
  `[image_start] + N×[placeholder] + [image_end]`；processor 会检查 placeholder
  数量与实际媒体数量一致，并按实际 resize 后的 grid 计算 `N`。
- tokenizer 是否增加特殊 token 由调用方的 `add_special_tokens` 控制，Vision
  processor 自身不额外硬编码 BOS。
- image 和 video 当前共用同一组 start/placeholder/end token ID。当同一 prompt
  混合 image/video、video 位于 image 前且二者 token 数相同时，两块 token 序列
  可能无法仅凭 token ID 区分；纯 image、纯 video 和不产生该歧义的混合输入不受
  影响。这是当前实现仍需补充处理的边界情况。

### 当前测试覆盖

新增单测覆盖以下行为：

1. 视频逐帧 patch 与分别按图片预处理的 patch 完全一致；
2. TCHW 与 THWC 视频输入得到相同结果；
3. 超过 temporal position embedding 上限时明确报错；
4. 多帧 BF16 ViT 的 temporal position addition 不会把输出提升为 FP32。

目前单测尚未覆盖 mixed image/video placeholder 顺序、完整 checkpoint 权重加载、
端到端生成以及 TP>1。

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
