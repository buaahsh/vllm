# YOCO Shared-KV：可选 FP8 KV cache

使用已有的 `kv_cache_dtype` 配置项，在引擎启动时固定 KV 精度。默认 `auto`
继续随 BF16 target 使用 BF16 KV；不增加按 batch、并发或长度切换精度的分支。

| 配置值 | Shared-KV 缓存 |
| --- | --- |
| `auto`（默认）、`bfloat16` | BF16，原有读取路径 |
| `fp8`、`fp8_e4m3` | CUDA E4M3，draft 按 target writer 的 K/V scales 反量化读取 |

FP8 KV 主要用于容量。相同 token 容量的 KV 数据每元素从2字节降到1字节，
attention 仍有量化、反量化和读取成本，不能由此推断吞吐更高。
当前 FP8 draft 在读取时反量化到 BF16 寄存器，沿用 Triton attention；
BF16 draft 保留原 FA4 路径，没有建立一份常驻 BF16 KV 副本。

## 开启和关闭

在已有 `vllm serve` 启动参数中添加：

```bash
--kv-cache-dtype fp8_e4m3
```

恢复原精度：

```bash
--kv-cache-dtype bfloat16
```

服务 YAML 使用同一配置项：

```yaml
kv-cache-dtype: fp8_e4m3
```

Python API 在已有模型和 draft 配置中设置：

```python
from vllm import LLM

config["kv_cache_dtype"] = "fp8_e4m3"  # config includes the paired target/draft.
llm = LLM(**config)
```

原生 benchmark 也接受相同开关：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
  .venv/bin/python -m tools.yoco_alignment.benchmark_mega_moe \
  --backend hybrid --fast-prefill --synchronous-engine \
  --kv-cache-dtype fp8_e4m3 --kv-cache-gib 4 \
  --max-model-len 12288 --concurrencies 1 8 --tokens 128 \
  --output /new/path/fp8-kv
```

对照 BF16 时改用 `--kv-cache-dtype bfloat16 --kv-cache-gib 8`，其余配置固定。
如果仍提供相同 KV 显存预算，引擎通常会分配更多 token 容量；进程总显存不一定下降。
需要降低实际显存时，应同时调小已有的 `kv_cache_memory_bytes` / `--kv-cache-gib` 预算。
流式 benchmark 的 config JSON 同样使用 API 键 `kv_cache_dtype`。

## 存储、尺度与边界

- Shared-KV draft 只读 target owner 的 E4M3 页面，K/V 分别使用对应的 FP32 scalar scale。
  不把缓存字节直接当作 BF16，也不静默假定 scale=1。
- scale 由 CUDA graph 中的 kernel 读取，更新原张量后重放可以读取新值。
  `fast-prefill` 仍要求固定尺度，保留已有的动态尺度冲突检查。
- 保留严格 prefix、SWA8192、空行、非连续 request slot、HND/NHD 布局与 int64 页面寻址。
  当前 CUDA Shared-KV 入口接受 E4M3，不扩大到 E5M2/FNUZ 或 DeepInfer 的 MXFP8/group-scale 格式。
- target/draft 的 BF16 hidden、raw h20、FP32 proposal/rejection 和 BF16 词表参数保持原配置。
  原生 target FP8 KV attention 可能同时量化 Q，质量数据按完整原生路径解读。

## 质量与验证

既有完整 MATH-500、每题128K输出上限的冻结对照：BF16 KV **490/500（98.0%）**，
E4M3 KV **486/500（97.2%）**，下降 **0.8 个百分点**。两侧各一题提取器格式误判经复核后
分别为491/487，差距不变。单种子配对区间为[-2.2,+0.6]个百分点，不能据此保证所有任务损失范围。

本次沿用其 FP8 读取算术；补充默认关闭/显式开启的集成验证和尺度错误检查。
配置、正确性、容量和诊断计时见[开关验收报告](fp8-kv-switch-20261022/REPORT.md)。
