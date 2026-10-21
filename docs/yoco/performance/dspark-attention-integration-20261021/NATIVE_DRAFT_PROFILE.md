# 真实 target 生成过程中的 draft 整步计时

Archived pre-integration experiment. Relative source/data paths below refer to
`/data/vllm_test/work/yoco-dspark-attention-20261021/`.

本轮直接在原生 `SharedKVSpeculator.propose` 入口及其 `run_fullgraph` 前后记录 GPU events。计时包含输入准备、真实四层骨干、154880 词表投影和 K=8 顺序概率采样；target 前向在 start event 之前，已排除。KV、h20 条件 hidden、位置和候选采样均来自真实 target 生成流程。

单 B200，TP1，原 current target MoE、同一 step30000 draft。原 Triton → FA4 顺序各跑一次。每种上下文、每档并发先热身两次，再正式生成两次，每请求 256 输出 token。每轮为一整批 C 个请求各提出 8 个候选，表内为完整 propose 的单轮耗时中位数。

## 8193-token 输入（有效历史窗口为 8192）

| 并发 C | 原 draft | FA4 draft | 加速倍数 | 耗时降低 | 统计轮数 原/新 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.308 ms | 2.494 ms | 1.326× | 24.61% | 236/255 |
| 4 | 3.455 ms | 2.692 ms | 1.284× | 22.09% | 175/169 |
| 8 | 3.595 ms | 2.913 ms | 1.234× | 18.96% | 139/144 |

当前真实执行中，draft 本身加速约 1.23–1.33 倍，一轮节省 0.68–0.81 ms。

## 短输入（24–30 token，再生成 256 token）

| 并发 C | 原 draft | FA4 draft | 耗时变化 |
| ---: | ---: | ---: | ---: |
| 1 | 2.386 ms | 2.449 ms | +2.67% |
| 4 | 2.515 ms | 2.580 ms | +2.60% |
| 8 | 2.649 ms | 2.713 ms | +2.43% |

短历史下 FA4 整步慢约 2.4%–2.7%，约 0.064–0.065 ms/轮。

## 口径与证据

统计选择实际 live=C 且 graph padded batch=C 的轮次；长上下文还要求每个请求 prefix>=8192。起始分批、长输入的未完成 prefill、结束时尾批和 padding 轮次均保留在各 `trials.json`，不混入上表固定形状的比较。`summary.json` 同时记录所有调用数、实际 prefix 范围、graph/propose 均值与中位数和两个 trial 的独立中位数。

此前 `draft-results.json` 的 1.26–1.36× 是独立微基准，使用真实权重及合成 KV/hidden，不能代替原生整步。当前问题以本轮真实生成中的 1.23–1.33× 为准。

GPU event 与 metadata clone 仅用于这次实验。prefix clone 排在 end event 之后，未计入 propose 耗时；两组使用同一套 instrumentation。这里不把带 profiling 开销的 generate 吞吐当作新的端到端性能面板；未插桩的 A/B/B/A 吞吐仍见上一轮报告。

两组 native 进程均退出 0。计时脚本为 `native_profile_backend.py`、`native_profile_benchmark.py`、`run-native-profile.sh`，原始数据在 `native-profile/`。GPU 7 已恢复 0 MiB，生产源码没有新增修改。
