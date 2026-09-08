# Qwen3 普通模式：P/D 路径存在所选 token log-prob 差异

日期：2026-09-07。本问题是本轮吞吐实验追加功能检查的发现，尚未定位根因，也未作修复。

同一个 Qwen3-30B-A3B-Instruct-2507 BF16 checkpoint，四组循环 token 输入长度511/512/513/4097，每组生成32 tokens，greedy、seed42、ignore_eos、独立cache salt：

- 原始单卡 GPU5、本地 P GPU4、本地 D GPU5 的已测生成 token 及所选 token log-prob 相同（数值最大差0）。
- 经过 Mooncake P/D 代理之后，4组生成token仍相同，但所选token log-prob最大绝对差分别为0.3886835575、0.2370193005、0.3592822552、0.1594758630。
- 另外两个带自然语言问题尾部的513/4097-token输入，本地P/D所选token log-prob差0，经过代理后的最大差分别0.1202327609/0.0899816751，生成token仍相同。

这把差异范围缩小到了P/D执行路径，但没有证明是浮点舍入，也没有区分增量计算、attention/GEMM路径、缓存交接元数据或返回概率路径。不能把这些数据解释为Qwen的数值对齐验收通过。当前性能数据只代表记录下来的普通Qwen服务实现，不是数值严格等价实现之间的受控对比。

`functional-comparison.json` 保存计时前检查。`post-timing-functional/SUMMARY.json` 和同目录逐长度JSON保存计时外复核（包含实际输入token IDs、完整响应、所选log-prob以及每组对比）。全部复核使用长测结束后仍运行的P/D引擎，没有重新加载权重；开始时间晚于长测完成与排空时间，未混入AIPerf计时。复核完成后再次排空并停止服务。

这里只比较已生成token的概率，没有比较全词表logits/log-prob；生成token相同只覆盖6组探针，并非全trace输出对照或通用bitwise保证。

复现入口：`post_timing_functional.py`。需要匹配的服务配置、完整长测排空证据及`active.json`拥有权记录。当前四卡Job已保留为空闲状态；重新启动服务属于新的测试轮次，应使用新结果目录与cache salt。
