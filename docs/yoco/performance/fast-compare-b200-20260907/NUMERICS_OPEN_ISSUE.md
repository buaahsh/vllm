# Fast 单卡 / P-D 数值探针差异（未定位）

日期：2026-09-07。来自正式AIPerf计时外的启动功能探针。

同一YOCO L3 checkpoint、固定token输入、temperature0、seed42，各生成32个token。比较Fast单卡GPU5与Fast 1P1D P4/D5。

| 输入token数 | 输出token相同 | 所选token log-prob最大绝对差 |
| ---: | --- | ---: |
| 511 | True | 0.13219839334487915 |
| 512 | True | 0.11818569898605347 |
| 513 | True | 0.08655852824449539 |
| 4097 | True | 0.0 |

四组生成token均相同；最大log-prob差0.13219839334487915。Fast不承诺bitwise，但此差异的来源尚未隔离，不能直接称为普通舍入。

当前比较包含单卡与P/D的执行和MoE后端策略差异。没有在本轮追加本地P/本地D/代理的控制实验，因此尚不能判定差异来自GEMM后端、增量计算、KV交接元数据还是概率返回路径。也没有比较全logits或全trace生成内容。

原始响应见standalone-functional.json和pd-functional.json，逐组结果见functional-comparison.json。吞吐/输出长度门槛与数值对齐结论分别记录。
