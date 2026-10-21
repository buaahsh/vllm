# Shared-KV draft 线性缓存与保原精度 norm 融合整合

将会话 `01a15cd9-0163-7163-b4cb-d68fa74edcd0` 最终采用的优化，
合入 `fhb-dev-aligndeepinfer-startpoint` 的 `fd79a66bfc` 基线上。
合并范围是冻结 BF16 线性权重计算副本，以及保持原 norm 数值语义的 RMS/residual-add 融合。

## 实现与来源

- 线性权重在加载后缓存 BF16 计算副本，减少每轮 draft 的重复转换；
  checkpoint 原参数与 FP32 residual 保留，计算缓存不写入 state_dict。
- norm 保留原 FP32 gamma、平方与 Torch mean 归约、输出 dtype 和舍入次序。
  两个 Triton 逐元素 kernel 包围原归约；residual-add 合并到第一步，关闭 FMA 重排。
- 原地更新计算副本，保持已捕获图的地址；普通同形状重载后图可继续回放。
  计算副本支持 `assign=True` 刷新并保留地址；全模型 `assign=True` 替换其他 FP32
  参数地址时，仍需重建相关图。
  当前 draft 新增约 1.199 GiB BF16 线性计算副本和 126 KiB FP32 norm gamma 副本。
- 当前分支已经有 FA4、FP32 proposal q-cache 和 FP8 target 兼容；本次复用已有实现。
  SWA 的 FA4 路径与 int64 物理页地址保护文件也与接受版一致，直接沿用。
  原会话补丁里的 OPD 采样/提议捕获 hook 与本次性能优化无关，未引入。
- 保留已修复的 fast-prefill 行为：仅纯中间 prompt batch 跳过后半段及 draft，
  输出步返回真实 raw h20，decode 依据拒绝数选择有效位置。

源模型文件和新 norm 文件的 SHA256 与最终接受记录一致；测试在当前套件上合并，
保留当前分支的拒绝/q-cache/FA4 测试，并增加原会话的缓存、精确 norm 和重载覆盖。
详见 [来源与选择记录](source-session.json)。实现使用 AI 辅助整合。

## 合并后的验证

本地模型与 runner 合计 **159 passed，4 项 SM90+ 用例跳过**；
B200 补齐后 **163 passed，零跳过**（模型/prefill/Shared-KV 89，runner/attention/sampler/rejection 74）。
新增 norm 检查使用 `rtol=atol=0`，覆盖 BF16/FP32、128/3072 维、原 gamma 提升规则、
residual 和 graph 捕获后重载；保留原有拒绝锚点、KV 因果性与 FA4 窗口测试。

两个隔离目录分别逐项核对父版 8033、合并版 8034 个源码/链接条目，
以及各 1996 个已有运行支持文件，全部一致。

完整命令与两次测试目录初始化问题见 [validation.json](validation.json)；
初始化问题均发生在收集阶段，已保留原始失败日志，最终成功测试未放宽断言。

## 合并后的端到端对照

| 输入 | C | 合并前 tok/s | 合并后 tok/s | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: |
| short | 1 | 328.46 | 336.18 | +2.35% |
| short | 4 | 888.05 | 896.82 | +0.99% |
| short | 8 | 1471.25 | 1476.13 | +0.33% |
| context8193 | 1 | 176.45 | 179.23 | +1.58% |
| context8193 | 4 | 242.70 | 243.23 | +0.22% |
| context8193 | 8 | 268.78 | 268.66 | -0.04% |

单 B200、TP1、FP8 block128 target、Mega auto dispatch、V2、K8、fast-prefill，
APC 关闭；输出固定 128 token，忽略 EOS，温度 1、standard rejection。
每格两次预热、三个正式 seed，分别启动父版与合并版原生引擎，测量完整 generate 墙钟。
两版未启用 event/profiler；沿用 benchmark 的图回放计数，未增加审计图。

正式 36 个 batch、156 个请求、19968 个输出 token；18 组正式配对和 12 组预热配对，
输出 token、accepted/proposed/draft-round 及逐位置接受计数均精确相等。
四条首 token top-20/logprobs 与 256 个完整行 teacher-forced 位置也精确相等。

两版 target 图均为 `[9,18,36,72]`，draft 图均为 `[8,16,24,32,40,64]`，
实际回放已核对：仍为 **target 4 份、draft 6 份**。

本次只有父版→合并版的一轮独立进程顺序，每格三个 seed；表格是整合验收点估计，
不能排除进程先后、温度和系统波动，也不替代真实文档分布或任务准确率评估。
原会话的大文档验证仍作为已有证据，本次没有重新计入。

见 [summary.json](summary.json)、[trials.csv](trials.csv) 和 [图与 worker 回执](worker-receipts.json)。

## 历史结果的适用范围

原会话在保留 FA4 和线性缓存后，单独加入保原精度 norm 融合，测得完整 K8 draft
设备延迟降低 10.8%–13.7%；这不是本次组合的端到端提升。
原会话 target 使用 FP8 block128、Mega OFF，本次使用当前 Mega auto dispatch。
[原 norm 对照统计](reference-norm-summary.json)仅作为来源证据，未混入本次计时。

## 复查路径

当前原始日志、输出与审计文件位于 `/data/vllm_test/work/yoco-draft-merge-20261021/`。
原始接受记录位于 `/data/vllm_test/work/yoco-draft-fp32-norm-20261021/acceptance.json`。

GPU6 在结束时为 0 MiB/0% utilization，advisory lock 已释放；
Pod UID、节点、restart=0 与 PID 7 保持，未切换线上服务。
完成相同工作负载后，Torch allocated 增加 **1242.12 MiB**；这是常驻计算副本及
该运行的分配差值，不是 reserved 或峰值显存测量。见 [环境回执](environment.json)。
