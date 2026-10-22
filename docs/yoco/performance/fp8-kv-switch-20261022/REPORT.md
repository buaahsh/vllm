# YOCO Shared-KV FP8 KV 开关验收

基线 `456d589ebf82262b0ca837e7ae988e24da5cd48d`，2026-10-22。
本次按用户要求提供启动开关，默认仍为BF16；FP8主要用于增加KV容量，不以提速为采用依据。

## 开关与实现

复用已有 `kv_cache_dtype`：CLI/YAML `--kv-cache-dtype fp8_e4m3` / `kv-cache-dtype: fp8_e4m3`，
Python配置 `kv_cache_dtype="fp8_e4m3"`。`fp8`为E4M3别名，`auto`/`bfloat16`保留原BF16路径。
原生benchmark新增同名参数，默认auto，并保存实际缓存dtype、形状、stride、字节数和owner scales。
[使用说明](../SHARED_KV_FP8.md)含完整命令和API/YAML示例。

Shared-KV原先拒绝量化target KV；本次允许CUDA E4M3，draft按writer的FP32 scalar K/V尺度
读取cache，将FP8值在寄存器反量化为BF16后做attention。没有常驻BF16 KV镜像。
保留严格prefix、SWA8192、HND/NHD、int64物理页地址、raw h20与decode当前token语义。
BF16继续使用原FA4路径；FP8使用支持owner scales的Triton读取路径。

dtype在引擎启动时固定。一个引擎只捕获其选定精度的图，不按M、并发或长度切换，
也不同时保留两套精度图。无新增全局环境精度开关；现有CacheConfig负责dtype与编译配置隔离。
不修改target权重量化、BF16 router、draft计算权重、FP32 proposal/rejection或调度策略。

## 质量：复用已完成的完整MATH-500

先前隔离评测目录 `/data/vllm_test/work/yoco-kv-precision-math500-20261022/`
已经完成BF16与E4M3各500题。本次沿用这份已验收读取算术，不重复生成整套题。

- BF16 KV：**490/500（98.0%）**；E4M3 KV：**486/500（97.2%）**，下降0.8个百分点。
- 自动分数配对：两侧都对481、仅BF16对9、仅FP8对5、两侧都错5；McNemar p=0.424，
  配对bootstrap95%差值区间[-2.2,+0.6]个百分点，单种子结果不能保证其他任务的损失范围。
- 每侧各有一题格式提取假阴性；人工复核491/487，差距仍为−0.8pp。自动分数与复核分别保留。
- 输出上限131072、context139264，temperature1/top_p1/seed20261022，native模板及boxed后缀。
  数据为冻结MATH-500 test，SHA256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`。
  BF16/FP8分别2/1题用满输出上限，均计错，未移出500题分母。
- 同B200/TP1/V2/K8/fast-prefill、BF16 router/draft、FP32 residual/statistics；APC与async关闭。
  Target为FlashInfer原生FP8 KV路径，可能同时量化Q，不将其视为只改缓存字节的数学隔离实验。

对比2524个运行时Python文件，除三个Shared-KV接入文件外，target、采样与其他运行时源码
均与原评测一致；FP8 `_window` 读取kernel和`paged_prefix_attention`在排除docstring后AST相同。
本次增加的wrapper检查只验证尺度类型/设备，不改有效输入的浮点运算。
详见[算术核对](quality-arithmetic-audit.json)、[原质量汇总](quality-summary.json)和[格式复核](format-review.json)。
原500题逐题输出、评分、source/harness审计与原始失败日志保留在上述评测目录。

## 本次集成验证

B200模型/配置/Shared-KV/prefill回归 **199项通过**，工具回归 **17项通过**。
新增8项FP8 kernel回归覆盖HND/NHD、K1/K8、全prefix/SWA8192、空行、非单位尺度、
重放时scale原地更新和未来/旧窗口污染；另4项回归拒绝缺失、非scalar、错误dtype或错误设备的scales。
默认auto与显式E4M3均完成原生生成、teacher-forced有限性和精确输出预算检查。

## 容量和计时

同一GPU顺序测auto/BF16与显式E4M3，target/draft与其他参数固定，使用8GiB/4GiB KV预算
保持相同容量。原生同步in-process，maxlen139264/maxseq8/batched tokens2048，APC关闭。
短输入与8193输入、C1/C8、每请求128输出、两个正式seed，各格两次预热。计时包含prefill、
采样、调度和GPU drain；加载/编译/预热/teacher-forced面板不计入。未开profiler。
这是单轮配置与开关诊断，不是重复容量压测或稳定性能结论；生成轨迹可因量化变化。

两组实际owner形状均为`[131072, 8, 16, 256]`，元素数`4294967296`；
auto为BF16、KV数据`8589934592` bytes，E4M3为uint8存储、`4294967296` bytes，
相同容量的缓存数据字节精确减半。两组writer K/V scale均为1，图数均为target4/draft6。

| 输入 | C | BF16 output tok/s | FP8 output tok/s | 变化 |
| --- | ---: | ---: | ---: | ---: |
| short | 1 | 332.09 | 343.95 | +3.57% |
| short | 8 | 1606.56 | 1517.38 | -5.55% |
| context8193 | 1 | 163.02 | 167.14 | +2.53% |
| context8193 | 8 | 255.70 | 268.46 | +4.99% |

相同KV预算通常可容纳更多token；不意味着进程的显存会自动降低一半。
本轮按相同容量减小KV预算，权重、workspace、graph与临时张量仍需额外显存。
FP8读取减少缓存字节，同时带来量化/反量化和不同attention kernel的成本，保留全部正负计时结果。

## 审计与产物

冻结源码8091文件、运行支持1996文件前后SHA256通过；
最终工作树5434个代码/配置文件与已测快照一致。Pod UID、node与restart=0保持，PID7存在；
任务GPU6释放为0MiB。审计见[audit.json](audit.json)，诊断数据见[performance-summary.json](performance-summary.json)。

完整本地产物：`/data/vllm_test/work/yoco-fp8-kv-integration-20261022/`。
冻结脚本、source/runtime-support清单、原生worker回执、逐请求token/metrics和日志均保留；
既有native Torch/DeepGEMM运行库未替换，服务/Job/Pod未修改。

归档与脚本SHA256见[artifacts.json](artifacts.json)，路径均相对完整本地产物目录。
