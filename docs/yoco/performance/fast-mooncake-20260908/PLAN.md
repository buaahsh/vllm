# fhb-dev-9-8 Fast Mooncake 复测（2026-09-08）

当前 inference 提交：416b83d83504af902b8da248452a19cc5d56b5db

只测试当前 YOCO L3 Fast，先单卡 GPU5，再 1P1D P4/D5；不重测 Qwen/Align。
每个拓扑先四组功能探针和 smoke50，再完整600秒到达窗口与排空。
复用2026-09-07 Fast的模型、物理GPU、软件依赖、完整server/client参数。
冻结trace：Mooncake FAST25 toolagent源时间300–900s、1x、3643请求，SHA256 680e526d49545258c0ca5b635d0004a44cce2ce990d533ac32024cefee1f0170.
BF16、TP1/DP1、maxlen81920、maxseq256、memory0.85；P/单卡budget32768，D8192。
请求FA4、FULL_AND_PIECEWISE；保留Fast实际按角色MoE backend选择，无外部autotune cache。
AIPerf0.12.0，concurrency512、workers32、record-processors1、timeout600、seed42；每case独立cache_salt。
完整保留逐请求、GPU、metrics、传输、排空与失败证据。共享节点、单次重复、无SLO，标为diagnostic。
只启停本轮拥有的服务进程，保留Job/Pod。源代码用当前git archive覆盖隔离runtime，复用已编译二进制，启动前后全量SHA校验。
更新持续表仅Fast实测行，保留其他历史行，并同步两仓库fhb-dev-9-8文档。
