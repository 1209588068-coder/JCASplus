# 固定后的作者源码对比合同

本合同只描述“作者源码 + 数据接口适配”的对比，不把本项目的选择器、特征支持集、clean-KL、双干净代理、残差生成器、域投影或分布正则混入外部方法。

## 源码身份

上传的源码先以 ZIP 哈希和归档 comment 中的 commit 标识绑定，再与 `third_party/author_baselines/` 中的代码文件逐字节核对。核对逻辑位于 `author_comparisons/source_registry.py`，只读取源码、不会加载归档内的数据集或 checkpoint。

| 方法 | 上传源码 | 原生任务 | 在本任务中的位置 |
|---|---|---|---|
| SPEAR | `SPEAR-main.zip` | 归纳式节点分类 | 主对比，source-backed |
| UGBA | `UGBA-main.zip` | 归纳式节点分类 | 主对比，source-backed |
| DPGBA | `DPGBA-main.zip` | 归纳式节点分类 | 主对比，source-backed |
| GTA/GraphBackdoor | `GraphBackdoor-main.zip` | 整图分类 | 只做单独的 pair-readout 结构迁移，不能并入同质主表 |
| Motif-Backdoor | `Motif-Backdoor-main.zip` | 整图分类 | 只保留源码审计，不强行适配 pair-node 监督 |
| CGBA | 未发现本地作者 ZIP | — | 不把手写实现称为作者源码对比 |

`author_comparisons/runner.py` 的主队列只注册前三个 source-backed 方法。GTA 和 Motif 的源码已提取/核对，但任务接口不同；CGBA 没有可核验的作者归档，因此三者不进入同一主表。

## 统一接口

- 数据：Argoverse 2 Motion Forecasting 派生的车辆 pair-node 图；只读原 TRAIN 内登记的 train/dev 场景。
- 监督：固定 pair 监督、固定目标池和 Moving/Slow 分层；不根据方法输出重新筛目标。
- 方法内部：保留作者原生选点、生成器、损失、优化和 checkpoint 逻辑；适配层只负责输入映射、隔离导入、日志、哈希和公共评估。
- 受害模型：统一 GraphSAGE 口径；clean reference 与 victim 使用匹配初始化、监督索引和 checkpoint 规则。
- 预算：在合同中记录节点/特征/结构权限的差异，不把不同权限包装成等价预算。
- 评估：整体、Moving、Slow 增量 ASR，clean AUC/AP/召回，邻接与非邻接增量误报，正类压制，修改代价和失败原因。
- 测试数组、内部 audit 数组和确认数组不加载；DEV 结果是开发证据，不冒充盲测。

## 重复与运行

新合同使用五个预先固定的比较种子：`20261201`–`20261205`。这五个种子与主方案的五个种子分开登记。旧的三种子记录不混入新汇总。

运行入口是：

```text
python3 -B -m author_comparisons.runner --check-only
```

`--prepare` 之前必须先通过源码审计和 GPU 资源验收；本轮代码修改只完成注册、来源和接口检查，不自动启动攻击训练。启动前应重新生成一个新的结果目录和 plan，旧结果目录不可覆盖。

## 结果解释

主表只比较同一 pair-node 任务中的三个 source-backed 方法。GTA/GraphBackdoor 的结构迁移若以后单独运行，必须使用单独表格和单独标题，明确它是 graph-classification 到 pair-readout 的迁移证据。Motif-Backdoor 只报告任务不匹配与源码审计结果。

旧的 `comparison_methods/` 手写算法实现保留用于历史审计和单元测试，不作为本合同的作者方法实现。
