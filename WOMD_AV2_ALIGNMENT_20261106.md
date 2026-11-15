# WOMD 对齐 AV2 修复版的匹配试跑（2026-11-06）

本次使用公开授权的 WOMD 数据和研究者自有 GNN，开展离线模型鲁棒性实验。以正在运行的 `av2_formal_main_10seed_20261106_v2` 冻结合同和源码为参考，保持 WOMD 既有数据池，运行一个全新模型链路的匹配 pilot。

2026-11-06 21:02（北京时间）已提交后台。短暂等待GPU余量后已进入真实CUDA训练；监督进程PID4140927，训练worker PID4140928。后续进度以[即时状态](results/womd_av2_aligned_pilot_20261106_v1/status.json)为准。该试跑尚未完成，不计作WOMD正式多种子结果。

## 固定数据池和投毒比例

| 项目 | AV2 当前参考 | 本次 WOMD |
|---|---:|---:|
| TRAIN场景 | 115,490 | 原512 |
| DEV场景／固定负类目标 | 128／112 | 原128／126 |
| AUDIT场景／固定负类目标 | 1,280／1,113 | 原256／249 |
| 合格TRAIN监督pair | 2,188,291 | 原57,570 |
| 干净监督pair | 437,658 | 原11,514 |
| 投毒pair | 5,865 | 154 |
| 投毒pair／(干净监督pair＋投毒pair) | 1.3223666% | 1.3198492% |

原 WOMD 26个投毒pair对应 `26/(11514+26)=0.2253033%`，与AV2当前主口径不匹配。本次在154、155之间使用精确有理数比较，选择离AV2比例最近的154，误差为 −0.0025174 个百分点。额外outer节点保持512，outer总数为666。

这不是5%的场景投毒率。选择器允许同场景多个pair；场景数、所有pair数和实际干净监督数不是同一分母。

旧WOMD v2的全部场景、顺序、已标准化特征、邻接、标签、车辆pair身份、motion分组及固定目标索引均逐字段逐字节保留。normalizer、干净监督indices和重叠诊断文件逐字节一致；共享索引函数也精确重现原seed20261104的名单。仅将旧缓存无损重打包为完整场景批次，不重新选场景、目标或拟合归一化。

DEV Moving目标仍为35，AUDIT Moving目标仍为68。三个角色使用既有source-shard隔离，不冒充AV2录制组；WOMD录制独立性未获证明，且这些目标已在先前pilot曝光，仍明确标为内部development/audit评价，不称盲测或官方test。

## 与 AV2 v2 的共同执行合同

- **训练阶段**：新GCN选点200轮、新MLP200轮、两个新GraphSAGE clean代理各200轮、base生成器200轮、uniform残差200轮、新语义支持200轮、新victim200轮、独立clean reference200轮。
- **选点和归因**：同一作者GCN/MLP前向、熵式排序和预测类轮转；继承AV2 v2的精确零熵项连续延拓修复。SAGE为128个TRAIN解释样本、32个背景、2048 permutations、batch64；样本哈希namespace和原作者构造参数与参考一致，实际数据身份另行准确记录。
- **支持**：每次重新生成top30；各阶段保护357/715；复制组仅取与新候选的交集，不补字段；最后一次exact24，不可行则记录失败。identity输出，不加域投影、分布判别器或pair_mean。
- **损失**：目标NLL按全局clean＋outer监督数归一，0.1 cosine按全TRAIN节点归一，两个clean代理KL各按outer节点归一后取平均；支持阶段另加一次 `((E[cost]-24)/legal_capacity)^2`。生成器/影子模型lr0.0002，支持门控lr0.01，GraphSAGE/选点模型lr0.01；其余模型、Adam和weight decay规则来自参考源码。
- **检查点**：GCN/MLP使用最早最高正常DEV accuracy；GraphSAGE使用最早最高正常DEV AUC；生成器和支持只使用固定末轮，不按ASR选模。
- **执行**：CUDA FP32，关闭AMP/TF32；稳定完整场景分批，最多16,384个pair节点；不采样场景或邻居。每轮全局梯度累计后一次Adam更新。
- **随机数**：主seed保留20261104；阶段偏移与AV2一致，selection0、proxy1/2为101000/102000、reference103000、anchor201000、support301000、victim401000。
- **评估**：共用参考评估函数及argmax阈值（平票class0）；整体／Slow／Moving的Incremental ASR、Absolute ASR、Conditional flip、clean activation、AUC/AP/正类召回、邻接／非邻接增量误报、正类压制及修改幅度一致。独立clean reference不参与触发梯度。

所有模型、排名、top30和learned24从头训练。没有复用AV2恢复检查点或旧WOMD权重。与旧WOMD pilot相比，投毒预算、分批执行、阶段种子偏移及归因样本哈希规则等已明确改变，不能把新旧结果差异单独归因于一个模块。

## 源码与验收

新快照以活动AV2 v2快照为基础，保留相同 `av2_formal.run_pipeline`。四个已有模块只叠加WOMD数据身份、`training_20s`来源边界和输出元数据适配；数学计算不改。差异记录为[code_alignment.diff](results/womd_av2_aligned_pilot_20261106_v1/code_alignment.diff)，不修改正在运行的AV2冻结源码。

34项回归通过，包含WOMD标签缓存的真实CUDA选点→归因→生成→支持→victim→评估链路、AV2关键数值等价测试、配置漂移和完成凭据校验。真实TRAIN最大消息边批次为16,332节点／1,738,002边，DEV最大批次为15,860节点／2,127,010边；六条前后向检查全部通过，零优化器更新，最大实际分配峰值约1.506GiB。仍保留8GiB启动余量门槛，无GPU资源时明确等待。

后台监督每10秒检查worker。每轮保存检查点，失败保留资产并明确标记；不会静默改设备、预算或重启。完成验收核对两组summary及全部三种评价CSV的哈希，并将目标CSV的场景、pair索引和两个车辆ID逐项与旧缓存目标名单比较，只有通过后才写整次完成凭据。

## 记录

- [固定试跑计划](results/womd_av2_aligned_pilot_20261106_v1/plan.json)
- [实时状态](results/womd_av2_aligned_pilot_20261106_v1/status.json)
- [训练日志](results/womd_av2_aligned_pilot_20261106_v1/seed20261104/run.log)
- [数据保留验收](results/womd_av2_aligned_pilot_20261106_v1/resources/data_preservation_acceptance.json)
- [启动前总验收](results/womd_av2_aligned_pilot_20261106_v1/resources/prelaunch_acceptance.json)

结果输出位于 `seed20261104/evaluation_dev/` 和 `evaluation_audit/`；只有批次根目录出现经过验证的 `completion.json` 才代表匹配试跑完整结束。完成后不自动扩展WOMD场景或启动正式多种子。
