# 同场景关系监督：解码器小规模验证

本实验限于本地 AV2 和研究者自训练 GNN 的授权离线鲁棒性研究。
它是**训练机制干预诊断**，不是原数据投毒能力边界内的新正式攻击实验。

## 固定假设与变量

假设：完整与反转的差异在已有表示中部分保留，但原监督没有充分约束关系选择性。
本轮检验固定编码层后，解码器是否能通过同场景配对监督同时保住完整响应、压低反转响应。
不能从一次短程失败推导所有配对训练都不可能成功。

初始模型固定为 `run_nfcqkj53/geometry_shared/best_model.pt`。编码层保持 eval、冻结参数，
只更新原 `decoder.mlp`；不增加网络模块，不修改模板、42D 表示或动态图标签规则。
解码器输入由原模型前向 hook 缓存，每次验证缓存前向与完整模型输出一致。
训练后检查整个非 decoder.mlp 部分的 state_dict SHA 完全不变。

两个对照同初始化、同 seed 20260621、相同更新次数和数据抽样顺序：

| 分支 | 辅助目标输入 | 辅助目标标签 |
|---|---|---|
| clean_replay | 同场景 clean／clean／clean | 0／0／0 |
| matched_relation | 同场景 clean／full／reversed | 0／1／0 |

辅助标签对目标两个有向边一致赋值；只有完整条件的目标为正。
这两个分支都另外回放相同原始干净监督，不使用干净模型蒸馏、梯度投影或额外排序损失。
损失为合并后有向边的普通 mean BCE，不是完整场景均匀抽样的原 5% 投毒训练。

每次更新：至多 1,024 条干净监督边，加 64 个辅助场景 × 3 条件 × 2 有向边。
辅助场景有放回抽样。一轮 epoch 是干净监督边遍历一次，**不表示每个辅助场景只出现一次**。
具体更新数与重复抽样数写入 history。

AdamW lr=1e-4、weight_decay=1e-4，沿用权重衰减分组策略，固定 5 epochs。
仅使用最后一个 epoch 的 checkpoint；不查看逐轮 develop 结果选 checkpoint，不改超参数。

## 数据与停止规则

- 训练辅助目标：上一轮诊断的 337 个低速 construct 目标（128 原正投毒、209 原反转负控制）。
- 干净回放：原 train/construct 的 1,024 个 SHA 固定抽样场景，排除本轮目标场景。
- 干净保持检查：原 train/develop 的 1,024 个 SHA 固定抽样场景，排除本轮目标场景。
- 响应检查：上一轮诊断的 64 个低速、64 个运动 develop 目标；运动不参与辅助训练，只作观察。
- 无监督干净场景跳过并记录，不另外补选。construct/develop 按 recording group 分离。
- 不读取 inner-audit 或原 validation/test 图；develop 已用于之前诊断，不能称新独立测试。
- 公共阈值不变，来自已完成 BCE pilot 的 clean_reference inner-develop pair F1。

预先固定的 gate（matched_relation，低速 develop）：

1. 完整触发 Incremental ASR ≥ 60%。
2. 反转触发**绝对激活率** ≤ 10%，不是只检查增量翻转。
3. 抽样干净保持检查的 pair AUC 相较**相同场景上的原模型**下降 ≤ 0.0015。

同时报告 clean activation、full-only activation、相邻／非关联边副作用及相对 clean_replay
的 full-only 增益。后者是机制归因的重要对照，不能只看 gate PASS 忽略两组是否相同。
小样本 AUC 保持也不等于完整公共池干净性能已通过。

最多 5 epochs 后停止，无论 PASS/STOP 均不自动启动后续训练或正式 validation。

## 运行与保存

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.relational_decoder_pilot --check-only
python3 -m jcas.workflows.relational_decoder_pilot --run --device cuda
```

每次创建 `record/reproduction/relational_poison_v1/bce_pilot/decoder_pilot_*/`，保存固定协议、
样本清单、初始模型绑定、缓存、两组最后 checkpoint、history、初始／两组逐目标响应、
干净 pair 分数、副作用计数、results、gate、summary 及证据 SHA。
不会覆盖旧模型或结果；不支持用已有目录续写。STOP 正常完成时返回 2，PASS 返回 0。

缓存仅用于固定编码层这项诊断；它不允许解释为重新训练了整个 GNN。
若本轮有改善，下一步仍需单独验证能否在原训练／攻击能力边界内成立。

## 2026-09-13 完成结果

目录：`record/reproduction/relational_poison_v1/bce_pilot/decoder_pilot_z_pl3i2t/`。
两个分支均完成 5 epochs，每轮 40 次更新，合计各 200 次更新；每个分支辅助场景共抽取
12,800 次（有放回），不是只看每个目标 5 次。每轮 BCE 分别记录在 history。
实际干净回放 923 个有监督图，干净保持检查 936 个图、21,143 个 pair。
辅助训练使用 337 个低速目标，编码层保持冻结。

以下响应均为同一批 64 个低速 develop 目标；AUC 来自另一批固定抽样的干净 develop 图。

| 指标 | 原模型 | 仅干净回放 | 同场景配对 |
|---|---:|---:|---:|
| 完整 Incremental ASR | 67.1875% | 56.2500% | 62.5000% |
| 反转绝对激活率 | 62.5000% | 54.6875% | 59.3750% |
| full 激活且 reversed 不激活 | 7.8125% | 3.1250% | 6.2500% |
| 干净保持检查 Pair AUC | 0.987063291 | 0.987069746 | 0.986811620 |
| 完整触发 adjacent incremental FP | 0.402145% | 0% | 0.804290% |

Gate 为 **STOP**：完整 Incremental ASR ≥60% 通过；干净 AUC 下降 0.000251671，
≤0.0015 通过；反转绝对激活率 59.375%，未通过 ≤10%。正常完成返回码为 2。

配对组相对干净回放提高完整响应 6.25 个百分点，但反转也更容易激活。
full-only 比例仅比回放高 3.125 个百分点，且低于原模型，不能据此宣称关系选择性改进。
在原 209 个反转负控制训练场景中，配对后的反转绝对激活率仍为 50.2392%。

本轮有限结论：冻结当前编码层、使用该配对 BCE 短程调整解码器，尚不能实现期望的关系选择性。
不能把前一轮问题简单归结为反转样本数量不足，也不能从 5 epochs 的失败推导所有可行训练都失败。
没有延长训练、改门槛或访问 inner-audit／原 validation／test；本轮实验停止。
