# Rotating v4-max2：最多两个训练目标的轮换对照

## 边界和假设

仅使用本地公开 AV2 和研究者自有模型，开展离线、授权的机器学习鲁棒性研究。
这是 `training_procedure_control` 开发实验，不属于冻结 v6.0 的纯数据投毒主结果。
不修改冻结 v6.0/v6.4-S、图、grouped-v5 manifest、旧计划、旧源码绑定或旧 STOP。

已完成 v4 诊断发现，扩大目标覆盖后，原目标响应下降，新增目标响应也偏弱。
存在局部梯度干扰，但诊断不能证明它是唯一原因，也不支持直接取消干净任务保护。

本轮检验：在每轮触发样本量固定的情况下，缩小同场景轮换集合，能否恢复目标
学习强度，并改善同一留出目标池的响应。只改变**每场景训练目标数量上限**，
不改变损失、投影、训练轮数或副作用门槛。

这是单种子对照，不能保证提升；覆盖量与单目标曝光次数会同时改变，
不能把结果解释成梯度干扰的独立因果效应。

## 唯一计划变量

在原1,037个可轮换场景中：

- 原 slot0 始终保留。
- 沿用原 v4 的哈希排序，从原训练备选目标中保留排序第一的一个；
  不用分数挑选、不重新搜索候选、不改变哈希 seed。
- 原哈希选择结果在当前绑定计划中恰好是旧 slot1。
- 其余训练备选从**正向触发训练角色**中移出，不改它们的干净标签，
  也不删掉对应场景或非目标监督。
- 原1,037个留出 pair 原样保留，不从移出目标中重新选留出。

| 项目 | 原 v4 | v4-max2 |
|---|---:|---:|
| 原投毒场景数 | 5,776 | 5,776 |
| 可轮换场景数 | 1,037 | 1,037 |
| 不轮换场景数 | 4,739 | 4,739 |
| 场景内最多训练目标数 | 4 | 2 |
| 可用训练目标总数 | 8,191 | 6,813 |
| 新增训练目标数 | 2,415 | 1,037 |
| 留出目标数 | 1,037 | 1,037 |
| 每轮触发样本数 | 5,776 | 5,776 |
| 每轮 moving / slow | 2,746 / 3,030 | 2,746 / 3,030 |

每场景每轮仍只呈现一个触发版本，奇数轮选原目标，偶数轮选保留的新增目标。
不轮换场景保持原目标。完整50轮中，每个可轮换场景的两目标各呈现25次；
被移出的1,378个目标和原留出目标正向触发训练曝光均为0。

checkpoint 仍由 clean validation loss 选择。因此审计中的实际曝光只计算到
新 checkpoint 的最佳轮次，不预设它仍是第37轮，也不把50轮曝光算给更早模型。

## 不变的训练设置

- seed20260621，从随机初始化训练，不继续训练旧 checkpoint。
- 50 epochs、batch size64、GENConv hidden128/layers3/dropout0.1/LayerNorm。
- 同一42维表示、动态风险标签5/1/4、5%场景变换率。
- 0.2 m相对位移、K10、minimum-jerk、velocity residual、alpha0.5。
- target0.1、ranking0.1、consistency1、adjacent5、margin1、clip5。
- 原 clean-reference 梯度投影、extra norm ratio1、AdamW lr0.001/weight decay0.0001。
- 完整 clean validation 用于 checkpoint 选择与原 Pair AUC 门槛。
- 不运行攻击 validation，不访问 test；即使 gate PASS 也不自动扩种子或继续评估。

首轮目标清单及呈现规则保持一致，但不承诺不同运行的 GPU 数值轨迹逐位相同。

## 为什么有两个计划文件

不修改原 source-bound 训练实现，新增入口：

```text
jcas.workflows.two_target_rotation_pilot
```

每次生成：

- `plan.json`：本实验的外层契约，明确 `effective_max_train_targets=2`，
  绑定原计划、诊断证据、源码、门槛、内层计划、目标表和 lineage。
- `engine_plan.json`：原引擎可直接读取的兼容格式。
- `targets.csv`：实际供训练读取的目标表，每个可轮换场景恰好两个训练 slot。
- `lineage.csv`：记录原2,415个新增目标中谁被保留、谁被移出。

旧引擎格式的 `max_train_targets=4` 表示格式允许的容量，因此内层仍保留4；
它不是本实验实际训练数量。外层契约、逐场景角色表复算及50轮日程核对共同
强制有效上限为2。不要手动把内层字段改成2，否则原引擎会拒绝该格式。

训练子目录名称仍可能包含旧引擎的 `rotation_v4` 字样。实验身份应由新的
`rotating_counterfactual_v4_max2` 父目录、外层计划和结果中的内层计划 SHA 判定，
不能将它与旧 v4 混为一谈。新候选在对照汇总中明确标作 `rotating_v4_max2`。

## 自动完成的 train-only 审计

先评估全部5,776个原目标，沿用原运动目标 gate。随后对以下四组报告结果：

| 同场景对照组 | 目标数 | max2 正向训练角色 |
|---|---:|---|
| rotatable_anchors | 1,037 | 保留 |
| retained_additional | 1,037 | 保留 |
| dropped_additional | 1,378 | 移出，但仍有干净/非目标监督 |
| heldout | 1,037 | 仍不作为正向触发目标 |

每组对照新 max2、旧 v4、固定 v3 和 clean reference。旧模型使用已验证且
与同一目标对应的缓存分数，不重复运行旧模型。新候选补充推理覆盖
2,415个原新增目标加1,037个留出目标，共3,452个干预；原目标子组复用本轮
完整原目标审计记录。

报告 pair 等权、场景等权、clean activation、概率/logit变化和可用的副作用。
原目标 CSV 没有子组逐干预副作用计数时留空，不用全5,776场景统计冒充1,037子组。
这些补充组没有事后新增的 gate。

留出池已经在此前 development 中查看过，不能称作新的盲测或 validation/test。
被移出的目标更不能当成从未训练过的统一留出池，因为旧 v4 曾将它们正向训练。

## 固定9项停止门槛

| 指标 | 门槛 |
|---|---:|
| 全部原运动目标 Incremental ASR | ≥30% |
| 原运动目标 mean logit delta | ≥1 |
| 原运动目标 clean activation | ≤5% |
| 原运动目标 adjacent-negative incremental positive | ≤0.5% |
| clean validation Pair AUC | ≥0.985 |
| 同一留出池 Incremental ASR | ≥30% |
| 同一留出池相对固定 v3 的增益 | ≥10个百分点 |
| 留出池 clean activation | ≤5% |
| 留出池 adjacent-negative incremental positive | ≤0.5% |

使用未四舍五入的值，全部通过才 PASS。增益门槛的基线仍为固定 v3，
不是偷偷改成较容易超过的其他模型。旧 v4 仅额外作为同池对照。
STOP 退出码为2；即使前面原目标指标不通过，仍完成本轮预定义的补充审计再
写出结果，不自动转入攻击 validation、其他种子或 test。

## 怎么跑

一条命令自动准备独立计划、训练单种子并完成所有 train-only 审计：

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.two_target_rotation_pilot --device cuda
```

仅预检（不创建目录、不加载图或模型、不训练）：

```bash
python3 -m jcas.workflows.two_target_rotation_pilot --check-only
```

准备阶段复用绑定的 train 角色表，无需重新加载轨迹。预检仍会核对旧资产和
完整日程，可能需要几分钟。CUDA 不可用时报错，可显式用 `--device cpu`。

每次自动创建新目录，不覆盖已有运行：

```text
record/reproduction/rotating_counterfactual_v4_max2/seed20260621_<随机后缀>/
  plan/plan.json
  plan/engine_plan.json
  plan/targets.csv
  plan/lineage.csv
  protocol.json
  train.log
  train/<旧引擎生成的运行名>/result.json
  train/<旧引擎生成的运行名>/best_model.pt
  anchors.log
  audit/anchors.json
  audit/anchors.records.csv
  audit/supplemental.records.csv
  audit/cached_comparators.records.csv
  audit/comparison.csv
  audit/gate.json
  audit/gate.txt
  audit/summary.json
```

完整结束会写 `audit/summary.json`，状态为 `PILOT_COMPLETE_REVIEW_REQUIRED`；
具体是否通过看其中 `gate_status`，它可能是 STOP。原 v4 的 STOP 保持不变。
跑完后提供输出目录，重点查看 `summary.json`、`comparison.csv` 和 `gate.json`。

## 只准备、使用已有计划、恢复审计

只生成计划，不训练（输出目录必须尚不存在）：

```bash
python3 -m jcas.workflows.two_target_rotation_pilot --prepare-only \
  --output-dir record/reproduction/rotating_counterfactual_v4_max2/my_plan
```

使用该外层计划训练，仍会创建新运行目录：

```bash
python3 -m jcas.workflows.two_target_rotation_pilot \
  --plan record/reproduction/rotating_counterfactual_v4_max2/my_plan/plan.json \
  --device cuda
```

如果训练已经完成、审计中断，指定**该次训练对应的外层计划**和完整 result：

```bash
python3 -m jcas.workflows.two_target_rotation_pilot \
  --plan /absolute/path/to/the/original/max2/plan/plan.json \
  --result /absolute/path/to/the/completed/max2/train/run/result.json \
  --device cuda
```

这只重新审计，不重新训练，也不是中断 epoch 的训练恢复。
不能传旧 v4 result，不能换成新生成的另一份计划，也不要把 `engine_plan.json`
误传给 `--plan`。恢复前会核对计划路径/SHA、6813训练目标覆盖及全部50轮日程。
源码或资产变化时应停下核查，不要手动更新旧计划摘要绕过检查。
