# 运动目标轮换：rotating_counterfactual_v4 单种子 pilot

## 边界与实验身份

仅使用本地公开 AV2 和研究者自有 GENConv，离线授权模型鲁棒性研究。
准备阶段与触发响应审计只读 train 图；训练沿用 clean validation loss 选择 checkpoint。
不访问 test，不自动运行攻击 validation，不改动冻结 v6.0/v6.4-S、图文件和 grouped-v5 manifest。

本实验允许控制训练流程，仍属于 training_procedure_control，
不是 v6.0 的纯数据投毒正式结果。“v4”是 counterfactual 开发迭代名称，
与冻结的 v6.4-S 不是同一个实验。

## 假设与唯一计划变量

此前匹配诊断显示：训练过的运动目标 A 响应较强，
同一场景中没有被指定为投毒目标的运动 pair B 响应明显下降。
这支持目标级泛化不足，但没有证明具体哪个网络层或损失造成它。

本轮只改变触发正向训练目标的覆盖方式：
在预定义的场景内候选集合中按 epoch 轮换，不改变每个 epoch 的触发样本总数。
固定目标 v3 seed20260621 的既有 checkpoint 作为对照，不额外训练一个重复固定模型。
两个模型将在同一批新留出的目标上使用同一阈值评估。

固定配置：

- 原 5,776 个场景；每 epoch 仍是 5,776 份触发样本，保留所有原干净训练图。
- 每 epoch 运动目标 2,746、慢速目标 3,030；慢速场景始终用原目标。
- seed20260621，随机初始化，50 epochs，batch size64，clean validation loss 选 checkpoint。
- GENConv、42维表示、0.2 m / K10 / minimum-jerk / velocity residual / alpha=0.5。
- 原 balanced 权重：target0.1、ranking0.1、consistency1、adjacent5、margin1、clip5。
- 原 projected_v3 全模型更新规则，rho1；AdamW lr0.001、weight decay0.0001。
- 不改阈值，不用留出响应选择 checkpoint，不增加训练 epoch 或投毒样本数。

改变目标会自然改变邻接边数及损失对应的边集合；它们是目标轮换的结果，
不能宣称逐步梯度或每个目标的曝光次数与固定训练完全相同。

## 候选与留出规则

只读取 train 轨迹和原动态标签，不使用任何模型预测选择候选。
候选必须满足原双向真负标签、严格监督、物理可行和双端连续11状态要求，
且两个端点速度的最小值至少0.5 m/s。

- 保留原目标为 slot0。
- 场景至少有三个合格运动 pair，才可能进行轮换。
- 按固定哈希 seed20260909 预留一个非原目标 pair。
  留出候选排除上次 A/B/C 诊断中所有已查看过的精确 scenario/pair。
- 再从其余候选中按固定哈希取最多三个训练替代目标，连同 slot0 最多四个。
- 候选不足或不能留出新目标的场景继续固定原目标。
- 每个 epoch 每个场景只选一个 slot，索引为 (epoch-1) % 该场景训练候选数。
  epoch1 与原固定目标 presentation 一致，不消费新的随机数流。
- 留出 pair 不以任何方向进入正向触发训练；其干净负标签和其他目标扰动时的
  非目标一致性监督仍保留。因此这是“留出触发目标角色”，不是全新干净样本，
  也不是独立 recording/test 集。

已经完成的覆盖检查：

| 项目 | 数量 |
|---|---:|
| 场景总数 | 5,776 |
| 可轮换运动场景 | 1,037 |
| 不轮换场景 | 4,739 |
| 可用训练目标总数 | 8,191 |
| 留出目标 | 1,037 |
| 1 / 2 / 3 / 4 个训练 slot 的场景数 | 4,739 / 272 / 152 / 613 |

1,037 个可轮换场景约占原 2,746 个运动场景的37.8%。
留出指标不能代表全部运动目标，更不能代表完整 validation 目标池。
准备门槛固定为至少300个留出场景；不足时拒绝训练，不能通过减小门槛继续。

## 实现与防错

新增独立计划文件与角色表，原 poison manifest 只用于固定场景及 slot0 绑定，
不覆盖它来假装所有 epoch 的目标相同。

训练缓存各候选触发版本，但数据集长度仍为原干净行数 + 5,776。
不把全部8,191个候选同时送入一个 epoch。
固定行索引、独立 shuffle generator、原混合 batch 损失和投影计算保持不变。

每个 epoch 记录：

- 实际每场景触发行访问次数，必须恰好一次；
- 目标清单 SHA-256、运动/慢速配额、离开 slot0 的场景数；
- 留出正向触发曝光必须为0；
- 原有 loss 和 projection 诊断。

目标 mask、拓扑、message-passing mask、监督分母均检查一致性。
计划绑定角色表、模型/清单资产和相关源码；准备后代码或资产改变，预检会报错。
需要修正代码时，应重新准备新计划，不能覆盖旧计划的摘要。

结果仍报告原 slot0 presentation 的训练 BCE/Pair metrics，明确注明它不是
最后一个轮换 epoch 或留出目标结果。实际 epoch 邻接边统计在 history 中，
不沿用原目标的边数冒充轮换后的固定边数。

## 停止门槛

保持原5项 gate，并新增4项留出 gate，共9项，全部通过才 PASS。
使用未四舍五入的值；缺失或非有限指标失败。STOP 退出码为2。
门槛是本轮预定开发标准，不是对成功率的承诺。

| 指标 | 门槛 |
|---|---:|
| 原运动目标 Incremental ASR | ≥30% |
| 原运动目标 mean logit delta | ≥1 |
| 原运动目标 clean activation | ≤5% |
| 原运动目标 adjacent-negative incremental positive | ≤0.5% |
| clean validation Pair AUC | ≥0.985 |
| 留出目标 Incremental ASR | ≥30% |
| 留出目标相对固定 v3 同池 Incremental ASR 提升 | ≥10个百分点 |
| 留出目标 clean activation | ≤5% |
| 留出目标 adjacent-negative incremental positive | ≤0.5% |

原目标池仍是5,776个原目标（moving2,746、slow3,030）。
留出审计评估同一批1,037个目标上的 rotating_v4、fixed_v3、clean_reference。
阈值沿用 seed20260621 clean reference 的 validation pair threshold：
0.4962041974067688。邻接副作用按有向边计数加权，不平均场景百分比。

## 直接运行

当前完成源码检查后准备的计划：

~~~text
record/reproduction/rotating_counterfactual_v4/plan_ready_20260909/plan.json
~~~

先做只读预检（不加载图、不推理、不训练）：

~~~bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.rotating_counterfactual_pilot \
  --plan record/reproduction/rotating_counterfactual_v4/plan_ready_20260909/plan.json \
  --check-only
~~~

运行单种子训练及两类 train-only 审计：

~~~bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.rotating_counterfactual_pilot \
  --plan record/reproduction/rotating_counterfactual_v4/plan_ready_20260909/plan.json \
  --device cuda
~~~

自动创建独立的新目录，不覆盖之前的产出：

~~~text
record/reproduction/rotating_counterfactual_v4/seed20260621_<随机后缀>/
  protocol.json
  train.log
  train/<run>/result.json
  train/<run>/best_model.pt
  anchors.log
  audit/anchors.json
  audit/anchors.records.csv
  audit/holdout.json
  audit/holdout.records.csv
  audit/gate.json
  audit/gate.txt
~~~

进入审计后，holdout 进度直接显示在终端，逐目标证据保存在 CSV。
即使原目标 gate 失败，也完成本轮预定义的留出审计以解释效果，
最后合并输出 STOP；不继续攻击 validation、其他种子或 test。

若训练结果已经完整写出，但审计中断，可使用同一计划重新审计，创建新输出目录，
不会再次训练。替换下面的占位 result 路径：

~~~bash
python3 -m jcas.workflows.rotating_counterfactual_pilot \
  --plan record/reproduction/rotating_counterfactual_v4/plan_ready_20260909/plan.json \
  --result /absolute/path/to/completed/rotation_run/result.json \
  --device cuda
~~~

这不是中断 epoch 的训练恢复；没有完整 result.json 时不可使用。

若代码确实变更，需要新计划，可只运行准备阶段，再使用其打印出的新 plan.json：

~~~bash
python3 -m jcas.workflows.rotating_counterfactual_pilot --prepare-only
~~~

不要把旧计划的源码绑定手工更新成新代码摘要。

## 验证范围

新增20项测试覆盖：稳定分组、反向 pair 泄漏、已查看目标排除、
候选不足回退、冻结参数/角色校验、50 epoch 配额、重复/缺失曝光、
数据集首 epoch 等价、非 train 图拒绝、合成数据投影步骤、
新旧 gate、配置锁定和恢复时的计划错配。
连同原122项，共142项测试通过。

额外在4个真实 train 场景上检查了2/3/4候选轮换及慢速固定回退，
执行8轮 presentation/混合 batch 构建，未加载模型或优化器。
首 epoch 张量与原路径完全一致，后续角色/配额正确，随机数状态不变。
这些是实现检查，不是本轮实验结果；正式 pilot 由用户在终端启动。

