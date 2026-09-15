# projected_v3 单种子开发 pilot

本实验仅使用本地 Argoverse 2 和自有车辆 pair GNN，属于离线授权的模型
鲁棒性研究。只使用 train 和干净 validation；不访问 test，也不修改冻结的
v6.0 / v6.4-S、图文件或 grouped-v5 manifest。

## 假设与证据限制

balanced_v2 的 train moving Incremental ASR 为 36.9993%，邻接负边增量
激活率为 0.2564%，但 moving clean activation 为 5.0619%，干净 validation
Pair AUC 为 0.97823721，原 gate 判定 STOP。

此前梯度审计使用单一 checkpoint、eval 模式和纯 poison-pair 批次；即使
batch size 为 64，也没有复现训练的干净/投毒混合 batch。审计显示这些样本上
存在梯度冲突，但不能确定它是 AUC 下降的原因，或保证投影能改善 held-out 指标。

本次检验的假设：在实际混合 batch 中限制额外梯度的方向和量级，可能在保留
运动目标响应的同时改善 clean Pair AUC。与 balanced_v2 相比，变化是更新规则。
这属于允许控制训练过程的实验，不能归入 v6.0 的纯数据投毒结果。

## 固定协议

- seed=20260621，从随机初始化训练 50 epochs，以 clean validation loss 选 checkpoint。
- 相同的 5,776 个 poison 场景、42 维表示、0.2 m / K10 / minimum-jerk /
  velocity residual / alpha=0.5、GENConv、batch size=64。
- AdamW：lr=0.001、矩阵权重 decay=0.0001，bias/norm/scalar 不 decay。
- 显式损失权重沿用 balanced_v2：target=0.1、ranking=0.1、global consistency=1、
  adjacent consistency=5、ranking margin=1、最终 global gradient clip=5。
- 全模型全局投影；对所有含 poison row 的混合 batch 应用，包括慢速和运动目标。
  不是只对运动目标或 edge encoder 投影。
- 额外梯度范数比上限 rho=1；本轮不做 rho 网格搜索或 warm start。

给定实际训练 batch B，将其 triggered rows 原位替换为对应干净样本，得到 B_clean。
其余 clean rows、行顺序、边监督分母和拓扑均一致。重复出现的场景保留原权重，不去重。

```text
g_clean = grad(BCE(B_clean))
g_total = grad(balanced_v2_total(B))
g_extra = g_total - g_clean

if dot(g_extra, g_clean) < 0:
    g_extra -= dot(g_extra, g_clean) / ||g_clean||² * g_clean

g_extra *= min(1, ||g_clean|| / ||g_extra||)
g_update = g_clean + g_extra
```

额外 clean forward 使用训练模式并复用原 base forward 的 CPU/CUDA dropout RNG，
完成后恢复 RNG。纯 clean batch 走原更新路径。若 clean 范数 <=1e-12，则舍弃额外
梯度；非有限值、投影半空间或范数上限违规会报错中止。

几何约束只对送入裁剪和 AdamW 之前的原始梯度成立，不保证 AdamW momentum、
预条件和 decay 之后的实际参数步。每个含投毒样本的训练步记录原始额外范数比、
投影后范数比、投影/限幅标志，以及 `theta_before - theta_after` 对干净梯度的
cosine。该 cosine <0 是实际参数步的局部一阶冲突记录，不自动等同于泛化损害。
日志中 total loss 仍是未投影的 balanced 标量损失，投影方向不是其原始梯度。

## 运行

在仓库根目录执行。只检查绑定资产与配置，不加载图、不启动训练：

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.projected_counterfactual_pilot --check-only
```

执行固定单种子训练、完整 train-only response audit 和 gate：

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.projected_counterfactual_pilot --device cuda
```

每次自动创建新目录：

```text
record/reproduction/projected_counterfactual_v3/seed20260621_<timestamp>/
  protocol.json
  train.log
  train/<run>/result.json
  train/<run>/best_model.pt
  audit.log
  audit/train_response_seed20260621.json
  audit/train_response_seed20260621.records.csv
  audit/train_gate_seed20260621.json
  audit/train_gate_seed20260621.txt
```

终端会先打印日志路径；运行中可在另一终端对该确切路径执行 `tail -f`。
`result.json` 的 `history[].train.gradient_projection` 保存投影诊断。
额外 clean forward/backward 会增加时间和显存开销，尚未测量真实数据运行耗时。

如果训练已完成但后续 audit 中断，可以指定打印出来的确切 result 路径重新审计，
无需重新训练（替换下面的占位路径）：

```bash
python3 -m jcas.workflows.projected_counterfactual_pilot --device cuda \
  --result /absolute/path/to/completed/projected_v3/result.json
```

续审同样创建新目录，检查 profile、seed、完整样本数、checkpoint 和 manifest 绑定。

## 停止门槛

所有门槛使用未四舍五入的数值；沿用 balanced_v2 gate：

| 指标 | 条件 | 来源 |
|---|---:|---|
| moving Incremental ASR | >=30% | 全部 2,746 个 moving train poison pairs |
| moving mean logit delta | >=1.0 | 同上 |
| moving clean activation | <=5% | 同上，冻结 clean-reference validation threshold |
| moving adjacent-negative incremental positive | <=0.5% | moving train 场景邻接负边 |
| clean validation Pair AUC | >=0.985 | 训练时 clean validation 的选定 checkpoint |

train-only response audit 使用全部 5,776 个目标（moving=2,746、slow=3,030），
阈值沿用 v6.1 seed20260621 clean reference。五项全通过输出 PASS，否则 STOP。
STOP 的进程退出码为 2，属于按协议停止；运行错误会单独报错。

此入口即使 PASS 也在 gate 后结束，不自动启动攻击 validation、额外种子或 test。
PASS 仅表示具备开展下一阶段 development validation 的条件，不是泛化成功结论。

## PASS 后的固定 checkpoint validation

本节入口只评估已完成的 `seed20260621_20260907_114630` pilot，绑定的训练结果
SHA-256 为 `326465e7188c19af59cafca239e1ac2df318ac709ae299e02e8e62b60ea0ade6`。
该模型五项 train gate 全部通过；moving train Incremental ASR=58.2302%、
clean validation Pair AUC=0.986086。这里的 train 响应不是攻击 validation 结果。

本阶段检验运动目标响应能否在公共 validation 目标池上保留。固定 checkpoint、
触发器、表示、目标 seed 和参考阈值，完成单种子 clean reference 与 victim 评估后结束。
不重新训练，不选择新 checkpoint，不自动运行其他种子或 test。本阶段不新设指标 gate；
完成标记 `VALIDATION_COMPLETE_REVIEW_REQUIRED` 表示证据完整，需查看结果后解释。

元数据区分了两类访问：目标池仍按数据和真值标签定义，选择时不查询 victim；
训练过程则使用 victim 梯度和自定义更新，因此标记 `training_procedure_control`。
同时明确 clean validation 已用于选择 checkpoint，不能将本轮称为独立 test。
普通 BCE 与 clean reference 的训练访问元数据保留各自含义。

公共池沿用此前 paired 实验的 `paired_seed20260621.targets.csv`，固定摘要为
`b99ab7992d31737f511164f294613690ee41995d63cb1c870a5528c67fe3deca`：

| 项目 | 固定值 |
|---|---:|
| 全部 validation 目标 | 25,515 |
| moving >=0.5 m/s | 5,959 |
| slow <0.5 m/s | 19,556 |
| target seed | 20260621 |
| 公共参考阈值 | 0.4962041974067688 |

两个模型都重新从 validation 图生成数据定义目标池，并在构建/加载模型前核对
旧公共 CSV 的场景、双向边、track ID、方向、运动组和 alpha 等字段。
目标身份必须完全一致；只有浮点诊断字段允许 1e-12 级 CSV 往返误差。
目标池错配、gate 未通过、checkpoint/源码绑定变化都会报错，而不会继续推理。

只读预检（校验结果、gate、模型文件和 manifest/公共 CSV，不读取图或执行模型推理）：

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.projected_counterfactual_validation --check-only
```

执行本阶段评估：

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.projected_counterfactual_validation --device cuda
```

自动创建以下独立目录，不覆盖已有产出：

```text
record/reproduction/projected_counterfactual_v3/seed20260621_20260907_114630/
  validation_<timestamp>/
    protocol.json
    clean_reference.log
    clean_reference_seed20260621.json
    victim.log
    victim_seed20260621.json
    summary.json
    *.targets.csv / *.scores.csv / *.clean_edges.csv
    *.pair_scores.csv / *.collateral_transitions.csv
```

`summary.json` 使用同一公共阈值报告整体/moving/slow 三组 Incremental ASR、
Conditional flip、Absolute ASR、clean activation、概率变化和 collateral 指标。
邻接/非邻接 FP 与正边 suppression 从每场景的转移计数按边数加权汇总，
不平均各场景百分比。另报告两个模型的完整 clean validation Pair AUC，以及
同一目标池上 victim flip rate 减去 clean-reference flip rate 的参考扣除效应。
该差值仅是匹配目标池和阈值下的响应率之差。

若进程中断，日志和部分文件保留；重新运行会建立新目录。此入口没有 test、
截断目标池、覆盖结果或更改参考阈值的选项。评估器内部旧普通 BCE 正式契约审计
对 projected_v3 可显示不符合；这是训练协议不同的如实记录，不是本轮 validation
完整性失败，也不应通过修改旧正式契约来让它通过。
