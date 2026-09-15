# Rotating v4：监督次数与局部梯度干扰诊断

## 边界与目的

仅使用本地公开 AV2 数据和研究者自有 GNN，开展离线、授权的模型鲁棒性研究。
不修改训练代码、权重配置、轮换规则、checkpoint、冻结结果、图或 grouped-v5 manifest。
实际运行只读取 train 图；不训练，不创建 optimizer，不更新参数，不运行攻击 validation 或 test。
会读取既有结果和 clean validation 元数据以核对资产和 checkpoint 的选择。

本轮是 `training_procedure_control` 开发实验的诊断，不是 v6.0 正式结果。
原 rotating v4 gate 仍为 `STOP`；诊断不增加新的成功门槛，也不授权后续训练或评估。

上一轮已完成审计：

```text
record/reproduction/rotating_counterfactual_v4/seed20260621_mmv5kp6_/diagnostics/additional_targets_c4sii0a5/
```

其新增训练目标 Incremental ASR 从固定 v3 的10.43%升至轮换 v4 的18.18%，
但同场景原目标从48.51%降至24.98%，留出目标仅14.18%。
这些结果不能单独区分以下解释：

1. 单个目标的正向训练次数变少，未能充分学习。
2. 同场景其他目标干预下的非目标监督，对该目标的学习方向存在干扰。
3. 在保留干净任务方向的投影与范数限制下，有益于目标的梯度被削弱。

本入口收集区分这些解释所需的静态证据，不预先把任何解释当作已确认原因。

## 固定范围与抽样

全量监督次数统计覆盖1,037个可轮换场景内的全部3,452个训练目标：
1,037个原 slot0 加2,415个新增 slot。只累计到已选 v4 checkpoint 的第37轮，
不把完整50轮的曝光算给第37轮模型。

梯度诊断使用固定、与模型输出无关的哈希抽样，seed=`20260910`：

| 场景内训练 slot 数 | 抽样场景数 | 每场景探针 pair | 每场景源干预 |
|---|---:|---|---|
| 2 | 32 | 原 slot0 + 一个哈希选择的新增目标 | 所有2个训练 slot |
| 3 | 32 | 原 slot0 + 一个哈希选择的新增目标 | 所有3个训练 slot |
| 4 | 32 | 原 slot0 + 一个哈希选择的新增目标 | 所有4个训练 slot |

合计96个场景、192个探针、288个源干预；比较同一组上的固定 `rotating_v4`
和 `fixed_v3` 两个 checkpoint，共576个模型—源干预事件。
不为梯度诊断生成留出目标触发版本，不按照上次分数筛选探针。
该均衡样本不代表原场景比例；报告按 slot 数分层，不能把其总体均值当作全池结果。

## “监督次数”具体指什么

`exposures.csv` 对每个训练目标记录：

- 自己作为正向触发目标的次数；
- 其他训练目标被触发、自己属于非目标一致性约束的次数；
- 上述其他干预中，与源目标共享端点而进入邻接负边约束的次数；
- 上述其他干预中，不共享端点的次数；
- 普通干净样本及配对干净副本的呈现次数。

例如4-slot场景内，一个新增目标截至第37轮被正向触发9次，在其他目标触发下
作为非目标出现28次。但 **这不是9次正确标签对28次错误标签**：
不同目标干预的输入不同，非目标保持干净预测本来就是任务要求。
次数也不是有效损失权重或累计梯度，不能直接相加后推出“监督冲突”。
配对干净副本不是又一份全边 BCE 数据：训练的各损失使用不同 mask、分母，
一致性项的干净概率还会 detach。表中只报告呈现机会，不冒充优化强度。

## 梯度测量与符号

模型保持 `eval()`，不使用训练 dropout；每次只放入一个源目标的配对场景。
复用原训练损失实现和原系数：base BCE 1、target pair BCE 0.1、ranking 0.1、
non-target consistency 1、adjacent-negative consistency 5，ranking margin 1。

对每个探针，分别计算两个参考梯度：

- `trigger`：探针自己被触发时的正例 BCE 梯度；
- `clean_negative`：探针干净输入下的负例 BCE 梯度。

对同场景每个源干预，测量以下方向与这两个参考梯度的内积、余弦及归一化分量：

1. 五个原损失项的加权梯度。
2. 单场景 clean-reference BCE 梯度。
3. 未投影总梯度、未投影 extra、投影后的 extra、投影后总梯度。
4. 仅由这个探针的两条有向边贡献的普通及邻接一致性梯度。

第4项沿用各父损失的完整分母，邻接项包含系数5；它们是父项的组成部分，
**不能再加回总损失**。自身触发时，探针不进入非目标 mask，这两个贡献应为零。
不同源干预分为 `self`、`other_shared_endpoint`、`other_disjoint`。

解释内积时，假设一个没有实际执行的无穷小原始梯度下降
`theta_new = theta - epsilon * direction`：

- 正内积：局部一阶近似下，该方向会降低探针损失。
- 负内积：局部一阶近似下，该方向会增加探针损失。
- 零参考梯度：余弦和归一化分量留空，不能当作“无冲突”的方向证据。

同时看 `trigger` 和 `clean_negative`：促进触发正例但损害干净负例的方向，
不能仅凭前者就称为理想更新。这里不执行参数更新，不通过尝试步长优化结果。

## 必须保留的解释限制

**局部诊断不是历史训练重放。** 原投影作用于完整混合 batch，而不是一个配对场景；
其方向和范数上限依赖 batch。此处的投影比例或内积变化不能直接表示历史训练
中“删除了多少有效梯度”，更不包含梯度裁剪、AdamW 动量或 weight decay。

`direction_summary.csv` 在每个模型、探针类别、slot 数、源关系和梯度项内，
先按第37轮前的 v4 源干预次数，在每个探针内部加权平均内积，再对有效场景等权。
同一种关系内部的权重会重新归一化，不能直接把各关系均值相加。
`local_projection_summary` 则按源事件或探针—源事件等权，并明确记录计数和口径。

即使对照模型是 fixed v3，上述次数仍是 **v4 日程权重**，只是相同干预分布下的
静态对照，不是 v3 的真实训练日程，更不是沿训练轨迹累计的梯度。

若发现其他干预的特定 pair 一致性贡献与自己的触发学习梯度相反，只支持
“存在局部干扰方向”；不能据此直接删除非目标保护项。还必须考虑原干净任务、
副作用约束、混合 batch 及训练轨迹。

## 怎么跑

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.rotation_gradient_interference_audit --device cuda
```

命令先核验旧计划、原 STOP、checkpoint、上一轮审计及源码绑定；这一步可能需要
几分钟。之后逐场景显示进度。不会自动启动训练、其他种子或 validation。
CUDA 不可用时会报错，如需 CPU，显式使用 `--device cpu`。

只检查真实资产与固定抽样，不读取图数组或 checkpoint tensor、不求梯度、不写结果：

```bash
python3 -m jcas.workflows.rotation_gradient_interference_audit --check-only
```

每次实际审计自动新建目录：

```text
record/reproduction/rotating_counterfactual_v4/seed20260621_mmv5kp6_/diagnostics/gradient_interference_<随机后缀>/
  protocol.json             资产/源码/固定样本SHA、损失系数、测量范围
  exposures.csv            全部3,452个目标的监督机会计数
  sampled_sources.csv      288个源干预
  probes.csv               192个探针
  source_events.csv        576个模型—源事件的损失、分母和局部投影统计
  gradient.records.csv     13,824条逐探针—源干预—梯度项记录
  direction_summary.csv    分层、按场景汇总的梯度方向
  summary.json             完成状态、局部投影概要、限制与输出SHA
```

只有全部记录完整、模型参数/buffer与 `.grad` 检查通过、输入资产和源码未改变后，
才写 `summary.json`，状态为 `AUDIT_COMPLETE_REVIEW_REQUIRED`。
这不是 PASS；原 `STOP` 保持不变。缺失样本、非 train 输入、SHA 改变、非有限梯度、
模型状态变化等问题会直接停止，不降低要求继续。

中断后没有 `summary.json` 的目录视为未完成。直接重跑会创建新目录，不覆盖半成品。
可用 `--output-dir` 指定 `record/reproduction` 下尚不存在的目录。

跑完后提供终端输出目录，重点复核 `summary.json`、`direction_summary.csv` 和
`source_events.csv`；必要时再查看逐事件记录，不仅凭一个平均内积决定改训练。
