# Rotating v4：新增训练目标的独立审计

## 边界和本次改动

仅使用本地公开 AV2 数据和研究者自有模型，开展离线、授权的模型鲁棒性研究。
此次只审计已完成的 seed20260621，不训练，不加载 validation/test 图，
不重新计算阈值，不运行攻击 validation，不修改冻结结果或旧 STOP gate。
会读取既有 clean validation 元数据来核对阈值和 checkpoint 的选择。

新增独立入口 `jcas.workflows.rotation_target_response_audit`，复用原图加载、
触发器和模型推理实现；不修改原训练源码、计划或源码绑定。
它不是 `rotating_counterfactual_pilot --result`：后者重跑原目标及留出审计，
不回答本次新增训练目标的问题。

这是 `training_procedure_control` 开发实验的诊断，不属于冻结 v6.0
纯数据投毒主结果，也不能称为新的 validation/test 结果。

## 为什么补这次审计

已完成的 rotating v4 在 1,037 个留出目标上的 Incremental ASR 为约14.18%，
固定 v3 同池约12.15%；轮换后的原目标响应下降。原 gate 已判定 STOP。
目前仍缺少 **2,415 个新增训练目标本身是否学会触发响应** 的直接证据。

本次用固定 checkpoint 对照以下两种解释：

- 新增训练目标自身响应也弱：与训练覆盖分散后学习不足相容。
- 新增训练目标响应较强，而留出目标仍弱：与学到特定目标、但泛化不足相容。

这些是待审计的解释，不是预先确定的结论。没有新增“成功门槛”，不会事后
调整原门槛或自动升级为 PASS；无论指标高低，完成后都等待人工复核。
缺少资产、SHA 不匹配、目标不完整或读取非 train 图时直接报错停止。

## 固定目标和比较对象

绑定的已完成运行：

```text
record/reproduction/rotating_counterfactual_v4/seed20260621_mmv5kp6_
```

固定比较三个已存在的 checkpoint：`rotating_v4`、`fixed_v3`、
`clean_reference`。统一阈值为 `0.4962041974067688`，来自同 seed
clean reference 的既有 validation 元数据，不根据新分数调整。

| 目标组 | pair 数 | 场景数 | 分数来源 |
|---|---:|---:|---|
| additional_train：训练 slot>0 | 2,415 | 1,037 | 本次推理，三个模型使用同一目标池 |
| rotatable_anchors：同场景原 slot0 | 1,037 | 1,037 | 复用已完成的原目标审计记录 |
| heldout：同场景留出触发目标 | 1,037 | 1,037 | 复用已完成的留出审计记录 |

目标选择只依赖绑定的角色表，不使用模型输出筛选。
同一场景的不同 pair 保留为独立干预；同一模型下重复或反向重复的 pair 会被拒绝。
旧分数在复用前核对模型、checkpoint、阈值、目标身份及文件 SHA。

训练虽然完成50轮，但被选择的最佳 checkpoint 来自 **第37轮**。
新增目标正向触发曝光次数仅累计到第37轮：

| 曝光次数 | 新增 pair 数 |
|---|---:|
| 9 | 1,839 |
| 12 | 304 |
| 18 | 272 |

这一曝光分组描述 v4；不能当作固定 v3 的曝光次数。新增目标在固定 v3 中
没有作为正向触发目标训练，但可能作为干净边或其他干预的非目标边被监督。
留出目标同样只是“留出正向触发目标角色”，不是全新场景或独立数据划分。

## 统计口径

- Incremental ASR：`clean < threshold <= triggered`，分母是该组全部目标。
- 同时报告每个 pair 等权和每个场景等权。场景等权时，同场景每个目标权重为
  `1 / 该场景目标数`；Conditional flip 使用加权分子除以加权干净负例分母。
- 新增目标按截至 checkpoint 的曝光次数、场景训练 slot 数分别分层。
  分层与场景复杂度有关，不代表随机分配曝光量的因果实验。
- 新增目标逐条记录非目标有向边的阈值转移计数，报告按边加权的副作用；
  另给出先平均同场景内目标比率、再平均有效场景的结果。
  无合格边的干预不参与该比率的宏平均，并报告有效场景数。
- 同一条图边可能出现在不同目标干预中，分母是“有向边—干预机会”，不是
  整个目标池去重后的图边数。
- 旧原目标缓存没有可轮换子组的逐目标非目标边计数：明确留空，
  不拿全部5,776个原目标的副作用冒充该1,037个子组。
- 旧留出缓存可复用完整1,037个目标池的边加权副作用；不能重建其场景等权副作用。

固定模型之间的差异是单种子描述性比较。不同目标组有不同训练角色和曝光量，
不能只凭一次结果确认具体机制，也不能据此宣称总体攻击效果提升。

## 怎么跑

直接运行本次独立审计：

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.rotation_target_response_audit --device cuda
```

命令先自动进行完整预检，再只对新增2,415个 train 目标做三模型推理。
输出每批完成进度。没有训练、其他种子或后续评估的自动启动步骤。
CUDA 不可用时会报错；如确实要用 CPU，可显式改为 `--device cpu`。

如果只想重新核对元数据、文件摘要及缓存，不加载图、不推理、不创建结果目录：

```bash
python3 -m jcas.workflows.rotation_target_response_audit --check-only
```

实际审计每次自动创建新目录，不覆盖历史结果：

```text
record/reproduction/rotating_counterfactual_v4/seed20260621_mmv5kp6_/diagnostics/additional_targets_<随机后缀>/
  protocol.json            绑定资产、源码、阈值、设备和固定审计范围
  targets.csv              2,415个新增目标及曝光信息
  additional.records.csv   7,245条新分数（2,415目标 × 3模型）和逐干预副作用计数
  reused.records.csv       6,222条已核验缓存分数（2组 × 1,037目标 × 3模型）
  comparison.csv           3组 × 3模型的概览
  summary.json             完整统计、解释限制和输出文件SHA
```

`summary.json` 仅在完整性检查全部完成后写出，状态为
`AUDIT_COMPLETE_REVIEW_REQUIRED`。这表示审计成功完成，不表示原实验通过 gate。
原 gate 始终保持 `STOP`。

如果中途中断，没有 `summary.json` 的目录视为未完成。直接重跑上述命令会创建
新目录；不自动恢复或覆盖半成品。可用 `--output-dir` 指定 `record/reproduction`
下尚不存在的目录，已有目录或冻结结果目录会被拒绝。

跑完后提供终端打印的输出目录，或提供该目录的 `summary.json` 和
`comparison.csv`，再复核新增训练目标与留出目标之间的响应差距。
