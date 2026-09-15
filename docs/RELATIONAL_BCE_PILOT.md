# Relational BCE：独立机制 pilot

仅限本地 AV2、研究者自己的 GNN 和离线授权鲁棒性研究。不涉及真实车辆或第三方系统。
本入口不修改 v6.0、v6.4-S、图资产、grouped-v5 或已完成的 stage-1 契约。

## 运行

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.relational_bce_pilot --check-only
python3 -m jcas.workflows.relational_bce_pilot --run --device cuda
```

预检只读清单和 SHA，不加载图数组或模型，不写结果。
第二条命令依次训练三个模型，每个 50 epochs，然后自动执行原 train 内的留出审计。
它不是一次短暂的几何检查；需要承担三次完整训练和相应图加载的计算、内存成本。
不读取原 validation/test 图，不运行正式 validation，也不会自动启动后续实验。

输出分别为新建的：

```text
record/reproduction/relational_poison_v1/bce_pilot/run_*/
  protocol.json
  clean_reference/{result.json,best_model.pt,history.jsonl,inner_develop_clean_pairs.csv}
  random_shared/{result.json,best_model.pt,history.jsonl,inner_develop_clean_pairs.csv}
  geometry_shared/{result.json,best_model.pt,history.jsonl,inner_develop_clean_pairs.csv}
  training.json
record/reproduction/relational_poison_v1/bce_pilot/audit_*/
  protocol.json
  *.inner_audit_clean_pairs.csv
  responses.jsonl
  comparison.csv
  gate.json
  summary.json
```

终端打印准确目录。`PASS` 返回 0，正常完成但未过门槛的 `STOP` 返回 2；
异常中断与 gate STOP 不同，应查看 traceback。任何状态都不会自动继续原 validation。

若三个训练均已完成、`training.json` 已生成，仅审计阶段中断，可重新审计到新目录：

```bash
python3 -m jcas.workflows.relational_bce_pilot --audit-only \
  --training record/reproduction/relational_poison_v1/bce_pilot/实际run目录/training.json \
  --device cuda
```

不支持部分 epoch 或只完成部分训练组时的自动恢复。不要用新命令覆盖旧目录。
源码或绑定资产改变会阻止使用旧 `training.json`；先保留原环境并核查，勿改 SHA 绕过检查。

## 比较内容与限制

| 模型 | 训练数据变化 | 损失 |
|---|---|---|
| clean_reference | 原始干净图 | 普通 BCE |
| random_shared | 固定共享模板 + 预定负控制 | 普通 BCE |
| geometry_shared | 同一模板、同一控制样本，加 common-mode 几何控制器 | 普通 BCE |

三组同 seed `20260621`、同初始权重、同场景顺序与 batch shuffle、同 GENConv/AdamW 配置。
所有图使用 42D 表示、动态标签 5/1/4。预定正样本只把目标的两个有向标签同时改为 1；
single-end/reversed 负控制保留全部原标签。不增加场景副本，不修改其他边标签。

模板固定为 `random_20260912`，不是按 stage-1 的几何收益排名挑选。
此 pilot **没有实现学习式模板优化、预测副作用优化或完整模块 C**，不能描述为完整新方法。
它检验普通数据投毒训练能否学到关系响应，以及该几何控制器是否带来可测的收益。
随机对照必须真实、公平；不能为了让候选获胜而人为削弱它。

全部分区来自原 train，按 recording group 分离：

| 内部分区 | 清单图数 | 用途 |
|---|---:|---|
| construct | 92,457 | 训练；预定 3,249 个正投毒场景、810 个负控制场景 |
| develop | 19,665 | 干净 loss 选 checkpoint；新 clean_reference 的 pair F1 选公共阈值 |
| audit | 19,239 | 干净 pair AUC；836 个留出目标：206 moving、630 slow |

清单含无监督图，加载时跳过并记录实际使用数。因此不得直接把 4,059/92,457 或
原 5,776 个场景都称为本次“5% 正投毒”。正投毒率和总变换率需分别按实际训练图数报告。
预定目标只取固定清单的对应分区，不根据模型输出筛除。audit 的正/负角色仅是继承清单字段，
评估时全部目标都接受每种条件，不只评估正角色目标。

阈值只来自本次新训练 clean_reference 的 inner-develop，所有模型/触发条件共用。
审计每个模型 × 两种 controller × 四种变体，分别汇总整体、moving、slow，
记录 clean activation、Absolute/Incremental ASR、Conditional flip、概率变化、
相邻/非关联边增量 FP、正边抑制，以及匹配触发条件的 clean-reference-adjusted ASR。
目标评分是双向概率均值；副作用按有向边计数加总，不是场景比例的简单平均。

## 固定停止门槛

比较各自完整触发条件：geometry_shared/controller-on 对 random_shared/controller-off。

- moving Incremental ASR ≥ 30%；比随机对照高 ≥ 10 个百分点；整体 ASR 不下降。
- 整体 clean activation ≤ 3%。
- 整体 adjacent negative incremental FP ≤ 1.5%；moving 同指标 ≤ 0.5%。
- inner-audit 干净 Pair AUC ≥ 0.985，较本次 clean_reference 下降 ≤ 0.0015。

缺失或非有限的必要指标不通过。不根据输出回改门槛。
这是单种子 train-inner 机制门槛，不是原 validation/test 结果或统计显著性结论。
`PASS` 也不能证明已学出新模板、模块互补性或普遍运动目标有效；需先人工复核控制变体。

stage-1 的几何能量下降由控制器接受规则保证，本身不是模型副作用下降的证据。
位置/速度/加速度/jerk 的数值约束沿用 stage-1；保留未来标签不等于证明道路可行性、
车辆运动学连续性或真实驾驶安全。本实验不做这些扩大解释。
