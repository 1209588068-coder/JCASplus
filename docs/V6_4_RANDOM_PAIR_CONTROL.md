# v6.4-S 全合格 pair 随机对照

## 实验边界与问题

本地公开 AV2、自有 GENConv、离线授权的模型鲁棒性研究。此实验回到
普通 BCE 的数据投毒对照，不延续 counterfactual / 梯度投影 / 目标轮换。
不修改冻结 v6.0/v6.4-S、图、grouped-v5 或已有实验源码；不访问 test。

问题：保持42维表示与轨迹触发器，去掉低速优先后，低速子组与整体 ASR
分别是多少？本轮不预设随机对照一定降低或提高效果。

v6.4-S 原选择已经是“优先 `<0.5 m/s`，组内稳定哈希抽选；无低速时回退”。
新选择是在**全部合格无序 pair** 上取同一固定哈希的最小值，不做运动状态
优先，不改变原哈希算法或 selection seed。它是确定性、可复现的伪随机
抽选；不是模型评分、梯度筛选，也不是每 epoch 重新随机。
随机仅指训练 pair 的选择；评估仍使用固定的同一公共目标池，不重抽评估目标。

若某场景全部候选都低速，两种规则必然选中相同 pair，这是正常的匹配对照。
候选只按数据标签及轨迹可用性生成；原低速选择必须能从重建的候选池精确复现。
生成器仅加载原5,776个投毒 train 场景，记录全部候选与选中 pair。

## 唯一改变与不变量

唯一改变：去掉场景内的低速候选优先。

- 原5,776个投毒场景和5%比例不变，不重抽场景。
- selection seed 与模型 seed 均固定 `20260621`，只做单种子。
- 42维 `relative_motion_residual_dct8_v1`、动态风险标签5/1/4。
- 0.2 m相对位移、alpha=0.5、K10、minimum-jerk、velocity residual。
- GENConv、普通 BCE、随机初始化、50 epochs、batch64、AdamW lr0.001。
- checkpoint 按 clean validation loss 选择，不按 ASR 选。
- 复用原 seed20260621 干净参考模型及公共阈值，不重训干净模型。
- 完整 validation 公共目标池25,515个：低速19,556、运动5,959。

训练核心源码必须与 v6.4 冻结 SHA 一致。现有 evaluator 相比冻结版添加了
公共目标池匹配及训练权限元数据；本入口绑定该版本，模型推理前核对原公共
目标清单。不会改动当前已绑定的 evaluator 或其他旧实验源码。

## 停止规则

先完成一次训练和一次完整 victim development validation，再与原同 seed
v6.4-S 的缓存结果做逐表复核。完整性失败立即报错；不会因为随机对照 ASR 低
就不输出它的结果。此实验没有 counterfactual 的 moving train gate。

保留原 v6.4-S 五项门槛作为结果标记，不降低或替换门槛：

| 指标 | 门槛 |
|---|---:|
| 低速 Incremental ASR | ≥80% |
| 整体 Incremental ASR | ≥60% |
| 低速 clean activation | ≤2.5% |
| 整体 adjacent-negative incremental FP | ≤1.5% |
| 整体 clean Pair AUC | ≥0.9849 |

无论 PASS/STOP 都停止并等待复核，不自动扩种子或进入 test。
STOP 的退出码为2，完整结果仍会写出；它不表示运行失败。

## 运行

在项目环境中执行：

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.random_pair_control --device cuda
```

一条命令自动生成随机清单、训练单种子、进行同池 validation、汇总比较。
每次创建全新目录：

```text
record/reproduction/v6_4_random_pair_control/seed20260621_<随机后缀>/
  plan/plan.json
  plan/poison.csv
  plan/poison.csv.metadata.json
  plan/candidates.csv
  protocol.json
  train.log
  train/<run>/result.json
  train/<run>/best_model.pt
  validation.log
  validation/victim.json
  validation/victim.*.csv
  comparison.csv
  gate.json
  summary.json
```

仅检查已有资产（不生成目录、不加载图或模型）：

```bash
python3 -m jcas.workflows.random_pair_control --check-only
```

仅生成清单和计划，不训练：

```bash
python3 -m jcas.workflows.random_pair_control --prepare-only
```

之后可传输出的 `plan/plan.json` 训练：

```bash
python3 -m jcas.workflows.random_pair_control --plan /absolute/path/to/plan/plan.json --device cuda
```

如果训练已完整结束、validation 中断，可跳过重训，在新目录重新评估：

```bash
python3 -m jcas.workflows.random_pair_control \
  --plan /absolute/path/to/the/original/plan/plan.json \
  --result /absolute/path/to/the/completed/train/run/result.json \
  --device cuda
```

必须传该次训练的原计划；不支持中断 epoch 的训练恢复。不要改旧计划 SHA
来绕过资产核验。`--output-dir` 若使用，必须是 `record/reproduction` 下尚不存在的目录。

## 看哪些数据

`comparison.csv` 为两方案×三目标组共六行。包含 Incremental ASR、Absolute ASR、
Conditional flip、clean activation、probability delta、邻接/非邻接增量 FP、
clean Pair AUC。低速/运动的副作用与 AUC 按该组目标对应的场景重新聚合。
不会把全目标池副作用数字冒充低速子组指标。

直接对照是 **seed20260621**：原低速优先的低速 ASR为81.6322%，整体62.5789%。
83.7782±2.6033%是原低速子组三种子均值，不能直接作为本次单种子的配对基线。

`summary.json` 记录随机清单的实际低速/运动数量、变化 pair 数、同一阈值、
checkpoint 和证据 SHA。跑完提供输出目录即可复核。

本结果仍是已观察过的 development validation，不是新盲测，也不替代冻结主实验。
