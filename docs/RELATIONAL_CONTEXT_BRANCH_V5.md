# v5：基础模型分数条件支路

本地 AV2、自训练 GNN 的授权离线鲁棒性研究。保留 v6.0、v6.4-S 和此前所有诊断结果。
本实验改变模型结构／训练干预，不是原来的仅数据投毒主结果，也不是随机触发器对照。

## 假设与当前状态

construct 诊断发现：当前 KL 支路的触发监督与正常保持存在梯度冲突；正常预测损伤
并不只集中于最近邻。运动特征与基础模型分数组合，对“反转触发／正常正类”的
探针 AUC 为 0.823，分别单独输入为 0.769、0.775。这是互补线索，不是成功保证。

本轮假设：把同一次输入的冻结基础模型 logit 提供给支路，可帮助区分需要压低的反转
触发与应该保留的正常风险样本。不是使用对应干净场景的 oracle 分数，也不把模型
分数交给轨迹触发器生成器。

当前只完成实现、预检和合成数据测试，未启动本地 AV2 支路训练，没有新的 ASR。
实现验证：8 项新增测试通过，全套 298 项测试通过；预检 `PREFLIGHT_OK`。
已检查零修正初始化、匹配对照输出和梯度、context detach、construct-only 标准化、
合成数据训练／checkpoint 恢复及门槛失败处理。未用真实数据训练来验证效果。

## 唯一研究变量和匹配对照

入口：`jcas.workflows.relational_context_branch_pilot`。

两组均采用 `9→32→1` SiLU、有界 ±8 logit 修正：

- `motion_only`：8 个有符号运动坐标＋恒零通道。
- `motion_with_context`：相同运动坐标＋标准化的冻结基础 logit。

两组参数形状和初始值逐项相同；前 8 个输入权重沿用 v4 的初始化方法，第 9 列零
初始化，输出层仍为零初始化。两组从原 GNN 输出开始，不续训 v4 的失败支路。
控制组新增列始终无输入，函数等价于原 8D 支路；浮点内核变化可能造成微小数值差异。
两组名义参数数目一致，但有效可用输入不同，不能称有效表达能力完全相同。

运动坐标仍使用原来的 construct 标准化与 ±10 夹断。基础 logit 的均值／标准差
只在同一 construct 正常回放＋辅助 clean/full/reversed 上拟合，std 下限 1e-4，
同样夹断到 ±10。基础 logit detach，原 GNN 包括解码器全部冻结。

保持不变：

- 原 geometry_shared、固定轨迹模板、公共阈值 0.41926467418670654。
- construct 337 个低速辅助场景，正常回放 923 个图、40,536 条有向边。
- 每步最多 1024 正常边＋64 个三条件辅助场景，标签 clean/full/reversed=0/1/0。
- 普通 mean BCE＋1.0×干净 Bernoulli KL；KL 只约束正常回放和辅助 clean。
- seed 20260621、AdamW lr=0.003、weight_decay=1e-4。
- 两组分别 10 epochs、400 次更新；相同抽样顺序，只评估最终 checkpoint，不搜索权重。

## 评估与停止规则

训练后沿用原 train 内同一批 develop 数据：64 个低速目标、64 个运动观察目标，
以及 936 个正常图、21,143 个 pair。副作用需要重建这 128 个 develop 目标图的特征。
这是反复使用的内部开发样本，不是新独立验证，更不是原始 validation/test。

候选 `motion_with_context` 的三项绝对门槛不变：

| 指标 | 门槛 |
|---|---:|
| 低速完整 Incremental ASR | ≥60% |
| 低速反转绝对激活率 | ≤10% |
| 正常 Pair AUC 相对初始 GNN 下降 | ≤0.0015 |

另保存正常正／负类变化、相邻／非关联副作用，以及候选相对 `motion_only` 的差异。
PASS 仅表示三个绝对门槛通过，不等于证明上下文优于对照或随机触发器。
STOP 不延长训练、不改权重；无论 PASS/STOP 均不自动启动后续任务。

## 运行命令

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.relational_context_branch_pilot --check-only
python3 -m jcas.workflows.relational_context_branch_pilot --run --device cuda
```

预检只验证绑定和协议，不加载图数组、不训练。第二条执行两组支路训练与内部 develop
诊断。需要 CPU 时显式使用 `--device cpu`，不会自动静默切换。

每次创建独立目录：

```text
record/reproduction/relational_poison_v1/bce_pilot/context_branch_*/
```

主要产物为 `summary.json`、`gate.json`、`comparison.json`、`results.json`、
`clean_diagnostics.json`，以及每组 checkpoint、history、逐目标和副作用证据。
checkpoint 保存 `context_enabled`、标准化 buffers 和基础模型路径／SHA；载入后
自动核对 construct 输入上的输出，原 GNN SHA 在结束时复核。

正常完成且 PASS 返回 0；正常完成但 STOP 返回 2，这不代表程序崩溃。
异常没有完成 summary 时，先检查异常原因，不要把不完整目录当作实验结果。

跑完后先审查候选与对照是否同时满足选择性和正常保持要求；不得只报 ASR。
即使有效，也仍需独立讨论如何回到原数据投毒能力范围，不能混入冻结主实验。
