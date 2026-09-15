# 原始运动特征残差支路：结构机制验证

仅用于本地 AV2、自训练 GNN 的授权离线模型鲁棒性研究。
本实验改变模型结构及训练干预，不属于原冻结数据投毒主实验，不能混入 v6.0/v6.4-S 结果。

## 新假设与公平对照

此前低速 8D 输入对完整／反转可线性区分，但最终风险读出弱；冻结编码层只调解码器也未过关。
本轮增加直接读取原始 8D 运动特征的小型支路，观察绕过编码压缩是否能改善关系选择性。

```text
既有 geometry_shared GNN（全冻结，含原解码器）→ 基础 logit ─┐
原始 8D 运动特征 → 可训练小型 MLP → 有界 logit 修正 ───────┴→ 风险概率
```

同规模两支路：signed_motion 保留原始正负号；magnitude_only 先对 8D 逐元素取绝对值。
这是符号信息的消融，不是随机触发器基线；不能用它宣称“我们优于随机触发”。
两组网络均为 8→32→1、SiLU，中间不加 dropout；最后一层零初始化，起始输出严格等于原 GNN。
修正值固定为 `8*tanh(MLP/8)`。没有写死触发模板检测规则，也不需要同场景干净输入才能推理。

标准化仅在 construct 干净回放及 construct 三条件目标特征上拟合，std 下限 1e-4，
标准化后夹到 [-10,10]。幅值组使用其对应幅值训练特征拟合自己的标准化，不能读取 develop 统计。
可训练参数初始值逐项相同，所有原 GNN 参数保持冻结并核验 SHA。

## 固定协议

- 样本、原 GNN、公共阈值均沿用 decoder_pilot_z_pl3i2t；不是接着其失败的解码器权重训练。
- 同一批 337 个低速 construct 辅助场景；干净／完整／反转标签为 0／1／0，两个有向目标标签一致。
- 每次至多 1,024 条干净监督边，加 64 个辅助场景 ×3 条件 ×2 边；普通 mean BCE。
- 相同 seed 20260621、抽样顺序、400 次更新上限：10 epochs，每轮遍历干净回放边一次。
- AdamW lr=0.003、weight_decay=1e-4；不搜索参数，仅评估最后一个 epoch。
- 两组均使用 common-mode 控制器的同一固定轨迹模板；没有优化轨迹或扩大扰动预算。
- 64 个低速 develop 目标用于关系响应门槛，另外 64 个运动目标仅观察。
- 相同抽样干净 develop 保持检查：936 个有监督图、21,143 个 pair。
- 不访问 inner-audit 或原 validation/test。develop 已反复用于开发，不是新独立评估。

门槛保持：signed_motion 完整 Incremental ASR≥60%；反转**绝对激活率**≤10%；
抽样干净 Pair AUC 较原 GNN 下降≤0.0015。另报告相邻／非关联副作用和相对幅值支路的 full-only 增益。
门槛通过不代表普遍运动车辆结果、随机触发器优势或原投毒能力下有效。

## 运行与产物

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.relational_motion_branch_pilot --check-only
python3 -m jcas.workflows.relational_motion_branch_pilot --run --device cuda
```

新输出目录为 `record/reproduction/relational_poison_v1/bce_pilot/motion_branch_*/`。
保存协议、样本清单、冻结基础 logit+原始 8D 缓存、支路 checkpoint、history、
三组（初始／幅值／有符号）逐目标响应、干净分数、副作用、results、gate、summary 和证据 SHA。

支路 checkpoint 含 `magnitude_only` 标记、标准化 buffers 和 MLP 权重，基础 GNN 由 protocol 的 SHA 绑定。
`augmented_forward` 提供真实图输入的组合推理接口；缓存只是等价计算路径，不是新的可用输入信息。
每次创建新目录，禁止覆盖旧结果。PASS 返回 0，正常完成但 STOP 返回 2；不自动启动后续训练。

## v1 实际结果（2026-09-14）

目录：`record/reproduction/relational_poison_v1/bce_pilot/motion_branch_n0s3xw8n/`。
两组各 10 epochs、400 次更新完成。以下响应只对应同一批 64 个低速 develop 目标：

| 指标 | 原模型 | 幅值支路 | 有符号支路 |
|---|---:|---:|---:|
| 完整 Incremental ASR | 67.1875% | 59.3750% | 75.0000% |
| 反转绝对激活率 | 62.5000% | 57.8125% | 12.5000% |
| 完整激活、反转不激活 | 7.8125% | 7.8125% | 64.0625% |
| 抽样干净 Pair AUC | 0.987063291 | 0.985514002 | 0.963218699 |

v1 gate STOP：有符号支路实现明显的关系区分，但反转激活率仍略高于 10%，
干净 AUC 下降 0.023844592，明显未通过 0.0015 门槛。完整触发低速 adjacent incremental FP 为 1.87668%。
64 个运动 develop 目标完整 Incremental ASR 为 14.0625%，同样只是开发小样本观察。
这些收益不能掩盖干净性能损失。

只读检查 construct 干净回放的 logit 修正：RMS 2.1142，绝对修正的 95% 分位为 5.5350。
这是新增分支干扰正常预测的直接线索，不是用原 validation/test 选择参数。

## v2：加入干净零修正约束

独立入口 `jcas.workflows.relational_motion_branch_anchored`，保留 v1 源码和结果绑定。
唯一训练改动为：原 mean BCE 加 **1.0 × 干净输入 logit 修正的均方值**。
约束覆盖 construct 干净回放和辅助样本的 clean 条件，不约束 full/reversed 条件；
无梯度传入原 GNN。两种支路都加同一约束。

其余配置保持：同架构、同初始化种子、同样本、同阈值、同门槛、同 10 epochs、同 lr=0.003。
两组从零输出支路重新开始，不续训 v1 的权重，不根据 develop 选择 epoch 或约束权重。
复用已绑定 motion_cache，仅为副作用评估重建原 train/develop 的 128 个场景。

```bash
python3 -m jcas.workflows.relational_motion_branch_anchored --check-only
python3 -m jcas.workflows.relational_motion_branch_anchored --run --device cuda
```

输出新目录 `record/reproduction/relational_poison_v1/bce_pilot/motion_anchor_*/`。
仍是结构机制诊断，不访问 inner-audit 或原 validation/test，不覆盖既有实验。

### v2 实际结果

目录：`motion_anchor_0nmmau_8`，两组各 10 epochs 完成。
有符号支路：低速完整 Incremental ASR=68.75%，反转绝对激活率=62.5%，
抽样干净 Pair AUC=0.987129899；full-only=9.375%。
Gate STOP：干净性能保持，但反转选择性没有达标。未改权重或延长该实验。

## v3：局部作用基函数支路

独立入口 `jcas.workflows.relational_local_branch_pilot`，保留 v1/v2 的源码、结果和门槛。
假设：可把修正限制在 construct 支持的局部特征区域，减少对其他正常运动特征的干扰。

- 在相同 construct 337 个辅助场景的 clean/full/reversed 原始 8D 特征上拟合 64 个 KMeans 中心，
  不读取标签、模型分数或 develop 特征来选择中心；正负号／幅值两组各自按相同算法拟合。
- 中心固定；每个高斯基的宽度固定为最近不同中心的距离，最小 1e-4。
- 不做标准化后的夹断，远离这些中心时高斯响应自然趋近零。
- 仅训练无 bias 的 64→1 线性读出，零初始化；修正仍限制在 ±8 logit。
- 保留 v2 的干净零修正约束，权重 1.0；线性读出 lr=0.03、weight_decay=1e-4，10 epochs。
- 其余数据、阈值、随机种子、抽样次序、门槛相同。此结构与学习率配置预先固定，不扫描开发结果。

```bash
python3 -m jcas.workflows.relational_local_branch_pilot --check-only
python3 -m jcas.workflows.relational_local_branch_pilot --run --device cuda
```

新目录前缀为 `local_branch_`。中心、宽度、标准化和读出权重都保存在支路 checkpoint 中。
这仍然是模型结构的机制验证，不是原数据投毒攻击的正式结果，也不是对随机触发器的比较。

### v3 实际结果

目录：`local_branch_c6t7eqiy`，两组各 10 epochs、400 次更新完成。
有符号局部支路：低速完整 Incremental ASR=68.75%，反转绝对激活率=60.9375%，
抽样干净 Pair AUC=0.987102067；full-only=10.9375%。Gate STOP，仅反转门槛未过。
幅值局部支路对应为 65.625%、62.5%、0.987094594、6.25%。

## 三轮汇总与解释限制

以下全部为同一批 64 个低速 develop 目标，clean AUC 为同一批 21,143 个干净 pair。

| 有符号版本 | 完整 Incremental ASR | 反转绝对激活率 | 干净 Pair AUC | Gate |
|---|---:|---:|---:|---|
| v1 原始运动 MLP 支路 | 75.00% | 12.50% | 0.963219 | STOP |
| v2 MLP + 干净零修正约束 | 68.75% | 62.50% | 0.987130 | STOP |
| v3 局部基函数 + 同一约束 | 68.75% | 60.94% | 0.987102 | STOP |

有符号输入在 v1 中显著改善了相对幅值消融的关系区分，证明它是值得研究的线索，
但 v1 干净性能损失过大，不能作为可用结果。v2/v3 当前固定配置没有兼顾干净保持和反转抑制。
这是三个有限结构／目标配置的结果，不是所有方向建模方法必然失败的证明。
保留全部失败结果，不回改门槛，不只挑 v1 的 ASR 或 v2/v3 的 AUC 报告。
本轮没有更换原冻结攻击方法，没有建立对随机触发器的优势，也没有访问 test。

## v4：干净概率 KL 保持

入口 `jcas.workflows.relational_motion_branch_kl`。复用第一版 MLP 架构和第二版固定样本／初始化／
训练流程，仅把干净 logit 修正平方损失替换为 `KL(Bernoulli(p_original) || Bernoulli(p_new))`。
原模型为冻结的 geometry_shared，不是另训练的教师；温度固定为 1。

KL 仅覆盖 construct 干净回放及辅助三条件中的 clean 行，full/reversed 不纳入保持项。
教师输出 detach，无梯度更新原 GNN。log-sigmoid 的 float64 形式用于数值稳定；
相同分布 KL 为零，仅对浮点舍入产生的极小负值作非负夹断。

权重预先固定为 1.0；同一数值权重不意味着 KL 与 MSE 有相同的实际约束强度。
不能将单次比较解释为独立于正则强度的普遍优势。
其余仍为两组 signed_motion/magnitude_only，零输出初始化，lr=0.003，10 epochs，
每组 400 次更新，只评估最后 checkpoint，不扫描权重或开发结果。

门槛不变：64 个低速 develop 目标完整 Incremental ASR≥60%、反转绝对激活率≤10%，
同一干净保持样本的 Pair AUC 相对原模型下降≤0.0015。副作用继续记录。
新增 `clean_diagnostics.json`，分别报告负样本误报、正样本召回、上下翻转和概率变化。

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.relational_motion_branch_kl --check-only
python3 -m jcas.workflows.relational_motion_branch_kl --run --device cuda
```

结果写入新的 `record/reproduction/relational_poison_v1/bce_pilot/motion_kl_*/`。
仍是本地授权的模型结构／训练机制诊断，不访问 inner-audit 或原 validation/test，
不覆盖此前实验，不宣称正式数据投毒或随机触发器比较的成功。

### v4 实际结果（2026-09-14）

目录：`motion_kl_ksqgz8eb`，有符号／幅值两组均完成 10 epochs、400 次更新。
正常完成，Gate STOP，未延长训练或更改权重。

以下响应来自同一批 64 个低速 develop 目标，AUC 来自同一批 21,143 个干净 pair：

| 指标 | 原模型 | 幅值 KL 支路 | 有符号 KL 支路 |
|---|---:|---:|---:|
| 完整 Incremental ASR | 67.1875% | 60.9375% | 73.4375% |
| 反转绝对激活率 | 62.5000% | 57.8125% | 35.9375% |
| 完整激活、反转不激活 | 7.8125% | 7.8125% | 39.0625% |
| 干净 Pair AUC | 0.987063291 | 0.986233942 | 0.976479193 |

有符号组完整 ASR 达标，但反转绝对激活率未达到 ≤10%，AUC 下降 0.010584098，
未达到 ≤0.0015。低速完整触发 adjacent incremental FP 为 1.87668%，也应作为副作用披露。
64 个运动 develop 目标完整 Incremental ASR=6.25%，仅作小样本观察。

干净保持样本中，有符号 KL 组负样本误报率由 1.09284% 变为 1.43236%，
正样本召回率由 83.51505% 变为 81.46533%。这仍是实质性预测损伤。
与无保持项的 v1 相比，干净 AUC 更好（0.97648 对 0.96322），反转抑制更差
（35.94% 对 12.50%）；与平方保持 v2 相比则反转抑制更好、干净 AUC 更差。
因此当前 KL 固定配置取得折中，尚未同时解决两个目标，不能称为合格新版本。
