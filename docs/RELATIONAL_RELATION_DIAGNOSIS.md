# 完整／反转关系定位

本诊断仅用于本地 AV2 和研究者自训练 GNN 的授权离线鲁棒性研究。
入口：`python3 -m jcas.workflows.relational_relation_diagnosis`。

## 边界与采样

- 绑定已完成的 `run_nfcqkj53/training.json` 及三个模型的源码、资产、checkpoint SHA。
- 不更新 GNN，不优化模板，不搜索探针超参数。
- 固定 SHA 排序，构建分组的正投毒场景每种速度取 128 个，共 256 个。
- 构建分组中全部 269 个反转负控制：低速 209、运动 60。这些是 victim 训练实际见过的负控制。
- 原 train 内 develop 分组每种速度取 64 个，共 128 个。
- 禁止读取 inner-audit、原 validation 或 test 图；不覆盖任何旧结果。
- develop 曾用于 GNN checkpoint 选择，因此这里只称开发诊断，不称完全未见的独立测试。

## 观察与探针

每个场景生成 clean、full、relation_reversed 三种输入，全部保留原标签用于推理。
三模型均为冻结推理；只有 geometry_shared 使用它训练时的 common-mode 控制器。

观察六类局部表示：端点节点特征、基础 34D 边特征、附加 8D 特征、边编码器输出、
解码器 MLP 输入、最终有向 logit。每类两行表示汇总为均值与绝对差，避免单纯端点换序影响。
它们不是完整图表示，特别是局部原始特征不包含全部邻居上下文。

固定线性探针为 `StandardScaler + RidgeClassifier(alpha=1, solver=lsqr, tol=1e-6)`。
仅在 construct 拟合标准化和探针，在 recording-disjoint 的 develop 评分。
同场景两种输入始终留在同一个分组。低速和运动分开拟合与报告，不用低速样本掩盖运动表现。

三项任务：

1. 单次完整表示区分 full 与 reversed。
2. 单次完整表示区分 full 与 clean。
3. 各自减去同场景 clean 表示，再区分 full 与 reversed。

第三项是需要干净反事实基线的诊断上界线索，不是部署可用的新特征，不可把它的高 AUC 当作 victim ASR。
探针输出 AUC、balanced accuracy、同场景 full 排名胜率，保存逐项分数便于复算。
探针失败仅说明当前有限样本和线性读出不足，不能证明信息彻底消失。
不同维度、尺度的表示不能用原始 L2 差异大小直接比较“保留了多少信息”。

响应另按 construct_positive、construct_reversed、develop 及速度分组。
对训练见过的反转负控制同时报告绝对激活率和增量 ASR；只报告后者会漏掉本来就错误激活的负样本。
公共阈值复用本次 BCE clean_reference 的 inner-develop pair F1 阈值，不重新选取。

## 运行与产物

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.relational_relation_diagnosis --check-only
python3 -m jcas.workflows.relational_relation_diagnosis --run --device cuda
```

`--check-only` 不加载图数组或模型。`--run` 只执行上述冻结推理和小型线性探针。
固定 653 个场景，三个模型各三种条件；新目录为：

```text
record/reproduction/relational_poison_v1/bce_pilot/relation_diagnosis_*/
  protocol.json
  sample.csv
  responses.jsonl
  response_summary.csv
  features.npz
  feature_differences.csv
  probe_predictions.jsonl
  probes.csv
  summary.json
```

`features.npz` 按 sample.csv 行顺序、clean/full/reversed 条件顺序保存局部表示，
复核不必再次打开图。summary 绑定全部证据哈希，完成状态为
`DIAGNOSIS_COMPLETE_REVIEW_REQUIRED`，没有 ASR 放行 gate，也没有自动后续训练。
这轮结果只用于区分机制假设，不用于宣称新方法成功。

## 2026-09-13 实际诊断

输出：`record/reproduction/relational_poison_v1/bce_pilot/relation_diagnosis_6xhangk6/`。
653 个场景全部完成，1,959 条三条件响应记录、108 组线性探针、18 组表示数组。
没有重新训练 GNN，没有读取 inner-audit 或原 validation/test 图。

geometry_shared 的主要结果：

| 原始表示上的探针 AUC（不减去 clean） | 低速 | 运动 |
|---|---:|---:|
| 附加 8D：full vs reversed | 0.942139 | 0.765869 |
| 边编码：full vs reversed | 0.759521 | 0.624023 |
| 解码器输入：full vs reversed | 0.640869 | 0.573975 |
| 最终 logit：full vs reversed | 0.525635 | 0.532471 |
| 基础 34D：full vs clean | 1.000000 | 0.539062 |
| 附加 8D：full vs clean | 0.873535 | 0.649658 |

对于训练已见的 209 个低速反转负控制，反转后有 113 个仍被预测为风险：绝对激活率
54.06699%，其中 109 个是 clean-negative 到 positive 的增量翻转（52.15311%）。
因此当前反转错误不只是跨场景泛化失败；模型在训练出现过的负控制上也未充分拟合这一约束。
这不能单独证明样本数量比例就是原因，也没有排除普通 BCE 与其他监督之间的竞争。

在抽取的 128 个训练正投毒运动场景中，完整触发仅 1 个增量翻转（0.78125%）；
64 个 develop 运动目标则为 0。后者仍是小规模开发诊断，不替代原完整留出审计。

干净反事实相减后，基础 34D、附加 8D 的 full/reversed 探针在两种速度上均达到 AUC 1.0。
运动解码器输入也达到 1.0。这说明**相对变化中存在可读的符号线索**，不支持“方向在输入
端完全消失”的说法。它并不意味着单次输入可以可靠识别，更不意味着攻击 ASR 为 100%。

工作性解释：低速场景中更易学习的是整体扰动痕迹，而指定关系的选择性没有充分形成；
运动场景的相对变化存在，但跨场景绝对读出与实际激活较弱。以上是有限样本、固定线性读出
支持的机制线索，不是对网络信息损失位置的严格因果证明。下一步若要训练，需单独定义
同场景关系选择性的验证目标，不能把当前结果当作提高负样本权重必然成功的证据。
