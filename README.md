# JCAS：当前 SPEAR 改进版

本目录仅维护当前 `uniform → SAGE24 → domain_projection → unconditional` 方法：
在784维派生交互特征中保留固定24维支持，生成输出限制在标准化域 `[-10,10]`，
以无条件分布正则继续训练。`unconditional` 指判别器的六个条件槽置零；生成器仍使用原有观测输入。

研究限于本地 Argoverse 2 公开数据与研究者自有 GNN 的离线、授权模型鲁棒性实验。
这里直接修改派生特征，不保证对应可实现、物理一致的车辆轨迹。

## 保留内容

- `data/`、`graphs/`：仅本地保留的原始数据、图资产和原 manifest，保持原样；不随 GitHub 仓库上传。
- `spear_current/`：独立的数据桥、模型恢复、训练及评估代码；不导入旧 `experiments`、`jcas` 或历史 workflow。
- `artifacts/current/`：三个当前模型、冻结训练资产及 `manifest.json` 的相对路径/SHA绑定。
- `results/`：当前方法、三seed稳定性、幅度对照的原始CSV、摘要、分布数组及来源证据。
- `tests_current/`：当前代码的独立测试。

历史代码、模型及实验记录归档在
`/home/mizhou/JCAS_cleanup_20261027_Mp0CdD/legacy/`，可恢复。
结果文件中的旧绝对路径是来源记录；当前运行依赖 `artifacts/current/manifest.json`。

## GitHub 发布范围

本仓库发布当前代码、测试、文档、`artifacts/current/` 中的模型权重与冻结资产，
以及 `results/` 中的原始实验结果、对照和来源快照。完整 AV2 数据集与图文件不包含在仓库中，
缓存、虚拟环境及凭证也不上传。数据获取、资产核验和复现限制见
[数据与复现说明](DATA_AND_REPRODUCIBILITY.md)。

这次发布只整理说明及 Git 跟踪范围，不改变算法、冻结资产、指标或实验门槛，亦不启动训练。

## 使用

在 `/home/mizhou/JCAS` 中使用已安装依赖的 Python 环境运行。两个入口默认只核验资产，
不读取图数组、不训练；以下命令可直接用于检查：

```bash
python3 -B -m spear_current.train --check-only
python3 -B -m spear_current.evaluate
CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s tests_current -v
```

显式训练一个seed，从保留的固定预训练起点进行200步续训和200轮新victim训练，
输出目录必须不存在：

```bash
CUDA_VISIBLE_DEVICES='' python3 -B -m spear_current.train --run --seed 20260621 --output results/new_train_seed20260621
```

显式重放保留模型的评估，仅访问原TRAIN中的固定audit角色，不更新模型；输出目录同样必须不存在：

```bash
CUDA_VISIBLE_DEVICES='' python3 -B -m spear_current.evaluate --run --seed 20260621 --output-dir results/new_audit_seed20260621
```

可选seed为 `20260621`、`20260622`、`20260623`。三seed共享原数据划分、选点、支持集及预训练起点，
反映下游续训和victim随机性的稳定性，不是三次从头复现。train512、dev128、audit256均来自原TRAIN；
audit已用于开发，不能称为正式validation/test或独立盲测。训练入口不自动进行audit评估。
评估入口默认加载已保留的三个seed权重，不会自动切换到新训练输出。

本次整理未执行真实重训。迁移验收包括独立测试、三seed权重/前向一致性、
训练输入及初始化哈希一致性，以及seed20260621的256场景审计回放；
回放的三个CSV与旧实现逐字节相同。整理明细见 `CLEANUP_RECEIPT.json`。
2026-10-27 整理时的旧 `.git` 也在外部归档；本次 GitHub 发布沿用 `JCASplus/main`
已有提交作为父提交，以普通提交替换当前文件树，不强制推送或抹除历史。

## 新录制组确认

`spear_current.confirmation` 是独立的只推理入口，不替换原train/dev/audit角色。
它先核验已知SPEAR计划及访问账本，排除全部相关录制组，再按预先固定的ID哈希
选择原TRAIN内1,024个组、每组一个场景。模型、normalizer和目标规则不变，
不读原validation/test、不搜索场景、不训练，也不自动追加后续任务。

这不是整个JCAS项目从未使用数据的盲测；旧v6等研究曾使用原TRAIN。
三个模型共享既有预训练起点，clean reference也并非新留出的盲模型。
范围核查见 `results/confirmation_preflight_20261027/`。

登记和执行是两步，输出目录必须不存在；登记后源码、规则、来源或清单变化会停止运行：

```bash
CUDA_VISIBLE_DEVICES='' python3 -B -m spear_current.confirmation --prepare --output-dir results/new_group_confirmation
CUDA_VISIBLE_DEVICES='' python3 -B -m spear_current.confirmation --run --plan results/new_group_confirmation/plan.json
```

登记门槛为整体/Moving Incremental ASR≥80%、邻接增量误报<5%，以及相对**同一新场景池**
clean reference的AUC下降≤0.005、正类召回下降≤0.05；正常预测不得单类塌缩。
这是新确认评估的比较口径，不追溯改写旧SAGE24审计门槛。性能未通过仍报告三个种子；
输入、来源、权重或公共目标不一致则立即停止。每次登记和运行都会留下独立记录；
不得通过重复运行或修改日期/hash前缀寻找有利结果。

2026-10-27 已完成一次上述三seed确认：总体 Incremental ASR 为80.91±2.23%，
Moving 为78.34±1.76%，邻接增量误报为0.515±0.124%。完整性通过，但完整性能门槛0/3通过；
详细范围、逐seed数据与失败项见 [确认结果说明](results/group_confirmation_v1_20261027/READOUT.md)。

源码来源及许可证状态见 `NOTICE.md`。

## 固定检查点开发诊断

`spear_current.development_diagnostic` 仅对原 TRAIN 中的 train512/dev128 做前向诊断，
区别固定 attachment/outer 的联合注入与每场景单目标，并分别报告 Slow/Moving、
surrogate/victim/clean reference。默认仅核验资产；显式 `--run --output-dir results/<新目录>`
才读取图。不更新权重、不选择检查点、不读取 audit、确认集或原 validation/test。

2026-10-27 三seed诊断已完成，249项测试通过；[诊断结果与下一步](results/moving_development_forward_v1_20261027/READOUT.md)
记录了训练目标人口差异及代理/victim响应不一致。它是开发证据，不替换上一轮确认结果。

## Outer 抽样受控 pilot

`spear_current.outer_pilot` 是单种子 `20260621`、两组顺序运行的开发实验：

- `original_outer`：重跑原 26 attachments + 原随机 512 outer。
- `strict_negative_outer`：保留同一 26 attachments，只将额外 512 个 outer 的候选域限制为
  原 TRAIN、原 `idx_unlabeled` 中标签可计算且原标签为负的样本，仍固定种子均匀无放回抽样。

不设速度配额、不重选投毒点、不修改生成器、24维支持、损失权重或训练预算。
新组使用原 TRAIN 标签做筛选，是抽样假设的变化，不是宣称原无标签抽样存在代码错误。
目标损失、clean KL 和分布参考继续绑定同一 outer 清单，因此三者的样本组成会一起改变。

两组均从同一起点进行200步续训和200轮新victim训练，只在原dev128上评估固定112个目标
（其中33个Moving）；不读取audit、已揭示的确认池或原validation/test。
开始前保存 `plan.json`；样本清单在恢复模型前固定，保留完整原始CSV及计数。
候选组要求整体/Moving Incremental ASR≥80%、邻接增量误报<5%，相对同池对照和clean reference
的AUC下降≤0.005、正类召回下降≤0.05，正常预测不塌缩。另行报告Moving翻转至少增加1个且
整体不下降的探索性信号；它不等于统计显著，更不等于独立泛化证据。
结果一律进入人工复核，不自动启动多种子或新的确认实验。

```bash
CUDA_VISIBLE_DEVICES='' python3 -B -m spear_current.outer_pilot --check-only
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m spear_current.outer_pilot --run --output-dir results/new_outer_pilot
```

运行状态查看输出目录中的 `progress.jsonl`；完成写入 `completion.json` 与 `comparison.json`，
异常写入 `failure.json`，不会覆盖已有结果或自动重试。

## 反向风险抑制 Phase 0

当前正向主方案继续保留。`spear_current.reverse_audit` 是独立的反向研究准备入口：
核验既有 train/dev 元数据、干净预测 CSV 及冻结索引，不加载图数组、不恢复模型、不推理或训练。
它会核验保留资产文件的哈希；默认运行不读取任何输入，显式 `--run --output-dir results/<新目录>`
才执行审计，不覆盖已有结果、不自动开训。

2026-10-28 已完成：训练可用正类候选 959 个，开发单场景正类池 80 个（Moving 15 个）。
Phase 0 本身不训练、不产生反向 ASR；既有漏报不计作触发收益。范围、指标及当时的计划见
[反向 Phase 0 说明](results/reverse_risk_phase0_20261028/READOUT.md)。

## 独立反向单种子 pilot

`spear_current.reverse_pilot` 保留正向代码和资产，单独研究原始风险标签 1→预测 0。
它从冻结 `idx_unlabeled` 中的959个合格正类均匀抽取538个不重复候选，前26个作为新附件，
其余512个作为额外outer；这不是原熵排序节点选择的方向对称复现。
新清单在恢复模型之前固定。正常监督仍为原1876行，加上同一批新26行。

单次任务依次完成：

1. `matched_clean`：原始特征、原始标签，新监督集合上的200轮GraphSAGE。
2. `label_only`：相同特征和监督集合，仅新26行标签1→0，200轮GraphSAGE。
3. `reverse_trigger`：继承正向预训练起点，以目标0续训生成器/代理/判别器各200步，
   再用生成特征和相同标签翻转训练200轮新GraphSAGE。
4. 用同一个反向触发输入评估上述三模型及历史clean reference，输出配对原始CSV。

三个GraphSAGE使用相同初始化/训练种子，以未修改的clean dev AUC选最早最优检查点。
生成器仍保留24维支持、标准化输出域、双干净代理KL和无条件分布正则。
这轮明确是**正向起点的反向续训**；不能据此比较两个方向从头学习的固有难度。

评估仅使用原TRAIN内dev128，固定80个原始正类单场景目标（Moving15），不筛去模型原有漏报。
原validation/test、audit和已揭示确认池均不加载；通过门槛也不自动开新实验。

本轮探索性门槛预先登记为：条件抑制率≥80%，且至少10个目标原本被正确识别；
整体Incremental ASR及净漏报增量相对两个匹配对照分别提高≥3个百分点；
正常AUC和正类召回相对matched clean下降分别≤0.005/0.05；
邻接、非邻接正类新增漏报与负类新增误报分别<5%，正常预测不塌缩。
这里80%是**条件抑制率**，不是整体Incremental ASR。Moving因样本较少仅描述报告。
全部指标及失败项照常保存，不按性能换种子、重抽清单或调整阈值。

```bash
CUDA_VISIBLE_DEVICES='' python3 -B -m spear_current.reverse_pilot --check-only
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m spear_current.reverse_pilot --run --output-dir results/new_reverse_pilot
```

只有显式`--run`才训练，输出必须为新目录。状态记录在`progress.jsonl`，
完成写`completion.json`，异常写`failure.json`；不覆盖已有实验。

2026-10-28 单种子 pilot 已完成：固定80个开发目标中新增漏报31个，
整体 Incremental ASR 为 **38.75%（31/80）**，条件抑制率为 **86.11%（31/36）**。
正常 Pair AUC 为0.911111，相对匹配干净模型0.916861下降0.005750，超过预设0.005门槛，
因此完整性能门槛未通过。条件抑制率不是整体 ASR；本轮未访问原 validation/test，
也未替换正向主方案。完整逐项门槛见 [comparison.json](results/reverse_risk_pilot_v1_20261028/comparison.json)，
运行状态见 [completion.json](results/reverse_risk_pilot_v1_20261028/completion.json)。
