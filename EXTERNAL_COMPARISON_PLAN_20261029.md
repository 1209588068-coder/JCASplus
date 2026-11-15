# 外部方法对比：AV2 派生 pair-node 任务

> 当前唯一新重跑合同：[AUTHOR_RERUN_RAW_20261029.md](AUTHOR_RERUN_RAW_20261029.md)。下方均为历史。

> **已被用户后续指令取代，旧队列已在训练前取消。** 用户要求使用作者原码，不接受本地算法重实现。
> 当前方案见 [作者原码对比合同](AUTHOR_SOURCE_COMPARISON_20261029.md)。本文件只保留历史登记，不能据此重新启动旧队列。
> 本轮取得DPGBA源码后确认其使用MLPAE；下文早先DOMINANT相关描述是当时未核验的假设，不是已核验源码事实。

## 范围与执行顺序

本轮是本地 Argoverse 2 公开数据和自训 GNN 上的离线、授权模型鲁棒性研究，不涉及真实车辆控制、网络入侵或第三方系统。

用户授权先完成对比代码，待当前 clean-KL 消融结束后自动运行。新代码独立放在 `comparison_methods/`；不修改正在运行的 `spear_current/*.py`，不覆盖保留结果。

固定队列是 **SPEAR、UGBA、CGBA、GTA × 20260621/22/23，共 12 个 case**，按种子优先、方法固定顺序串行执行。每个 case 包含方法自身的训练，以及新的 matched clean reference 和 victim。它不是只有 12 次神经网络拟合。

前置任务是 `results/clean_kl_stage_ablation_v1_20261029/` 的九臂消融。只有其完成记录、九臂清单、汇总哈希及进程退出全部核验成功，才启动对比。性能门槛未通过仍可正常完成；完整性或数值错误则停止，不自动重试、改参数或跳过失败方法。

## 方法身份与来源

| 名称 | 本轮保留机制 | 必须说明的适配 |
|---|---|---|
| SPEAR | 熵驱动选点、全局 SAGE 特征选择、特征生成与代理优化 | 从未投毒起点独立训练；特征预算由原适配 30 调为 24；固定 TRAIN-only SAGE 估计预算，不继承历史 26 个附件或历史支持集 |
| CGBA | clean-label 选点、GraphLIME 平均特征重要性、共享特征触发生成 | 基于论文的本地连续特征重实现，不是作者代码；只改已有正常监督正类的特征，不翻转标签；论文未明确的超参数单独登记 |
| UGBA | 聚类／度数选点、生成子图、同质性正则、内外层优化 | 在 AV2 派生 pair-node 图上新增每目标 3 个抽象节点；原节点特征和原边不改；最终生成器取固定末轮，属于明确的检查点适配 |
| GTA | 节点任务中的生成式子图注入、联合生成器／代理训练 | 采用 UGBA 工程中的 GTA 节点分类／数据投毒适配思路，**不是**原 GTA 论文的预训练模型木马权限或完整复现 |

来源：

- [SPEAR 论文](https://ponderly.github.io/pub/SPEAR_WWW2025.pdf)、[源码](https://github.com/yhDing/SPEAR)。本地 ZIP SHA-256 为 `be18c0145089c0f063a6f863bf76b7de5d7654f65e1aa7427a54afe2505e0757`；归档 commit 标识 `187d2cdc1f2454795dc8b37467ee2103e2613b15`，逐文件来源记录在实现中。
- [CGBA，AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/35466/37621)、[GraphLIME](https://yichang-cs.com/jlu/TKDE23_GraphLIME.pdf)。CGBA 未核验到完整作者实现，本轮严格标记为 paper-guided reimplementation。
- [UGBA 论文](https://arxiv.org/abs/2303.01263)、[UGBA 源码](https://github.com/ventr1c/UGBA)，核验 commit `c1d945a215dd16fd806f8a231381d303a0b811f7`。本地 UGBA/GTA 适配记录相关源码路径、机制差异和训练预算。
- [GTA 论文](https://www.usenix.org/conference/usenixsecurity21/presentation/xi)。本轮结果须标成 GTA node/data-poisoning adaptation，不以原论文完整权限作结论。

**DPGBA 未实现、未排队。** [论文](https://arxiv.org/abs/2405.10757)及[作者仓库](https://github.com/zzwjames/DPGBA)已定位，但作者核心代码获取未完成，DOMINANT 参考筛选、判别器图构造及目标嵌入损失的实现细节尚未核验。本轮不会用简化替代版充当完整 DPGBA，代码显式拒绝启动它。

## 共同数据与评估合同

- 使用保留的原 TRAIN 内 train512/dev128、相同 TRAIN-only 784 维标准化；不是旧 v6.0 的车辆节点图实验。
- 每个派生节点对应一个车辆 pair，输入为 `[A358, B358, AB34, BA34]`。UGBA/GTA 新增的是抽象 pair-node，不声称新增真实车辆或可由轨迹实现。
- 固定公共 DEV 112 个非风险目标，其中 Moving 33、Slow 79；不以干净模型预测筛除目标。三种子重复不增加独立目标数。
- 仅加载原 TRAIN 内 train/dev；原 validation/test、内部 audit 与确认池数组均不加载。
- Victim 为 fresh GraphSAGE，hidden=80、dropout=0.5、lr=0.01、weight_decay=0.0005，200 epoch；matched clean reference 同样 200 epoch。
- 同一方法、同一种子下 clean/victim 使用相同初始化、监督索引及初始 RNG，clean 使用原始特征和原始标签，victim 使用该方法训练输入。
- 二者都按正常 DEV AUC 严格改善选最早最优检查点，不按 ASR 选轮。新 clean reference 不参与触发优化，但复用开发数据，**不称独立盲测**。
- 每个方法投毒 26 个节点。SPEAR/UGBA/GTA 从合格负类 reservoir 自行选点并改为目标标签 1；CGBA 从已有监督正类自行选点，标签不变。
- 特征方法选择 24 个坐标；结构方法每目标新增 3 个节点。二者不是等维数、等修改代价或等权限比较。
- 方法特定 selector/pretraining/GraphLIME/SAGE 成本另行记录；不声称所有方法总计算量相同。

特别注意：CGBA 仍使用原 1876 个监督节点；dirty-label 方法通常将 26 个原 reservoir 节点加入监督，共 1902 个。clean/victim 在各方法内部匹配，但各方法之间监督量并非完全相同。

当前保留主方案经过多阶段训练，不能把它的历史结果直接加入此表称为“同总预算从头比较”。本轮先建立外部适配证据；主方案完整预算匹配与最终确认另行设计，不由队列自动启动。

## 模型及输入验收

特征方法不得改变拓扑或目标外行；结构方法不得改变原始节点特征、原始边或把不同场景的触发分量相连。GraphSAGE 忽略边权，因此无效生成边必须真正从 `edge_index` 中删除，不能只将边权置零。

所有合成节点仅参加消息计算，不进入监督或评估分母。模型可以为全图输出 logits，但统计只读取原始节点前缀。

每个 case 保存生成器、实际投毒训练图、选点和支持记录、clean/victim 参数、检查点轨迹、代码／数据／模型哈希及运行耗时。候选不足、越权修改、非有限数值或结果复算不一致均停止队列。

## 输出与汇总

每个 case 的 `development/` 保存：

- `targets.csv`：逐目标 clean／trigger 预测及身份。
- `clean_pair_scores.csv`：全部合格正常 pair 的预测。
- `triggered_pair_scores.csv`：目标、邻接与非邻接原 pair 的触发响应。
- `modification_cost.csv`：实际改动坐标、L1/L2/L∞、新增节点／边、六组副本检查。
- `summary.json`：整体／Slow／Moving Incremental ASR、Absolute ASR、Conditional flip、clean activation、AUC、AP、固定阈值召回、邻接／非邻接增量 FP 和真实正类压制。

副本检查仅表示六组已核验的原目标 pair 内重复字段，不代表全部物理约束。对不改原特征的子图方法，该项记录为不适用，不把“原特征未改”包装成生成节点一致性合格。

原始 CSV 保存后重新读取并复算指标。每个 case 完成后写 `completion.json`；12 个 case 全部完成后才生成总 `summary.json` 和 `EXTERNAL_COMPARISON_COMPLETE_REVIEW_REQUIRED` 完成记录。弱 ASR 照常报告，不换种子或事后改配置。

## 入口及状态

代码：`comparison_methods/{spear,cgba,subgraph,evidence,runner}.py`。

```bash
python3 -B -m comparison_methods.runner --check-only
python3 -B -m comparison_methods.runner --prepare \
  --output-dir results/external_method_comparison_v1_20261029 \
  --after results/clean_kl_stage_ablation_v1_20261029
python3 -B -m comparison_methods.runner --enqueue \
  --plan results/external_method_comparison_v1_20261029/plan.json
```

上述为首次准备／排队入口，不要对已有目录重复运行。实际队列状态以 `launch.json`、`wait.jsonl` 为准；开始训练后才有 `execution_claim.json`、`progress.jsonl`。等待期间创建该结果目录下的 `CANCEL` 文件可阻止启动；这不是运行中杀进程接口。

等待上限 72 小时，每 30 秒检查一次；使用 PID、进程启动 tick 与 boot ID 识别前置进程。计划绑定源码、环境、数据清单和前置任务身份，运行期间更改已绑定代码会停止队列，不自动忽略变更。

## 验收记录

2026-10-29：新增对比合成测试 44 项，连同现有测试共 **539 项 unittest 通过**。覆盖四方法接口、来源关键运算对照、clean-label、消息边裁剪、原节点监督／分母、数值梯度、CSV 复算和队列失败／取消保护；包括前置进程尚未写完完成文件的等待竞态。测试通过不是实际 AV2 效果或忠实原论文数值复现的证据。

当前 clean-KL 计划的源码和资产合同已重新核验，保持不变。本文件不预填任何新对比 ASR，也不作为后台完成标记。
