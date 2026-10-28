# 确认评估前置核查

状态：`CONFIRMATION_SCOPE_REVIEW_REQUIRED`。本轮仅读取已有JSON/CSV元数据；没有训练、加载图数组或启动确认评估，没有读取test结果。

## 已核实事实

- 当前 `unconditional` 三seed固定为20260621/22/23，保留权重，不增加模块、不调参。
- 当前模型使用原TRAIN内train512/dev128/audit256；已有audit目标224，其中Moving60。
- 历史 `spear_formal_evaluation_v1/evaluation_20261024_seed20260621_v1` 已对全部29,071个grouped-v5 validation图完成评估，覆盖2,420个录制组。
- 历史validation报告记录：registered_scenes=processed_scenes=29,071，targets=25,515，original_val_arrays_loaded=true。
- 当前manifest的validation有效图数量仍为29,071。因此不能把现有validation重新切片，称为全新独立盲测。
- 本轮没有检查或使用任何新的test指标。历史plan本身已有test暴露声明；不能据此再次启用test。

## SPEAR历史清单初筛

递归核对37份历史SPEAR plan，无JSON解析失败。显式原TRAIN场景/录制组并集为896/896；将场景ID保守回映射至manifest后，并集未增加。当前U三个seed的六份训练/审计访问账本全部为原TRAIN，且与固定角色一致。

原TRAIN现成有图资产为131,361场景/11,584组。排除上述896组后，初筛余120,916场景/10,688组。这只是候选空间：4份plan不内嵌场景行（training_trace、supplied clean_diagnosis、checkpoint_auc、distribution_audit），本轮未穷尽这些引用；也未查全项目旧v6使用。不得将初筛余量表述为已证明“从未使用”。

validation另有1,718条无图元数据，其中22个组没有现成图；本轮不重建图，也未证明这些原始场景未使用，故不将其计作可直接评估的独立数据。

## 后续范围需要明确

可考虑从原TRAIN中排除所有已登记SPEAR训练、开发、审计录制组，另行固定新的组外确认集。这可检查当前方法在新增场景/录制组上的泛化，但原TRAIN曾用于旧项目研究，不能声称它是整个项目从未使用过的独立数据。

在新评估前应冻结场景清单、模型及归一化资产、三seed、公共目标规则和指标。不得根据预测结果筛场景、选seed或调参。若要求整个项目首次接触的独立确认，则需另行提供确实未用过的数据。

## 证据

历史validation摘要：
`/home/mizhou/JCAS_cleanup_20261027_Mp0CdD/legacy/record/reproduction/spear_formal_evaluation_v1/evaluation_20261024_seed20260621_v1/val/summary.json`

SHA-256：`065fbc6252c50c2f63f1339b4cbc3a88249fa9e5b4015fe2132b6b42a7609bee`

grouped-v5 manifest SHA-256：`32a0ab0720ac78db455ae79e4b4002dc3b313af8c4c86fc83a3fae304fbb010b`。

## 后续更新：2026-10-27 确认评估已完成

上文保留为最初前置核查记录，不代表当前执行状态。用户随后同意在“不宣称全项目首次使用”的限定下开展新增录制组确认。

`history_reference_audit.json` 已补全四份间接计划引用，状态为 `PASS_KNOWN_SPEAR_HISTORY`，未解决引用为空；最终排除清单仍为 896 个场景/896 个录制组。核查并绑定 139 份来源文件；它不证明旧 v6 等其他项目路径未使用剩余原 TRAIN 数据。

后续固定了 1,024 个原 TRAIN 录制组、每组一个场景，并完成三个保留模型的前向评估。没有训练，没有访问原 validation/test 数组。总体 Incremental ASR 为 80.9149±2.2337%，Moving 为 78.3404±1.7567%，邻接增量误报为 0.5149±0.1242%，Clean Pair AUC 为 0.898007±0.002377。三个种子均未满足 Moving≥80%，完整门槛通过数为 0/3；其余质量检查通过，seed20260622 另未通过总体≥80%。

完整说明与原始证据见 [新增录制组确认报告](../group_confirmation_v1_20261027/READOUT.md)。原确认计划和结果未修改，没有自动启动后续任务。
