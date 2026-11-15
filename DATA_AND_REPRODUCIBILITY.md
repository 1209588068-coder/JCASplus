# 数据与复现说明

本仓库用于本地、授权的 Argoverse 2 / 自有 GNN 模型鲁棒性研究，不涉及真实车辆控制或第三方系统。

## 本次发布包含什么

| 路径 | 内容 |
| --- | --- |
| `spear_current/` | 当前 SPEAR 派生方法、数据桥、训练和评估代码 |
| `tests_current/` | 合成输入及合同测试 |
| `artifacts/current/manifest.json` | 保留资产的相对路径、SHA-256、配置和内部场景角色 |
| `artifacts/current/assets/` | 三种子保留权重、固定预训练起点、标准化参数、支持集和训练索引 |
| `results/` | 已有实验的原始 CSV、摘要、日志、权重、诊断数组和源码快照 |

结果中保留成功和未通过门槛的实验，不因本次发布重新筛选结果。
源码快照中的旧目录名仅用于追溯，不是当前代码的导入依赖。
这不是 SPEAR 原论文的完整逐点复现，也不是三个从零开始、完全独立的训练流水线；
具体预训练起点、数据角色与评价限制以 README 和各实验计划为准。

## 不包含的完整数据

`data/` 和 `graphs/` 均不上传，也不会因发布而在本地删除。
不上传的内容包括原始 AV2 场景、冻结 graph-v4 NPZ，以及完整 grouped-v5 manifest。
`artifacts/current/manifest.json` 是较小的实验资产清单，不等同于完整数据集 manifest。

原始数据来自 [Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html)。
AV2 数据适用 [CC BY-NC-SA 4.0 及 Argoverse 使用条款](https://argoverse.github.io/user-guide/terms_and_conditions.html)，
数据归属为 © 2021 Argo AI, LLC。数据条款不构成本仓库全部源码的统一许可证；源码来源说明见 `NOTICE.md`。

**仅从 AV2 官网下载数据，并不能自动得到本项目字节一致的冻结图。** 当前代码读取既有图资产，
本次精简发布不包含原始 AV2 到 graph-v4/grouped-v5 的完整历史构建流水线。
从头重建图及划分需要另行提供相应历史构建合同、工具和验证，不应宣称本仓库已包含该能力。

## 资产核验与运行前提

安装依赖后，可以在仓库根目录运行：

```bash
python3 -B -m spear_current.train --check-only
python3 -B -m spear_current.evaluate
CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s tests_current -v
```

前两个入口默认只核验保留资产，不读取图数组或训练。单元测试使用合成输入；
上述检查通过不代表完整数据集已安装，也不代表已重新取得论文结果。

真实训练或评估仍需字节一致的冻结图。保留合同中的图目录为：

```text
/home/mizhou/JCAS/graphs/av2mf_graph_v4
```

完整 `manifest_grouped_v5.csv` 的冻结 SHA-256 为：

```text
32a0ab0720ac78db455ae79e4b4002dc3b313af8c4c86fc83a3fae304fbb010b
```

各已登记场景的图哈希也保留在实验资产清单中。不得以不同图、不同划分替代后仍声称复现同一合同。
原 train 中的内部 train/dev/audit 角色不等于原数据集的 train/validation/test；audit 已有开发使用历史。

## 路径与来源记录

冻结清单和结果包含原运行机器的绝对路径。模型资产通过清单内的相对路径恢复，
但图目录、部分审计计划及跨文件哈希仍绑定原环境。这些路径不是通用下载地址。
迁移到另一台机器时，需要恢复对应资产并显式登记路径迁移；不能静默修改冻结计划后继续声称哈希未变。
本次上传没有修改任何已有模型、清单或实验结果来伪装为可直接跨机器重放。
`.gitattributes` 禁止自动换行转换，以保留冻结源码、CSV 和资产的原始字节及哈希。

历史大工程归档位于原机器的 `/home/mizhou/JCAS_cleanup_20261027_Mp0CdD/legacy/`，
该外部归档不包含在本次仓库中。旧文件树仍可从 GitHub 的前序提交查看；本次替换不删除远端提交历史。
