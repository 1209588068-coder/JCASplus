# JCASplus

当前车辆交互图神经网络后门研究代码与正式运行摘要。

本发布包含：

- `spear_current/`：当前方法、AV2/WOMD 数据接口、生成器、训练和评价代码；
- `author_comparisons/`、`comparison_methods/`：对比方法适配与运行接口；
- `scripts/`、`tests_current/`：运行脚本和回归测试；
- `results/published/formal_runs_20261115/`：AV2 五种子正式结果和 WOMD 当前运行快照。

完整数据集、图张量、模型检查点、原始分数表和运行缓存不随仓库发布。请先阅读 `DATA_AND_REPRODUCIBILITY.md`、`NOTICE.md` 以及对应实验协议。

WOMD 快照在采集时仍有后续种子运行中；其中记录的状态文件反映采集时刻，不代表后续运行结果。
