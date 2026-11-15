# 来源与声明

当前模型中的 GraphSAGE、GCN、GraphTrojanNet 定义提取自用户提供的 SPEAR 源码，
经本项目已有兼容版本迁入 `spear_current/models.py`。保留原SPEAR作者归属；本次整理不新增许可证授权。
数据桥、残差生成器、支持限制、域投影与分布正则为本项目的本地适配代码；不能据此声称完整逐点复现原论文。

原始输入：`/home/mizhou/SPEAR-main.zip`。

ZIP SHA-256：

```text
be18c0145089c0f063a6f863bf76b7de5d7654f65e1aa7427a54afe2505e0757
```

来源性质为用户提供的归档，未声明其对应已验证的作者Git commit。
原逐文件哈希记录随历史目录保留在
`/home/mizhou/JCAS_cleanup_20261027_Mp0CdD/legacy/third_party/SPEAR_supplied/SOURCE_BINDING.json`。

在提供的SPEAR文件集合中未见独立许可证文件；本项目不将其标记或声明为MIT授权。
PyTorch、PyTorch Geometric及其他已安装依赖仍适用各自的来源声明和许可证。

## 2026-10-29 外部对比适配

`comparison_methods/` 是本地任务适配，不声称逐点复现论文数值。SPEAR模块核对了上述用户
归档（ZIP comment 的 commit 标识为 `187d2cdc1f2454795dc8b37467ee2103e2613b15`），
新增来源记录不将归档标识自动等同于完整远端仓库一致性证明。
UGBA及GTA节点／数据投毒适配核对 `ventr1c/UGBA` 的
`c1d945a215dd16fd806f8a231381d303a0b811f7`；CGBA和GraphLIME按论文机制重实现，
不是CGBA作者代码。所有来源链接、已知偏离和适配预算见
[对比合同](EXTERNAL_COMPARISON_PLAN_20261029.md)及各模块的来源记录。
不因独立适配声明而改变上游作者归属或新增未取得的许可证授权。

## 后续更正：直接作者原码

用户随后否定自写对比算法，旧队列在训练前取消。`third_party/author_baselines/` 保留
SPEAR作者归档、UGBA作者git对象、直接克隆的DPGBA和GTA作者仓库；来源固定版本、
取得方式和未见标准许可证的限制见[作者原码对比合同](AUTHOR_SOURCE_COMPARISON_20261029.md)。
`author_comparisons/` 只桥接本地数据、作者调用、兼容运行及评价；逐文件diff/hash保存在各case中。
这不授予任何超出上游授权的再发布权。本轮没有对GitHub推送任何源码。
