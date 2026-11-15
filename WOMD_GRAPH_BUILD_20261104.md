# WOMD到JCAS车辆图与pair派生图

本轮按用户要求，将已授权下载的Waymo Open Motion Dataset v1.3.1转换为当前车辆交互风险任务的图资产。只构图，不训练、不生成触发器，不修改AV2图或历史实验。

## 全量完成结果

北京时间2026-11-04 01:49:02完成，8个CPU进程耗时约19分58秒；1000/1000分片成功处理，执行失败0。

| 统计 | 数量 |
|---|---:|
| 扫描原始场景 | 70,541 |
| 成功构图场景 | 52,495（74.42%） |
| 显式排除场景 | 18,046 |
| 车辆节点 | 1,041,380 |
| 有向车辆边 | 24,667,218 |
| pair派生节点 | 12,333,609 |
| 有向pair派生边 | 983,066,444 |
| 合格监督pair | 5,465,166 |
| 风险／非风险监督pair | 330,557／5,134,609 |
| Moving／Slow监督pair | 332,001／5,133,165 |
| NPZ文件 | 157,485（每个保留场景3个） |
| NPZ总大小 | 9,119,643,748字节，约9.12GB |

节点/pair数量是跨场景实例总和，不代表独立真实车辆或独立录制样本数。

排除原因：累计时间网格偏差10,382，单步间隔偏差6,651，合格车辆不足2辆877，轨迹长度与时间轴不齐135，原始时间轴不足110样本1。前两项是本次保持旧0.1秒特征/标签合同的兼容性限制，不表示官方文件损坏；没有重采样或换窗提高覆盖率。

全量无重复scenario ID组或相同payload组。这只排除了这些精确重复，尚未证明录制片段没有重叠。

验收：pilot全部114场景342个NPZ回读通过；全量固定抽取34个分片各首个成功图（102个NPZ）独立回读通过，全部1000个分片账本哈希及70,541条manifest记录核对一致。不是声称对全量每个文件都额外重新执行了一遍图算法。

- [最终统计](graphs/womd_motion_v1_3_1_graph_v1_20261104/summary.json)
- [完成凭据](graphs/womd_motion_v1_3_1_graph_v1_20261104/completion.json)
- [逐场景清单](graphs/womd_motion_v1_3_1_graph_v1_20261104/manifest.csv)
- [独立验收](graphs/womd_motion_v1_3_1_graph_v1_20261104/independent_acceptance.json)

状态为`GRAPH_BUILD_COMPLETE_SPLIT_REVIEW_REQUIRED`：图生成完成，后续仍需录制重叠审查、数据角色划分和WOMD TRAIN归一化。尚未训练模型，不将原始来源目录名直接当作本任务的训练划分。

## 输入与输出

- 原始输入：`/home/mizhou/datasets/waymo_open_motion/training_20s/`，1000个官方TFRecord分片，92,523,163,949字节。
- 全量输出：`/home/mizhou/JCAS/graphs/womd_motion_v1_3_1_graph_v1_20261104/`。
- 小样本写盘验收：`graphs/womd_adapter_pilot_20261104_v1/`，只用前两个分片，不根据模型结果选场景。
- 当前实现：`spear_current/womd_source.py`、`vehicle_graph_core.py`、`womd_graph_builder.py`。
- 正式状态以输出目录的`status.json`、`summary.json`、`completion.json`为准，本文不是完成凭据。

每个源场景最多生成一个窗口，目录按原分片分开：

```text
plan.json / registration.json       原始对象、源码、构图和标签配置绑定
source_snapshot/                    本轮源码与官方proto来源索引
shards/00000/
  vehicles/*.npz                    车辆图：x_node358、edge_attr34、消息mask、未来标签材料
  pairs/*.npz                       当前模型输入图：x784、y、eligible、edge_index、pair身份
  source_audit/*.npz                原时间戳与完整SDC轨迹，仅用于来源/重叠审查
  records.jsonl                     所有记录的结果，包括排除原因
  completion.json                  分片完成与来源哈希
manifest.csv                        所有源记录的图路径、哈希、规模和状态
duplicate_report.json               完全相同ID/payload检查，不等同录制重叠审计
summary.json                        总体规模、覆盖率、排除统计
```

车辆图中的未来轨迹仅供标签计算；模型直接读取pair图的`x`和`edge_index`。`source_audit`包含完整SDC轨迹，不能拼进模型输入或用于归一化。

## 与现有方法保持一致的部分

1. 车辆观察窗口50帧、未来标签窗口60帧，名义采样0.1秒。
2. 车辆观察有效数至少40，最后有效观察索引至少48；沿用原速度、加速度、跳变过滤及仅在观察段内的缺失填充。
3. 末观察位置距离不超过60米时保存双向车辆边；沿用分层消息/运动候选/严格监督mask。
4. 358维车辆特征和34维边特征公式直接取自原graph-v4构建器，不重新设计特征。
5. pair派生图直接调用当前`spear_current.data.line_graph_from_graph`：按track-ID字符串固定端点顺序，拼接A358、B358、AB34、BA34为784维；共享车辆且两pair消息可用时相连。
6. 风险标签复用`spear_current.risk_labels`，参数为5米基础距离、1秒反应时间、4m/s²安全减速度，未来距离的5点局部回归估计接近速度，至少3个连续风险帧。
7. 未来只影响标签及监督可计算性，不进入观察特征与消息图。不可监督pair仍作为上下文，`y=-1`、`eligible=False`。

`edge_label`/`edge_label_dynamic_risk`才是当前任务标签。`edge_label_proximity`等10米字段只为追溯旧构图诊断保留，不能替代动态风险真值。

## WOMD需要显式适配的差异

| 差异 | 本次处理 |
|---|---|
| 20秒原始记录未指定本任务的观察截止 | 固定取源索引0—109；0—49观察、50—109未来。每场景一个窗口，不按标签或效果择窗 |
| 少量时间间隔与0.1秒不一致 | 要求每步误差不超过1ms、相对理想窗口网格偏差不超过5ms；不符合的场景记账排除。不重采样，不寻找替代窗口 |
| 轨迹states长度可能与timestamps不同 | 整场景记账排除，不静默截断或推测时间对应关系 |
| 无AV2的focal/category字段 | focal维占位0且另存known=False；category=-1。SDC独立记录，不冒充AV2 focal，不用tracks_to_predict/objects_of_interest选车 |
| 类型体系不同 | 使用WOMD的TYPE_VEHICLE；不凭空细分bus或补入行人/骑行者 |
| lane缺少AV2 is_intersection | 保留未知；不把interpolating当路口，也不把未知当明确非路口而硬删边 |
| lane左右邻接按区段定义 | 聚合为现有规则所需的lane级邻接集合，并记录这一粗化；entry/exit映射前后继 |
| 缺乏已核验的全局录制组ID | 构图暂不分train/val/test，split=unassigned。源目录training_20s只是来源标签 |

因为地图字段不完全相同，不能声称WOMD与AV2消息图在所有场景下具有完全等价的物理mask。缺失信息采用保守的未知处理，差异记录在图和计划元数据中。

本次不拟合归一化参数，不套用AV2的均值/尺度；后续划分确定后只用WOMD自己的TRAIN拟合。图数据完成不表示训练角色已经具备独立性。

## 验收与执行

10项合成合同测试通过，覆盖未来隔离、pair映射、未知地图/focal、观察缺失、时间异常、轨迹长度及CRC损坏。独立检查确认12个核心特征/规则函数与旧构建器AST一致；合成场景的共有图数组逐位一致。

前两个分片实际包含149条记录：117条通过时间/长度解析，其中3条经过原车辆质量过滤后不足2辆车，最终写出114组车辆图和pair图。

- 2,150个车辆节点，39,142条有向车辆边。
- 19,571个pair节点，1,054,340条pair有向边。
- 7,643个合格监督pair，其中583个风险、7,060个非风险。
- 35个场景排除：20个累计时间偏差、11个单步时间偏差、1个轨迹长度歧义、3个合格车辆不足。
- 两分片图资产约17.3MB，按同分布粗估1000分片约8.7GB；正式规模以实际产出为准。

全量按分片多进程CPU执行，逐场景写出，不拼成单个巨图，不限制车辆/pair数量以压低开销。每个分片再次核对官方大小、MD5及每条TFRecord的双CRC；文件SHA和产物SHA写入账本。出现未知实现错误停止，不能将其当正常数据排除。磁盘剩余不足20GiB停止。

运行使用系统Python的NumPy，加上既有隔离审计环境的protobuf/CRC依赖，不修改系统Python或训练环境。官方proto固定commit为`99a4cb3ff07e2fe06c2ce73da001f850f628e45a`，计划绑定实际源文件和生成文件哈希。

```bash
cd /home/mizhou/JCAS
export PYTHONPATH=/home/mizhou/tools/womd_audit_venv/lib/python3.12/site-packages
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B -m spear_current.womd_graph_builder --prepare \
  --output-dir graphs/womd_motion_v1_3_1_graph_v1_20261104 --workers 8
python3 -B -m spear_current.womd_graph_builder \
  --run graphs/womd_motion_v1_3_1_graph_v1_20261104/plan.json
```

这是复现入口说明；若目录已经准备或运行，不能重复prepare或覆盖。实际启动由本轮助手执行，用户无需另开相同任务。
