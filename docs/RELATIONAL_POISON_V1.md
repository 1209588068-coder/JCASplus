# Relational poison v1：第一阶段，轨迹与数据契约可行性

## 边界与交付状态

仅使用本地公开 AV2 和研究者自有模型开展离线授权鲁棒性研究。
本入口不加载模型、不训练、不查询模型输出，不读取 val/test 图或结果。
不修改 `record/v6/**`、`record/v6_4/**`、原图、grouped-v5、旧训练器或评估器。

**这是三模块研究方案的第一阶段，不是已经实现的完整攻击。**

已实现：

- A：观察历史上的共享低维平滑轨迹、固定随机模板、预算归一化、双端变换、
  节点/边特征重新计算、端点交换与节点重排一致性。
- B 的几何原型：保持相对轨迹变化，通过有限共同平移降低邻接距离变化；
  不是代理模型上的预测副作用优化，也不声称降低 FP。
- 数据契约：原 5,776 场景、随机 pair、4,621 正向/1,155 负控制的计划；
  recording-group 级 construct/develop/audit 分组与 SHA 绑定。
- train-only 数值可行性审计、完整/单端/关系反向对照及观察—未来衔接诊断。

未实现：代理模型上的预测邻接约束、跨场景普通 BCE 重训练优化（C）、
最终 victim 训练、攻击 ASR gate，以及地图/曲率/完整车辆运动学可行性判定。
没有训练命令或自动启动下一阶段的选项。所有输出带 `training_ready=false`；
`planned_targets.csv` **不能作为旧训练器的 poison manifest**。

## 假设与不提前宣称的结果

假设：同一相对运动信号在不改变拓扑的条件下，存在满足预算、且邻接几何影响
较小的实现。先验证数据变换与预算是否成立，再研究它是否能被普通 BCE 学习。

几何能量下降不等于预测 FP 下降；随机模板可行不等于随机攻击有效。
本阶段不会选出“最佳攻击模板”，不会产生 ASR 或 PASS 攻击结论。
旧 adaptive/codeword 的失败不能通过新命名变成成功：本参数化只作为基础设施，
后续必须单独证明预测局部性与跨 pair 学习增益。

## 固定配置

沿用 graph-v4 字节、grouped-v5 和 5/1/4 动态风险标签。
训练 pair 完全复用已完成的 all-eligible random 对照，不重新筛选：

```text
record/reproduction/v6_4_random_pair_control/seed20260621_dfxklaqf/plan/plan.json
```

只读取该计划中的 train 清单和图契约，不遍历其 validation/模型资产。
旧 v6.0 和 v6.4-S 源码归档的固定 SHA 也会验证，但不会解压或覆盖它们。

| 项目 | 固定设置 |
|---|---|
| seed | 20260912（分组、预算角色、抽样） |
| 场景数 | 5,776，不添加训练记录 |
| 计划正向投毒 | 4,621 场景 |
| 计划负控制 | 1,155 场景，三种控制各 385 |
| 时间窗口 | 最后 10 个间隔，K+1 连续有效状态 |
| 位置预算 | 每端 ≤0.1 m、相对 ≤0.2 m |
| 速度/加速度/jerk 增量预算 | 从原 0.1 m minimum-jerk 指令残差推导 |
| 共同平移 | ≤0.02 m，包含在每端 0.1 m 总预算内 |
| 新模板相对信号预算份额 | 全局 90%，为共同平移留余量；有/无控制共用同一信号 |
| 输入表示 | 重新计算 base34 + 原 DCT8，共 42 维 |
| 图拓扑 | 不变 |

4,621/1,155 是全体场景的角色计划，不表示每个 inner split 都有相同比例。
输出逐组、逐运动状态计数。将来使用 construct 子集训练代理模型时，必须按实际
子集计数，不能宣称用了完整 5,776 场景。总受控比例约 5%，真正翻转标签的比例
低于旧的 5% 全正向方案，不能冒充严格相同的投毒率。

共享相对信号 `r(t)` 使用三条固定平滑多项式基函数，每条有 LOS/切向两维系数。
第一条基函数包含原 minimum-jerk ramp；不是声称已经实现可学习样条生成器。
模板只做一次全局预算归一化，不按场景或速度缩放。新模板统一使用 90% 增量
预算，避免双端对称信号把某个预算用满后共同平移无可行空间；不是放宽总预算，
也不是依据模型 ASR 调整。三个随机模板 seed 为
20260912、20260913、20260914。另保留原 ramp 的 100% 预算模板作为参考，
它的共同平移可能必然为零，不用于声称相同实际信号强度的比较。报告实际扰动大小，
不同模板的平均能量不保证相同，因此当前结果不能直接作为公平 ASR 对比。

```text
delta_p_first  = c(t) - r(t)/2
delta_p_second = c(t) + r(t)/2
```

端点按当前几何位置规范排序，不用 track ID 或模型分数决定信号。
因此节点重排和端点交换不改变物理结果；该排序不保证全局旋转等变性，
后续应将旋转/坐标系敏感性作为检查项，不能预先宣称跨朝向不变。

共同平移拟合三基函数的 ridge 最小二乘，使用邻接距离的一阶几何近似；
固定回退步长只接受满足全部预算且使**实际**邻接距离平方变化减小的结果。
不修改 r(t)，不删除目标；没有可接受平移时 c=0 并记录。原信号若用满预算，
可能没有可用平移空间，这是本阶段需要暴露的限制，不会自动扩大预算。

## 分组与审计范围

原 train recording groups 按稳定哈希排序，以组数的 70%/15%/15% 划为
construct/develop/audit。输出所有 train 场景的角色表，不改变原 manifest。
不使用模型输出或攻击结果做分组。

本阶段只从 **construct** 中固定抽取 64 个低速和 64 个运动目标；
develop/audit 的轨迹均不打开。这个均衡诊断样本不是全目标池估计。

每个目标 × 4 模板 × 5 设置，共 2,560 条记录：

1. 无共同平移的完整模板。
2. 有共同平移的完整模板。
3. 仅规范排序后的第一个端点。
4. 仅第二个端点。
5. 交换两端残差，反转关系。

控制样本不额外归一化。反向关系是一个可复现对照，不保证被模型识别为不同
触发器；只有后续模型实验才能回答。所有样本均报告，不能删除无共同平移收益者。
控制版本的 `common_mode` 诊断对应消融前完整双端变换；各版本实际预算单独
计算，不能把父变换的几何收益当作单端/反向控制自身的收益。

## 物理与真值解释限制

位置和速度按 residual 方式更新。两端的原/新速度均 ≥0.5 m/s 时，保留原
heading 与速度方向之间的角度差；否则保留 heading，并报告 heading 变化。
这个阈值切换和原数据噪声可能导致 heading 不平滑，本阶段不作车辆动力学认证。

当前动态风险 oracle 只读取 future 数组；本变换不修改 future，所以原风险标签
按构造保持不变。**这不是观察—未来轨迹连续或标签语义完全合理的证据。**
输出第一未来点与最后观察点之间步长的变化；缺少第一未来点时记 null，
不能当作零变化。本阶段也没有道路边界/曲率/绝对加速度门槛。

## 运行

在已有 JCAS Python 环境中，无需 GPU。先进行不写文件、不加载图的预检：

```bash
cd /home/mizhou/JCAS
python3 -m jcas.workflows.relational_poison_prepare --check-only
```

准备计划，命令会打印确切 plan 路径：

```bash
python3 -m jcas.workflows.relational_poison_prepare --prepare-only
```

再使用打印的路径进行 train-only 审计，替换下面的占位路径：

```bash
python3 -m jcas.workflows.relational_poison_prepare --audit \
  --plan /absolute/path/to/stage1_directory/plan.json
```

计划和审计分别创建独立的新目录：

```text
record/reproduction/relational_poison_v1/stage1_<random>/
```

计划目录包含 `plan.json`、`planned_targets.csv`、`inner_train_roles.csv`、
`feasibility_sample.csv`。审计目录包含 `protocol.json`、
`feasibility.records.jsonl`、`summary.csv`、`summary.json`。

`--output-dir` 仅允许指定该实验根目录下尚不存在的新子目录；拒绝冻结路径、
已有目录和符号链接逃逸。源码或绑定资产改变后旧计划会拒绝运行，不能修改 SHA
绕过检查。中断后没有 summary 的目录视为未完成，重新运行创建新目录。

完整完成时：

```text
status = FEASIBILITY_COMPLETE_REVIEW_REQUIRED
asr_gate = NOT_RUN
training_ready = false
```

非法轨迹、目标错配、越界读取、SHA 错误或证据不全立即报错，不写完成标记。
完成后只复核共同平移是否有空间、实际扰动大小、heading 与衔接诊断，再决定
预测层 B 和重训练层 C 的实现细节。不会自动训练、扩大预算或访问 validation/test。

## 测试

```bash
python3 -m unittest tests.test_relational_poison -v
python3 -m unittest discover -s tests -q
```

覆盖随机性、预算、图不变性、特征重算、控制版本、未来数据隔离、规范排序、
recording-group 分离、路径/SHA 防护与合成数据端到端审计。
