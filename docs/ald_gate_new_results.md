# ALD NEW fallback 实验记录

正式两臂训练尚未启动；用户要求暂停后，自动队列已停止。候选仅在 legacy gate 未保留任何当前正 NEW 时追加其中最强的一个，共同采用 legacy 融合；KD/SEP=0。它有可能增加错误 CAM，尤其当随机 crop 已经裁掉原图中的新对象。正式收益以本实验自己的 legacy 控制臂为准。主实验legacy在4,000次低于off，6,000次已反超，step1的8,000次终点整体高0.347点、新5类高0.760点，step2仍在训练；这说明中途分数方向会变化，本候选基于 gate 机制诊断提出，不以旧 ALD 无效为前提。

## 已完成验证

- 纯 gate 的20项 CPU 测试通过，独立训练副本的12项 CPU合同检查由实现审计代理完成；核心条件是 OLD 标签不变、仅追加原本为正的 NEW、原 fallback 归属依据追加前标签。
- 实际 CUDA 原 helper 比对覆盖两次增量边界、float32/float16 和24个合成样本，legacy gate 与原 helper 精确一致：`.runtime/ald-gate-policy-cuda-verification.json`。
- 同输入 CUDA 标签筛选、融合、像素 BCE 及像素 logits 梯度，在两个增量边界精确一致。类别权重使用实际训练的全1设置：`.runtime/ald-gate-policy-cuda-loss-gradient-verification.json`。该检查约束像素路径，不宣称整网所有参数梯度一致。
- 独立两臂 CUDA smoke 四个阶段全部真实 exit0；每阶段4次训练、8张验证图。28项计数、前驱checkpoint、配置/输入/权重/指标/计数日志SHA及归档均通过总结工具校验。候选 active 时 step1/step2 分别发生19/20次 NEW rescue，控制臂为0，证明规则已实际执行。

smoke 的分数不作为性能证据；summary 的 `performance_evidence=false`。原始及总结位于 `runs/ald_gate_new_v1_smoke/`，比较图明确标注 SMOKE NOT PERFORMANCE。

## 数值等价的范围

两次独立执行的 legacy smoke 完整权重没有逐位相同。step1/step2 最大参数绝对差为 0.00021248 / 0.00042646，全参数 RMSE 为 3.37e−7 / 1.28e−6；整体 mIoU 差仅为 +0.001728 / +0.000434 个百分点，但不能用这个小样本差异估计正式误差条。详细收据：`.runtime/ald-gate-smoke-legacy-crossrun-comparison.json`。

相同输入 gate 和像素路径合同精确通过，独立全模型 CUDA 轨迹有微小差异。PyTorch2.7 文档说明 CUDA 的 `interpolate` 梯度可能不确定，训练中存在该操作；它是可能来源，本次没有定位到唯一原因。来源：[官方 interpolate 文档](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.interpolate.html)。因此正式 gate 效果必须比较本轮自己的 legacy 控制，不能把主实验 legacy 分数直接当作该候选的配对控制。

## 冻结来源和调度

所有文件仍在同一个 evoproto 项目内；`experiments/ald_gate_new_v1` 仅包含独立入口、Trainer、统计的三个最小副本，用于保持正在训练的主实验源码冻结。42份训练依赖指纹继承主37份源码并增加这五份新源：三副本、纯 gate、独立runner。smoke源码归档共126个独立文件，SHA256：`9ff97f5e732fcb6b9cc798f70eb18df48c3a57c966f265cb7e203edbe35b2742`。

GPU0、1上主实验各自的step1和step2均完成8000次、真实退出0且哈希链通过后，再接新正式两臂；主GPU2、3继续自己的训练。正式runner重新检查空闲显存至少20000MiB和主任务无残留，不提前创建正式输出，不中断其他任务。自动调度队列曾实际处于waiting；现已按用户要求终止，主训练也已停止，正式输出未创建。队列每45秒检查一次，事件保存在 `.runtime/ald-gate-launch-queue.jsonl`，源码 SHA256 为 `458e95db81279fe87353b62a59c4c277bbb6099ec10d910767ed6f3e728da08c`。只有通过全部启动条件才会发起正式runner。

后续监控与归档已经准备，正式训练仍未开始：`tools/monitor_ald_study.py`按真实runner PID及/proc/stat启动时间每45秒记录，日志仅在项目.runtime，物理GPU总量明确包含共享任务。`tools/archive_ald_source.py`只接受已存在study.json的项目内study，拒绝覆盖已有归档。其唯一独立CPU检查位于`.runtime/archive-tool-checks/new-gate-smoke-v1/`：142个普通唯一相对文件（42训练＋77列表＋23辅助），现有两种summary来源契约通过；tar SHA为`6f8ddede069934d27fe6745a562ca80ca87a75921d40f3b6f63b3134d07b1860`。原smoke归档保留，此CPU归档不是新正式训练证据。

正式study实际出现后，在已激活私有环境中运行`python -B tools/archive_ald_source.py --study runs/ald_gate_new_v1`；该工具SHA为`924f5d514bff42d00ed349b49c59ea78dda0dc3a481c755f46aa4c1d3b075982`。辅助与文档SHA仅标记读取时快照，不属于冻结训练42指纹；失败返回1并保留诊断输出，不把存在tar视为训练完成。正式归档及监控实际启动后将补充记录。

## 已确认的机制范围

独立源码审查表明，此规则放行已有PAR的NEW区域，不能重新分配此前由OLD/BG赢得的区域。OLD gate对旧像素监督没有直接筛选路径，但会通过混合池fallback影响NEW保留；共享参数更新仍可能间接改变OLD、分类和PTC。指纹及行号：`.runtime/ald-gate-dataflow-audit.json`。主实验相同active预算前缀的覆盖率已保存，不把覆盖量视为CAM准确率；见 [主实验记录](ald_fusion_results.md)。

## 待正式结果

正式预算为两臂各两个8000次阶段，共32000新增optimizer step，共享原正式step0，不重训初始化。终点需报告 previous/current5/all、最终initial10/cumulative10/all21，以及warmup/active/all的28项整数总计和比例。单seed0结果只支持本次配置的结论，不提供显著性或泛化保证。

协议见 [ald_gate_new_protocol.md](ald_gate_new_protocol.md)。完整正式实验完成后执行：

```bash
python -B tools/summarize_ald_gate_study.py --study runs/ald_gate_new_v1
```

## 接续进程检查的实际范围

驱动返回的compute PID在容器/proc中不可见；PyTorch elastic rank另建session，不能把launcher进程组等同于rank组。已仅重启并加强排队器，训练37/42源保持冻结。它在admission及实际启动前按原训练ENTRY和off/legacy work_dir完整token扫描本项目launcher/rank/继承argv的worker，要求非僵尸匹配数为0，再核验真实exit0、双阶段SHA链和显存。暂停前的真实检查匹配12个本实验过程；用户暂停后本任务训练过程已全部终止，不再接续排队。它不把不可见driver PID当作退出证据，也不修改其他任务。收据：`.runtime/ald-launch-process-namespace-verification.json`。
