# ALD NEW rescue：独立 gate 单变量实验协议

本协议定义 `experiments/ald_gate_new_v1` 下的独立后续实验。当前四组主实验 `runs/ald_fusion_v1` 的冻结训练源码保持原样；本实验仅比较 gate 的新类补救规则，两臂共同采用 `legacy` 像素融合。它不同时改变 ignore 策略、阈值、分类目标、KD/SEP 或训练预算。

本文不提供候选性能结论。纯 gate 模块20项 CPU测试、独立训练副本12项 CPU合同和两臂4/4阶段 CUDA smoke 已通过。CUDA同输入 gate/标签/BCE/像素logits梯度精确一致；独立全模型执行存在微小数值漂移，不能宣称权重逐位一致。验收范围与收据见 [验证记录](ald_gate_new_results.md)。正式后续实验尚未启动。

## 已有诊断提供的动机

主实验的 `runs/ald_fusion_v1/early-active-gate-diagnostics.json` 保存了 step1、`legacy` 组迭代 2050–2550 的 11 个日志区间。它们合计 550 次训练迭代、4,400 次图像抽样，位于像素损失已经启用的阶段。

| early-active 诊断 | 整数计数 / 比例 |
| --- | --- |
| 当前 NEW 正类出现次数 | 5,310 |
| gate 删除 NEW 正类 | 3,773 / 5,310 = 71.05% |
| 直接达到 raw 12.5 的 NEW 正类 | 564 / 5,310 = 10.62% |
| 原 fallback 恢复 NEW | 973 次 |
| 拒绝的新类 CAM 像素 `R` / 筛选前新类 CAM 像素 | 81,480,990 / 133,481,828 = 61.04% |

这些比例先对整数计数求和，再相除。直接通过阈值的 NEW 数由 `5310−3773−973=564` 得到；原规则的每图 fallback 最多恢复一个正类。它们提示某些已知新类未获得通过 gate 的 CAM 监督，但没有衡量 CAM 是否正确、补救后标签是否更准确或最终 mIoU 是否提高。该区间也不是完整训练轨迹。

step1 的 5,542 张原始训练图都至少包含一个当前 NEW 图像级标签。然而数据增强会随机缩放、裁剪，训练仍沿用原图标签。原图存在该类不保证当前 crop 仍含该对象；恢复最强 NEW 可能增加错误背景或旧类覆盖。这是候选的重要代价，不能因图像级标签可靠就假定像素 CAM 可靠。

## 唯一变化：在 legacy gate 上追加 NEW fallback

gate 量与主实验相同：多尺度合并主 CAM 在归一化前的全图动态范围，包含 padding；它不是概率或经过校准的置信度。阈值固定为 12.5，达到阈值即保留。图像级正类池由 teacher 分类 logits > 0 的旧类与当前新类图像标注组成。

`tools/ald_gate_policies.py` 的接口为：

```python
gate_ald_classes(
    cam_peak, cls_labels,
    old_classes=old_classes, threshold=12.5,
    policy="legacy" or "new_fallback",
)
# ALDGateResult(gate_labels, legacy_gate_labels, new_rescue_mask)
```

峰值和标签为同设备 `[B,C]` 张量，峰值有限且非负，标签为二值；`old_classes` 划分旧前景和当前 NEW。输出保留标签 dtype/device，不修改或 alias 输入，最终 gate 与 legacy gate 也不互相 alias。最大峰值相同选较低类别索引。

| runner 实验臂 | gate 行为 | 共同像素融合 |
| --- | --- | --- |
| `legacy` | 保留达到 12.5 的原正类；若所有正类都未通过，从 OLD+NEW 混合池恢复最强正类 | `ald_mode=legacy` |
| `new_fallback` | 先执行完全相同的 legacy；若原正 NEW 非空、但 legacy 中没有任何 NEW，再额外恢复其中 raw peak 最强的一个 NEW | `ald_mode=legacy` |

候选完整保留 legacy 已选的 OLD 和 NEW，不恢复原标签为负的类别，不改分类/辅助分类损失目标或辅助 PTC 标签。它可在已有 OLD 通过阈值时补救 NEW，也可在原 fallback 选择 OLD 后追加 NEW；已有 NEW 通过或被原 fallback 恢复时，不再追加。

两臂都在 PAR 后按各自 gate 筛选 CAM，再以有效当前 NEW CAM 覆盖 teacher 像素 argmax。拒绝区域由 teacher 旧类或背景填回，不采用 `preserve_rejected` 或 `preserve_background`。融合后共同将 box 外 padding 恢复 ignore=255；因此有效 box 内 `final_ignore_pixels=0`，`padding_labeled_pixels=0`。当前 NEW 为 step1 的 11–15、step2 的 16–20。

## 筛选作用的数据流边界

给定相同PAR标签和teacher，融合先复制teacher，再只用当前NEW区域覆盖。OLD CAM被gate保留为OLD或过滤为255，都不会直接改变旧类像素监督；OLD正类和raw响应仍可通过混合池的“是否全部拒绝”及fallback胜者，间接决定NEW是否留下。

gate数值在PAR前计算，像素筛选在PAR后应用；PAR依然使用原图像级标签，包括正OLD的softmax竞争。候选只放行已经由PAR分给NEW的区域，不重跑PAR，不能将此前OLD/BG赢得的区域重新分配给NEW。分类、辅助分类和PTC均继续使用原标签；CTC在该训练实现中已注释。两pixel损失共享筛选/融合后的标签。逐文件行号与指纹记录在 `.runtime/ald-gate-dataflow-audit.json`。

论文中可陈述“当已知正NEW全部未被legacy保留时，额外放行响应最强的一个NEW，使其已有PAR区域进入teacher/CAM融合监督”。不能据此声称修正OLD伪标签、改善PAR定位、保证新增CAM正确，或保证共享优化后的旧类表现不变。

## 共同预算与初始化

| 设置 | 正式两臂 | 独立 smoke 两臂 |
| --- | --- | --- |
| 每臂训练阶段 | step1、step2，共 4 个新增阶段 | 同样 4 阶段 |
| 每阶段迭代 | 8,000 | 4 |
| 增量 seed / batch / workers | 0 / 8 / 4 | 0 / 8 / 4 |
| 学习率 warmup / loss warmup | 2,000 / 2,000 | 1 / 1 |
| KD / SEP 权重 | 0 / 0 | 0 / 0 |
| 主像素 / 原型像素辅助权重 | 0.1 / 0.1 | 0.1 / 0.1 |
| raw gate 阈值 / fusion | 12.5 / legacy | 12.5 / legacy |

两臂从同一个旧正式实验 `runs/fixed_baseline_v1/shared/10-5/step0/checkpoints/model_final.pth` 初始化；该共享 step0 原 seed=0、训练 20,000 次，本轮不重训或复制大 checkpoint。每臂 step2 从自己的 step1 final 初始化，并使用自己的 step1 模型作为冻结 teacher。正式新增预算为 32,000 次 optimizer step，不包括共享 step0。

其余训练设置沿用主实验：完整训练/验证阶段列表、ImageNet 预训练 ViT、crop448、学习率2e-5、PolyWarmupAdamW、分类和 PTC 目标、关闭 confusion reweight。前 2,000 次仅分类、辅助分类和 PTC 产生有效优化目标；第 2,001 次起启用两项像素损失。gate 诊断需区分 warmup 与 active，不能把 warmup 构造标签计为已使用的像素监督。

## 隔离实现、合同与溯源

新增的最少训练副本为 `experiments/ald_gate_new_v1/train.py`、`trainer.py`、`ald_stats.py`；复用独立纯模块 `tools/ald_gate_policies.py`。实验入口新增 `--ald_gate_policy legacy|new_fallback`，独立编排工具为 `tools/run_ald_gate_study.py`。这些文件与原主实验依赖分别记录指纹；不得在原 `continual/Trainer.py`、`utils/camutils.py` 或其他冻结训练源上热修改，也不将新代码写入冻结包目录。

控制臂与主 `legacy` 实现在 device/layout 处理和附加诊断之外应保持同一训练语义。正式运行前需要 CUDA smoke 合同：在相同 checkpoint、初始化和 batch 上检查控制臂 gate、CAM 筛选、融合标签和 padding 精确一致；检查实际 active 像素损失与训练梯度按预先规定的浮点容差一致。纯模块测试包括冻结 helper 的 AST 回归，但不能证明完整 GPU 训练副本等价。smoke 缩短预算，仅验证管线和合同，不用于判断性能。

source fingerprint 继承主 manifest 的 37 个冻结源码，加隔离的三个训练文件、纯 gate 模块和新 runner，共 42 个文件；实际导入的主依赖保持对应版本。数据列表、共享/前驱 checkpoint、预训练权重也须绑定哈希。保存可校验的源码归档、配置、命令、进程退出码、终点评价与 checkpoint/计数日志 SHA。正式完整性和当前工作区源码状态分别记录，避免将后续工作区变化误认为历史实验文件损坏。

## NEW rescue 的独立统计

原 25 个整数计数保留定义；额外记录 `new_rescue_images`、`new_rescued_class_occurrences`、`direct_threshold_new_classes`。OLD/NEW gate 拒绝数依据最终 gate；原 fallback trigger 仍是所有原正类都未达到阈值，其 OLD/NEW 归属必须依据**补救前 `legacy_gate_labels`**。

若原 fallback 选择 OLD，随后候选追加 NEW，最终 gate 同时有两类。用最终 gate 重建原 fallback 归属会把同一图双计，破坏原分区；不能把这种 NEW rescue 算作原 fallback_NEW。独立统计需检查：

```text
fallback_images = fallback_old_images + fallback_new_images
new_rescue_images = new_rescued_class_occurrences
positive_new_classes = gate_rejected_new_classes
                       + direct_threshold_new_classes
                       + fallback_new_images
                       + new_rescued_class_occurrences
```

legacy 臂 rescue 两计数为 0。两臂仍检查新 CAM 筛选和 teacher 填回的像素分区、完整日志覆盖及零 padding 监督。新增比例分别为 rescue/images、rescued-class-occurrences/positive-NEW-occurrences、direct-threshold-NEW/positive-NEW；分别对 all、warmup、active 的整数求和后计算，分母为零用 null。

## CLI 与正式启动条件

以下 smoke 命令已实际完成，正式命令等待主GPU0/1各自完成两个正式阶段后执行：

```bash
cd /ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto
source scripts/activate_baseline_env.sh
python -B tools/run_ald_gate_study.py --smoke --output runs/ald_gate_new_v1_smoke --gpus 0,1 --workers 4 --seed 0
python -B tools/run_ald_gate_study.py --output runs/ald_gate_new_v1 --gpus 0,1 --workers 4 --seed 0
```

runner 两臂命名 `legacy`、`new_fallback`，共同配置 `ald_mode=legacy`；默认 GPU 为 0、1，smoke 和正式 output 分离。缓存、日志和输出仅位于项目授权范围，写前检查实际路径、符号/硬链接及嵌套挂载，不修改数据原件或共享 checkpoint。

CPU 准备可与主实验并行。CUDA smoke 需要明确的设备调度；若共享正在运行主任务的 GPU，runner 须先检查至少 20,000 MiB 实际空闲显存并记录物理设备占用与共享范围，不得中断或挤出主任务。smoke 检查通过不等于允许在同一设备启动正式后续实验。正式 runner 只能在目标 GPU 原主任务对应两阶段链的 completion/process、退出码及哈希实际通过检查，且拥有的进程已退出、设备无该任务残留后启动；不能依据估算进度或某个中途 checkpoint 提前发起，也不能中断主实验腾出设备。

## 评价与解释边界

最终以各臂 step1/step2 的 8k 终点评价比较 gate 规则，先看同阶段匹配控制，再看两阶段完整结果。报告 previous/current5：step1 前景1–10 / 11–15，step2 前景1–15 / 16–20；all 分别含背景0–15 / 0–20。原始 `old_miou/new_miou` 在 step2 仍为初始10 / 累计新增10，应与15/5分组区分。逐类 IoU 的 null 排除，并报告有限类别数和背景。

本实验单 seed，不提供误差条或显著性结论。不同阶段验证图集及有效 GT 类别范围不同，跨阶段分数下降不能直接称为遗忘。gate 保留率、NEW rescue 次数和监督覆盖量证明规则是否作用；最终 mIoU 才能检验这种作用是否有益。旧固定 BASE 和主四组结果提供背景，新 gate 的因果比较以本实验自己的 legacy 控制臂为准。

## 自动接续队列

队列已启动并实际等待GPU0/1主任务完整退出；事件见 `.runtime/ald-gate-launch-queue.jsonl`。它锚定主PID/starttime、两个manifest SHA和NEWsmoke的42源指纹，在真实阶段SHA/exit0/进程组/显存检查通过后仅启动一次正式runner；等待期间不创建正式输出。当前手工启动记录为：

```bash
python -B tools/queue_ald_gate_study.py --primary-pid 983111
```

已有队列持有私有exclusive lock时会拒绝第二队列；该命令属于本次原主PID的调度记录，不是换机器通用PID。

## 接续进程检查的实际范围

驱动返回的compute PID在容器/proc中不可见；PyTorch elastic rank另建session，不能把launcher进程组等同于rank组。已仅重启并加强排队器，训练37/42源保持冻结。它在admission及实际启动前按原训练ENTRY和off/legacy work_dir完整token扫描本项目launcher/rank/继承argv的worker，要求非僵尸匹配数为0，再核验真实exit0、双阶段SHA链和显存。当前真实检查匹配12个本实验过程，因此继续等待。它不把不可见driver PID当作退出证据，也不修改其他任务。收据：`.runtime/ald-launch-process-namespace-verification.json`。
