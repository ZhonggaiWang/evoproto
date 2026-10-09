# Seed-supported teacher correction (experimental, not yet adopted)

## Problem

A previous-stage teacher has no novel-class output. Similar novel objects can therefore be assigned to old foreground classes, rather than background. Incomplete novel CAMs leave these regions under erroneous hard old-class supervision. The goal is to identify and correct this old-class absorption of novel regions, while preserving real old objects.

## Method

The existing learnable 512-D prototypes, student-confusion graph, SEP, KD and old-prototype direction loss remain active. OLC and legacy ALD are off.

1. Select foreground anchors where current main and auxiliary CAM agree at >=.7, permitted image tags and PAR agree. Old anchors additionally require teacher agreement. Select background references where teacher/PAR agree on background and both CAM maxima are <.25.
2. On reliable novel anchors, measure a directed teacher-confusion relation: row = novel seed class; column = previous-stage teacher's prediction. EMA decay .99, synchronized across GPUs, at least 3 distinct images per edge; retain top2 supported old-foreground columns. Background contributes to row mass but is never a correction target. This is distinct from the unchanged student-confusion graph used by SEP/KD.
3. For each image, pool normalized frozen teacher encoder features at each reference class's anchors (minimum 2 feature-grid pixels). These temporary 768-D seed prototypes are not learned parameters and do not replace the original learnable prototypes.
4. A novel region can expand only through spatially connected feature-grid pixels which are at least as similar as the 10th percentile of its seed similarities, more similar to that novel prototype than other available reference prototypes, and supported by both CAMs at >=.25. At least one competing reference is required. No new trainable network or retained image/feature bank is introduced.
5. Correct only pixels currently assigned to old foreground, in these expanded regions, whose teacher old class matches the directed confusion relation. Never change reliable old anchors on the feature grid, current novel labels, background labels, padding or existing ignore pixels. Restore image-resolution pair gating after interpolation.
6. Replace their old hard labels with the supported novel class for both main/prototype segmentation BCE. Remove contradictory PTC/SEP/KD constraints on the same pixels. Original student-confusion estimation and every other loss remain unchanged. Activate after the original 2000-update warmup; update teacher-confusion evidence throughout training.

The graph uses the current synchronized batch's detached seed evidence. It never uses validation pixels or pixel GT. New image tags are the only current-stage training annotations; old tags remain teacher predictions.

## Predeclared comparisons

- baseline: existing completed full restored EvoProto, no OLC.
- correct: seed-supported correction gated by teacher confusion.
- ignore: identical evidence/gating but removes those old labels; preserve original BCE denominator to avoid rescaling other pixels.
- no_graph: identical prototype/seed/spatial evidence but no teacher-confusion pair gate.

All new arms start from the same fixed Step0, have their own Step1->Step2 chain, each stage8000 updates, seed0, crop448, global batch8=4/GPU, physical4090 GPUs5/6. No continuation, best-iteration selection or extra training. Final weights only; stage1 retained while needed as teacher.

Evaluate main and fixed .5 main/prototype fusion under square448 and aspect-preserving area672. Test-time segmentation uses neither image tags nor this correction module.

## Diagnostics

CPU and two-rank tests cover direction, unique-image support, graph synchronization, connected expansion, preservation of reliable old anchors, no-seed/no-reference cases, padding and input immutability.

Initial offline diagnosis calibrates the teacher graph on 200 deterministic training images through the image-only dataset, then measures corrections on 200 disjoint validation images. On final trained arms, the saved online training graph is used instead. Pixel masks enter metric calculation only after proposals are fixed. Record per-image results to expose concentration in a few objects.

At2000/4000/6000/8000 updates, 80 fixed validation images diagnose current seed precision and old-on-new vs new-on-old pseudo-label errors without updating the graph or training. These are prospective correction diagnostics; main segmentation validation remains tag-free. No intermediate model weights are saved.

The initial baseline audit found Step1 correction precision97.52%, with93576 novel pixels rescued from old labels and956 true-old pixels relabeled novel; ungated correction precision96.03%, with96150 rescues and1824 old-to-new errors. This is a limited offline diagnosis, not a final segmentation improvement. Step2 corrections were sparse (2353 valid pixels in200 images). These hypotheses still require the complete training comparison.

## Reproduction

`CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 .runtime/restore_proto/venv/bin/python -B -m torch.distributed.run --master_addr=127.0.0.1 --master_port=49376 --nproc_per_node=2 experiments/restore_proto_seed_v1/test_mechanism.py --ddp`

Start the authorized GPU5/6 reservation helper, then:

`.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_seed_v1/run.py --smoke --arms baseline off correct ignore no_graph`

`.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_seed_v1/run.py --arms correct ignore no_graph`

The reservation remains held between jobs and on failures until the active goal ends. Release explicitly at goal completion. Full source hashes, checkpoint lineage, commands, metrics, graph state and cleanup receipts are saved under runs/restore_proto_seed_v1.

## Limits

One seed; adaptively inspected VOC validation. Seed quality may be worse early in training than in the initial final-checkpoint audit. Teacher features may fail to separate visually similar classes. Connected expansion and relative prototypes limit but do not eliminate mistakes. The method's novelty and final segmentation benefit are not established by this pilot.

## 中文说明：混淆关系如何指导伪标签校正

这是一个训练中的伪标签校正实验，目前尚未证明最终分割收益。原始可学习 prototype、SEP、KD 都保留。

旧教师没有新类输出，因此“旧类置信度很高”也不能排除这个像素实际上属于新类。当前方法先从主、辅助 CAM 与 PAR 一致的位置取得可靠新类种子，再问两个问题：这个新类经常被教师认成哪个旧类？图像里哪些相邻像素与这个新类种子的特征相似？两类证据同时成立，才考虑撤掉错误旧标签并改成新类。

- **类别层面的信号**：以新类种子类别为行、旧教师预测为列，在线累计有方向的混淆关系。例如“狗种子经常被教师认成猫”。这是种子估计，不是使用像素真值得到的混淆矩阵。
- **像素层面的证据**：在当前图像内，把同类种子的冻结教师编码特征平均成一个临时原型。候选像素需要与种子空间连通、具有双 CAM 支持、足够接近新类原型，并且比其他参考原型更接近它。没有可靠种子或竞争参考时，不执行校正。
- **最终动作**：只处理当前被分配给旧前景类、且命中上述有方向混淆关系的像素。将其主分支与 prototype 分支的旧类硬标签改为新类，同时撤掉该处矛盾的 PTC、SEP、KD 约束。背景、已有新类标签、忽略区和填充区不参与这种替换。

这里有两套用途不同的关系：原有学生混淆关系继续控制 SEP/KD；新增教师混淆关系控制本次伪标签校正。临时种子原型使用 768 维冻结编码特征，原有 512 维可学习原型分支仍照常训练，两者不能混称。

前 2000 次更新只积累新关系，随后在同一段 8000 次训练内启用校正，不增加继续训练阶段。推理仍使用普通分割输出，不需要图像类别标签，也不执行本校正流程。

三组新实验分别回答：直接改成新类是否有效；收益是否仅来自忽略错误旧标签；混淆关系筛选是否比只依赖种子和特征更有用。三组均从同一个 Step0 开始，各自完整训练 Step1 和 Step2，并采用相同训练预算。

必须区分三类数字：种子准确率；拟校正像素的准确率与覆盖率；最终分割 mIoU。前两项改善不保证第三项改善。训练期间的 80 图诊断是当前模型对固定验证图像的拟校正分析，不是整个训练过程的累计像素准确率。每 50 次记录的训练像素数量也只是该次采样 batch 的统计。

最终分割错误分析以 Step2 的类别划分为主：旧前景为类别 1–15，新类为 16–20，背景不计入旧前景。另保留“初始 10 类 / 后续引入 10 类”的描述性统计，避免把 Step1 引入的类在 Step2 中误称为当前新类。该分析来自无图像标签推理的混淆矩阵，分母分别是真实新类或真实旧类像素数，与伪标签校正准确率含义不同。

## 当前阶段结果（完整实验尚未结束）

`correct` 组 Step1 已完成 8000 次更新。方形 448 输入下，原基线 / 加入校正的全部类别 mIoU 为 74.6066 / 74.2752，旧前景为 76.6679 / 76.6363，新类为 66.7365 / 65.7514。4000 次时出现过 +1.3376 的总体优势，但 6000 次回落至 -0.2538，末次为 -0.3313；目前没有 Step1 最终增益证据。

原型方向更新检查通过，Step2 已使用本组 Step1 末次权重启动。末次固定 200 图上的拟校正精度为 93.9465%，纠正 28090 个新类被标成旧类的像素，同时将 984 个真实旧类像素误改为新类。该诊断与中途固定 80 图诊断的样本范围不同。

同一组 200 图上，在本次替换操作之前生成的伪标签，其新类被标成旧类数量由基线 393266 降至 332318，旧类被标成新类由 144159 降至 112674。这些是像素加权的伪标签指标，不能替代类别平均的分割 mIoU。完整 Step2 和 `ignore`、`no_graph` 训练对照尚未完成，因此本方法仍为实验状态。原始数据和限制见 [interim_step1.json](interim_step1.json)。
