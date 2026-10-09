# EvoProto 恢复实验（4090，物理 GPU 5、6）

本实验保留“混淆建模 → 原型蒸馏／分离 → 增量分割”的主线。完整模型及两组核心消融链已完成，核验后的结果与局限见下文。新增 OLC 的同预算对照另见 [OLC 说明](../restore_proto_olc_v1/README.md)。

## 共同实验协议

- VOC 10-5，固定类别顺序与 seed 0。
- 共享初始阶段使用已经训练 20,000 次的 fixed_baseline_v1 checkpoint；其 SHA-256 为 `4f298e14721630cf3c66ba4f6af04bd198ebeaf77adc6b7b07ab8403ad0893b5`。
- 初始阶段曾使用像素标签；两个增量阶段只使用当前新类图像标签和冻结的上一阶段模型。这里不是所有阶段都只用图像标签。
- 每个增量阶段训练 8,000 次，总 batch **8 = 每卡 4 × 两卡**。每组 step2 继承自己训练得到的 step1，不能借用其他组的最终模型。
- 输入随机缩放后裁剪到 448；学习率 2e-5，decoder 与分类头倍率 10；前 2,000 次只启用分类和 PTC。
- 每 2,000 次验证一次，仅在阶段结束写 `model_final.pth`。日志与混淆统计保留；smoke 权重完成检查后清理。
- 训练图片与验证标签直接读取 `/data/zhonggai/coco/PascalVOC12`。增量训练不读取像素标签；验证不使用图像级真值标签筛选预测类别。

## 原型是什么

保留原实现的 512 维可学习类别原型。旧原型从上一阶段继承，新原型按固定种子初始化，并通过像素辅助监督学习。设归一化特征为 f，归一化原型为 p，原型预测为 `z_proto(x,c) = dot(f_x, p_c) / 0.1`。

主分割头依然存在。原型辅助监督、原型预测 KD 和原型 SEP 同时作用于原型与共享 decoder 特征；另有较弱的主头 SEP 将同一混淆关系作用于最终预测。最终主头分数和原型分支分数分别报告，不能把主头直接称为原型推理。

## 混淆关系从哪里来

1. 当前主、辅助 CAM 的最大类别一致，二者归一化强度都至少 0.7，且与 PAR 精炼标签一致，才作为前景锚点。
2. 旧类锚点还要求上一阶段 teacher 的分割预测一致；新类必须属于当前图片给定的新类标签。
3. 对这些锚点，统计 student 主头把类别 a 预测成其他前景 b 的有向混淆。背景不参加图中的边。
4. 混淆像素计数和锚点数量分别以 0.99 EMA 更新，按锚点类别归一化。一条边至少需要 3 张不同训练图片支持，重复采样同一图片不增加独立支持数。
5. SEP 每类最多选择两个有证据支持的混淆对象；KD 根据类别参与混淆的程度加权。图完全停止梯度，更新不使用验证真值。

## KD 和 SEP 如何优化原型

- **KD**：在有旧类 CAM 支持、teacher 可信且非新类区域，分别保持 teacher 的旧前景主头分布和旧前景原型预测分布。采用温度 2 的条件 KL，不直接约束背景与新类的竞争。当前新类 PAR 区域完全关闭 KD；新 CAM 越强，KD 权重越低。按类别支持量的平方根平衡样本影响。
- **SEP**：只在可信锚点和图中选中的混淆对象之间施加预测间隔。当锚点类别的原型 logit 已高于对手 0.5 时停止推动。主头使用同一有向关系、较低权重。不会对所有原型无差别施加正交约束。
- **轻量原型方向约束**：保持旧原型与 teacher 的对应方向，作为稳定项。

增量 warmup 后，基础损失为两个分类损失、0.2 PTC、0.1 主头像素 BCE、0.1 原型像素 BCE。新增部分为：

`0.10 主头 KD + 0.05 原型 KD + 0.05 原型 SEP + 0.02 主头 SEP + 0.01 旧原型方向约束`。

这组系数是预先确定的初始候选，不是已经证实最优的参数。

## 三组完整链路

| 组 | 混淆选择 | 原型训练约束 | 用途 |
| --- | --- | --- | --- |
| full | 实测图、每类最多两个对手、混淆程度加权 KD | 全部启用 | 恢复版 |
| without_confusion | 均匀使用全部其他前景，KD 类别混淆权重固定 1 | 全部启用 | 检验混淆建模是否有用 |
| without_proto | 与 full 相同 | 关闭原型像素辅助、原型 KD／SEP、方向约束；保留主头 KD／SEP | 检验原型训练路径是否有用 |

旧 BASE／KD／SEP 的历史结果仅供参照。当前两卡划分和运行环境不同，不能宣称跨历史实验逐步采样及浮点运算完全一致。三组新实验共享同一代码、训练预算和运行方式。

## ALD 和推理设置

此轮先关闭 ALD。历史 V9 依赖已有最终阶段参考模型，其最终模型细化结果不能直接当成新完整增量链的证据。若后续引入 ALD，需明确合法前驱与参考模型的获取过程，并另做匹配比较。

最终 stage2 在完整 1,449 张 VOC val 上评价：

- square448：缩放为 448 × 448。
- aspect672：保持原长宽比，面积约为 672 × 672，各边取最接近的 16 倍数；与此前 71.x 记录使用同一规则。

两种推理方式分别报告。分辨率收益、混淆建模收益和原型收益不能混写。单 seed、反复查看验证集以及候选选择的局限需在最终报告中说明。

## 运行与核查

使用 `.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_v1/run.py`。runner 在准备和阶段间维持显存预留，训练时让出所需显存；子任务退出或失败后恢复预留。只有整个 goal 结束后才停止预留进程。

状态与日志位于 `runs/restore_proto_v1/control`。正式各组位于 `runs/restore_proto_v1/formal`。每组保存配置、代码 hash、前驱 checkpoint hash、逐阶段指标和混淆统计。

测试已覆盖原型／共享特征梯度、teacher 停止梯度、新类 KD 排除、独立图片支持门槛、双 CAM 门控、关键消融，以及双进程梯度与合并 batch 的一致性。smoke 只检查链路，不作为性能证据。

## 与原论文的对应关系

已核对原稿第 3-5 页：保留可学习原型、原型像素监督、teacher/student、混淆指导的分离与知识保持，以及原来的增量问题设置。当前候选不是原公式逐项复现。

需要重写原公式 (2)-(7)：可信区域的有向 EMA 混淆、最多两个对手、原型预测间隔损失和条件 KD。尤其原文对高混淆类别降低全局原型 MSE 权重；此候选则只在有证据的旧类区域加强保留，并在新类区域关闭／衰减 KD，两者的作用范围和权重方向不同，不能沿用原式或声称完全不改论文。总损失 (11) 与结构图中的 KD／SEP 输入也需同步修订。

完整核对记录保存于 `runs/restore_proto_v1/control/paper_alignment_audit.json`。ALD 暂未并入本轮三组；后续是否加入以合法链路实验为依据。

显存管理已升级为运行期间额外预留每卡 8 GiB、阶段之间预留每卡 20 GiB；实际训练仍为每卡 batch4。运行控制版本及交接日志保存于 control 目录，不改动本次已经冻结的训练代码。

## 运行环境与占用程序

关键依赖版本位于 `environment.json`。项目环境继承该机 `dl` 环境中的 PyTorch，再安装项目内依赖，不修改共享环境。完整依赖清单位于 `.runtime/restore_proto/requirements-locked.txt`。

当前运行的第二版占用程序源码亦保存在 `tools/restore_proto_reservation_v2.py`，与运行副本内容一致。它读取 `runs/restore_proto_v1/control/reservation_request.json` 中的 `mode`（`reserve` / `training` / `stop`）；交接标记 `reservation_v2_adopt` 存在后写正式 heartbeat。研究 runner 负责在训练、验证与间隔之间切换模式。已有占用／训练进程运行时不要再启动一份。

从干净目录恢复本机运行时，须先确认物理5、6号卡空闲，准备所需前驱权重和项目环境，再创建 control 目录、写入 `{"mode":"reserve"}` 请求、创建 `reservation_v2_adopt` 标记，并以这两张卡对应的 CUDA UUID 启动占用程序，记录其 PID 到 `reservation.pid`。收到新鲜 heartbeat 后再启动研究 runner。整个 goal 完成时发出 `stop` 并核实本任务进程已退出。

## 固定网格的推理融合诊断

`tools/evaluate_restore_proto_fusion.py` 不训练参数，比较原型概率权重 alpha = 0、0.25、0.5、0.75、1。主头与原型头都采用 BCE 训练，因此诊断使用 `(1-alpha) * sigmoid(main_logits) + alpha * sigmoid(prototype_cosine / 0.1)`；先将 logits 插值回原图，再变换概率。端点直接以各分支 logits 取 argmax，避免 sigmoid 饱和造成端点数值平局。

该诊断默认在 CPU 上运行，可与 GPU 训练并行；先用共享 step0 核对 GPU 主头分数。每张图片的混淆矩阵会保留为紧凑 NPZ，供之后核验或配对分析。alpha 网格在最终增量结果出现前固定。若采用融合推理，必须公开全部取值，并给需要比较的实验使用同一 alpha；任何验证集选参都不能被表述为独立测试证据。主头的 square448 与 aspect672 原协议结果仍须保留。

## 最终权重核查

阶段保存并生成 `final_receipt.json` 后，可运行：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .runtime/restore_proto/venv/bin/python -B tools/audit_restore_proto_checkpoint.py --stage-dir runs/restore_proto_v1/formal/full/10-5/step1
```

该检查只用 CPU，核对 checkpoint 与前驱的 SHA256、冻结训练源码、8,000 次更新记录、完整模型严格加载、所有参数有限值，以及各阶段 512 维原型参数的形状和范数。旧原型相对前驱的变化量也会保存到阶段目录的 `checkpoint_audit.json`。参数存在或发生变化本身不能证明性能贡献，仍须结合实际损失／梯度与完整链路消融。

清理权重时，最终采用方法的共享 step0、step1 前驱和 step2 最终权重需要一起保护。其他候选须在完成所需评估与比较后再决定是否删除；不按文件时间或名称模糊批量删除。

`tools/postprocess_restore_proto.py` 是有限的 CPU 后处理队列：等待每组正式回执，核查各阶段最终权重，等待官方双卡评估完成后，对最终模型执行同一固定融合网格。它检查两种协议的 CPU/GPU 端点差异、1449 张图像唯一性和逐图混淆矩阵汇总；结果写入各组 step2 的 `fusion_evaluation.json`、`.npz` 和 `fusion_verification.json`。该任务不会修改权重，训练继续使用原两张卡。队列状态在 control/postprocess_state.json；若正式协调进程结束而所需结果缺失，队列报错退出。

`tools/plot_restore_proto_confusion.py --stage-dir ... --output output/figures/name.png` 可从 full 组保存的训练统计生成 PNG 与 SVG。左图为受独立图片支持的 EMA 混淆率，橙框为 SEP 对手；右图仅显示当前阶段旧类实际可用于 KD 的权重增幅。新类虽然参与图统计，但被 KD 门控排除，不能把它们的 degree 当作实际蒸馏强度。图中数值是指定更新时点的快照，不是训练均值或验证集混淆矩阵。

## Standalone evaluation correction (2026-10-09)

Palette masks are uint8. The original standalone evaluator multiplied these labels by the class count before casting, which overflowed for class IDs 13–20 at the final 21-class stage. The corrected evaluator explicitly uses int64 before encoding histogram indices; the training validator already used long integers and is unaffected. Neither training losses nor model parameters were changed. `tools/test_restore_proto_evaluation.py` reproduces the defect and checks all classes plus void pixels for 11, 16 and 21 classes.

The initial full-method standalone result (~26.55) is invalid and retained only as `evaluation_invalid_uint8.json` for diagnosis. Use only revised evaluation files with `histogram_label_dtype: int64`; final recomputation is in progress. A local evaluation-only source amendment preserves the original file and before/after hashes alongside the original training manifest. The checkpoint audit accepts precisely this recorded evaluation-file change while continuing to verify every training-source hash and checkpoint lineage. A fresh clone starts from the corrected evaluator. An existing pre-fix manifest requires explicit reconciliation before restarting the coordinator; do not silently replace its training provenance.

CPU reevaluation while GPU training continues is supported with `--device cpu --threads 8`; the default remains two-rank CUDA evaluation.

### Legacy flags versus effective restoration losses

The inherited CLI flags `w_proto_kd=0`, `w_proto_sep=0` and `confusion_reweight=false` disable the old coordinate-MSE/geometry separation and old class-weight code paths. They do **not** switch off the restored `ConfusionProto` module. After iteration 2000 the full arm uses main/prototype BCE 0.1/0.1, main/prototype prediction KD 0.1/0.05, main/prototype prediction SEP 0.02/0.05, and old-prototype direction consistency 0.01. `without_proto` sets the four prototype-specific contributions to zero; `without_confusion` substitutes a uniform foreground graph and unit KD degree. `relation_metrics.jsonl` records the restored terms, and `tools/summarize_restore_proto.py` lists their effective settings alongside each stage.

### Fixed inference comparison after the full-model grid

The full-model validation grid selected equal probability fusion (`alpha=0.5`): square448 68.8482 and aspect672 70.0644, versus main-only 68.7125 and 69.8902. Main-only remains the common primary ablation metric. The fixed fusion coefficient applies to **prototype-enabled** arms; `without_proto` uses main only, since blending its untrained incremental prototype predictions would create a misleading control. Full fused versus no-prototype main is a whole-module comparison and must be labeled separately. This selection used VOC validation, not an independent held-out test set.
