# 固定权重 prototype baseline：方法与评价协议

本文记录 `runs/fixed_baseline_v1/study.json` 对应的已实现协议，供实验复现与结果解释使用。它不包含尚未完成的正式结果，也不以 smoke 测试推断性能。正式比较需等待共享初始阶段及四组的两个增量阶段全部完成，即 9 个阶段均具有预定最终迭代的评价记录和 `model_final.pth`。

## 任务与共同设置

采用 VOC 10-5：step0 学习背景及前景类别 1–10，step1 加入 11–15，step2 加入 16–20。类别次序固定为当前代码中的 VOC 次序，seed 为 0。step0 使用像素标注；增量阶段读取图片与新类图像级标签，不读取训练图片的像素真值。增量四组都使用冻结的上一阶段 teacher 提供旧类信息。

| 设置 | 固定值 |
| --- | --- |
| Backbone | ImageNet 预训练 `vit_base_patch16_224`，aux layer = -3 |
| 输入 crop / local crop | 448 / 96 |
| 训练随机缩放 / CAM 多尺度 | [0.5, 2.0] / [1.0, 0.5, 1.5] |
| 全局 batch size | 8 |
| 初始 / 每个增量阶段训练长度 | 20,000 / 8,000 iterations |
| 初始 / 增量初始学习率 | 6e-5 / 2e-5 |
| 优化器 | PolyWarmupAdamW，weight decay = 0.01，betas = (0.9, 0.999)，power = 0.9 |
| 学习率 warmup | 每阶段 2,000 iterations |
| ALD / confusion reweight | 均关闭 |
| 验证间隔 | 2,000 iterations |

具体调用参数保存在每阶段的 `config.json`；study manifest 保存了训练核心文件的 SHA-256。共享 step0 使用四张 GPU、每卡 batch 2；四组增量训练各使用一张 GPU、batch 8。各增量组采用相同训练设置和 seed，但此协议没有通过记录 batch 序列证明跨进程的每次采样、增强完全相同。

## 原型、主分割头与共同像素辅助损失

decoder 产生 512 维像素特征 $f_x$，可学习类别原型为 $p_c$，背景索引为 0。原型与主分割头 `conv8` 的分类权重是独立参数。对特征和原型分别做 L2 归一化：

\[
\hat f_x = \operatorname{normalize}(f_x),\qquad
\hat p_c = \operatorname{normalize}(p_c),\qquad
z^{\mathrm{proto}}_{x,c}=\hat f_x^\top\hat p_c/T,\quad T=0.1.
\]

所有组共同训练原型像素辅助分支，权重 `w_proto_seg = 0.1`。主分割分支权重 `w_seg = 0.1`。两分支均使用 `get_seg_loss`：对 one-hot 标签做逐类 sigmoid BCE，在有效像素上求和，再除以有效像素数；**没有再除以类别数**。本实验 class weight 均为 1，背景也参与这两个像素损失，ignore index 为 255。全 ignore crop 使用分母下界 1，返回可反传的零损失。

step0 的像素标签来自真值，尚未学习的类别设为 ignore。增量阶段的像素目标来自 teacher 分割结果的 argmax，并由 CAM/PAR 精炼结果中的本阶段新类覆盖相应像素；旧类图像标签来自 teacher 分类 logits > 0，新类图像标签来自图像级标注。这些共同的 teacher 伪标签机制在 BASE 组中仍然存在。

因此，BASE 表示“关闭新增原型 KD／SEP 约束的对照”，其原型辅助分支和 teacher 伪标签均已启用。验证预测取自主分割头 `conv8`，不是原型相似度分支。

## 原型方向蒸馏 KD

令 $\mathcal O_s$ 为阶段 $s$ 的已学旧前景集合：step1 为类别 1–10，step2 为类别 1–15。student 和 teacher 中相同类别按相同索引对齐：

\[
\mathcal L_{\mathrm{KD}}
=\frac{1}{|\mathcal O_s|}
\sum_{c\in\mathcal O_s}
\left[1-\operatorname{clip}
\left(\hat p^{S\top}_c\,\operatorname{sg}(\hat p^T_c),-1,1\right)\right].
\]

`sg` 表示 stop gradient。实现中 teacher 原型显式 `detach()`，teacher 全部参数冻结、置于 eval 模式，teacher forward 在 `torch.no_grad()` 中执行。KD 排除背景，不约束本阶段新前景原型；不存在旧前景时返回可反传的零损失。

这是**类别原型方向的参数蒸馏**：保留同类原型方向，而非像素预测概率、segmentation logits 或特征图的蒸馏。原型是与输入图片无关的学习参数，该损失不包含 logit 温度或 KL 分布匹配，也不保持原型向量的模长。

## 前景原型分离 SEP

令 $\mathcal F_s$ 为目前已学的全部前景类别，$K=|\mathcal F_s|$。固定 margin $m=0$：

\[
\mathcal L_{\mathrm{SEP}}
=\frac{1}{\binom K2}
\sum_{\substack{i,j\in\mathcal F_s\\i<j}}
\left[\max\left(0,
\operatorname{clip}(\hat p_i^\top\hat p_j,-1,1)-m\right)\right]^2.
\]

每个无序前景对仅计算一次，排除背景和自身配对，包含旧–旧、旧–新与新–新配对。cosine 不大于 0 的配对没有惩罚；少于两个前景原型时返回可反传的零损失。SEP 不要求所有前景原型的 cosine 为某个固定负值。

KD 和 SEP 的直接梯度只作用于 student 原型；它们不直接约束 `conv8` 或 backbone。共同的原型像素辅助损失同时训练原型与 decoder 特征，是这两个约束影响主分割分支的共享特征联系。几何约束的优化不能自动等同于主分割预测改善。

## 四组权重、损失 warmup 与 checkpoint 链路

| 组名 | \(\lambda_{\mathrm{KD}}\) | \(\lambda_{\mathrm{SEP}}\) |
| --- | ---: | ---: |
| BASE (`base`) | 0 | 0 |
| KD (`kd`) | 0.1 | 0 |
| SEP (`sep`) | 0 | 0.1 |
| KD+SEP (`kd_sep`) | 0.1 | 0.1 |

权重和 margin 在两个增量阶段内保持固定，不使用论文中的动态权重。各增量阶段前 2,000 次迭代仅训练分类、辅助分类和权重为 0.2 的 PTC；主像素损失、原型像素损失、KD、SEP 的目标系数均为 0。从第 2,001 次迭代开始使用：

\[
\mathcal L=
\mathcal L_{\mathrm{cls}}+\mathcal L_{\mathrm{cls,aux}}
+0.2\mathcal L_{\mathrm{PTC}}
+0.1\mathcal L_{\mathrm{seg}}
+0.1\mathcal L_{\mathrm{proto\text{-}seg}}
+\lambda_{\mathrm{KD}}\mathcal L_{\mathrm{KD}}
+\lambda_{\mathrm{SEP}}\mathcal L_{\mathrm{SEP}}.
\]

这里的 loss warmup 与学习率 warmup 是两个独立设置。零系数分支仍保留在 DDP backward graph 中；零目标梯度不意味着相关参数绝对不变，例如 AdamW 的 weight decay 仍可能作用于参数。

共享 step0 不使用 KD 或 SEP，采用分类、辅助分类、权重为 0.1 的主像素损失和权重为 0.1 的原型像素损失；初始阶段没有增量的 loss warmup 或 PTC 目标。四组都从同一个 step0 final checkpoint 出发：

```text
shared step0 final
├── BASE step1 final    → BASE step2
├── KD step1 final      → KD step2
├── SEP step1 final     → SEP step2
└── KD+SEP step1 final  → KD+SEP step2
```

每个增量阶段的 student 继承其上一阶段参数，新增类别参数除外；teacher 完整加载上一阶段模型。新 `conv8` 头在 checkpoint 加载后从已加载的背景权重初始化，新类原型保留该阶段的随机初始化。加载兼容 `module.` 前缀，并检查继承参数缺失及意外参数；step2 各组继承各自的 step1，不重新回到共享 step0。

## 指标与验证集合

IoU／mIoU 报告为百分数；相对同阶段 BASE 的差值报告为百分点（pp）。mIoU 对指定集合中的有限类别 IoU 求平均；非有限类别值保存为 null 并排除，汇总工具同时记录有效类别数量。

| 指标 | step1 | step2 |
| --- | --- | --- |
| `old_miou` / `old_initial10_miou` | 前景 1–10 | 前景 1–10 |
| `old_initial_with_background_miou` | 类别 0–10，共 11 类 | 类别 0–10，共 11 类 |
| `new_miou` / `new_since_initial_miou` | 前景 11–15 | 前景 11–20 |
| `previous_foreground_miou` | 前景 1–10 | 前景 1–15 |
| `previous_with_background_miou` | 类别 0–10，共 11 类 | 类别 0–15，共 16 类 |
| `current_foreground_miou` | 前景 11–15 | 前景 16–20 |
| `background_iou` | 类别 0 单独报告 | 类别 0 单独报告 |
| `all_miou` / `all_with_background_miou` | 类别 0–15，含背景 | 类别 0–20，含背景 |
| `foreground_miou` / `foreground_all_miou` | 前景 1–15 | 前景 1–20 |

step2 应同时提供“初始 10／累计新增 10”和“上一阶段 15／当前新增 5”两种前景分组，不能把第一种口径直接称为 15/5。含背景的初始旧类平均始终使用类别 0–10、共 11 类；含背景的上一阶段旧类平均在 step2 使用类别 0–15、共 16 类。二者均不能与相应的旧前景平均混用。

训练代码中的 stage 索引 $s$ 读取 `val_10-5_step_{s+1}.txt`：step0 为 869 张图，step1 为 1,240 张，step2 为 1,449 张。三个集合逐步包含，step2 列表与本地 VOC2012 官方 val 图像集合一致。四组在同一阶段使用同一列表，可以做同阶段对照；前两个阶段为阶段相关子集，不能默认等价于每阶段均使用完整 VOC val 的外部论文评价协议。

验证 GT 保留 VOC 原始类别 ID，不把未来类别映射为背景；`utils/evaluate.py` 的 `_fast_hist` 仅计入 `0 <= GT < num_classes` 的像素，三个阶段的 `num_classes` 分别为 11、16、21。例如 step0 验证图 `2007_000129` 含 person（ID 15），step1 验证图 `2007_000452` 含 sofa（ID 18），相应未来类像素在早期被排除，后期才纳入统计。若这些像素被预测为旧类，早期不会计入该旧类的 false positives，后期会计入，因此有效 GT 像素范围及旧类 IoU 分母也随阶段变化。同阶段四组采用完全相同的 GT 处理；最终 step2 使用全部 21 类的原始 GT，255 仍为 ignore。

## 结果解释的限制

1. 当前 KD 仅保存原型方向，主分割头通过共同像素辅助损失与原型间接联系；不能把结果表述为已经检验了常规像素／logit 蒸馏的效果，也不能仅凭原型几何变化声称性能提升。
2. 当前实验只有一个 seed、一个类别顺序和一组固定权重／margin，没有跨 seed 误差条或显著性证据。两个损失分别对类别及配对取平均，相同系数 0.1 不代表相同的梯度强度。
3. 验证图片集合及有效 GT 像素范围均随阶段变化，跨阶段旧类 mIoU 也受评价样本和旧类 false-positive 分母变化影响。解释或量化遗忘需要固定共同验证图集，并统一各阶段的 GT／标签空间处理（包括未来类像素和有效像素 mask），再重新评价各阶段模型；仅固定图集仍不足以排除这些混杂因素，不能把跨阶段分数下降直接归因于遗忘。
4. step2 继承各组自己的 student 和 teacher，最终组间差异反映完整的两个阶段轨迹；当前设计不能单独分离“只在 step2 增加某个损失”的边际贡献。

正式性能表以完整的最终 `results.json` 及各阶段最终评价记录为准。`prototype_diagnostics.json` 仅提供 checkpoint 原型几何诊断；`comparison.png/.pdf` 展示同阶段相对 BASE 的指标变化。smoke 输出仅验证程序流程，不作为方法有效性的证据。

## 实现依据

- `model/losses.py`：原型 KD、SEP 和像素 BCE 的确切归一化方式。
- `model/decoder/conv_head.py`：主头、独立可学习原型和归一化像素相似度分支。
- `continual/Trainer.py`：teacher 冻结、checkpoint 继承、共同伪标签、warmup、最终指标切片。
- `datasets/voc.py`、`datasets/voc/incremental_split/`：增量训练输入与阶段验证列表。
- `tools/run_baseline_study.py`、`tools/summarize_baseline_study.py`：四组执行链路、最终完成检查与补充指标口径。
- `runs/fixed_baseline_v1/study.json` 和各阶段 `config.json`：本次实际配置及冻结训练代码 hash。

## 本工作区复现

使用已经准备好的项目私有环境、VOC 输入和预训练权重，无需另建系统环境。启动或重新编排时，同一 output 目录由一个 runner 管理：

```bash
cd /ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto
source scripts/activate_baseline_env.sh
python -B tools/run_baseline_study.py --output runs/fixed_baseline_v1 --workers 4
```

GPU 默认使用 0、1、2、3。runner 先核对 study 配置和 7 个训练核心文件 hash；已有完成阶段还会核对阶段配置、前驱 checkpoint hash 和预训练权重 hash，匹配后复用。它不加载中途 optimizer 状态：缺少 final checkpoint 的阶段从该阶段的初始状态重新训练，而非恢复到中断迭代。欲独立重跑，将 `--output` 改为项目内一个新的目录，数据和预训练权重仍沿用已准备的输入。

在上述激活环境中，用以下 CPU 工具生成汇总、原型诊断和科学比较图；若采用新 output，也同步替换三个命令中的 study 路径：

```bash
CUDA_VISIBLE_DEVICES='' python -B tools/summarize_baseline_study.py runs/fixed_baseline_v1
CUDA_VISIBLE_DEVICES='' python -B tools/analyze_baseline_prototypes.py --study runs/fixed_baseline_v1
CUDA_VISIBLE_DEVICES='' python -B tools/plot_baseline_study.py --study runs/fixed_baseline_v1
```

汇总和原型诊断允许报告未完成状态；plot 严格要求 `results.json` 中 9 个阶段都具有预定最终迭代的评价，缺失时报告 incomplete，不生成替代性能图。输出均写入对应 study 目录。激活脚本将临时目录和相关缓存指向项目 `.runtime`，并关闭 Python bytecode 写入。

依赖版本锁位于 `.runtime/requirements-locked.txt`，已有环境的导入及运行验证回执位于 `.runtime/verification.json`。25 个 CPU 单元测试已通过，日志为 `.runtime/baseline-unit-tests.log`，回执为 `.runtime/baseline-unit-tests-receipt.json`；回执记录测试前后 7 个训练核心文件及 5 个测试文件的 hash，并确认训练核心与冻结 study 一致。这些是环境与实现验证证据，不是第二个增量阶段的性能结果。
