# ALD 伪标签融合：四组实验协议

本协议定义 `tools/run_ald_study.py` 的四组比较、监督计数和完成条件。正式评价采用独立的 `runs/ald_fusion_v1`，旧 `runs/fixed_baseline_v1` 保留为历史实验和共享初始化来源。本文不填写性能结果；方法是否有效须根据同阶段最终评价及实际监督计数判断。

## 问题与共同伪标签流程

增量训练采用 VOC 10-5 固定类别次序。step0 已学习前景 1–10；step1 增加 11–15，step2 增加 16–20。背景为 0，ignore 为 255。各组 step1 从同一个已完成的 step0 final checkpoint 初始化；各组 step2 继承自己的 step1 student，并以自己的 step1 模型作为冻结 teacher。

teacher 提供旧前景及背景像素预测的 argmax。图像级分类目标由 teacher 分类 logits > 0 的旧类标签与本阶段新类的图像级标注拼接而成。主 CAM 经过多尺度合并和 PAR 精炼，只有当前新增类的有效 CAM 标签覆盖 teacher 像素标签。

ALD 的 gate 从原始图像级标签的副本生成。分类损失、用于 CAM/PAR 的原始图像标签以及辅助 PTC 标签生成仍使用原来的 `cls_label`；gate 只在 PAR 后筛选像素 CAM。融合函数不原位修改 before/after CAM 或 teacher 标签，便于比较真正被拒绝的区域与最终监督去向。

## 四组定义

记 `V` 为 `[top,bottom,left,right]` 的 `img_box` 有效区，`P` 为 ALD 筛选前的 PAR 标签，`Q` 为筛选后标签，`T` 为 teacher 标签。当前新增类集合为 `N`，旧前景集合为 `O`。三组启用 ALD 的拒绝区定义为：

\[
R=V\cap\{P\in N\}\cap\{Q=255\},\qquad
R_{\mathrm{BG}}=R\cap\{T=0\},\qquad
R_{\mathrm{OLD}}=R\cap\{T\in O\}.
\]

`R` 只包含新类 CAM 因本次 gate 从有效类变为 ignore 的像素；不包含原先 PAR 就是 255 的区域、旧类 CAM 被拒区域或 box 外 padding。

| `ald_mode` | gate | 最终 `R` 内监督 | 有效区中参与像素损失的数量 |
| --- | --- | --- | --- |
| `off` | 关闭，`Q=P` | 未产生拒绝；保留有效新增类 CAM | `|V|` |
| `legacy` | 开启 | teacher OLD/BG 全部填回 | `|V|` |
| `preserve_rejected` | 开启 | 整个 `R` 在融合后保留 255 | `|V|−|R|` |
| `preserve_background` | 开启 | `R_BG` 保留 255，`R_OLD` 仍保留 teacher 旧类 | `|V|−|R_BG|` |

未被 gate 拒绝的当前新增类 CAM，在四组中都覆盖相应 teacher 标签。其他有效区域由 teacher 旧类或背景填回，包括筛选前已为 255 的 CAM 区域。`preserve_background` 检验“拒绝新类 CAM 后，不立即把 teacher 背景当作监督”的选择；保留 teacher 旧前景的策略本身并不保证旧类预测正确。

旧融合会把 PAR 留下的 box 外 255 也填回 teacher 标签。本实验在四组融合后统一将 `~V` 恢复为 255，要求 `padding_labeled_pixels=0`。因此 `off` 必须用当前共同实现重新训练；旧 BASE 只能作历史参照，其差值混有 padding 处理变化，不能直接归因于 ALD。

## 固定 gate：raw CAM 12.5 与 fallback

启用 ALD 的三组完全共用原有 gate。主 CAM 在每个尺度合并原图与水平翻转预测，应用 ReLU，再将尺度 `[1.0,0.5,1.5]` 的响应求和，得到 `A_c`。gate 的量为归一化前全图动态范围：

\[
a_c=\max_{x\in\mathrm{crop}}A_c(x)-\min_{x\in\mathrm{crop}}A_c(x).
\]

图像标签为正且 `a_c >= 12.5` 的类别保留；`a_c < 12.5` 的正类被 gate 移除。此量包含 padding，不是归一化 CAM 的峰值、概率或经过校准的置信度，阈值也不属于 `[0,1]`。它受模型响应尺度、多尺度求和和 crop 影响。本轮保持 12.5，以隔离融合策略的作用。

若该图所有正类都未通过阈值，则 fallback 恢复原始正类中 `a_c` 最大的一个；没有正类时不恢复。正类集合同时包含 teacher 预测的旧前景与有标注的新前景。最大值相同采用当前 `argmax` 的索引规则。单正类图因此不会因 gate 丢掉唯一正类；多正类图的 fallback 可能只恢复旧类，而使新类 CAM 被拒。记录 `fallback_old_images` 和 `fallback_new_images`，避免把阈值以下的恢复误记为通过 12.5。

gate 保持背景类别有效，并在 PAR 后把未保留类别的像素标签改为 255；新增的融合策略随后决定其中 `R` 的最终监督。PAR 仍在全图传播，最后按 box 取有效区，本轮没有改变其传播算法。

## 共同训练预算与损失

| 设置 | 正式四组 | smoke 四组 |
| --- | --- | --- |
| 新训练阶段 | 每组 step1、step2，共 8 阶段 | 同样 8 阶段 |
| 每阶段迭代 | 8,000 | 4 |
| 全局 batch / 每组 GPU 数 | 8 / 1 | 8 / 1 |
| GPU 映射 | off→0；legacy→1；preserve_rejected→2；preserve_background→3 | 相同 |
| 增量 seed / DataLoader workers | 0 / 4 | 0 / 4 |
| 初始学习率 | 2e-5 | 2e-5 |
| 学习率 warmup / loss warmup | 2,000 / 2,000 | 1 / 1 |
| 验证迭代 | 2k、4k、6k、8k | 4 |
| 监督计数记录间隔 | 50 iterations | 1 iteration |
| train / val 图像上限 | 0 / 0，完整阶段列表 | 32 / 8 |
| KD / SEP 权重 | 0 / 0 | 0 / 0 |
| 原型像素辅助权重 / margin | 0.1 / 0 | 0.1 / 0 |
| confusion class reweight | 关闭 | 关闭 |

正式新增预算是总计 64,000 次 optimizer step，按 batch 8 计为 512,000 次训练图像抽样；这不是独立图像数量。smoke 为 32 次 optimizer step、256 次图像抽样。共享原 step0 已训练 20,000 iterations，本轮不重新训练或复制其大 checkpoint。每组两个阶段串行，四组并行；具体运行时间依 GPU 共享负载及验证开销确定，不把 smoke 时间直接外推为正式预算。

backbone 为 ImageNet 预训练 `vit_base_patch16_224`，aux layer −3；crop/local crop 为 448/96，训练缩放 `[0.5,2.0]`。PolyWarmupAdamW 使用 weight decay 0.01、betas `(0.9,0.999)`、power 0.9。共同主分割头权重 `w_seg=0.1`、辅助 PTC 权重 `w_ptc=0.2`，原型辅助分支保留；KD 和 SEP 虽可计算与记录，其目标系数始终为 0。

前 2,000 次正式增量迭代仅使用分类、辅助分类及 PTC 目标。第 2,001 次开始加入主像素损失和原型像素辅助损失：

\[
\mathcal L=\mathcal L_{cls}+\mathcal L_{cls,aux}
+0.2\mathcal L_{PTC}+0.1\mathcal L_{seg}+0.1\mathcal L_{proto\text{-}seg}.
\]

像素损失对各类 sigmoid BCE 求和，再除以非 ignore 像素数，没有额外除以类别数；背景参与，class weight 均为 1。改变 ignore 数同时改变监督区域和这一分母。因此监督数量下降不能单独证明标签质量提升或梯度尺度一致。

## 监督计数与 warmup 解释

各阶段 `ald_metrics.jsonl` 由 rank0 按 interval 写入整数计数。正式每行累计 50 个 batch、400 张抽样图像；smoke 每行 1 个 batch、8 张。计数在写入后清零，跨行汇总比例应先求分子、分母的和，再相除，不能直接平均各行比例。

| 字段 | 含义 |
| --- | --- |
| `valid_box_pixels` / `total_pixels` | box 内有效像素 / crop 全部像素 |
| `before_new_cam_pixels` / `after_new_cam_pixels` | 有效区筛选前 / 后的新类 CAM 像素 |
| `rejected_new_cam_pixels` | `|R|` |
| `rejected_teacher_old_pixels` / `rejected_teacher_bg_pixels` | `|R_OLD|` / `|R_BG|` |
| `retained_ignore_pixels` | 本策略在 `R` 中最终保留 ignore 的数量 |
| `rejected_final_ignore_pixels` / `rejected_final_old_pixels` / `rejected_final_bg_pixels` | 被拒新类 CAM 最终去向 |
| `final_new_pixels` / `final_old_pixels` / `final_bg_pixels` / `final_ignore_pixels` | 有效区最终标签分区 |
| `final_valid_pixels` | 有效区中最终不是 ignore 的像素数量 |
| `padding_labeled_pixels` | box 外最终不是 ignore 的数量，必须为 0 |
| `positive_*_classes` / `gate_rejected_*_classes` | 按图片计的正类 / gate 移除类出现次数，区分 OLD/NEW |
| `fallback_images` / `fallback_old_images` / `fallback_new_images` | fallback 图像数及所恢复类别的归属 |

监督统计的OLD/NEW按当前增量边界定义：step1 OLD为前景1–10、NEW为11–15；step2 OLD为前驱前景1–15、NEW为16–20。teacher OLD像素、OLD正类及gate移除OLD类计数均采用该边界，背景单独记0；它与step2原始old_miou的initial1–10口径不同。

四个比例分别为 `R/before_new_cam`、`R/valid_box`、`final_valid/valid_box`、`retained_ignore/R`。分母为零保存 null，而非人为记为 0。`off` 不计算 gate，其拒绝与 fallback 计数为 0。

每行含 `step`、`iteration`、`ald_mode`、`interval_batches` 和 `segmentation_loss_active`。正式迭代 50–2,000 的 active 为 false，2,050–8,000 为 true；smoke 迭代 1 为 false，2–4 为 true。warmup 内仍产生完整伪标签和计数，但像素损失系数为 0：这些是候选监督的统计，不能当作已用于像素优化的证据。报告时应分别累计 warmup 与 active 区间。

验收检查整段记录序列、batch 数、active 边界与下列整数恒等式；对于每个 mode，还检查四组定义对应的 teacher OLD/BG→ignore 闭环：

```text
before_new_cam = after_new_cam + rejected_new_cam
rejected_new_cam = rejected_teacher_old + rejected_teacher_bg
rejected_new_cam = rejected_final_ignore + rejected_final_old + rejected_final_bg
retained_ignore = rejected_final_ignore = final_ignore
valid_box = final_valid + final_ignore
final_valid = final_bg + final_old + final_new
final_new = after_new_cam
padding_labeled = 0
```

## 指标口径与比较对象

预测取自主分割头，不取原型相似度辅助分支。IoU/mIoU 报告为百分数，差值用百分点；先以同阶段重新训练的 `off` 为比较对象，再讨论两个阶段的完整轨迹。最终结果使用 8k 记录，不以 2k/4k/6k 中最好的一次替代终点。

| 指标 | step1 | step2 |
| --- | --- | --- |
| `old_miou` | 初始旧前景 1–10 | 初始旧前景 1–10 |
| `new_miou` | 累计新增前景 11–15 | 累计新增前景 11–20 |
| `all_miou` | 0–15，含背景 | 0–20，含背景 |
| `foreground_miou` | 1–15 | 1–20 |
| 补充 previous 前景口径 | 1–10 | 1–15 |
| 补充 current 前景口径 | 11–15 | 16–20 |

`class_iou` 保留逐类信息，背景应单独报告；含背景的旧类均值需明确类别集合。step2 的 `old_miou/new_miou` 不是 15/5。mIoU 对有限类别 IoU 求平均，缺失类存为 null；汇总应报告有限类别数。`classification_f1` 和 `cam_miou` 是辅助指标，不能替代主分割 mIoU。

验证沿用阶段列表：step0/step1/step2 分别有 869/1,240/1,449 张，step2 与本地官方 VOC2012 val 图像集合一致。早期 GT 保持原始类别 ID，`_fast_hist` 仅计入 `0 <= GT < num_classes`，未学习的未来类像素被排除而非映射为背景。因而跨阶段的图片集合、有效 GT 像素及旧类 false-positive 分母都会变化；同阶段四组的评价范围一致。不能直接把跨阶段旧类分数下降解释为遗忘量。

## 来源、源码冻结与完成条件

共享 checkpoint 以绝对路径只读引用：

```text
/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/runs/fixed_baseline_v1/shared/10-5/step0/checkpoints/model_final.pth
SHA-256: 4f298e14721630cf3c66ba4f6af04bd198ebeaf77adc6b7b07ab8403ad0893b5
```

runner 将实际文件 SHA 与旧正式四组 step1 `inputs.json` 的前驱 SHA 全部匹配，核对原 20k 最终评价、来源配置和预训练权重 SHA。预训练权重 SHA 为 `80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c`。

旧方法在 ALD 改动前保存源码、依赖锁与数据列表快照，共 152 个文件；归档不包括图像、掩码、大模型权重或整个运行环境：

| 保存文件 | SHA-256 |
| --- | --- |
| `runs/fixed_baseline_v1/source_snapshot.tar.gz` | `e4a9c503c8967c631f3c53955f516744f645aefdf3dd68e71086ddb387a888d1` |
| `runs/fixed_baseline_v1/source_snapshot.json` | `929293f6cbe6d7d5628925967e88f69bc9e6ea031101cee3491a4c7bfc486eea` |

正式 ALD 冻结版本另有 127 文件归档，包含训练源码、测试、数据列表及私有环境验证记录；权重、图像与整个 conda 环境仍独立保存：

| 保存文件 | SHA-256 |
| --- | --- |
| `runs/ald_fusion_v1/source_snapshot.tar.gz` | `b4e4298a13b612346ccc6182f206124f0f81db4e87aae4cbe7141eeac090b891` |
| `runs/ald_fusion_v1/source_snapshot.json` | `f30b8a683ef22dd4f7073af9195a832dbf5f5556e632e32fd5eb9e384e40edbe` |

新 study 的 `study.json` 锁定训练入口、runner、tasks、model/continual/datasets/utils 的 Python 源码，包含 `ald.py`、`ald_stats.py`、`camutils.py` 和 PAR；另记录 VOC 列表、数据路径、初始化与预训练 SHA。每阶段 `inputs.json` 保存完整命令、预期配置和前驱来源，`config.json` 保存实际参数，`process.json` 保存返回码。`completion.json` 绑定最终评价、监督计数、输入/config/checkpoint/metrics/ald_metrics 的 SHA。

正式完成要求四组全部两个阶段的 8k 评价、完整 50–8,000 监督计数、final checkpoint 和 exit0 完成记录，共 8/8 新阶段，并且总 runner exit0；此时生成 `results.json`，`status.json` 为 complete。smoke 对应 4iter 和 1–4 监督计数。阶段验证后的中间 checkpoint 不等同于最终完成。

已有未完成阶段或不匹配配置不被覆盖；runner 用独占锁防止同 output 的双启动。训练失败保留产物并记录失败，清理范围限于该 runner 创建的子进程。输出、日志和缓存均限制在授权的项目目录内，其他 GPU 用户任务保持原样。

历史 `runs/ald_fusion_v1_smoke` 保留其当时的源码 SHA 与产物。其训练实现与本次正式实现一致；正式启动前仅增强 runner 对 `ald_metrics.jsonl` 的 SHA 和计数闭环验收，模型/Trainer/CAM 源码未因此变化。历史 smoke 的 manifest/completion 不补写；新 runner 的 SHA 因验收补丁而不同，不能把两份 runner manifest 说成逐字一致，也不复用新源码去覆盖历史 smoke。

## 可复现命令

使用项目私有环境和已准备的 VOC/预训练输入：

```bash
cd /ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto
source scripts/activate_baseline_env.sh

# 将来重跑 smoke 时使用新 output，保留历史 smoke。
python -B tools/run_ald_study.py --smoke --output runs/ald_fusion_v1_smoke_reproduce --seed 0 --workers 4 --gpus 0,1,2,3

# 正式协议：四组共用同一 step0，各自串行训练两个 8k 阶段。
python -B tools/run_ald_study.py --output runs/ald_fusion_v1 --seed 0 --workers 4 --gpus 0,1,2,3
```

GPU 0–3 分别绑定四个 mode，每个子任务由 `torch.distributed.run --standalone --nproc_per_node=1` 启动。同一 output 只允许一个 runner；独立复现正式实验应改用新的项目内目录，例如 `runs/ald_fusion_v1_reproduce`。`--seed` 只改变增量随机种子，共享初始化仍为原 seed0；若声称多 seed 完整训练稳定性，需要另外训练并记录各 seed 的 step0。

## 研究解释的边界

1. 当前仅 seed0、VOC 10-5、固定类别顺序与 raw threshold 12.5。没有跨 seed 误差条、显著性证据或跨数据集结论；gate 也没有概率校准。
2. 监督计数核验的是像素标签去向，不是伪标签正确率。ignore 策略同时改变有效样本数量、损失分母和 teacher OLD/BG 的影响，不能仅凭被拒数量增加声称去噪成功。
3. box 外像素在最终主分割/原型像素损失中统一忽略，但 raw gate 的全图动态范围和 PAR 传播仍可能受 padding 影响。辅助 PTC 使用特征分辨率 CAM 而 `img_box` 来自原图分辨率的旧坐标问题本轮保持原样，不能声称已修复所有 padding 影响。
4. 早期未来类 GT 排除及阶段验证子集影响跨阶段比较。需要固定图集、有效像素和标签空间处理后重新评价，才能较可靠地讨论遗忘。
5. step2 的 teacher 与初始化均来自本组 step1，最终差异代表两个阶段的完整学习轨迹，不能解释为“只改变 step2 融合”的边际效果。各组固定 seed 与配置，没有另外记录逐 batch 的采样/增强序列以证明完全一致。
6. 正式终点及监督计数优先于 smoke 或中间验证趋势。性能和覆盖统计须由独立结果汇总报告；本协议本身不构成提升证据。
