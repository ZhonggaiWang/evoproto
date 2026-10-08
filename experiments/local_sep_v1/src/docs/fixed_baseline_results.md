# 固定权重原型 baseline：正式结果与后续方向

生成时间：2026-10-04T01:22:03.675763+00:00。VOC 10-5、seed 0；共享初始阶段及四组的两个增量阶段全部完成，共 9/9 个预定终点。主 runner 正常退出（exit code 0）。本报告使用最终评价，不使用 smoke 或中途分数。

## 本次实现与配置

KD 使用匹配旧前景原型的平均 `1 − cosine`，teacher 原型 detach；SEP 使用全部已学前景无序配对的平均 `relu(cosine − margin)^2`，排除背景及自身配对。启用项权重固定为 0.1，margin 固定为 0，ALD 和 confusion reweight 均关闭。KD 是原型方向参数蒸馏，主分割预测取自独立的 `conv8`。

BASE 仅关闭新增 KD/SEP，仍有权重 0.1 的共同原型像素辅助损失和 teacher 硬伪标签。四组的其他监督一致。初始阶段使用像素真值；增量阶段读取图像及新类图像级标签，不读取训练像素真值。冻结 teacher 提供旧类图像标签及像素伪标签。

全局 batch=8，crop=448，初始/增量学习率=6e-5/2e-5，训练预算=20,000/8,000 iterations。增量前 2,000 步为分类与 PTC loss warmup，此后 KD/SEP 系数固定。step0 用四卡、每卡 batch 2；四组增量各用单卡、batch 8。step1 均继承同一 step0，step2 各自继承自身 step1。所有实际参数、前驱 checkpoint 和预训练 SHA 记录在阶段 `config.json`/`inputs.json`。

共享 step0 最终初始 10 个前景 mIoU 为 **82.266**，包含背景的全部 11 类 mIoU 为 **83.556**。

## 各增量阶段的旧类、新类与整体 mIoU

单位为百分数。旧/新均只计前景；整体包含背景。step1 的旧/新为 1–10 / 11–15；step2 为 1–15 / 16–20。相对 BASE 的差值为同阶段百分点（pp）。

| 组别 | step1 旧10 | step1 新5 | step1 全16类 | step2 旧15 | step2 新5 | step2 全21类 | step2 Δ整体 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 76.353 | 65.764 | 74.109 | 71.901 | 53.652 | 68.475 | +0.000 |
| KD | 76.079 | 64.751 | 73.613 | 71.481 | 56.047 | 68.758 | +0.283 |
| SEP | 76.505 | 65.169 | 74.011 | 71.965 | 55.571 | 68.997 | +0.522 |
| KD+SEP | 76.659 | 66.079 | 74.400 | 72.150 | 52.942 | 68.505 | +0.030 |

训练原始 `old_miou` 始终指初始前景 1–10，`new_miou` 在 step2 指累计新增 11–20；它们不是 step2 的 15/5 分组。以下同时提供原始口径、背景与全部前景：

| 组别 | step2 初始10前景 | step2 累计新增10前景 | 背景 IoU | 全20前景 mIoU |
| --- | ---: | ---: | ---: | ---: |
| BASE | 74.764 | 59.913 | 91.198 | 67.339 |
| KD | 74.443 | 60.801 | 91.468 | 67.622 |
| SEP | 75.080 | 60.653 | 91.608 | 67.866 |
| KD+SEP | 75.006 | 59.689 | 91.646 | 67.348 |

## 结果解释

1. 本次 SEP 的最终整体分数最高，较 BASE 提高 **0.522 pp**。旧15仅提高 **0.064 pp**，当前新5提高 **1.919 pp**；增益主要来自当前新增类别。它在 step1 的整体分数低于 BASE 0.099 pp，不能说两个阶段均改善。
2. KD 最终整体提高 **0.283 pp**，但旧15下降 **0.420 pp**，当前新5提高 **2.395 pp**。原型方向更稳定没有转化为更高的旧类分割成绩。
3. 联合组 step1 整体提高 **0.290 pp**；step2 仅提高 **0.030 pp**。最终旧15提高 **0.249 pp**、新5下降 **0.710 pp**。当前结果没有显示两个约束叠加产生持续优势。
4. 这是一个 seed、一个类别顺序和一组固定权重的 baseline，无误差条或显著性证据。上述排序描述本次运行，尚不能证明稳定优于 BASE，也不能据此排除其他权重或蒸馏形式。

## 原型几何：检验约束是否起作用

漂移为匹配原型的平均 `1 − cosine`，越低越接近参照；分离惩罚为全部前景配对的平均正 cosine 平方。step1 的 teacher 旧类为 10 个，step2 为前驱 15 个。step2 teacher 是各组自身的 step1；初始10累计漂移另用共同 step0 作为参照。

| 组别 | step1 旧10对teacher漂移 | step1 分离惩罚 | step2 旧15对teacher漂移 | step2 初始10对共享step0漂移 | step2 分离惩罚 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BASE | 0.052636 | 0.159389 | 0.062512 | 0.139572 | 0.185455 |
| KD | 0.013702 | 0.139348 | 0.027782 | 0.046542 | 0.151601 |
| SEP | 0.056678 | 0.070082 | 0.052516 | 0.111763 | 0.084309 |
| KD+SEP | 0.016065 | 0.076810 | 0.024018 | 0.042708 | 0.078247 |

KD/联合组更贴近 teacher 方向；SEP/联合组明显减小正 cosine 强度，说明新增约束确实改变了原型几何。不过所有组两个增量终点的前景 pair cosine 仍全部为正：margin=0 是软惩罚，没有实现所有配对 cosine≤0 的硬约束。联合组最终分离惩罚甚至比 SEP 更低，但分割分数更低；几何指标不能代替主头性能。

KD/SEP 的直接梯度只作用于原型；主头通过共同原型像素辅助损失与共享 decoder 特征间接受到影响。保留原型不等于保留主头预测。旧–旧 SEP 与 KD 对正相关 teacher 原型存在目标竞争的可能；只分离含新类的配对是可另行验证的假设，本次没有实现或比较。

## 评价协议与证据边界

验证集分别为 869 / 1,240 / 1,449 张，最终集合与 VOC2012 官方 val 一致；同阶段四组使用相同图集。早期 `_fast_hist` 只计入 GT<num_classes 的像素，未来类别像素被排除；后期有效 GT 范围扩大，也会改变旧类 false-positive 分母。因此跨阶段分数下降不能直接称为纯遗忘。量化遗忘需固定图集，并统一 GT/标签空间与有效像素处理。最终 step2 评价全部21类原始 GT，255仍 ignore。

9条 `results.json` 记录逐条与原始 `metrics.jsonl` 最后一行严格相等，迭代为 20,000/8,000；各阶段已见类别 IoU 齐全且有限。9个最终 checkpoint 均经 CPU weights_only 加载、张量有限/头尺寸核验；前驱实际 SHA、预训练 SHA、配置及7个冻结核心 SHA一致。CPU 汇总和原型诊断均完整9/9，无 warning/failure。

环境位于 `.runtime/env`，21项依赖导入及 CUDA forward/backward 验证通过。缓存均限制在项目内：激活脚本的相关缓存位于 `.runtime`，训练入口将 torch.hub 缓存设为项目 `pretrained`，实际使用已核验 SHA 的 `pretrained/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth`。环境内部链接全部解析到该环境；关键解释器/依赖文件与源安装 inode不同。12031份增强标注均为自有独立文件，6个split hash与准备回执一致、训练/验证集合无交集；准备阶段1449张验证增强标注数组与原始 GT完全一致。

25个CPU测试通过，测试源码、训练源码和日志 hash与回执一致；覆盖KD/SEP不变量与梯度、真实decoder原型/像素特征梯度和checkpoint继承、ignore/palette标签、增量JPEG-only读取及日志图释放。9阶段CUDA smoke（每阶段4步）对应相同7个核心hash，只用于集成流程验证；本报告性能来自完整正式预算。teacher整网冻结/eval/优化器排除另有实际源码证据，CPU单测不声称验证整网GPU位级确定性。

## 下一轮优先检验的 ALD 改动

当前 ALD 用归一化前 CAM 峰值阈值12.5决定新类CAM能否覆盖teacher。旧类图像标签仍来自teacher logits>0；分类loss没有ALD门控。ALD产生的255经过 `get_mixed_label` 后由teacher旧类/背景填回，过滤没有以ignore形式保留到像素监督。这是一个可明确定位的融合策略选择，不能仅凭代码断定改变它会提高精度。

建议先在step1做固定teacher的两组对照：均从同一shared step0出发，均开启现有ALD，保留BASE的KD=SEP=0、seed0、batch8和8000/2000训练预算；唯一差别是实验组在融合后保持“本次被ALD新拒绝的新类像素”为255。现有ALD-off BASE作为已有参照。

只屏蔽 `过滤前属于当前有效新类` 且 `过滤后变为255` 的像素，不能把原有全部CAM ignore区域一并屏蔽，以免同时删除旧类teacher监督。记录拒绝比例、有效像素覆盖率与旧/新/整体mIoU，再决定是否扩展到第二轮及与SEP叠加。三态旧类图像标签门控另做实验，避免一次改变多个监督来源。ROI crops/flags目前没有进入启用loss（CTC被注释且forward不编码crops），后续也应先确认新增模块的真实梯度链路。以上均为未实施、待验证的下一轮假设。

代码位置：[Trainer.py](../continual/Trainer.py) 591–643行，[camutils.py](../utils/camutils.py) 252–275行；方法细节见[协议](fixed_baseline_protocol.md)。

## 产物与复现

- [完整未舍入最终结果](../runs/fixed_baseline_v1/results.json)、[各阶段补充口径汇总](../runs/fixed_baseline_v1/summary.json)。
- [正式对照图 PNG](../runs/fixed_baseline_v1/comparison.png) / [PDF](../runs/fixed_baseline_v1/comparison.pdf)。图中step2使用初始10/累计新增10口径，不能当作15/5。
- [原型诊断](../runs/fixed_baseline_v1/prototype_diagnostics.json)、[冻结配置与核心hash](../runs/fixed_baseline_v1/study.json)。
- [测试回执](../.runtime/baseline-unit-tests-receipt.json)、[测试日志](../.runtime/baseline-unit-tests.log)、[环境版本锁](../.runtime/requirements-locked.txt)。

独立复现使用项目内新output（同一output只由一个runner管理）；实际数据和预训练输入已准备。完整方法、指标和复用/重训规则见[协议](fixed_baseline_protocol.md)。

```bash
cd /ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto
source scripts/activate_baseline_env.sh
python -B tools/run_baseline_study.py --output runs/fixed_baseline_reproduction --workers 4
```
