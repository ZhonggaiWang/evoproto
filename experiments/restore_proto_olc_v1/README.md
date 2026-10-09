# EvoProto + OLC (Old-class Label Correction)

OLC replaces the rejected extra-training ALD candidate. Each incremental stage still receives exactly 8,000 optimizer updates, batch size 8 (4 on each of physical GPUs 5 and 6), seed 0, and the original learning-rate schedule. Step 1 starts from the shared step-0 final checkpoint; step 2 starts from its own step-1 final checkpoint. There is no additional refinement stage, same-stage reference network, partial-set loss, or reference KL loss.

## Method

The frozen previous-stage teacher already produces main and auxiliary image-classification logits. For each image ID and each head, OLC maintains an exponential moving average of sigmoid probabilities over the randomly augmented training views observed so far. On its first visit, memory equals the current prediction. Subsequent visits use `memory = 0.5 * memory + 0.5 * current`. Duplicate IDs within one global batch are averaged before updating. Both ranks maintain identical detached memory. The fixed momentum 0.5 is an initial design choice, not a measured optimum.

An old class is positive when both averaged head probabilities exceed 0.5, negative when neither exceeds 0.5, and unknown otherwise. The teacher weights remain frozen; it is the per-image prediction memory that evolves inside the original 8,000-step training.

- Old-class classification BCE uses positive and negative decisions. Unknown entries contribute zero gradient, with the original batch-times-class denominator preserved. Current-class ground-truth image labels are unchanged.
- Unknown classes remain in CAM competition so that their spatial evidence is not automatically relabeled as background. Their winning old-class CAM/PAR labels are then ignored for hard supervision and PTC.
- Teacher old-class pixel labels are retained only for OLC-positive image classes. Rejected old-class labels become ignore (255), never background. Valid current-class pseudo labels retain priority.
- The original confusion/prototype mechanism is unchanged. Its existing class-tag inputs now receive corrected positives; hence unreliable old-class evidence cannot create anchors or reliable KD support.

No old-class annotations or incremental pixel masks are inputs to OLC. Offline label audits join image annotations only after predictions have been produced. Whole-image diagnostic quality does not establish the quality of EMA predictions under random crops; online memory will be audited separately.

## Evaluation

Primary comparison: main-head mIoU, same square-448 and aspect-preserving-672 protocols as the completed no-OLC baseline. Secondary comparison: the previously fixed 0.5 main/prototype probability fusion; do not retune its coefficient for OLC. Report old/new class performance as well as overall mIoU. Label diagnosis reports precision, recall counting abstained positives as misses, coverage, and wrong negatives.

At 2,000/4,000/6,000/8,000 updates, record segmentation, prototype, main CAM and auxiliary CAM mIoU. CAM diagnostics use validation image-level tags, as in the historical CAM evaluation; segmentation inference uses no image tags. These protocols must be identified separately.

Only final model weights are saved per stage. The small prediction-memory file is overwritten at evaluation boundaries. Necessary predecessor/final method weights are protected.

## Reproduce

Run from the project root with the configured runtime:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo .runtime/restore_proto/venv/bin/python -B -m torch.distributed.run --master_addr=127.0.0.1 --master_port=49374 --nproc_per_node=2 tools/test_olc.py --ddp
.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_olc_v1/run.py --smoke --arms baseline control olc
.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_olc_v1/run.py --arms olc
```

The runner records source hashes, predecessor hashes and commands, uses the existing GPU reservation guard, and restores reservation after each job or failure. It refuses to overwrite partial runs. The `control` arm uses the new trainer with OLC disabled; the `baseline` arm uses the original trainer for smoke equivalence checking. Formal comparison reuses the completed, identical-budget original full chain only after this equivalence check.

## 已完成的同预算结论

**当前 EMA 双头一致版 OLC 提高了旧类标签精确率和 F1，但没有提高最终分割。正式推荐仍为无 OLC 的原型恢复主线，OLC 保留为实验候选。**

两组均完成各自 Step 1 → Step 2，每阶段 8,000 步，总 batch 8、seed 0；没有追加续训。下表来自同一 GPU、相同输入的配对评估，融合系数 0.5 在 OLC 开跑前已经固定。

| 输入 / 预测方式 | 无 OLC | OLC | 变化（点） |
| --- | ---: | ---: | ---: |
| square448 / 主头 | 68.7124 | 68.7027 | -0.0097 |
| square448 / 固定 0.5 原型融合 | 68.8482 | 68.7678 | -0.0803 |
| aspect672 / 主头 | 69.8904 | 69.8479 | -0.0425 |
| aspect672 / 固定 0.5 原型融合 | 70.0644 | 69.8638 | -0.2006 |

aspect672 主头的旧前景变化为 -0.1544 点，新前景为 +0.3231 点。四项配对图像 bootstrap 区间均包含 0，不能称为显著提升或下降。这仅是单 seed、反复查看过的 VOC 验证集结果；图像 bootstrap 不衡量训练种子方差。

### 在线标签诊断

| 阶段 | 规则 | 精确率 % | 召回率 % | F1 % | 覆盖率 % |
| --- | --- | ---: | ---: | ---: | ---: |
| Step 1 | 同记忆，仅主头 | 54.22 | 93.60 | 68.66 | 100.00 |
| Step 1 | OLC | 60.21 | 90.56 | 72.33 | 98.50 |
| Step 2 | 同记忆，仅主头 | 46.38 | 93.98 | 62.11 | 100.00 |
| Step 2 | OLC | 49.35 | 92.69 | 64.40 | 98.39 |

这是对同一预测记忆的筛选规则比较，诊断真值仅在独立程序中拼接，不参与训练。它不等价于无 EMA 的独立训练消融。完整图像的静态诊断与随机裁剪记忆输入不同，Step 2 还使用了不同 Step 1 链路产生的教师，不能把两者差异全部归因于 EMA。

### 完整训练曲线（square448 主头）

| 阶段 | 更新 | 无 OLC | OLC | 变化（点） |
| --- | ---: | ---: | ---: | ---: |
| Step 1 | 2000 | 47.0837 | 45.9333 | -1.1504 |
| Step 1 | 4000 | 71.5185 | 73.7363 | +2.2178 |
| Step 1 | 6000 | 73.9737 | 74.5784 | +0.6048 |
| Step 1 | 8000 | 74.6066 | 75.2892 | +0.6827 |
| Step 2 | 2000 | 50.2123 | 53.7851 | +3.5728 |
| Step 2 | 4000 | 68.0165 | 64.6884 | -3.3281 |
| Step 2 | 6000 | 68.1181 | 68.4893 | +0.3712 |
| Step 2 | 8000 | 68.7124 | 68.7027 | -0.0097 |

Step 1 的最终提升没有转化为最终 21 类分割提升。Step 2 在 4,000 步的 bird / sheep 回落随后恢复，不能用中间最高或最低点替代 8,000 步结论。

### 机制与复核边界

- 混淆统计和原型 KD / SEP 保持有效；两个阶段继承原型的归一化方向均发生更新，排除了仅有权重衰减缩放的解释。
- 未知分类标签零梯度、原损失分母、旧类忽略掩码、双进程记忆一致和重复图像合并测试均通过。
- 两阶段 GPU 冒烟通过。关闭 OLC 的差异为 0.00214 mIoU 点，原入口重复运行差异为 0.00172 点；没有逐位确定性承诺。
- 历史负证据可能延迟接纳目标：记忆 0.02 与当前双头概率 0.90 平均后为 0.46，仍判负。这是规则行为的反例诊断，并未证明它导致某个实际类别退化。
- 旧 ALD 的额外续训结果不纳入 OLC 成绩。保留原主线作为正式配置，不能把“标签更准”直接写成“最终分割更好”。

机器可读的完整对照、类别差异、权重哈希、原型更新核查和训练源码哈希见 [results.json](results.json)。最终模型及必要前驱仍保存在远程工作区；数据和权重不上传 Git。
