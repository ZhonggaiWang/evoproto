# ALD 融合实验记录
最近已验证可加载的周期权重：off/legacy的step2均保存到2,000次，两ignore方案的step1均保存到6,000次；off/legacy的step1完整8,000次权重及既有baseline保留。停止时尚未到下一保存点的更新未保存。停止进程身份、退出码及四份权重CPU加载收据：`.runtime/ald-user-stop-receipt.json`。


用户要求暂停，训练、监控和后续排队进程已停止；已审计off/legacy的step1终点，共2/8个阶段完成，全实验尚未完成，不能据此选定最终方法。所有优化判断以本次新训练的 off 对照为准；历史固定 BASE 只作参考。

## 已完成验证

- 私有 conda 环境中的 47 项 CPU 测试全部通过，覆盖融合边界、teacher 不被原地修改、ignore 像素梯度、统计分区及既有原型/数据流程。收据：`.runtime/ald-unit-tests.receipt.json`。
- 四种模式的 CUDA smoke 共 8/8 个阶段以 exit0 完成，均保存最终权重。每阶段仅 4 次训练、8 张验证图，分数不用于性能比较。
- smoke 所有区间均满足真实图像有效区分区、teacher OLD/BG 拒绝分区、各模式 ignore 数量与零 padding 监督约束。原始日志与 SHA 收据位于 `runs/ald_fusion_v1_smoke/supervision-verification.json`。
- smoke 后只加强了 runner 的 coverage 完整性检查；模型与训练源码未变。新检查通过实际 8 个 smoke 阶段，并拒绝篡改 padding 数及缺少最终区间的临时副本：`.runtime/ald-runner-coverage-verification.json`。

## 正式对照

| 模式 | GPU | 新拒绝的 NEW CAM 像素 |
|---|---:|---|
| off | 0 | 不进行 ALD 过滤；teacher 与新类 CAM 融合 |
| legacy | 1 | 全部由 teacher 标签填回 |
| preserve_rejected | 2 | 全部保留 ignore |
| preserve_background | 3 | teacher 为背景时保留 ignore，teacher 为旧类时保留旧类监督 |

训练日志原字段`old_miou/new_miou`始终按initial10/cumulativeNEW统计：step1对应1–10/11–15，step2对应1–10/11–20；它们不是step2的previous15/current5。总结工具从逐类IoU分别重算两种类组，并核验原字段与initial/cumulative一致。step2 warmup首条off记录虽然raw new_miou为31.790，但本次新5类16–20的IoU全部为0，不能误读为本次5类已达到31.790。该2,000次记录是中途评估，不是最终性能。

监督统计的OLD/NEW按当前增量边界定义：step1 OLD为前景1–10、NEW为11–15；step2 OLD为前驱前景1–15、NEW为16–20。teacher OLD像素、OLD正类及gate移除OLD类计数均采用该边界，背景单独记0；它与step2原始old_miou的initial1–10口径不同。

四组共享旧正式 seed0 的 20,000 次 step0 权重；每组 step1、step2 各训练 8,000 次，全局 batch8，学习率 2e-5，2000 次 warmup。KD/SEP=0，共同原型像素损失权重0.1，主像素损失权重0.1，confusion reweight 关闭。四组均在融合后将图像框外设为255，这是共用修正，不是 ALD 单独的收益。

启动后只读审计确认：四组53项共同配置、37份训练源及77份数据列表完全匹配记录；四个实际 worker 使用项目私有 Python，各10项缓存环境变量均指向 `.runtime`。GPU 利用率与本 runner 进度每45秒记录在 `runs/ald_fusion_v1/gpu_utilization.jsonl`；启动时物理GPU2/3已有共享任务，随后GPU0/1也出现共享任务，目前四张卡都在共享。记录中的物理总利用率不能全部归为本实验；以本任务进程和逐50次更新记录确认持续训练。保持既定训练预算，不按较慢GPU的墙钟时间提前截断。

原始配置、训练命令、输入 SHA、逐评估指标、逐50次监督计数、阶段退出码及权重都保存在 `runs/ald_fusion_v1/`。每组完成 step1 后立即接自身 step2，不等待其他组。

## 可复现来源

冻结源码：`runs/ald_fusion_v1/source_snapshot.tar.gz`，SHA256：`b4e4298a13b612346ccc6182f206124f0f81db4e87aae4cbe7141eeac090b891`，共127个独立归档文件。每个训练源均与 `study.json` 指纹一致；归档不包含数据图像、conda环境或共享权重。

旧 baseline 的原始源码单独保留在 `runs/fixed_baseline_v1/source_snapshot.tar.gz`，原始结果及权重不变。实验协议见 [ald_fusion_protocol.md](ald_fusion_protocol.md)。

## Warmup 公平性诊断

前400次 warmup 的8个50次区间中，四组图像数、总像素、有效图框像素、旧/新图像级正标签数与过滤前新类 CAM 像素数逐区间完全相等：3,200次图像抽样、642,252,800 crop像素、520,393,152有效像素、2,765/3,854旧/新正标签出现次数、31,104,075过滤前新类像素。三个 ALD 组共拒绝22,387,418个新类 CAM像素；legacy最终保留ignore为0，preserve_rejected为22,387,418，preserve_background为17,940,246。

这证明区间统计一致，不能证明每个batch的图像/增强/完整CAM逐项相同。前2000次像素损失权重为0，这些仅为候选监督去向，不属于性能提升或有效像素优化证据。收据：`runs/ald_fusion_v1/warmup-prefix-verification.json`。

## 已到达的中间评估（不用于选定最终方法）

四组 step1 在 2,000 次 warmup 终点的逐类 IoU 完全相同：旧10类 65.735、新5类 0.000、含背景全部16类 46.500。收据：`runs/ald_fusion_v1/all-arm-warmup-evaluation-verification.json`。这对应像素损失尚未启用的训练前缀，不能解释为 ALD 带来的效果。

| step1 4,000次 | 旧10类 | 新5类 | 含背景全部16类 |
|---|---:|---:|---:|
| off | 75.298 | 61.089 | 71.966 |
| legacy | 74.985 | 56.528 | 70.300 |
| preserve_rejected | 74.055 | 54.942 | 69.254 |
| preserve_background | 75.769 | 55.290 | 70.417 |
| legacy − off（百分点） | −0.312 | −4.561 | −1.666 |

四组均已到达同一4,000次中途评估，此处只选择各组iteration4000记录，不将off/legacy后来8,000次的结果与ignore组4,000次混排。两个ignore候选此时整体指标均低于off；不能据此选定最终方法，仍运行既定每阶段8,000次预算。四臂原始逐类IoU、重算分组、共同配置、读取时日志SHA及UTC：`runs/ald_fusion_v1/step1-4000-allarm-intermediate-verification.json`。早期两臂证据`step1-4000-intermediate-verification.json`保留。

| step1 6,000次 | 旧10类 | 新5类 | 含背景全部16类 |
|---|---:|---:|---:|
| off | 75.930 | 63.411 | 73.087 |
| legacy | 76.278 | 65.168 | 73.864 |
| preserve_rejected | 76.627 | 64.620 | 73.912 |
| preserve_background | 75.874 | 63.816 | 73.193 |

四组已取得同预算6,000次中途评估。原ALD及两ignore方案的整体指标此时都高于off，和4,000次的方向不同。preserve_rejected整体比legacy高0.047点，但旧10类高0.349点、新5类低0.548点，存在取舍；这不能证明终点收益或统计显著性。任何最终选择仍使用每阶段8,000次终点和完整两阶段轨迹。四臂原值/相对off及legacy的差值/观察时日志SHA见`runs/ald_fusion_v1/step1-6000-allarm-intermediate-verification.json`；早期两臂收据`step1-6000-intermediate-verification.json`保留。NEW rescue仍基于门控监督机制提出，不以原ALD最终失败为前提。

第二阶段2,000次warmup中途评估，两组记录现已齐全：

| step2 2,000次 | previous15前景 | 本次新5类 | initial10前景 | 累计新增10类 | 含背景全部21类 |
|---|---:|---:|---:|---:|---:|
| off | 68.881 | 0.000 | 71.531 | 31.790 | 53.490 |
| legacy | 68.620 | 0.000 | 70.407 | 32.522 | 53.294 |

该点最后5类IoU均为0，像素损失系数仍为0；两组分别继承自己step1的不同权重。差值不能单独归因于step2的ALD像素过滤，亦不能跨step1/step2不同验证集合解释为纯遗忘。各自初始checkpoint SHA、逐类IoU、两套类组、源码切片及观察时日志SHA：`runs/ald_fusion_v1/step2-2000-off-legacy-intermediate-verification.json`。完整step2终点尚未完成。

legacy 已启用像素监督的 2050–2550 次区间合计删除 3,773/5,310=71.05% 的 NEW 正类出现次数、拒绝 81,480,990/133,481,828=61.04% 的 NEW CAM 像素；收据：`early-active-gate-diagnostics.json`。这提示 gate 的监督损失可能值得检验，并不证明被拒绝的 CAM 正确。已准备只追加最强正 NEW 的单变量对照，采用自己的 legacy 控制臂，保持共同融合和其余配置相同；见 [NEW gate 协议](ald_gate_new_protocol.md) 与 [验证记录](ald_gate_new_results.md)。

## 已完成的 step1 8,000 次终点

总结工具以明确的 `--partial` 模式审计，2/8正式阶段通过最终checkpoint/指标/计数/配置/输入SHA、真实exit0及来源归档校验；全研究仍未完成。off与legacy已自动启动过自身step2，现随用户暂停要求停止。统计类组全部有有限IoU值：旧10类、新5类、含背景16类。

| step1终点 | 旧10类 | 新5类 | 含背景全部16类 |
|---|---:|---:|---:|
| off | 76.431 | 64.921 | 73.879 |
| legacy | 76.582 | 65.681 | 74.226 |
| legacy − off（百分点） | +0.151 | +0.760 | +0.347 |

两ignore候选尚未到同等终点，step2也尚未完成，不进行完整方法排名。`runs/ald_fusion_v1/summary.json`、`comparison.png/pdf`明确标注未完成，仅显示已经通过审计的阶段终点，不引用中途checkpoint代替终点。

完整step1 active期间（更新2001–8000）两组各6000次更新、48000次图像抽样。legacy拒绝246,691,275/1,774,926,900=13.90%的筛选前NEW CAM像素，等于有效图框像素的3.13%；其中201,575,120由teacher BG填回、45,116,155由teacher OLD填回，最终有效监督覆盖100%。这与早期600次active前缀60.80%的拒绝率差异很大，说明早期诊断不能代表全阶段的平均作用。该计数只衡量标签去向，不证明被删CAM错误或门控随训练变得更准确。

## 相同 active 前缀的监督覆盖率

四组共同日志区间端点2050–2600，对应有效更新2001–2600：每组600次更新、4800次图像抽样、12条记录。逐区间25项整数分区、模式去向、padding为零和像素监督激活均通过。以下比例由每组整数总计求得，未平均日志比例。

| 模式 | 拒绝/筛选前NEW CAM | 拒绝/有效图框 | 有效监督/有效图框 | 最终NEW/有效图框 | 移除NEW类/正NEW类 |
|---|---:|---:|---:|---:|---:|
| off | 0.00% | 0.00% | 100.00% | 19.30% | 0.00% |
| legacy | 60.80% | 11.38% | 100.00% | 7.34% | 70.70% |
| preserve_rejected | 62.33% | 11.59% | 88.41% | 7.01% | 70.99% |
| preserve_background | 63.35% | 11.81% | 89.71% | 6.83% | 71.77% |

legacy把拒绝NEW区域填回teacher OLD/BG，因而监督覆盖仍为100%；两种ignore候选相应减少监督量。各组已接受不同有效梯度，后续CAM和参数可分化，因此不能把比例差当作同一批像素被不同策略分配的精确反事实；它也不衡量CAM准确性或mIoU。收据：`runs/ald_fusion_v1/active-prefix-comparison-verification.json`，含读取时各日志SHA/字节数、选中行SHA与迭代集合，不把进行中日志SHA当作未来终点哈希。

对齐4,000次评估的完整active窗口（更新2001–4000）：每组2000次更新、16000次图像抽样、40条完整50次区间。warmup和后续更新均排除。

| 模式 | 拒绝/筛选前NEW CAM | 拒绝/有效图框 | 有效监督/有效图框 | 最终NEW/有效图框 | 移除NEW类/正NEW类 |
|---|---:|---:|---:|---:|---:|
| off | 0.00% | 0.00% | 100.00% | 21.59% | 0.00% |
| legacy | 36.53% | 7.39% | 100.00% | 12.84% | 55.19% |
| preserve_rejected | 35.85% | 7.37% | 92.63% | 13.19% | 54.12% |
| preserve_background | 34.02% | 7.09% | 93.92% | 13.75% | 53.25% |

该窗口两ignore组的监督覆盖均高于最早600次active前缀，进一步说明筛选强度随训练变化。此处仅把覆盖计数和4,000次性能评估对齐；不同参数轨迹下CAM并不相同，覆盖减少也不等于损失或梯度按相同比例减少，不能据此宣布性能差异的原因。原始整数、teacher OLD/BG分区、fallback OLD/NEW数、选中行SHA与读取时日志SHA见`runs/ald_fusion_v1/active-2001-4000-allarm-verification.json`。

## 门控随迭代的实际轨迹

已用CPU生成step1三面板科研图 `runs/ald_fusion_v1/diagnostics.png/pdf` 及逐点证据 `diagnostics.json`，脚本 `tools/plot_ald_diagnostics.py`。本次更新快照off/legacy到8000，两个ignore组到6150，整图醒目标注 INCOMPLETE TRAJECTORIES；曲线只到各自完整日志，不补齐或外推。warmup阴影表示像素loss为0，此时只是候选标签覆盖。

legacy第一条active区间（端点2050）的NEW正类保留率29.55%，最后一条区间（端点8000）80.38%；对应拒绝NEW像素/筛选前NEW CAM为59.51%与2.75%。整个active阶段合并整数计数后分别是66.05%与13.90%，不能用两端或区间比例均值代替总体。有效图框内监督覆盖始终100%，因为拒绝区域由teacher填回。

这些是标签覆盖的时间变化，不证明CAM正确性、门控校准程度或因果解释。固定raw阈值在不同学习阶段的实际筛选强度明显不同，NEW fallback的后续实验将检验补救规则，而不是从曲线直接宣布其收益。进行中日志的SHA仅绑定本次快照，正式终点仍以completion及SHA链为准。

## 与原论文对照时的口径和稿件问题

原PDF第5页§5.1（稿件行528–529）定义初始类别、累计新增类别及全部已见类别；第6页Table1、第7页Table4、第8页Table5均以1–10 / 11–20 / All标注VOC10-5终点分组。正文Table1完整方法的74.7 / 64.5 / 70.7应按这个类组顺序读取。不能把前两项与本实现的最终previous15/current5直接比较；本报告同时提供initial10/cumulative10和15/5。

论文未明确说明All是否包含背景，Old是否额外包含背景也未明示；本实现的foreground组排除0、All包含0–20共21类。不能把代码定义直接当作原表的明示定义，或把加权反推的背景值作为论文实测值。

原稿附录存在标注冲突：第13页Table9以NewClasses / OldClasses列出同一74.7 / 64.5，第12页稿件行1299–1302也将74.7称为新类、64.5称为旧类，与正文表头顺序冲突。为提高论文可核验性，应统一正文/附录标注，并明确三组mIoU各自类别ID及背景处理；当前不自行判定哪处是笔误。独立只读审查与原PDF SHA：`.runtime/ald-paper-metric-scope-verification.json`。

本次ALD收益仍以同阶段、本研究新训练off为对照；历史固定BASE与论文仅作上下文。即使类号对应，训练组件和评价协议的其他差异也不支持把本次实验称为完整论文复现。

## 待正式结果补齐

报告将记录两阶段 previous/current/all mIoU，以及最终 initial10/cumulative10/all21 口径；整体含背景，组别 foreground 不含背景。覆盖率分别统计 warmup、像素损失有效期间和全部期间，先加总整数计数再求比例。

当前只有单次 seed0、VOC10-5；早期阶段的验证集合与未来类别处理不同，跨阶段下降不能直接解释为遗忘。若候选优于 off，仍需后续种子/任务验证；若未优于 off，将结合真实拒绝比例和旧/新类代价决定下一项单变量改动。
