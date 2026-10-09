"""Build the Chinese interim methods report from verified restoration records."""
from pathlib import Path
import io
from html import escape
from itertools import groupby
from fontTools.ttLib import TTFont as FontInfo
import matplotlib
matplotlib.use('Agg')
from matplotlib import mathtext
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/evoproto_restoration_methods.pdf'
assert OUT.parent.resolve().is_relative_to(ROOT)
OUT.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('CN','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
CN_CHARS=FontInfo('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf').getBestCmap()
def runs(value):
 for font,chars in groupby(value,lambda ch: 'CN' if ord(ch)>127 and ord(ch) in CN_CHARS else 'Latin'):
  yield font,''.join(chars)
def rich(value):
 return ''.join('<font name="'+font+'">'+escape(part)+'</font>' for font,part in runs(value))
W,H=595.28,841.89
c=canvas.Canvas(str(OUT),pagesize=(W,H))
c.setTitle('EvoProto 恢复版：实验方法与阶段记录')
c.setAuthor('EvoProto experiment record')
original_draw=c.drawString
def mixed_draw(x,y,value):
 size=c._fontsize
 for font,part in runs(value):
  c.setFont(font,size);original_draw(x,y,part)
  x+=pdfmetrics.stringWidth(part,font,size)
def mixed_right(x,y,value):
 width=sum(pdfmetrics.stringWidth(part,font,c._fontsize) for font,part in runs(value))
 mixed_draw(x-width,y,value)
c.drawString=mixed_draw
c.drawRightString=mixed_right
body=ParagraphStyle('body',fontName='CN',fontSize=10.5,leading=18,wordWrap='CJK',textColor=HexColor('#25364a'))
small=ParagraphStyle('small',parent=body,fontSize=9,leading=15)
y=0

def page(n,title,kicker):
 global y
 if n>1:c.showPage()
 c.setFillColor(HexColor('#123349'));c.rect(0,H-10,W,10,fill=1,stroke=0)
 c.setFont('CN',9);c.drawString(42,H-35,'EvoProto / 恢复版实验方法');c.drawRightString(W-42,H-35,'2026-10-09 · 中期记录')
 c.setFillColor(HexColor('#157f83'));c.setFont('CN',10);c.drawString(42,H-68,kicker)
 c.setFillColor(HexColor('#123349'));c.setFont('CN',22);c.drawString(42,H-99,title)
 c.setStrokeColor(HexColor('#d6e2e7'));c.line(42,48,W-42,48)
 c.setFont('CN',8);c.setFillColor(HexColor('#637482'));c.drawString(42,32,'训练进行中；阶段成绩不等于最终 20 类成绩。');c.drawRightString(W-42,32,str(n))
 y=H-127

def text(s,compact=False):
 global y
 p=Paragraph(rich(s),small if compact else body);_,h=p.wrap(W-84,H)
 if y-h<65:raise RuntimeError('Page overflow: '+s[:40])
 p.drawOn(c,42,y-h);y-=h+9

def head(s):
 global y
 y-=5;c.setFont('CN',13);c.setFillColor(HexColor('#157f83'));c.drawString(42,y-14,s);y-=27

def eq(s):
 global y
 data=io.BytesIO();mathtext.math_to_image('$'+s+'$',data,dpi=260,format='png',color='#123349')
 data.seek(0);image=ImageReader(data);iw,ih=image.getSize();width=min(iw*72/260,W-108);height=width*ih/iw
 if y-height<65:raise RuntimeError('Formula overflow')
 c.drawImage(image,54,y-height,width=width,height=height,mask='auto');y-=height+15

page(1,'要恢复什么，当前做到哪里','01 / 阅读导引')
text('本轮目标是在保留原论文主线的前提下恢复 EvoProto：先估计哪些类别容易混淆，再通过显式类别原型指导知识保持和类别分离。最终效果由完整增量链路及消融验证，不能只靠保留模块名称来判断。')
head('核心机制')
text('每个类别有一个可学习的 512 维 prototype。图像特征与原型的余弦相似度形成原型预测。主分割头与原型分支共享 decoder 特征，因此原型监督可以影响实际分割特征。默认结果仍使用主分割头，另行报告原型分支成绩。')
text('混淆性建模只使用训练图像的弱标签、CAM、PAR 伪标签和上一阶段教师。得到的混淆关系实际进入 KD 权重和 SEP 对手选择；所有混淆统计停止梯度，不读取增量训练的像素真值。')
head('当前证据')
text('共享初始阶段：主头 83.5556，原型分支 83.3298 mIoU。恢复版第一增量阶段已完成 8,000 步：主头 74.6066，原型分支 73.8913 mIoU。后者在 1,240 张验证图像、15 个前景类加背景上计算，输入为 448 × 448。')
text('第一阶段最终权重已通过完整模型加载、参数有限值、原型形状与来源哈希核查。第二阶段已继承这份权重开始训练。最终 20 个前景类加背景的成绩，以及两组消融的最终成绩，尚未产生。')
head('本文如何使用')
text('第 2 页解释符号与原型；第 3 页说明混淆关系；第 4-5 页给出 KD、SEP 与完整损失；第 6 页规定数据和评估协议；第 7 页记录复现、权重保留和论文修改边界。')
text('本报告描述本轮实际代码，而非逐式复现原稿。ALD 在当前三组实验中关闭；历史 V9 的最终阶段细化结果不直接作为本轮完整增量链的证据。',True)

page(2,'符号、类别原型与预测','02 / 从一个像素开始')
text('像素位置用 x 表示，类别用 a、b、c 表示。背景编号为 0；O 是本阶段已知的旧前景类别集合，N 是本阶段新增前景类别集合。旧类别数记为 |O|，这里不包含背景。')
text('f(x) 是 decoder 输出的 512 维像素特征；p(c) 是类别 c 的 512 维可学习原型。上标 S 表示当前 student，上标 T 表示冻结的上一阶段 teacher。带帽变量表示按向量长度归一化；epsilon 是防止除零的极小正数，尖括号表示向量内积。')
eq(r'\widehat f(x)=\frac{f(x)}{\max(\|f(x)\|_2,\epsilon)},\qquad \widehat p_c=\frac{p_c}{\max(\|p_c\|_2,\epsilon)}')
eq(r'z^{P}_c(x)=\frac{\langle\widehat f(x),\widehat p_c\rangle}{0.1}')
text('z 的上标 P 表示原型预测，M 表示主分割头预测。主头由可学习卷积分类器产生 logits；原型分支由特征和原型相似度产生 logits。原型辅助 BCE、原型 KD 和原型 SEP 都可以更新原型与共享特征。')
head('原型如何获取')
text('本轮没有引入额外聚类或验证集初始化。初始阶段已经学习的旧原型随权重继承；新增类别原型使用固定随机种子初始化，再随弱监督训练优化。teacher 原型固定，student 原型可更新。')
head('训练标签如何使用')
text('增量阶段只保留本阶段新增类别的图像级真值标签；旧类别图像标签由 teacher 预测替代。主、辅助 CAM 结合这些允许类别生成伪标签，并经 PAR 精炼。原型像素 BCE 与主头像素 BCE 使用伪标签。')
text('共享 step0 的历史训练使用像素标注，因此不能把整个系统描述成从零开始完全不使用像素真值。验证阶段使用像素真值计算指标，但预测本身不使用验证图像级真值标签。')

page(3,'混淆关系如何变成可用的图','03 / 核心创新的实际入口')
text('先筛选可信锚点：主、辅助 CAM 的最大类别一致，归一化响应都不低于 0.7，类别在图像允许标签中，且 PAR 标签相同。旧类锚点还要求上一阶段 teacher 的密集预测一致；padding 区域排除。')
text('若可信锚点属于 a，而 student 主头把它判成其他前景类别 b，就记录一次 a → b 混淆。每批统计这类像素数 n(a,b) 和属于 a 的可信锚点数 m(a)，跨两卡汇总后更新：')
eq(r'E_{ab}\leftarrow0.99E_{ab}+0.01n_{ab},\quad M_a\leftarrow0.99M_a+0.01m_a')
eq(r'C_{ab}=\frac{E_{ab}}{\max(M_a,10^{-6})}\,\mathbf{1}[U_{ab}\geq3],\quad a\ne b')
text('U(a,b) 是该有向混淆得到支持的不同训练图片数。重复裁剪同一张图片不会增加独立图片支持数。背景行、背景列和对角线均置零；图不要求对称。')
head('SEP：选谁作为对手')
text('每个锚点类别只保留 C 中最多两个最大的受支持对手，再将保留边除以该行边权之和（分母下限为 0.000001），记为 G。若该行无有效边，SEP 不对该类施加对手约束。旧-旧、新-新和旧-新关系都允许。')
head('KD：哪里更值得保留')
eq(r'd_c=1+\operatorname{clip}\!\left(\sum_b C_{cb}+\sum_a C_{ac},0,2\right)')
text('d(c) 表示类别参与混淆的程度。它只在下一页定义的可靠旧类区域中放大 KD，不能覆盖新类排除规则。SEP 用稀疏行归一化图 G，KD 用入边与出边共同构成的 d。')
text('实现依据：experiments/restore_proto_v1/mechanism.py。每步同步统计，保存 EMA、锚点量和不同图片支持数；独立测试验证改变混淆关系会改变优化目标。',True)

page(4,'KD：只在可靠旧类区域保持知识','04 / 原型预测与主头预测共同蒸馏')
text('令 t(x) 为 teacher 密集预测类别。可靠旧类位置必须满足：有效非 padding 像素、t(x) 是旧前景类、对应旧类 CAM 至少 0.25，且 PAR 没有把该处标为当前新类。记这一门控为 I(x)。')
text('u(x) 是 t(x) 对应旧类 CAM，v(x) 是当前新类 CAM 的最大响应；s(x) 是 teacher 主头获胜类别 logit 的 sigmoid 值。可靠度与混淆加权为：')
eq(r'w(x)=I(x)\,[\max(2s(x)-1,0)]^2\,u(x)\,[1-v(x)]^2\,d_{t(x)}')
text('因此 teacher 不自信或新 CAM 很强时，KD 会减弱；PAR 明确指向新类时，KD 完全关闭。背景和新类不进入条件类别分布，避免直接把新类概率压回旧类别。')
head('只在旧前景类别之间做条件 KL')
eq(r'q^{h}_{c}(x)=\frac{\exp(z^{h}_c(x)/\tau)}{\sum_{j\in O}\exp(z^{h}_j(x)/\tau)},\quad c\in O,\quad\tau=2')
eq(r'\ell_h(x)=\tau^2\sum_{c\in O}q^{h,T}_{c}(x)\log\frac{q^{h,T}_{c}(x)}{q^{h,S}_{c}(x)}')
text('h 分别取主头 M 或原型头 P，得到两项 KD。teacher 全部停止梯度；原型 KD 作用于原型产生的像素分布，因此会反向更新 student 原型及特征。温度用小写希腊字母 tau 表示，避免与 teacher 上标 T 混用。')
head('避免像素多的类别占据全部损失')
text('令 n(c) 是本批两卡中由 I 选出的 teacher 类别 c 的像素数。按其平方根平衡：')
eq(r'L^{h}_{KD}=\frac{\sum_x\ell_h(x)w(x)/\sqrt{\max(n_{t(x)},1)}}{\max(\sum_{c\in O}\sqrt{n_c},1)}')
text('没有可靠像素的类别贡献为零。实现同步全局计数并修正分布式梯度尺度；双卡损失和梯度已与合并 batch 的计算对照。',True)

page(5,'SEP 与完整优化目标','05 / 只分开真正混淆的预测')
text('对于可信锚点 x，其类别为 a(x)。G(a,b) 是混淆图选择并归一化后的对手权重。原型头与主头分别计算：')
eq(r'\ell^{h}_{SEP}(x)=\sum_bG_{a(x)b}\,\max(0,0.5-z^h_{a(x)}(x)+z^h_b(x))')
text('当正确类别的 logit 已比对手高至少 0.5 时，该对手不再产生推动。这个约束关注可信位置上的预测差异，不要求所有类别原型彼此正交。')
text('SEP 也按类别有效锚点数的平方根平衡；只有 G 行非空的可信锚点计入归一化。得到主头分离项和原型分离项。另加轻量旧原型方向稳定项：')
eq(r'L_{dir}=\frac{1}{|O|}\sum_{c\in O}\left(1-\langle\widehat p_c^S,\widehat p_c^T\rangle\right)')
head('完整损失与训练时序')
text('前 2,000 步只优化主、辅助图像分类损失和 0.2 倍 PTC。PTC 是原工程的像素特征一致性项，用伪标签像素对拉近同类、分离异类。本轮没有把混淆 KD／SEP 提前加入 warmup。')
eq(r'L_{base}=L_{cls}+L_{cls,aux}+0.2L_{PTC}+0.1L_{seg}^{M}+0.1L_{seg}^{P}')
eq(r'L=L_{base}+0.10L_{KD}^{M}+0.05L_{KD}^{P}')
eq(r'\qquad+0.05L_{SEP}^{P}+0.02L_{SEP}^{M}+0.01L_{dir}')
text('从第 2,001 步起使用上述完整目标。分割 BCE 使用伪标签。各系数是本轮预先确定的初始候选，不应描述成已知最优参数。')
head('两组必要消融')
text('without_confusion：对手改为全部其他前景类的均匀权重，KD 的 d 固定为 1，其他保持一致。比较的是整个混淆选择与加权模块。')
text('without_proto：增量阶段关闭原型 BCE、原型 KD／SEP 与方向项，保留主头 KD／SEP 和混淆图。共享 step0 曾使用原型监督，模型结构仍含原型参数，所以不能把它叫作从头训练的完全无原型模型。')

page(6,'数据、训练预算与评估口径','06 / 能公平比较，才有结论')
head('VOC 10-5 设置')
text('共享 step0 学习前 10 个前景类和背景；step1、step2 各新增 5 类。初始权重来自既有固定基线的 20,000 步最终模型，三组实验使用完全相同的起点。每组 step2 必须继承自己的 step1。')
text('本机数据：/data/zhonggai/coco/PascalVOC12。训练图片数依次为 6,139／5,542／2,145；对应验证图片数为 869／1,240／1,449。已经核查 split 内无重复、各训练集合与最终验证集无交集。')
head('固定训练配置')
text('ViT-B + 512 维 decoder；每个增量阶段 8,000 次更新；物理 GPU 5、6；每卡 batch 4，总 batch 8。基础学习率为 0.00002，头部倍率 10；使用原工程 PolyWarmupAdamW。训练裁剪 448×448，尺度范围 0.5 至 2.0，随机种子 0。ALD 当前关闭。')
head('两种推理协议分别报告')
text('square448：把输入缩放为 448×448。aspect672：保持长宽比，目标面积约为 672×672，每条边取最接近的 16 倍数。后者不是“最长边为 672”，也不是本轮把训练裁剪改成 672。')
text('最终在 1,449 张验证图像上计算 21 类 mIoU，同时报告旧前景、新前景和逐类 IoU。主头与原型分支分开报告。分辨率带来的收益不得算成混淆模块或原型的收益。')
head('推理融合仅作为额外诊断')
text('另预设主头／原型 sigmoid 概率融合网格：原型占比为 0、0.25、0.5、0.75、1，原型 cosine 先除以训练温度 0.1。已用共享初始权重核查 CPU／GPU 端点差异小于 0.001 个百分点。')
text('初始阶段融合的最大变化仅约 +0.005 个百分点，不能据此预言最终收益。若最终选择融合比例，须披露验证集选择，并对比较组应用一致设置，同时保留原始主头成绩。')

page(7,'复现、保留权重与论文修改','07 / 结论必须有对应证据')
head('目前可复核的工件')
text('代码分支：codex/restore-proto-4090。训练实现位于 experiments/restore_proto_v1；运行记录位于 runs/restore_proto_v1/formal。manifest 保存代码哈希、种子、预算和初始权重来源；每阶段保存配置、命令、前驱哈希、指标、混淆统计和最终权重回执。')
text('第一阶段原型参数为 [11,512] 和 [5,512] 两块，完整模型严格加载通过，参数均为有限值。第二阶段配置中的前驱路径与 SHA256 和已核查的第一阶段权重一致。')
head('权重策略')
text('不保存每轮／每次验证的快照，只保存阶段最终权重。最终采用方法的共享 step0、step1 前驱和 step2 最终模型一起保护。低性能候选只在评估完成、确认无后续依赖后再清理；日志、指标和核查记录保留。')
head('论文哪些要保留，哪些要改')
text('保留问题设置、teacher/student、显式可学习原型，以及“混淆指导分离与保持”的主线。当前实现改变了混淆估计、对手选择、SEP 和 KD 的具体公式，需要修订原式 (2)-(7)、总损失式 (11) 及相关结构图箭头。')
text('特别是原稿对高混淆类别减弱全局原型 MSE；本轮则仅在可信旧类区域加强条件 KD，并在新类区域关闭或衰减 KD。权重方向与约束范围都不同，不能说原公式原样保留。')
head('尚未完成与结论边界')
text('当前尚待完整第二阶段、两条消融链路、最终推理协议对照与最终候选选择。单 seed 和验证集选型不能支持统计显著性声明。已有单元测试与分布式梯度核查证明实现路径有效，但不能代替最终性能消融。')
text('研究目标仍是保留核心机制并争取同协议接近 71 mIoU。是否达到、混淆和原型分别贡献多少，将以完整结果为准。现阶段不得把 74.61 的第一阶段成绩当成最终达标。')
text('来源：本项目实际训练配置与日志、机制源码、checkpoint_audit.json、数据协议核查，以及原稿第 3-5 页。历史结果只作背景参照；本报告未加入未运行的实验成绩。',True)
c.save()
print(OUT)
