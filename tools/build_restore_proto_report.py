"""Build the Chinese methods report from verified restoration and OLC records."""
from pathlib import Path
import io
import json
from datetime import datetime, timezone, timedelta
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
c.setTitle('EvoProto 与 OLC：实验方法和验证记录')
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
 c.setFont('CN',9);c.drawString(42,H-35,'EvoProto / 恢复版实验方法');c.drawRightString(W-42,H-35,datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')+' · 实验记录')
 c.setFillColor(HexColor('#157f83'));c.setFont('CN',10);c.drawString(42,H-68,kicker)
 c.setFillColor(HexColor('#123349'));c.setFont('CN',22);c.drawString(42,H-99,title)
 c.setStrokeColor(HexColor('#d6e2e7'));c.line(42,48,W-42,48)
 c.setFont('CN',8);c.setFillColor(HexColor('#637482'));c.drawString(42,32,'结果以已完成记录为准；OLC 与旧 ALD 分别标识。');c.drawRightString(W-42,32,str(n))
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

comparison_path=ROOT/'runs/restore_proto_olc_v1/comparison.json'
comparison_data=json.loads(comparison_path.read_text()) if comparison_path.exists() else {'status':'waiting'}

page(1,'要恢复什么，当前做到哪里','01 / 阅读导引')
text('本轮目标是在保留原论文主线的前提下恢复 EvoProto：先估计哪些类别容易混淆，再通过显式类别原型指导知识保持和类别分离。最终效果由完整增量链路及消融验证，不能只靠保留模块名称来判断。')
head('核心机制')
text('每个类别有一个可学习的 512 维 prototype。图像特征与原型的余弦相似度形成原型预测。主分割头与原型分支共享 decoder 特征，因此原型监督可以影响实际分割特征。主头是消融的共同主指标；另行报告原型分支及固定比例融合。')
text('混淆性建模只使用训练图像的弱标签、CAM、PAR 伪标签和上一阶段教师。得到的混淆关系实际进入 KD 权重和 SEP 对手选择；所有混淆统计停止梯度，不读取增量训练的像素真值。')
head('当前证据')
text('共享初始阶段：主头 83.5556，原型分支 83.3298 mIoU。恢复版第一增量阶段已完成 8,000 步：主头 74.6066，原型分支 73.8913 mIoU。后者在 1,240 张验证图像、15 个前景类加背景上计算，输入为 448 × 448。')
text('无 OLC 完整模型的两阶段增量训练均已完成。最终 21 类主头 mIoU 为 68.7125（square448）和 69.8902（aspect672）；等比例融合为 68.8482 和 70.0644。两组核心消融链路也已完成。')
if comparison_data.get('status')=='complete':
 r=comparison_data['comparisons']['aspect672']
 text('加入 OLC 后，同设备 aspect672 主头为 '+format(r['main']['olc_miou'],'.4f')+'，变化 '+format(r['main']['candidate_minus_reference_pp'],'+.4f')+' 点；固定等比例融合为 '+format(r['fixed_half_prototype_fusion']['olc_miou'],'.4f')+'，变化 '+format(r['fixed_half_prototype_fusion']['candidate_minus_reference_pp'],'+.4f')+' 点。两阶段均为原定 8,000 步，详细对照见第 12 页。')
else:
 text('在线 OLC 的同预算完整增量链仍在验证；阶段记录见第 12 页。')
head('本文如何使用')
text('第 2 页解释符号与原型；第 3 页说明混淆关系；第 4-5 页给出 KD、SEP 与完整损失；第 6 页规定数据和评估协议；第 7 页记录复现、权重保留和论文修改边界；第 8-9 页给出完整模型与核心消融；第 10-12 页说明 OLC、标签诊断及同预算对比。')
text('本报告描述本轮实际代码，而非逐式复现原稿。核心三组实验均关闭 ALD。额外续训的旧 ALD 已被排除；新 OLC 专指原 8,000 步内的旧类图像标签校正。',True)

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
text('共享 step0 的历史训练使用像素标注，因此不能把整个系统描述成从零开始完全不使用像素真值。分割验证仅用像素真值计分，预测不使用验证图像真值标签。另列的 CAM 定位诊断使用验证图像级标签过滤类别，不能与无标签分割混称。')

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
text('ViT-B + 512 维 decoder；每个增量阶段 8,000 次更新；物理 GPU 5、6；每卡 batch 4，总 batch 8。基础学习率为 0.00002，头部倍率 10；使用原工程 PolyWarmupAdamW。训练裁剪 448×448，尺度范围 0.5 至 2.0，随机种子 0。旧 ALD 关闭；OLC 作为同预算附加模块单独比较。')
head('两种推理协议分别报告')
text('square448：把输入缩放为 448×448。aspect672：保持长宽比，目标面积约为 672×672，每条边取最接近的 16 倍数。后者不是“最长边为 672”，也不是本轮把训练裁剪改成 672。')
text('最终在 1,449 张验证图像上计算 21 类 mIoU，同时报告旧前景、新前景和逐类 IoU。主头与原型分支分开报告。分辨率带来的收益不得算成混淆模块或原型的收益。')
head('融合选择与公平消融')
text('另预设主头／原型 sigmoid 概率融合网格：原型占比为 0、0.25、0.5、0.75、1，原型 cosine 先除以训练温度 0.1。已用共享初始权重核查 CPU／GPU 端点差异小于 0.001 个百分点。')
text('完整模型验证网格选择了原型占比 0.5；这属于验证集选型。所有组的主要消融仍比较主头。0.5 融合只用于保留原型监督的组；without_proto 的增量原型未受训练，因此仅使用主头，避免人为拉低对照。')

page(7,'复现、保留权重与论文修改','07 / 结论必须有对应证据')
head('目前可复核的工件')
text('代码分支：codex/restore-proto-4090。训练实现位于 experiments/restore_proto_v1；运行记录位于 runs/restore_proto_v1/formal。manifest 保存代码哈希、种子、预算和初始权重来源；每阶段保存配置、命令、前驱哈希、指标、混淆统计和最终权重回执。')
text('完整模型最终原型参数为 [11,512]、[5,512]、[5,512] 三块。两阶段完整模型严格加载、有限值和前驱 SHA256 核查均通过。单独评估已与训练时验证对齐，主头 mIoU 差异小于 0.001 个百分点。')
head('权重策略')
text('不保存每轮／每次验证的快照，只保存阶段最终权重。本次保留共享 step0，以及无 OLC 主线和 OLC 候选各自的 step1、step2，共五个必要权重。已核验的冗余消融和废弃 ALD 权重在最终比较后清理；数据、日志、指标与核查记录保留。')
head('论文哪些要保留，哪些要改')
text('保留问题设置、teacher/student、显式可学习原型，以及“混淆指导分离与保持”的主线。当前实现改变了混淆估计、对手选择、SEP 和 KD 的具体公式，需要修订原式 (2)-(7)、总损失式 (11) 及相关结构图箭头。')
text('特别是原稿对高混淆类别减弱全局原型 MSE；本轮则仅在可信旧类区域加强条件 KD，并在新类区域关闭或衰减 KD。权重方向与约束范围都不同，不能说原公式原样保留。')
head('结论边界')
text('核心消融均已完成。混淆消融同时改变 SEP 对手与 KD 权重，没有隔离 KD 权重方向。单 seed、反复查看的验证集及融合选型，不能支持独立测试或跨种子显著性声明。旧 ALD 续训结果不纳入当前方法。')
text('无 OLC 主线的 aspect672 融合为 70.0644；主头的混淆收益约 0.82 点，原型增量监督收益约 0.11 点，后者较小，需要谨慎解释。OLC 的完整链收益另见第 12 页；第一阶段的 16 类结果不能替代最终 21 类结果。')
text('来源：本项目实际训练配置与日志、机制源码、checkpoint_audit.json、数据协议核查，以及原稿第 3-5 页。历史结果只作背景参照；本报告未加入未运行的实验成绩。',True)
page(8,'完整模型：已核验的最终阶段结果','08 / 成绩与归因分开报告')
import json
base=ROOT/'runs/restore_proto_v1/formal/full/10-5/step2'
evaluation=json.loads((base/'evaluation.json').read_text())
fusion=json.loads((base/'fusion_evaluation.json').read_text())
verification=json.loads((base/'fusion_verification.json').read_text())
assert evaluation['histogram_label_dtype']=='int64'
assert evaluation['images']==fusion['images']==verification['images']==1449
assert fusion['checkpoint_sha256']==verification['checkpoint_sha256']
assert verification['per_image_histograms_verified']
text('以下均为同一份 full 第二阶段 8,000 步最终权重，在 VOC 验证集 1,449 张图像上计算。全部类别指标含背景，共 21 类；旧前景为 15 类，当前新增前景为 5 类。数值单位为百分数。')
from reportlab.platypus import Table, TableStyle
rows=[['输入 / 预测方式','全部类别','旧前景','新增前景']]
for mode in ('square448','aspect672'):
 for key,label in [('main','主头'),('prototype','原型'),('fusion','0.5 融合')]:
  record=fusion['results'][mode]['0.5'] if key=='fusion' else evaluation['results'][mode][key]
  rows.append([mode+' / '+label]+[format(record[k],'.4f') for k in ('miou','previous_foreground','current_foreground')])
rows=[[Paragraph(rich(cell),small) for cell in row] for row in rows]
table=Table(rows,colWidths=[211,86,86,128-0.72])
table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e2f0f0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#f5f8fa'),HexColor('#ffffff')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9),('LINEBELOW',(0,0),(-1,0),.6,HexColor('#b4cbcf'))]))
_,height=table.wrap(W-84,H)
table.drawOn(c,42,y-height);y-=height+15
head('这些结果能说明什么')
text('保持长宽比和提高推理面积使主头提升 1.1777 个百分点；同一 aspect672 输入下，等比例融合再提升 0.1742 个百分点。这是推理设置的变化，不能代替混淆建模或原型训练收益的消融证据。')
text('原型分支自身可产生接近主头的密集预测，说明它是实际参与预测和优化的分支。完整 without_proto 对照已完成：aspect672 主头差约 0.1080 点，不宜描述为显著或大幅提升。')
head('评估校验与结果范围')
text('已修正独立评估中 uint8 标签乘类别数导致的直方图溢出；改为先转 int64。训练期验证原本使用 long，不受此问题影响。修正后的独立评估与训练验证一致，融合两端也与各单独分支一致。',True)
text('核心消融见下一页。旧 ALD 曾追加 1,200 步，用户已否定这一额外训练方案，因此不与本页 8,000 步主线混排。OLC 的全部新增机制在原训练预算内生效。',True)

def result_table(rows,widths):
 global y
 rows=[[Paragraph(rich(str(cell)),small) for cell in row] for row in rows]
 table=Table(rows,colWidths=widths)
 table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e2f0f0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#f5f8fa'),HexColor('#ffffff')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
 _,height=table.wrap(W-84,H)
 if y-height<65:raise RuntimeError('Result table overflow')
 table.drawOn(c,42,y-height);y-=height+15

page(9,'核心消融：混淆与原型各贡献多少','09 / 相同预算，完整增量链')
text('以下三组均从同一 step0 出发，各训练两个 8,000 步增量阶段；step2 继承本组 step1。主要对比统一采用主头，避免把未训练的增量原型混入 without_proto 结果。')
rows=[['方法','square448 主头','aspect672 主头','aspect672 融合']]
for variant,label in [('full','完整主线'),('without_confusion','关闭混淆'),('without_proto','关闭增量原型监督')]:
 d=ROOT/'runs/restore_proto_v1/formal'/variant/'10-5/step2'
 e=json.loads((d/'evaluation.json').read_text());f=json.loads((d/'fusion_evaluation.json').read_text())
 rows.append([label,format(e['results']['square448']['main']['miou'],'.4f'),format(e['results']['aspect672']['main']['miou'],'.4f'),'-' if variant=='without_proto' else format(f['results']['aspect672']['0.5']['miou'],'.4f')])
result_table(rows,[170,111,115,115])
head('混淆建模')
text('主头收益为 square448 的 0.7120 点和 aspect672 的 0.8221 点；固定 0.5 融合的 aspect672 收益为 0.7713 点。该对照支持整个混淆模块在本轮设置中的作用，但同时改变 KD 加权和 SEP 对手，不能分别归因。')
text('高混淆类别上调 KD 的前提是教师可靠：本实现先排除新类、低旧类 CAM 和低教师置信度区域。但这一权重方向仍是研究假设；若要单独证明，需要保留 SEP 不变，仅比较 KD 的上调、恒定或下调。')
head('原型监督')
text('第一增量阶段，完整主头较关闭增量原型监督高 1.0189 点；最终阶段差距缩小为 square448 的 0.2413 点、aspect672 的 0.1080 点。原型确实参与优化和预测，但最终主头收益很小，不能夸大。')
text('按验证图像配对 bootstrap，最终 aspect672 主头的混淆差值区间约为 [0.0524, 1.6201]，原型差值区间为 [-0.7971, 1.0918]。这些区间只反映验证图像抽样，不反映训练种子波动；验证集已被多次查看。',True)

page(10,'OLC：训练过程中校正旧类标签','10 / Old-class Label Correction')
text('OLC 专门处理“这张图里是否存在某个旧类”的错误判断。它从第一步开始工作，沿用冻结的上一阶段 teacher，复用主、辅助图像分类头，不添加额外模型，也不追加训练。')
text('i 表示训练图像，c 表示旧前景类别，v 表示这张图被再次采样的次数。g 是图像分类 logit；h 为主图像分类头 main 或辅助头 aux。r 是 sigmoid 概率，横线表示该图像的概率记忆。这里的 g 不同于前文像素 logit z。')
eq(r'r^{h,(v)}_{ic}=\operatorname{sigmoid}(g^h_c(x_i^{(v)})),\quad h\in\{\mathrm{main},\mathrm{aux}\}')
eq(r'\bar r^{h,(v)}_{ic}=0.5\bar r^{h,(v-1)}_{ic}+0.5r^{h,(v)}_{ic}')
text('首次访问直接用当前概率初始化。以后每次随机缩放、翻转和裁剪产生新的预测并更新记忆；teacher 权重不变。两卡共享相同记忆，同一全局 batch 中重复图像先取平均，再更新一次。')
head('正、负、未知三种状态')
eq(r'\pi^+_{ic}=\mathbf{1}[\bar r^{\mathrm{main}}_{ic}>0.5]\,\mathbf{1}[\bar r^{\mathrm{aux}}_{ic}>0.5]')
eq(r'\pi^-_{ic}=\mathbf{1}[\bar r^{\mathrm{main}}_{ic}\leq0.5]\,\mathbf{1}[\bar r^{\mathrm{aux}}_{ic}\leq0.5]')
text('正类指示量为 1 表示旧类存在，负类指示量为 1 表示不存在；两者都为 0 时为未知。不能把未知当成负类。0.5 的记忆系数是本轮预先固定的简单候选，尚未证明最优。')
head('怎样进入训练')
text('主、辅助分类 BCE 对未知旧类置零，保留原始 batch × 类别数分母，新类图像真值照常使用。未知类保留 CAM 竞争资格，但其旧类获胜标签不作硬监督。teacher 给出的旧类像素标签仅在该图像的旧类被接纳为正类时保留，否则置为 ignore，而非背景；有效新类伪标签优先。')
text('混淆图、原型、KD 和 SEP 的公式不改，只将其原有允许类别输入换为校正后的可信正类。OLC 会通过标签质量影响这些模块，因此并非统计上完全独立。',True)

page(11,'标签准确性：看精确率，也看遗漏','11 / 独立诊断，禁止用旧类真值训练')
text('静态诊断使用本阶段训练图像的完整图像缩放输入。预测完成后，才在独立程序中拼接旧类图像真值计分；这些真值不进入 OLC 记忆或训练。此诊断用于判断设计是否有希望，不能直接证明随机裁剪下的在线 EMA 有效。')
rows=[['阶段 / 标签规则','精确率','召回率','覆盖率']]
for stage in (1,2):
 d=json.loads((ROOT/f'runs/online_ald_label_audit_v1/step{stage}.json').read_text())
 for key,label in [('main_threshold_0.5','原主头'),('dual_head_consensus','双头一致')]:
  r=d['policies'][key];rows.append([f'Step {stage} / {label}']+[format(100*r[k],'.2f') for k in ['precision','recall','coverage']])
result_table(rows,[208,101,101,101])
text('精确率：接纳的旧类正标签中有多少为真；召回率：真实存在的旧类有多少被接纳，未知中的真实正类也计为遗漏；覆盖率：所有图像-旧类组合中，有多少作出了正或负判断。只看被接纳标签的准确率，会掩盖大量弃用标签的问题。')
head('训练内在线记忆诊断')
rows=[['阶段 / 更新 / 规则','精确率','召回率','F1','覆盖率']]
for stage in (1,2):
 directory=ROOT/f'runs/restore_proto_olc_v1/formal/olc/10-5/step{stage}'
 files=list(directory.glob('olc_label_audit_*.json'))
 if files:
  d=max((json.loads(p.read_text()) for p in files),key=lambda r:r['iteration'])
  for key,label in [('main_head_same_ema','仅主头'),('olc_memory','OLC')]:
   r=d['policies'][key]
   rows.append([f'{stage} / {d["iteration"]} / {label}']+[format(100*r[k],'.2f') for k in ['precision','recall','f1','coverage']])
 else:rows.append([f'{stage} / 待诊断','-','-','-','-'])
result_table(rows,[191,80,80,80,80])
text('在线表中的“仅主头”与 OLC 共用同一预测记忆，用于观察双头一致筛选的取舍。上表采用完整图像，下表记忆来自随机裁剪；Step 2 两表的教师也来自不同的 Step 1 链路，不能将上下表差异全部归因于 EMA。本实验没有独立隔离 EMA 的训练收益。',True)
text('两个冻结的教师头仍可能同时误判。历史负证据也可能延迟接纳目标：旧记忆为 0.02、当前双头均为 0.90 时，平均后只有 0.46，仍判负类。这是规则的滞后现象，不足以证明某一类别回落的原因。在线更新不保证标签质量单调提高；是否采用应结合完整链分割结果。',True)

page(12,'OLC 是否改善最终分割','12 / 同预算验证与采用条件')
comparison=ROOT/'runs/restore_proto_olc_v1/comparison.json'
data=json.loads(comparison.read_text()) if comparison.exists() else {'status':'waiting'}
if data.get('status')=='complete':
 text('两条完整增量链的预算、配置、前驱、最终权重和验证图像均已核对。OLC 在 Step 1 的 square448 主头提升 0.6827 点；下表为最终 21 类的同 GPU 配对结果，单位为 mIoU 百分数，变化列为 OLC 减无 OLC。')
 rows=[['输入 / 预测方式','无 OLC','加 OLC','变化']]
 for mode in ['square448','aspect672']:
  for head_key,label in [('main','主头'),('fixed_half_prototype_fusion','0.5 融合')]:
   r=data['comparisons'][mode][head_key]
   rows.append([mode+' / '+label,format(r['baseline_miou'],'.4f'),format(r['olc_miou'],'.4f'),format(r['candidate_minus_reference_pp'],'+.4f')])
 result_table(rows,[208,101,101,101])
 r=data['comparisons']['aspect672']['main']
 text('aspect672 主头：旧前景变化 '+format(r['old_delta_pp'],'+.4f')+' 点，新前景变化 '+format(r['new_delta_pp'],'+.4f')+' 点。配对图像 bootstrap 区间为 ['+', '.join(format(v,'.4f') for v in r['paired_image_bootstrap_95_percentile_interval'])+']。这不是跨训练种子的显著性证明。')
else:
 text('OLC 完整增量链或最终验证尚未全部完成，因此暂不能判断是否改善最终分割。下面只列已经写入验证日志的最新阶段记录，不将中途数字当成最终效果。')
 rows=[['阶段 / 更新','无 OLC','OLC','主头变化','原型','CAM / 辅助 CAM']]
 for stage in [1,2]:
  p=ROOT/f'runs/restore_proto_olc_v1/formal/olc/10-5/step{stage}/metrics.jsonl'
  if p.exists():
   r=json.loads(p.read_text().splitlines()[-1])
   baseline_rows=[json.loads(line) for line in (ROOT/f'runs/restore_proto_v1/formal/full/10-5/step{stage}/metrics.jsonl').read_text().splitlines()]
   b=next(row for row in baseline_rows if row['iteration']==r['iteration'])
   rows.append([f'Step {stage} / {r["iteration"]}',format(b['all_miou'],'.3f'),format(r['all_miou'],'.3f'),format(r['all_miou']-b['all_miou'],'+.3f'),format(r['prototype_miou'],'.3f'),format(r['cam_miou'],'.3f')+' / '+format(r['auxiliary_cam_miou'],'.3f')])
  else:rows.append([f'Step {stage} / 待验证','-','-','-','-','-'])
 result_table(rows,[108,72,72,74,68,117])
head('本次结论与实验边界')
if data.get('status')=='complete':
 text('本次 OLC 提高了旧类标签精确率和 F1，但两个输入协议的最终主头均未提升。固定融合的 aspect672 结果下降约 0.20 点，四项配对图像区间均包含 0。建议正式主线暂时保留无 OLC 的原型恢复版，将当前 OLC 作为实验候选；不能把标签更准写成分割增益，也不能据此声称显著退化。')
else:
 text('先检查旧类标签精确率是否提升，并同时检查召回率、覆盖率和错判为负类的数量。再看同输入协议下的最终分割及旧／新类别表现，不能只展示标签指标。')
text('无 OLC 的完整主线 aspect672 主头为 69.8902，固定半原型融合为 70.0644。OLC 不重新搜索融合比例，不追加 1,200 步，不引入同阶段参考模型。只在这些条件一致时讨论新增收益。')
head('复现检查与命名')
text('实现：experiments/restore_proto_olc_v1。双卡记忆同步、重复图像合并、未知标签零分类梯度及两阶段 GPU 冒烟通过。关闭 OLC 的冒烟差约 0.00214 mIoU 点；不修改代码的原入口重复运行差约 0.00172 点，因此没有逐位确定性承诺。',True)
text('旧 ALD 的额外续训方案仅作历史记录。新 OLC 只指本报告定义的在线旧类标签校正；两者不能混用名字、训练预算或实验成绩。最终方法的共享初始权重、必要前驱和最终权重均受保护。',True)

c.save()
print(OUT)
