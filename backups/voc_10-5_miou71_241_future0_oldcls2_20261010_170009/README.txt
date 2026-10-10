VOC 10-5：总 mIoU 71.241307，step2 第 4000 次验证。

source/ 为实际实验使用的 Python 与 shell 源码，code_sha256.json 记录原文件哈希。
training_config.json 记录训练超参数及参考结果。
step0 未来类别作为背景 0；增量阶段旧类图像候选采用严格 old_cls > 2.0。
ALD 阈值为 0，像素 CE 权重为 0.2，原型分离采用 hinge，CPA 保持旧前景原型方向。
本目录不含运行元数据、数据文件、日志、预训练文件或训练 checkpoint。
