import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 3 x 3 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)

def conv1x1(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 1 x 1 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, dilation=dilation, bias=False)

class IncrementalConv(nn.ModuleList):
    def forward(self, input):
        out = []
        for mod in self:
            out.append(mod(input))
        sem_logits = torch.cat(out, dim=1)
        return sem_logits

    def init_weights(self):
        first = self[0]  # nn.Conv2d
        last = self[-1]  # nn.Conv2d

        if isinstance(first, nn.Conv2d) and isinstance(last, nn.Conv2d):
            with torch.no_grad():
                ref_weight = first.weight[0].clone()
                for i in range(last.weight.shape[0]):
                    last.weight[i] = ref_weight.clone()
                if first.bias is not None and last.bias is not None:
                    ref_bias = first.bias[0].item()
                    last.bias.fill_(ref_bias)

class Prototype(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super(Prototype, self).__init__()
        # 使用 nn.Parameter 来定义学习的原型
        self.prototype = nn.Parameter(F.normalize(torch.randn(num_classes, embed_dim), dim=1))

    def forward(self):
        return self.prototype


class IncrementalPrototypes(nn.ModuleList):
    def __init__(self, classes_list, embed_dim):
        super(IncrementalPrototypes, self).__init__()
        for i, num_classes in enumerate(classes_list):
            # 将每个类别的原型封装成 Prototype 对象
            self.append(Prototype(num_classes, embed_dim))

    def forward(self):
        # 这里可以选择返回所有原型
        return [mod() for mod in self]


class LargeFOV(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=5, classes_list=None):
        super(LargeFOV, self).__init__()
        self.embed_dim = 512
        self.dilation = dilation
        self.conv6 = conv3x3(in_planes=in_planes, out_planes=self.embed_dim, padding=self.dilation, dilation=self.dilation)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = conv3x3(in_planes=self.embed_dim, out_planes=self.embed_dim, padding=self.dilation, dilation=self.dilation)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = IncrementalConv(
            [conv1x1(in_planes=self.embed_dim, out_planes=c, padding=0) for c in
             classes_list]
        )

        # 新增：可学习类别原型 [num_classes, embed_dim]
        self.classes_list = classes_list
        self.class_prototypes = IncrementalPrototypes(classes_list, self.embed_dim)
        if len(classes_list) > 1:
            self.conv8.init_weights()

    def _init_weights(self):
        # 初始化 conv6, conv7
        # for m in [self.conv6, self.conv7]:
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(last.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

        # 初始化 conv8 中的每个 1x1 卷积
        self.conv8.init_weights()

    def forward(self, x, cal_sim=False):
        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        out = self.conv8(x)

        # 计算相似度
        B, C, H, W = x.shape  # [B, embed_dim, H, W]
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, embed_dim]

        # 正则化特征和所有任务的原型
        x_flat = F.normalize(x_flat, dim=1)

        # 获取所有任务的类别原型，并进行正则化
        prototypes = torch.cat([F.normalize(self.class_prototypes[t](), dim=1) for t in range(len(self.classes_list))],
                               dim=0)

        # 计算相似度 [B*H*W, total_num_classes]
        sim_logits = torch.matmul(x_flat, prototypes.t())

        # 重新 reshape 回 [B, total_num_classes, H, W]
        total_num_classes = sum(self.classes_list)
        sim_logits = sim_logits.view(B, H, W, total_num_classes).permute(0, 3, 1, 2)

        # # ----- 原型对比损失 -----
        # proto_sim = torch.matmul(prototypes, prototypes.t())  # [N, N]
        # identity = torch.eye(proto_sim.size(0), device=proto_sim.device)
        # proto_contrastive_loss = ((proto_sim - identity) ** 2).mean()
        if not cal_sim:
            return out, sim_logits, prototypes
        else:
            return out, sim_logits, prototypes


class ASPP(nn.Module):
    def __init__(self, in_planes, out_planes, atrous_rates=[6, 12, 18, 24]):
        super(ASPP, self).__init__()
        for i, rate in enumerate(atrous_rates):
            self.add_module("c%d"%(i), nn.Conv2d(in_planes, out_planes, 3, 1, padding=rate, dilation=rate, bias=True))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
        return None
    def forward(self, x):
        return sum([stage(x) for stage in self.children()])
