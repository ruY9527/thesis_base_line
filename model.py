import torch
import torch.nn.functional as F
from torch import nn

'''
    双分支、带通道注意力和多层残差连接的卷积融合网络
    先把低分辨率高光谱图像插值放大，再利用两路输入学习需要补充的修正量
    ChannelAttention	根据当前特征，为每个特征通道生成一个权重
    ResBlock	        卷积提取特征，经过通道注意力后加回输入
    Net	                组织两路输入、特征融合、深层重构和最终输出
'''

'''
    如何判断哪些特征通道更值得保留
    先给每张特征图生成一个整体摘要，再根据摘要决定通道权重。
'''
class ChannelAttention(nn.Module):
    """通道注意力：空间全局平均池化 -> LayerNorm -> MLP 重加权各通道"""
    # 注意力对所有通道进行加权，并没有筛选后只留下高权重通道。低权重通道的响应被压低，仍然参与后续计算
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = x.mean(dim=(2, 3))                     # (B, C) 空间全局平均池化
        w = self.mlp(self.norm(w))                 # (B, C)
        return x * w.unsqueeze(-1).unsqueeze(-1)

'''
    学习局部修正，并保留原始特征
'''
class ResBlock(nn.Module):
    """残差块：两层 3x3 卷积 + 通道注意力"""

    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act = nn.GELU()
        self.att = ChannelAttention(dim)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))
        return x + self.att(out)


'''
    各模块分别搭建什么
    
'''
class Net(nn.Module):
    """
    LR-HSI / HR-MSI 融合网络（双分支特征提取 + 残差融合 + 全局残差基线）

    输入:
        lrhs: (B, spn, h, w)  低分辨率高光谱图像 (CAVE: spn=31)
        hrms: (B, ch,  H, W)  高分辨率多光谱图像 (RGB: ch=3), H = h * scale
    输出:
        (B, spn, H, W)  重建的高分辨率高光谱图像

    调用约定与 train_cave_M.py 一致:
        cnn = Net(31, 3); output = cnn(lr_hsi, hr_msi)
    spn	高光谱波段数，同时决定输出通道数
    ch	多光谱输入通道数
    dim	网络内部特征宽度
    num_blocks	深层重构使用的残差块数量
    """

    def __init__(self, spn=31, ch=3, dim=64, num_blocks=8):
        super(Net, self).__init__()
        self.spn = spn

        # HSI 分支侧重利用丰富的光谱信息，MSI 分支侧重利用高分辨率空间信息。将 HSI 特征放大到与 MSI 相同的尺寸
        # LR-HSI 光谱特征提取（在低分辨率下进行，节省计算量）
        self.hsi_head = nn.Sequential(
            nn.Conv2d(spn, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        # HR-MSI 空间细节特征提取
        self.msi_head = nn.Sequential(
            nn.Conv2d(ch, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        # 双分支特征融合
        self.fuse = nn.Conv2d(dim * 2, dim, 3, 1, 1)
        # 深层残差重构;列表推导式创建 8 个独立的 ResBlock，* 将这些模块展开，传给 nn.Sequential
        # ResBlock1 → ResBlock2 → ... → ResBlock8; 各块的参数独立，没有共享权重
        self.body = nn.Sequential(*[ResBlock(dim) for _ in range(num_blocks)])
        # 处理深层特征
        self.body_out = nn.Conv2d(dim, dim, 3, 1, 1)
        # 映射回光谱维；形状变化： (B,64,H,W) → (B,31,H,W)
        self.tail = nn.Conv2d(dim, spn, 3, 1, 1)

    def forward(self, lrhs, hrms):
        H, W = hrms.shape[2], hrms.shape[3]
        # 全局残差基线：LR-HSI 双线性上采样到 HR 分辨率
        # 这里是 “HSI 插值基线”，“插值”是把低分辨率图像放大
        hsi_up = F.interpolate(lrhs, size=(H, W), mode='bilinear', align_corners=False)
        f_hsi = self.hsi_head(lrhs)                                # (B, dim, h, w)
        f_hsi = F.interpolate(f_hsi, size=(H, W), mode='bilinear',
                              align_corners=False)                 # (B, dim, H, W)
        f_msi = self.msi_head(hrms)                                # (B, dim, H, W)
        #  拼接并融合
        f = self.fuse(torch.cat([f_hsi, f_msi], dim=1))            # (B, dim, H, W)
        # 深层重构与长残差链接;它在 8 个残差块外面又增加了一条跳跃连接，保留刚完成融合的特征。
        f = f + self.body_out(self.body(f))                        # 长残差连接
        # 将预测的修正量与插值图像，按相同波段、相同像素位置逐元素相加
        return hsi_up + self.tail(f)                               # (B, spn, H, W)



