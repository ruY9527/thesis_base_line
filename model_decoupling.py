"""双分支 baseline + MossFuse 核心解耦与空间/光谱聚合。

本文件自包含，只依赖 PyTorch，不导入 model.py 或 research 目录的模块。
来源：dusongcheng/MossFuse，固定版本
d6c7eb4a487046aecae5265cabf81b0c23506465：
  - MossFuseNet.py：两路共享/互补编码及共享 -> 空间 -> 光谱的聚合顺序。
  - net.py：Transformer、门控及两种聚合的计算逻辑。
  - LKNet.py：UniRepLKNetBlock 的 k=0/7 默认训练与普通推理路径。
  - utils.py：cal_decomp_loss 的全 batch 余弦配对。
https://github.com/dusongcheng/MossFuse/tree/d6c7eb4a487046aecae5265cabf81b0c23506465

相对原版的适配：沿用当前 model.py 的 LR-HSI stem、HR-MSI stem 和
残差重构；先提取 LR-HSI 特征再上采样。没有迁入 BlindNet、PSF/SRF 估计、
辅助自监督解码器或部署重参数化。这是核心模块迁移，不是完整 MossFuse 复现。

普通推理（与现有 baseline 相同的输入顺序和 Tensor 输出）：
    net = Net(31, 3)
    prediction = net(lrhs, hrms)

训练接入示例（需要使用者手动调整训练脚本，本文件不会自动加入损失）：
    prediction, aux = net(lrhs, hrms, return_aux=True)
    loss_sc = cal_decomp_loss(
        aux["msi_shared"], aux["msi_complementary"],
        aux["hsi_shared"], aux["hsi_complementary"],
    )
    loss = F.l1_loss(prediction, ground_truth) + 0.1 * loss_sc

所有四路特征均为 (B, dim, H, W)。共享指表征，两个共享编码器不共享参数。
Net 的前四个参数兼容 baseline；旧 baseline 的完整 state_dict 不兼容新架构。
测试方法统一放在 FunctionTest.py：
    python FunctionTest.py --model decoupling
"""

import math
import numbers

import torch
import torch.nn.functional as F
from torch import nn


MOSSFUSE_SOURCE_COMMIT = "d6c7eb4a487046aecae5265cabf81b0c23506465"

__all__ = [
    "Net", "Encoder_MSI", "Encoder_HSI", "ChannelAttention", "ResBlock",
    "TransformerBlock", "UniRepLKNetBlock", "Spatial_Aware_Aggregation",
    "Spectral_Aware_Aggregation", "cal_decomp_loss",
]


def _check_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


# 1. 当前 baseline 的通道注意力与残差块，计算方式保持不变。
class ChannelAttention(nn.Module):
    """空间全局平均池化 -> LayerNorm -> MLP -> 通道重加权。"""

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.mlp(self.norm(x.mean(dim=(2, 3))))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    """两层 3x3 卷积 + 通道注意力 + 局部残差。"""

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act = nn.GELU()
        self.att = ChannelAttention(dim)

    def forward(self, x):
        return x + self.att(self.conv2(self.act(self.conv1(x))))


# 2. MossFuse net.py 的原语；reshape/transpose 替代 einops，轴含义不变。
def to_3d(x):
    return x.flatten(2).transpose(1, 2)  # B,C,H,W -> B,HW,C


def to_4d(x, h, w):
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], h, w)


def _to_heads(x, num_heads):
    b, c, h, w = x.shape
    return x.reshape(b, num_heads, c // num_heads, h * w)


def _from_heads(x, h, w):
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2], h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """MossFuse 二维 LayerNorm：每个空间位置沿特征通道归一化。"""

    def __init__(self, dim, LayerNorm_type="WithBias"):
        super().__init__()
        if LayerNorm_type not in ("BiasFree", "WithBias"):
            raise ValueError("LayerNorm_type must be 'BiasFree' or 'WithBias'")
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == "BiasFree" else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """原版 GDFN：逐点扩展 -> 深度卷积 -> GELU 门控 -> 逐点投影。"""

    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        if hidden_features < 1:
            raise ValueError("ffn_expansion_factor produces no hidden channels")
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, 3, padding=1,
            groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    """原版 MDTA；注意力矩阵在每个 head 的特征通道之间计算。"""

    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        num_heads = _check_int("num_heads", num_heads)
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = F.normalize(_to_heads(q, self.num_heads), dim=-1)
        k = F.normalize(_to_heads(k, self.num_heads), dim=-1)
        v = _to_heads(v, self.num_heads)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return self.project_out(_from_heads(attn @ v, h, w))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


# 3. LKNet.py 的必要依赖闭包。支持原 MossFuse 使用的 k=0、k=7 默认路径。
class GRNwithNHWC(nn.Module):
    """NHWC 上的 Global Response Normalization，保留原 gamma/beta。"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return (self.gamma * nx + 1) * x + self.beta


class NCHWtoNHWC(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, 1, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, 1, bias=True)
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, x):
        weight = torch.sigmoid(self.up(self.nonlinear(self.down(F.adaptive_avg_pool2d(x, 1)))))
        return x * weight


class DilatedReparamBlock(nn.Module):
    """原 k=7 的多分支计算；eval 时仍使用原分支，不做部署合并。"""

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        if kernel_size != 7:
            raise ValueError("This extracted DilatedReparamBlock supports kernel_size=7 only")
        self.lk_origin = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.origin_bn = nn.BatchNorm2d(channels)
        self.kernel_sizes = [5, 3, 3]
        self.dilates = [1, 2, 3]
        for kernel, dilation in zip(self.kernel_sizes, self.dilates):
            self.add_module(
                f"dil_conv_k{kernel}_{dilation}",
                nn.Conv2d(
                    channels, channels, kernel, padding=(dilation * (kernel - 1) + 1) // 2,
                    dilation=dilation, groups=channels, bias=False,
                ),
            )
            self.add_module(f"dil_bn_k{kernel}_{dilation}", nn.BatchNorm2d(channels))

    def forward(self, x):
        out = self.origin_bn(self.lk_origin(x))
        for kernel, dilation in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f"dil_conv_k{kernel}_{dilation}")
            bn = getattr(self, f"dil_bn_k{kernel}_{dilation}")
            out = out + bn(conv(x))
        return out


class UniRepLKNetBlock(nn.Module):
    """保留原版默认局部块；没有 timm、CUDA 大核插件或部署合并依赖。

    k=0 仅跳过深度卷积与前置 BN，仍执行 SE、FFN、GRN、后置 BN 和残差。
    对应原版 deploy=False、drop_path=0、with_cp=False、use_sync_bn=False。
    """

    def __init__(self, dim, kernel_size, layer_scale_init_value=1e-6, ffn_factor=4):
        super().__init__()
        if kernel_size not in (0, 7):
            raise ValueError("This extracted UniRepLKNetBlock supports kernel_size=0 or 7 only")
        self.dwconv = nn.Identity() if kernel_size == 0 else DilatedReparamBlock(dim, 7)
        self.norm = nn.Identity() if kernel_size == 0 else nn.BatchNorm2d(dim)
        self.se = SEBlock(dim, dim // 4)
        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(NCHWtoNHWC(), nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(nn.GELU(), GRNwithNHWC(ffn_dim))
        self.pwconv2 = nn.Sequential(
            nn.Linear(ffn_dim, dim, bias=False), NHWCtoNCHW(), nn.BatchNorm2d(dim),
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value is not None and layer_scale_init_value > 0 else None
        )
        self.drop_path = nn.Identity()

    def compute_residual(self, x):
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.reshape(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, x):
        return x + self.compute_residual(x)


# 4. 四路解耦。baseline stem 已完成输入映射，这里不再叠加 patch_embed。
class Encoder_MSI(nn.Module):
    """输入已对齐的 MSI 特征；输出共享特征和空间互补特征。"""

    def __init__(self, dim=64, num_blocks=1, num_heads=8, ffn_expansion_factor=2,
                 bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        num_blocks = _check_int("encoder_blocks", num_blocks)
        options = dict(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                       bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(**options) for _ in range(num_blocks)])
        self.baseFeature = nn.Sequential(
            UniRepLKNetBlock(dim, 7), UniRepLKNetBlock(dim, 0), UniRepLKNetBlock(dim, 0),
            TransformerBlock(**options),
        )
        self.detailFeature = nn.Sequential(
            UniRepLKNetBlock(dim, 7), UniRepLKNetBlock(dim, 0), UniRepLKNetBlock(dim, 0),
        )

    def forward(self, feature):
        feature = self.encoder_level1(feature)
        return self.baseFeature(feature), self.detailFeature(feature)


class Encoder_HSI(nn.Module):
    """输入已对齐的 HSI 特征；输出共享特征和光谱互补特征。"""

    def __init__(self, dim=64, num_blocks=1, num_heads=8, ffn_expansion_factor=2,
                 bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        num_blocks = _check_int("encoder_blocks", num_blocks)
        options = dict(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                       bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(**options) for _ in range(num_blocks)])
        self.baseFeature = nn.Sequential(
            TransformerBlock(**options), UniRepLKNetBlock(dim, 7),
            UniRepLKNetBlock(dim, 0), UniRepLKNetBlock(dim, 0),
        )
        self.detailFeature = TransformerBlock(**options)

    def forward(self, feature):
        feature = self.encoder_level1(feature)
        return self.baseFeature(feature), self.detailFeature(feature)


# 5. 原版空间/光谱聚合；x 提供 Q/K/V，y 提供互补门控和局部补充。
class Spatial_Attention(nn.Module):
    """互补特征 -> B,1,H,W 的空间门控。"""

    def __init__(self, dim=64):
        super().__init__()
        if dim < 8:
            raise ValueError("Spatial_Attention requires dim >= 8")
        self.spatial_atten = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim // 8, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, 1, 1, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.spatial_atten(x)


class Spectral_Attention(nn.Module):
    """空间池化 -> B,dim,1,1 通道门控，广播到输入空间尺寸。"""

    def __init__(self, dim=64):
        super().__init__()
        if dim < 8:
            raise ValueError("Spectral_Attention requires dim >= 8")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spectral_atten = nn.Sequential(
            nn.Linear(dim, dim // 8, bias=False), nn.GELU(),
            nn.Linear(dim // 8, dim // 8, bias=False), nn.GELU(),
            nn.Linear(dim // 8, dim, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        weight = self.spectral_atten(self.avg_pool(x).reshape(b, c)).reshape(b, c, 1, 1)
        return weight.expand_as(x)


class Cross_Spatial_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        num_heads = _check_int("num_heads", num_heads)
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.Spatial_Att = Spatial_Attention(dim)
        self.Spectral_Att = Spectral_Attention(dim)
        self.DWConv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)

    def forward(self, x, y):
        h, w = x.shape[-2:]
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = q * self.Spatial_Att(y)
        q = F.normalize(_to_heads(q, self.num_heads), dim=-1)
        k = F.normalize(_to_heads(k, self.num_heads), dim=-1)
        v = _to_heads(v, self.num_heads)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = _from_heads(attn @ v, h, w)
        y = self.DWConv(y) * self.Spectral_Att(out)
        return self.project_out(out + y)


class Cross_Spectral_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        num_heads = _check_int("num_heads", num_heads)
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.Spatial_Att = Spatial_Attention(dim)
        self.Spectral_Att = Spectral_Attention(dim)
        # 原版虽名为 DWConv，这里实际是普通 1x1 卷积（groups=1），混合通道。
        # 空间聚合中的同名模块才是 groups=dim 的 3x3 深度卷积。
        self.DWConv = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, y):
        h, w = x.shape[-2:]
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = q * self.Spectral_Att(y)
        q = F.normalize(_to_heads(q, self.num_heads), dim=-1)
        k = F.normalize(_to_heads(k, self.num_heads), dim=-1)
        v = _to_heads(v, self.num_heads)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = _from_heads(attn @ v, h, w)
        y = self.DWConv(y) * self.Spatial_Att(out)
        return self.project_out(out + y)


class Spatial_Aware_Aggregation(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Spatial_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        x = x + self.attn(self.norm1(x), y)
        return x + self.ffn(self.norm2(x))


class Spectral_Aware_Aggregation(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Spectral_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        x = x + self.attn(self.norm1(x), y)
        return x + self.ffn(self.norm2(x))


# 6. 原 cal_decomp_loss 的配对与归约；不在 Net.forward 内自动计算。
def cal_decomp_loss(msi_shared, msi_complementary, hsi_shared, hsi_complementary):
    """MossFuse L_SC，返回标量；保持原版整个 batch 展平的余弦计算。

    正对：MSI shared / HSI shared。
    负对：MSI shared / MSI complementary；HSI shared / HSI complementary；
          MSI complementary / HSI complementary。
    logsumexp 与原 -log(exp(pos)/(exp(pos)+sum(exp(neg)))) 数学等价。
    半精度特征转 FP32 计算，但保留梯度链；非连续特征可用 reshape。
    """
    features = (msi_shared, msi_complementary, hsi_shared, hsi_complementary)
    if any(not isinstance(x, torch.Tensor) for x in features):
        raise TypeError("All decomposition features must be torch.Tensor")
    reference = features[0]
    if reference.ndim != 4 or reference.numel() == 0:
        raise ValueError("Decomposition features must be non-empty B,C,H,W tensors")
    if any(x.shape != reference.shape for x in features):
        raise ValueError("All four decomposition features must have the same B,C,H,W shape")
    if any(x.device != reference.device for x in features):
        raise ValueError("All decomposition features must be on the same device")
    if any(not x.is_floating_point() for x in features):
        raise TypeError("Decomposition features must have floating point dtypes")
    flattened = [
        (x.float() if x.dtype in (torch.float16, torch.bfloat16) else x).reshape(-1)
        for x in features
    ]
    ms, mc, hs, hc = flattened
    scores = torch.stack([
        F.cosine_similarity(ms, hs, dim=0),
        F.cosine_similarity(ms, mc, dim=0),
        F.cosine_similarity(hs, hc, dim=0),
        F.cosine_similarity(mc, hc, dim=0),
    ])
    return torch.logsumexp(scores, dim=0) - scores[0]


class Net(nn.Module):
    """双分支输入 -> 四路解耦 -> 空间/光谱聚合 -> baseline 残差重构。

    lrhs: B,spn,h,w；hrms: B,ch,H,W；输出: B,spn,H,W。
    按 hrms 的实际尺寸对齐，不硬编码 scale=32 或 CAVE 的波段数。
    num_blocks 控制 baseline body，encoder_blocks 控制各模态预编码深度。
    默认返回 Tensor；return_aux=True 返回 (prediction, 四路特征字典)。
    """

    def __init__(self, spn=31, ch=3, dim=64, num_blocks=8, *, num_heads=8,
                 encoder_blocks=1, ffn_expansion_factor=2, bias=False,
                 LayerNorm_type="WithBias"):
        super().__init__()
        self.spn = _check_int("spn", spn)
        self.ch = _check_int("ch", ch)
        self.dim = _check_int("dim", dim, minimum=8)
        self.num_blocks = _check_int("num_blocks", num_blocks, minimum=0)
        self.num_heads = _check_int("num_heads", num_heads)
        self.encoder_blocks = _check_int("encoder_blocks", encoder_blocks)
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if (isinstance(ffn_expansion_factor, bool)
                or not isinstance(ffn_expansion_factor, numbers.Real)
                or not math.isfinite(ffn_expansion_factor)
                or ffn_expansion_factor <= 0 or int(dim * ffn_expansion_factor) < 1):
            raise ValueError("ffn_expansion_factor must produce positive finite hidden channels")

        # 保留 baseline 的输入 stem：HSI 在 LR 提取后才上采样。
        self.hsi_head = nn.Sequential(
            nn.Conv2d(spn, dim, 3, 1, 1), nn.GELU(), nn.Conv2d(dim, dim, 3, 1, 1),
        )
        self.msi_head = nn.Sequential(
            nn.Conv2d(ch, dim, 3, 1, 1), nn.GELU(), nn.Conv2d(dim, dim, 3, 1, 1),
        )
        options = dict(dim=dim, num_blocks=encoder_blocks, num_heads=num_heads,
                       ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                       LayerNorm_type=LayerNorm_type)
        self.msi_encoder = Encoder_MSI(**options)
        self.hsi_encoder = Encoder_HSI(**options)

        # 原 Decoder_HR 的顺序：MSI shared 在前，HSI shared 在后。
        self.fuse = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        aggregation_options = dict(dim=dim, num_heads=num_heads,
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type)
        self.spatial_aggregation = Spatial_Aware_Aggregation(**aggregation_options)
        self.spectral_aggregation = Spectral_Aware_Aggregation(**aggregation_options)

        self.body = nn.Sequential(*[ResBlock(dim) for _ in range(num_blocks)])
        self.body_out = nn.Conv2d(dim, dim, 3, 1, 1)
        self.tail = nn.Conv2d(dim, spn, 3, 1, 1)

    def _validate_inputs(self, lrhs, hrms):
        if not isinstance(lrhs, torch.Tensor) or not isinstance(hrms, torch.Tensor):
            raise TypeError("lrhs and hrms must be torch.Tensor")
        if lrhs.ndim != 4 or hrms.ndim != 4:
            raise ValueError("Inputs must have B,C,H,W dimensions; call net(lrhs, hrms)")
        if any(size == 0 for size in lrhs.shape) or any(size == 0 for size in hrms.shape):
            raise ValueError("Inputs must have non-empty batch, channel and spatial dimensions")
        if lrhs.shape[0] != hrms.shape[0]:
            raise ValueError("lrhs and hrms must have the same batch size")
        if lrhs.shape[1] != self.spn or hrms.shape[1] != self.ch:
            raise ValueError(f"Expected lrhs channels={self.spn}, hrms channels={self.ch}")
        if lrhs.device != hrms.device or lrhs.dtype != hrms.dtype:
            raise ValueError("lrhs and hrms must have the same device and dtype")
        if not lrhs.is_floating_point() or not hrms.is_floating_point():
            raise TypeError("Inputs must have floating point dtypes")

    def forward(self, lrhs, hrms, *, return_aux=False):
        self._validate_inputs(lrhs, hrms)
        height, width = hrms.shape[-2:]
        hsi_up = F.interpolate(lrhs, size=(height, width), mode="bilinear", align_corners=False)
        f_hsi = F.interpolate(
            self.hsi_head(lrhs), size=(height, width), mode="bilinear", align_corners=False,
        )
        f_msi = self.msi_head(hrms)

        msi_shared, msi_complementary = self.msi_encoder(f_msi)
        hsi_shared, hsi_complementary = self.hsi_encoder(f_hsi)
        f = self.fuse(torch.cat([msi_shared, hsi_shared], dim=1))
        f = self.spatial_aggregation(f, msi_complementary)
        f = self.spectral_aggregation(f, hsi_complementary)
        f = f + self.body_out(self.body(f))
        prediction = hsi_up + self.tail(f)

        if return_aux:
            # 不缓存到 self；训练调用者决定何时计算损失并释放计算图。
            return prediction, {
                "msi_shared": msi_shared,
                "msi_complementary": msi_complementary,
                "hsi_shared": hsi_shared,
                "hsi_complementary": hsi_complementary,
            }
        return prediction


