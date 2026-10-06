import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ==============================================================================
# 1. 基础组件 (MLP 与 Window 划分)
# ==============================================================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features or in_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features or in_features, out_features or in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ==============================================================================
# 2. 全局分支: 无阈值的全局线性注意力 (Global Linear Attention)
# ==============================================================================
class GlobalLinearAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_relu = F.relu(q)
        k_weighted = F.relu(k)

        memory = k_weighted.transpose(-2, -1) @ v

        k_sum = k_weighted.sum(dim=-2, keepdim=True)
        denorm = q_relu @ k_sum.transpose(-2, -1)

        out = (q_relu @ memory) / (denorm + 1e-6)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out * self.alpha


# ==============================================================================
# 3. 局部分支: 最纯净的窗口注意力 (Standard Window Attention)
# ==============================================================================
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        self._init_relative_position_index()
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _init_relative_position_index(self):
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_pos_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_pos_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# ==============================================================================
# 4. 组装 Block：纯局部 Window + 纯全局 Linear (直接相加融合)
# ==============================================================================
class Block(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)

        # 局部分支
        self.local_attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # 全局分支
        self.global_attn = GlobalLinearAttention(dim=dim, num_heads=num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # Shift Mask 配置
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # 1. 全局特征提取
        x_global = self.global_attn(x)

        # 2. 局部特征提取
        x_spatial = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x_spatial, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_spatial

        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)

        attn_windows = self.local_attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x_local = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x_local = torch.roll(shifted_x_local, shifts=(self.shift_size, self.shift_size), dims=(1, 2)).view(B, L, C)
        else:
            x_local = shifted_x_local.view(B, L, C)

        # 3. 双分支直接相加融合 (无阈值、无方差引导)
        x_fused = x_local + x_global

        # 4. FFN
        x = shortcut + self.drop_path(x_fused)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ==============================================================================
# 5. 深层特征提取器桥接 & 顶层网络拓扑
# ==============================================================================
class Baseline_FeatureExtraction(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                Block(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                    window_size=window_size, shift_size=shift, mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
                )
            )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        for blk in self.blocks:
            x_flat = blk(x_flat)

        x_out = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x_out


class Baseline_Net(nn.Module):
    def __init__(self, hsi_channel=31, msi_channel=3, upscale_factor=8, n_feats=64,
                 depth=6, num_heads=8, window_size=8, drop_path_rate=0.1):
        super().__init__()
        self.up_factor = upscale_factor

        self.head_hsi = nn.Conv2d(hsi_channel, n_feats, 3, 1, 1)
        self.head_msi = nn.Conv2d(msi_channel, n_feats, 3, 1, 1)
        self.head_fuse = nn.Conv2d(2 * n_feats, n_feats, 3, 1, 1)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.body = Baseline_FeatureExtraction(
            dim=n_feats, input_resolution=(64, 64), depth=depth, num_heads=num_heads,
            window_size=window_size, mlp_ratio=4., drop_path=dpr
        )

        self.tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.final = nn.Conv2d(n_feats, hsi_channel, 3, 1, 1)

    def forward(self, x, y):
        x_up = F.interpolate(x, scale_factor=self.up_factor, mode='bicubic', align_corners=False)

        feat_hsi = self.head_hsi(x_up)
        feat_msi = self.head_msi(y)
        feat = self.head_fuse(torch.cat([feat_hsi, feat_msi], dim=1))
        res = feat

        feat = self.body(feat)

        feat = self.tail(feat) + res
        out = self.final(feat) + x_up

        return out