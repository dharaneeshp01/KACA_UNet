#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dekaht_v2.py
============
DeKAHT-UNet v2 : Swkat-GR-KAN encoder + KAN patch-expand decoder + GR-KAN
pyramid pooling, for semantic segmentation of NATURAL images.

Self-contained. Replaces `dekaht_unet_unified.py` entirely.

WHAT CHANGED vs dekaht_unet_unified.py  (every item was a measured bug)
----------------------------------------------------------------------
A1  Encoder restructured into Swin-compatible `BasicLayer` (blocks + downsample).
    The old flat ModuleList produced keys `layers.0.0.norm1` where the official
    Swin checkpoint has `layers.0.blocks.0.norm1`, so `load_state_dict(strict=False)`
    matched 4/183 tensors = 0.01% of encoder params and silently discarded the
    rest. The 100-epoch COCO run trained a 62M transformer FROM SCRATCH.

A2  `remap_swin_to_dekaht()` : function-preserving ImageNet transfer.
    Handles the only two remaining mismatches (mlp.fc* Linear -> 1x1 Conv, and
    the KAN-only sub-modules) so GR_KAN_Conv computes EXACTLY Swin's MLP at
    step 0:  depthwise -> dirac kernel, gate -> sigmoid(0)=0.5, fc2 *= 2.
    Measured: 100% of encoder params warm-started, max|KAN(x)-SwinMLP(x)| = 0.

A3  (H, W) threaded explicitly through every block.
    The old `SwkatBlockOpt.forward` inferred H,W from `round(sqrt(L))` behind a
    chain of `try/except: <fallback reshape>` handlers. Non-square input silently
    reshaped WRONG instead of raising. The model is now resolution-agnostic and
    aspect-preserving evaluation is possible.

A4  Window padding done once, correctly, outside the attention.
    The old code padded inside window_partition and then sliced the relative
    position bias table `full_bias[:N,:N]` when the runtime window was smaller
    -- which is not a valid position bias. Now the feature map is padded to a
    multiple of window_size, so N == ws*ws always and the bias is exact.

A5  GR-KAN Pyramid Pooling at the bottleneck.
    At 448 input the deepest stage is 14x14 with window_size 7 -> 2x2 windows.
    NOTHING in the old network ever saw the whole image. Medical images are
    centred and scale-consistent so it never showed; natural images are neither.
    Zero-init output conv => exact no-op at step 0.

A6  `FinalPatchExpandX4` removed; logits predicted at 1/4 and bilinearly
    upsampled inside the loss. The old head materialised (B,96,448,448) and
    (B,81,448,448) activations (~1.7 GB) before the loss even ran.

A7  Real head. Was Conv2d(96, 81, 1) = 7.8k params for 81 classes, while the
    2-block bottleneck held 33.1M (30.6% of the model). Now a 256-ch conv head
    + an auxiliary FCN head at 1/16 (deep supervision, dropped at eval).

A8  Gate mode configurable. The old gate was Conv2d(hidden, hidden, 1) --
    at dim 768 that is 9.4M params PER BLOCK, 4x the cost of fc1. 'depthwise'
    is 3.1k and empirically as good: 122.7M -> 62.8M total params.

A9  All `except Exception: <silent fallback>` blocks deleted. Shape errors now
    raise instead of quietly producing garbage.

A10 `torch.cuda.amp` / `nn.init` deprecation warnings cleaned up.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GRKANFusionHead", "BoundaryHead", "DeKAHTUNetV2", "dekaht_unet_v2", "remap_swin_to_dekaht", "slim_kan_gates",
    "GR_KAN_Conv", "SwkatBlock", "WindowAttention", "BasicLayer",
    "GRKANPyramidPooling", "SegHead", "STANDARD_VARIANTS", "count_params",
]

STANDARD_VARIANTS: Dict[str, Dict[str, Any]] = {
    "swkat-tiny":  {"embed_dim": 96,  "depths": (2, 2, 6, 2),  "num_heads": (3, 6, 12, 24)},
    "swkat-small": {"embed_dim": 96,  "depths": (2, 2, 18, 2), "num_heads": (3, 6, 12, 24)},
    "swkat-base":  {"embed_dim": 128, "depths": (2, 2, 18, 2), "num_heads": (4, 8, 16, 32)},
    "swkat-large": {"embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48)},
}


# =============================================================================
# DropPath
# =============================================================================
def drop_path(x: torch.Tensor, p: float = 0.0, training: bool = False) -> torch.Tensor:
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x * mask / keep


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        return drop_path(x, self.p, self.training)

    def extra_repr(self):
        return f"p={self.p:.3f}"


# =============================================================================
# GR-KAN FFN  (the DeKAHT contribution; replaces Swin's Linear MLP)
# =============================================================================
class GR_KAN_Conv(nn.Module):
    """Grouped-rational KAN-style FFN operating on (B, C, H, W).

        fc1 (1x1)  ->  grouped scale  ->  GELU  ->  depthwise 3x3  ->  gate  ->  fc2 (1x1)

    gate_mode:
        'depthwise' : Conv2d(hid, hid, 3, groups=hid)   <- DEFAULT, 3.1k params
        'full'      : Conv2d(hid, hid, 1)               <- original, hid^2 params
                                                           (9.4M/block at dim 768!)
        'none'      : no gating
    """

    def __init__(self, channels: int, hidden_channels: Optional[int] = None,
                 groups: int = 8, drop: float = 0.0, spatial_kernel: int = 3,
                 gate_mode: str = "depthwise"):
        super().__init__()
        hidden_channels = int(hidden_channels or channels * 4)
        assert hidden_channels % groups == 0, \
            f"hidden_channels ({hidden_channels}) must be divisible by groups ({groups})"
        assert gate_mode in ("depthwise", "full", "none"), f"bad gate_mode {gate_mode}"

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.groups = max(1, groups)
        self.gate_mode = gate_mode
        self.is_kan = True

        self.fc1 = nn.Conv2d(channels, hidden_channels, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1, bias=True)

        k = max(1, spatial_kernel)
        self.spatial_kernel = k
        self.depthwise_spatial = (
            nn.Conv2d(hidden_channels, hidden_channels, k, padding=(k - 1) // 2,
                      groups=hidden_channels, bias=True)
            if k > 1 else nn.Identity()
        )

        self.group_scale = nn.Parameter(
            torch.ones(self.groups, hidden_channels // self.groups))

        self.act = nn.GELU()
        if gate_mode == "full":
            self.gate = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 1, bias=True),
                                      nn.Sigmoid())
        elif gate_mode == "depthwise":
            self.gate = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                                                groups=hidden_channels, bias=True),
                                      nn.Sigmoid())
        else:
            self.gate = None

        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in (self.fc1, self.fc2):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        if isinstance(self.depthwise_spatial, nn.Conv2d):
            nn.init.trunc_normal_(self.depthwise_spatial.weight, std=0.02)
            nn.init.zeros_(self.depthwise_spatial.bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate[0].weight)
            nn.init.zeros_(self.gate[0].bias)     # sigmoid(0) = 0.5  (see remap)

    def extra_repr(self):
        return (f"channels={self.channels}, hidden={self.hidden_channels}, "
                f"groups={self.groups}, gate={self.gate_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.fc1(x)
        if self.groups > 1:
            cpg = self.hidden_channels // self.groups
            h = (h.view(B, self.groups, cpg, H, W)
                 * self.group_scale.view(1, self.groups, cpg, 1, 1)).view(B, -1, H, W)
        h = self.act(h)
        h = self.depthwise_spatial(h)
        if self.gate is not None:
            h = h * self.gate(h)
        h = self.drop(h)
        return self.drop(self.fc2(h))


# =============================================================================
# Windows  (no internal padding -- callers pad to a multiple of window_size)
# =============================================================================
def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, ws, ws, C).  H and W MUST be multiples of ws."""
    B, H, W, C = x.shape
    assert H % ws == 0 and W % ws == 0, f"({H},{W}) not divisible by window {ws}"
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(win: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws, ws, C) -> (B, H, W, C)."""
    B = win.shape[0] // ((H // ws) * (W // ws))
    x = win.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window MHSA with a learnable relative position bias.

    FIX A4: the old version sliced `bias[:N,:N]` whenever the runtime window was
    smaller than `window_size`, which is not a valid relative position bias.
    Callers now guarantee N == ws*ws, so the table is always used exactly.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim, self.window_size, self.num_heads = dim, int(window_size), num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        ws = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws),
                                            indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("relative_position_index", rel.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Bn, N, C = x.shape
        assert N == self.window_size ** 2, \
            f"WindowAttention got N={N}, expected {self.window_size ** 2}"
        qkv = (self.qkv(x).reshape(Bn, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(N, N, self.num_heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0).to(attn.dtype)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(-1))
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj_drop(self.proj(out))


# =============================================================================
# Swkat block  (H, W passed in explicitly -- FIX A3)
# =============================================================================
class SwkatBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4.0, drop_path_rate: float = 0.0, groups: int = 8,
                 gate_mode: str = "depthwise"):
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.shift_size = int(shift_size) % self.window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_conv = GR_KAN_Conv(dim, int(dim * mlp_ratio), groups=groups,
                                    gate_mode=gate_mode)
        self._mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def extra_repr(self):
        return f"dim={self.dim}, window={self.window_size}, shift={self.shift_size}"

    @torch.no_grad()
    def _attn_mask(self, Hp: int, Wp: int, device) -> torch.Tensor:
        key = (Hp, Wp)
        m = self._mask_cache.get(key)
        if m is not None and m.device == device:
            return m
        ws, ss = self.window_size, self.shift_size
        img = torch.zeros((1, Hp, Wp, 1), device=device)
        cnt = 0
        for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img[:, h, w, :] = cnt
                cnt += 1
        mw = window_partition(img, ws).view(-1, ws * ws)
        m = mw.unsqueeze(1) - mw.unsqueeze(2)
        m = m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)
        self._mask_cache = {key: m}          # keep exactly one; resolutions rarely mix
        return m

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W, f"SwkatBlock: L={L} != H*W={H * W}"
        ws, ss = self.window_size, self.shift_size

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        if ss > 0:
            x = torch.roll(x, (-ss, -ss), dims=(1, 2))
            mask = self._attn_mask(Hp, Wp, x.device)
        else:
            mask = None

        win = window_partition(x, ws).view(-1, ws * ws, C)
        win = self.attn(win, mask)
        x = window_reverse(win.view(-1, ws, ws, C), ws, Hp, Wp)

        if ss > 0:
            x = torch.roll(x, (ss, ss), dims=(1, 2))
        if pad_b or pad_r:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x.view(B, H * W, C))

        y = self.norm2(x).transpose(1, 2).reshape(B, C, H, W)
        y = self.mlp_conv(y).flatten(2).transpose(1, 2)
        return x + self.drop_path(y)


# =============================================================================
# Patch embed / merge / expand
# =============================================================================
class PatchEmbed(nn.Module):
    """FIX: added the LayerNorm that official Swin has (patch_embed.norm), so the
    checkpoint transfers 100% instead of leaving that tensor on the floor."""

    def __init__(self, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        ps = self.patch_size
        _, _, H, W = x.shape
        pad_b, pad_r = (ps - H % ps) % ps, (ps - W % ps) % ps
        if pad_b or pad_r:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        x = self.proj(x)
        _, _, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), (H, W)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or dim * 2
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"PatchMerging: L={L} != H*W={H * W}"
        x = x.view(B, H, W, C)
        if H % 2 or W % 2:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            H, W = H + H % 2, W + W % 2
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x)), (H // 2, W // 2)


class PatchExpand(nn.Module):
    """Spatial inverse of PatchMerging: Linear(C -> 2C) -> 2x2 shuffle -> C/2."""

    def __init__(self, in_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim // 2
        self.expand = nn.Linear(in_dim, 2 * in_dim, bias=False)
        self.norm = nn.LayerNorm(in_dim // 2)
        self.proj = (nn.Linear(in_dim // 2, out_dim)
                     if out_dim != in_dim // 2 else nn.Identity())

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"PatchExpand: L={L} != H*W={H * W}"
        x = self.expand(x).view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, 4 * L, C // 2)
        return self.proj(self.norm(x)), (2 * H, 2 * W)


# =============================================================================
# Encoder stage  (Swin-compatible naming: layers.{i}.blocks.{j} / .downsample)
# =============================================================================
class BasicLayer(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float, dpr: Sequence[float], groups: int,
                 downsample: bool, gate_mode: str):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwkatBlock(dim, num_heads, window_size,
                       0 if i % 2 == 0 else window_size // 2,
                       mlp_ratio, float(dpr[i]), groups, gate_mode)
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim, dim * 2) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        skip, sH, sW = x, H, W
        if self.downsample is not None:
            x, (H, W) = self.downsample(x, H, W)
        return skip, sH, sW, x, H, W


class SwkatGRKAN(nn.Module):
    """DeKAHT encoder. Resolution-agnostic (FIX A3)."""

    def __init__(self, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96,
                 depths: Tuple[int, ...] = (2, 2, 6, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24),
                 window_size: int = 7, mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.1, groups: int = 8,
                 gate_mode: str = "depthwise"):
        super().__init__()
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.dims = [embed_dim * 2 ** i for i in range(self.num_stages)]
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(0.0)

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        cur = 0
        for i, d in enumerate(depths):
            self.layers.append(BasicLayer(
                dim=self.dims[i], depth=d, num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio,
                dpr=dpr[cur:cur + d], groups=groups,
                downsample=(i < self.num_stages - 1), gate_mode=gate_mode))
            cur += d
        self.norm = nn.LayerNorm(self.dims[-1])
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x) -> List[Tuple[torch.Tensor, int, int]]:
        """Returns [(feat, H, W)] per stage; the last entry is LayerNorm'd."""
        x, (H, W) = self.patch_embed(x)
        x = self.pos_drop(x)
        feats = []
        for layer in self.layers:
            skip, sH, sW, x, H, W = layer(x, H, W)
            feats.append([skip, sH, sW])
        feats[-1][0] = self.norm(feats[-1][0])
        return [tuple(f) for f in feats]


# =============================================================================
# FIX A5 : GR-KAN Pyramid Pooling  (global context)
# =============================================================================
class GRKANPyramidPooling(nn.Module):
    def __init__(self, dim: int, scales: Sequence[int] = (1, 2, 3, 6),
                 reduction: int = 4, groups: int = 8, gate_mode: str = "depthwise"):
        super().__init__()
        mid = max(groups, (dim // reduction) // groups * groups)
        self.scales = tuple(scales)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, mid, 1, bias=False),
                          nn.GroupNorm(groups, mid), nn.GELU(),
                          GR_KAN_Conv(mid, mid * 2, groups=groups, gate_mode=gate_mode))
            for _ in self.scales
        ])
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim + mid * len(self.scales), dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False))
        nn.init.zeros_(self.out_proj[-1].weight)      # residual identity at step 0

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"PPM: L={L} != H*W={H * W}"
        f = x.transpose(1, 2).reshape(B, C, H, W)
        feats = [f]
        for s, br in zip(self.scales, self.branches):
            p = br(F.adaptive_avg_pool2d(f, min(s, H, W)))
            feats.append(F.interpolate(p, (H, W), mode="bilinear", align_corners=False))
        return x + self.out_proj(torch.cat(feats, 1)).flatten(2).transpose(1, 2)


# =============================================================================
# Decoder stage
# =============================================================================
class UpStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, depth: int,
                 window_size: int, mlp_ratio: float, groups: int,
                 dpr: Sequence[float], gate_mode: str):
        super().__init__()
        self.expand = PatchExpand(in_dim, out_dim)
        self.fuse = nn.Linear(2 * out_dim, out_dim)
        self.blocks = nn.ModuleList([
            SwkatBlock(out_dim, num_heads, window_size,
                       0 if i % 2 == 0 else window_size // 2,
                       mlp_ratio, float(dpr[i]), groups, gate_mode)
            for i in range(depth)
        ])

    def forward(self, x, H, W, skip, sH, sW):
        x, (H, W) = self.expand(x, H, W)
        if (H, W) != (sH, sW):                       # odd-size encoder stages
            B, _, C = x.shape
            x = F.interpolate(x.transpose(1, 2).reshape(B, C, H, W), (sH, sW),
                              mode="bilinear", align_corners=False)
            x = x.flatten(2).transpose(1, 2)
            H, W = sH, sW
        assert x.shape[1] == skip.shape[1] and x.shape[2] == skip.shape[2], \
            f"skip mismatch: up {tuple(x.shape)} vs skip {tuple(skip.shape)}"
        x = self.fuse(torch.cat([x, skip], dim=-1))
        for blk in self.blocks:
            x = blk(x, H, W)
        return x, H, W


# =============================================================================
# FIX A7 : real head
# =============================================================================
class SegHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 dropout: float = 0.1, n_conv: int = 2):
        super().__init__()
        layers, c = [], in_dim
        for _ in range(n_conv):
            layers += [nn.Conv2d(c, hidden, 3, padding=1, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
            c = hidden
        layers.append(nn.Dropout2d(dropout))
        self.body = nn.Sequential(*layers)
        self.cls = nn.Conv2d(hidden, num_classes, 1)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.zeros_(self.cls.bias)

    def forward(self, x):
        return self.cls(self.body(x))


# =============================================================================
# Full model
# =============================================================================
# =============================================================================
# V3-A : GR-KAN multi-scale fusion head
# -----------------------------------------------------------------------------
# WHY. In v2 the classifier saw ONLY the final 1/4 feature (96 ch). Everything
# the decoder computed at 1/32 (768 ch), 1/16 (384 ch) and 1/8 (192 ch) reached
# it only after being squeezed through two more upsample stages. Every strong
# natural-image segmenter fuses all scales directly at the head instead:
# UperNet uses an FPN head, SegFormer an all-MLP head. This is the KAN-native
# version, and it was the largest remaining structural gap after the v2 fixes.
# Cheap: laterals are 1x1 convs to fuse_dim (128), so this adds ~1.6M params,
# not the ~12M a full UperNet FPN head would.
# =============================================================================
class GRKANFusionHead(nn.Module):
    def __init__(self, in_dims: Sequence[int], fuse_dim: int, head_dim: int,
                 num_classes: int, groups: int = 8, dropout: float = 0.1,
                 gate_mode: str = "depthwise"):
        super().__init__()
        fuse_dim = max(groups, (fuse_dim // groups) * groups)
        head_dim = max(groups, (head_dim // groups) * groups)
        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d, fuse_dim, 1, bias=False),
                          nn.GroupNorm(groups, fuse_dim), nn.GELU())
            for d in in_dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_dim * len(in_dims), head_dim, 1, bias=False),
            nn.GroupNorm(groups, head_dim), nn.GELU(),
            GR_KAN_Conv(head_dim, head_dim * 2, groups=groups, gate_mode=gate_mode),
            nn.Conv2d(head_dim, head_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, head_dim), nn.GELU(),
            nn.Dropout2d(dropout))
        self.cls = nn.Conv2d(head_dim, num_classes, 1)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.zeros_(self.cls.bias)

    def forward(self, pyramid):
        """pyramid: list of (tokens (B,L,C), H, W), COARSE -> FINE."""
        tH, tW = pyramid[-1][1], pyramid[-1][2]
        outs = []
        for lat, (t, h, w) in zip(self.lateral, pyramid):
            B, L, C = t.shape
            f = lat(t.transpose(1, 2).reshape(B, C, h, w))
            if (h, w) != (tH, tW):
                f = F.interpolate(f, (tH, tW), mode="bilinear", align_corners=False)
            outs.append(f)
        feat = self.fuse(torch.cat(outs, 1))
        return self.cls(feat), feat


# =============================================================================
# V3-B : boundary head (deep supervision on object edges)
# -----------------------------------------------------------------------------
# WHY. After the v2 fixes the remaining worst classes are all THIN or heavily
# occluded: potted plant 0.220, handbag 0.233, skis 0.255, baseball bat 0.349,
# knife 0.383, toothbrush 0.391. Per-pixel CE is dominated by object interiors,
# so the few boundary pixels on a ski never move the loss. An explicit binary
# edge task forces the shared fused feature to retain the high-frequency detail
# that a 1/4-resolution classifier would otherwise throw away.
# =============================================================================
class BoundaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1))
        nn.init.normal_(self.body[-1].weight, std=0.01)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return self.body(x)


class DeKAHTUNetV2(nn.Module):
    """Logits are returned at 1/4 of input resolution (FIX A6). The loss and the
    metric upsample them to label size -- do NOT upsample inside the model."""

    def __init__(self, patch_size: int = 4, in_chans: int = 3, num_classes: int = 81,
                 embed_dim: int = 96, depths: Tuple[int, ...] = (2, 2, 6, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24),
                 decoder_depths: Tuple[int, ...] = (2, 2, 2), bottleneck_depth: int = 2,
                 window_size: int = 7, mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.2, groups: int = 8,
                 gate_mode: str = "depthwise", head_dim: int = 256,
                 aux_head: bool = True, use_ppm: bool = True,
                 ppm_scales: Sequence[int] = (1, 2, 3, 6),
                 use_fusion: bool = True, fuse_dim: int = 128,
                 boundary_head: bool = True):
        super().__init__()
        assert len(depths) == 4 and len(num_heads) == 4
        assert len(decoder_depths) == 3
        self.num_classes = num_classes
        self.patch_size = patch_size

        self.encoder = SwkatGRKAN(patch_size, in_chans, embed_dim, depths, num_heads,
                                  window_size, mlp_ratio, drop_path_rate, groups, gate_mode)
        self.dims = self.encoder.dims
        deep = self.dims[-1]

        self.ppm = (GRKANPyramidPooling(deep, ppm_scales, groups=groups, gate_mode=gate_mode)
                    if use_ppm else None)

        b_dpr = torch.linspace(0, drop_path_rate, max(bottleneck_depth, 1)).tolist()
        self.bottleneck = nn.ModuleList([
            SwkatBlock(deep, num_heads[-1], window_size,
                       0 if i % 2 == 0 else window_size // 2,
                       mlp_ratio, float(b_dpr[i]), groups, gate_mode)
            for i in range(bottleneck_depth)
        ])

        d_dpr = torch.linspace(0, drop_path_rate, sum(decoder_depths)).tolist()
        self.up_stages = nn.ModuleList()
        cur = 0
        for k in range(3):
            d = decoder_depths[k]
            self.up_stages.append(UpStage(
                in_dim=self.dims[3 - k], out_dim=self.dims[2 - k],
                num_heads=num_heads[2 - k], depth=d, window_size=window_size,
                mlp_ratio=mlp_ratio, groups=groups, dpr=d_dpr[cur:cur + d],
                gate_mode=gate_mode))
            cur += d

        self.norm_up = nn.LayerNorm(self.dims[0])

        # V3-A: fuse 1/32, 1/16, 1/8, 1/4 (coarse -> fine) directly at the head
        self.use_fusion = use_fusion
        if use_fusion:
            self.fusion_head = GRKANFusionHead(
                in_dims=[self.dims[3], self.dims[2], self.dims[1], self.dims[0]],
                fuse_dim=fuse_dim, head_dim=head_dim, num_classes=num_classes,
                groups=groups, gate_mode=gate_mode)
            self.seg_head = None
        else:
            self.fusion_head = None
            self.seg_head = SegHead(self.dims[0], head_dim, num_classes)

        self.aux_head = (SegHead(self.dims[2], max(64, head_dim // 2), num_classes, n_conv=1)
                         if aux_head else None)
        # V3-B: boundary supervision on the shared fused feature
        self.boundary_head = (BoundaryHead(head_dim)
                              if (boundary_head and use_fusion) else None)
        self._init_decoder()

    def _init_decoder(self):
        def fn(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for mod in (self.bottleneck, self.up_stages, self.norm_up):
            mod.apply(fn)

    def forward(self, x):
        """Logits at 1/4 resolution.
        train -> dict(main=..., aux=..., boundary=...)
        eval  -> Tensor (main logits only)
        """
        feats = self.encoder(x)                      # [(f,H,W)] x4, 1/4 .. 1/32
        d, H, W = feats[-1]

        if self.ppm is not None:
            d = self.ppm(d, H, W)
        for blk in self.bottleneck:
            d = blk(d, H, W)

        pyramid = [(d, H, W)]                        # 1/32, dims[3]
        aux = None
        for k, up in enumerate(self.up_stages):
            skip, sH, sW = feats[2 - k]
            d, H, W = up(d, H, W, skip, sH, sW)
            pyramid.append((d, H, W))                # 1/16, 1/8, 1/4
            if k == 0:
                aux = (d, H, W)                      # 1/16, dims[2]

        d = self.norm_up(d)
        pyramid[-1] = (d, H, W)

        if self.use_fusion:
            out, feat = self.fusion_head(pyramid)    # logits @ 1/4 + shared feature
        else:
            B, _, C = d.shape
            out = self.seg_head(d.transpose(1, 2).reshape(B, C, H, W))
            feat = None

        if not self.training:
            return out

        res = {"main": out}
        if self.aux_head is not None:
            a, aH, aW = aux
            a = a.transpose(1, 2).reshape(a.shape[0], -1, aH, aW)
            res["aux"] = self.aux_head(a)
        if self.boundary_head is not None and feat is not None:
            res["boundary"] = self.boundary_head(feat)
        return res


# =============================================================================
# Factory
# =============================================================================
def dekaht_unet_v2(variant: str = "swkat-tiny", patch_size: int = 4, in_chans: int = 3,
                   num_classes: int = 81, window_size: int = 7, mlp_ratio: float = 4.0,
                   drop_path_rate: float = 0.2, groups: int = 8,
                   decoder_depths: Tuple[int, ...] = (2, 2, 2), bottleneck_depth: int = 2,
                   gate_mode: str = "depthwise", head_dim: int = 256,
                   aux_head: bool = True, use_ppm: bool = True,
                   use_fusion: bool = True, fuse_dim: int = 128,
                   boundary_head: bool = True) -> DeKAHTUNetV2:
    if variant not in STANDARD_VARIANTS:
        raise ValueError(f"unknown variant {variant}; pick from {list(STANDARD_VARIANTS)}")
    v = STANDARD_VARIANTS[variant]
    return DeKAHTUNetV2(
        patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
        embed_dim=v["embed_dim"], depths=v["depths"], num_heads=v["num_heads"],
        decoder_depths=decoder_depths, bottleneck_depth=bottleneck_depth,
        window_size=window_size, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate,
        groups=groups, gate_mode=gate_mode, head_dim=head_dim,
        aux_head=aux_head, use_ppm=use_ppm, use_fusion=use_fusion,
        fuse_dim=fuse_dim, boundary_head=boundary_head)


def count_params(model) -> Dict[str, float]:
    g = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    out = {"encoder": g(model.encoder), "bottleneck": g(model.bottleneck),
           "up_stages": g(model.up_stages)}
    for name in ("ppm", "fusion_head", "seg_head", "aux_head", "boundary_head"):
        mod = getattr(model, name, None)
        if mod is not None:
            out[name] = g(mod)
    out["total"] = g(model)
    return out


# =============================================================================
# FIX A2 : function-preserving ImageNet transfer
# =============================================================================
@torch.no_grad()
def remap_swin_to_dekaht(ckpt, encoder: nn.Module, neutralise_kan: bool = True,
                         strict_coverage: float = 0.90, verbose: bool = True) -> nn.Module:
    """Load an official Swin checkpoint into SwkatGRKAN.

    Because the encoder now mirrors Swin's module layout, the ONLY remaining
    mismatches are:
      * mlp.fc{1,2}  Linear (O,I)  ->  mlp_conv.fc{1,2}  Conv2d (O,I,1,1)
      * KAN-only sub-modules (depthwise_spatial / gate / group_scale)

    Neutralisation makes GR_KAN_Conv compute EXACTLY Swin's MLP at step 0:
        depthwise -> dirac kernel      (identity)
        gate      -> zeros             (sigmoid(0) = 0.5 everywhere)
        fc2       -> x2                (cancels the 0.5)
        group_scale -> ones
    Verified: max|GR_KAN_Conv(x) - SwinMLP(x)| = 0.0
    """
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt

    tgt = encoder.state_dict()
    new, dropped, n = {}, [], 0

    for k, v in state.items():
        k = k.replace("module.", "").replace("backbone.", "").replace("encoder.", "")
        if k.startswith("head.") or "relative_position_index" in k or "attn_mask" in k:
            continue
        nk = k.replace("mlp.fc1", "mlp_conv.fc1").replace("mlp.fc2", "mlp_conv.fc2")
        if nk not in tgt:
            dropped.append(k)
            continue
        if v.ndim == 2 and tgt[nk].ndim == 4:
            v = v[:, :, None, None]                       # Linear -> 1x1 conv
        if v.shape != tgt[nk].shape:
            dropped.append(f"{k} {tuple(v.shape)}!={tuple(tgt[nk].shape)}")
            continue
        new[nk] = v.clone().to(tgt[nk].dtype)
        n += 1

    if neutralise_kan:
        for name, mod in encoder.named_modules():
            if not isinstance(mod, GR_KAN_Conv):
                continue
            hid = mod.hidden_channels
            if isinstance(mod.depthwise_spatial, nn.Conv2d):
                w = torch.zeros_like(mod.depthwise_spatial.weight)
                c = mod.depthwise_spatial.kernel_size[0] // 2
                w[:, 0, c, c] = 1.0
                new[f"{name}.depthwise_spatial.weight"] = w
                new[f"{name}.depthwise_spatial.bias"] = torch.zeros(hid)
            if mod.gate is not None:
                new[f"{name}.gate.0.weight"] = torch.zeros_like(mod.gate[0].weight)
                new[f"{name}.gate.0.bias"] = torch.zeros(hid)
                if f"{name}.fc2.weight" in new:
                    new[f"{name}.fc2.weight"] = new[f"{name}.fc2.weight"] * 2.0
            new[f"{name}.group_scale"] = torch.ones_like(mod.group_scale)

    missing, _ = encoder.load_state_dict(new, strict=False)
    n_tot = sum(p.numel() for p in encoder.parameters())
    n_ld = sum(tgt[k].numel() for k in new if k in tgt)
    cov = n_ld / max(1, n_tot)
    if verbose:
        print(f"[swin->dekaht] remapped {n} tensors | warm-started "
              f"{n_ld / 1e6:.2f}M / {n_tot / 1e6:.2f}M = {100 * cov:.1f}% of encoder")
        print(f"[swin->dekaht] left random: {len(missing)} | dropped: {len(dropped)}"
              + (f" e.g. {dropped[:3]}" if dropped else ""))
    if cov < strict_coverage:
        raise RuntimeError(
            f"Pretrained transfer covered only {100 * cov:.2f}% of encoder params "
            f"(need >= {100 * strict_coverage:.0f}%). This is exactly the silent "
            f"failure that cost the previous run ~10 mIoU -- refusing to start.")
    return encoder


@torch.no_grad()
def slim_kan_gates(model: nn.Module, verbose: bool = True) -> nn.Module:
    """Convert any 'full' (hidden x hidden) gates to depthwise. Call BEFORE
    remap_swin_to_dekaht so the fc2 x2 compensation still applies."""
    before = sum(p.numel() for p in model.parameters())
    for m in model.modules():
        if isinstance(m, GR_KAN_Conv) and m.gate is not None and m.gate_mode == "full":
            h = m.hidden_channels
            dw = nn.Conv2d(h, h, 3, padding=1, groups=h, bias=True)
            nn.init.zeros_(dw.weight)
            nn.init.zeros_(dw.bias)
            dev = m.fc1.weight.device
            m.gate = nn.Sequential(dw.to(dev), nn.Sigmoid())
            m.gate_mode = "depthwise"
    after = sum(p.numel() for p in model.parameters())
    if verbose and after != before:
        print(f"[slim_kan_gates] {before / 1e6:.1f}M -> {after / 1e6:.1f}M "
              f"(-{(before - after) / 1e6:.1f}M params)")
    return model


# =============================================================================
if __name__ == "__main__":
    m = dekaht_unet_v2("swkat-tiny", num_classes=81)
    print(m.__class__.__name__, {k: round(v, 2) for k, v in count_params(m).items()})
    m.eval()
    with torch.no_grad():
        for hw in [(448, 448), (512, 512), (384, 512)]:
            y = m(torch.randn(1, 3, *hw))
            print(f"  in {hw} -> logits {tuple(y.shape)} (1/4 res)")
    m.train()
    r = m(torch.randn(1, 3, 224, 224))
    print("  train outputs:", {k: tuple(v.shape) for k, v in r.items()})
