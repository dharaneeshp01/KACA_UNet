#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dekaht_unet_bcss5_UC1.py -- KAT-UNet architecture, UC1 release (BCSS 5-class).

A single self-contained architecture file that can build EVERY checkpoint produced
so far, so they can be weight-souped and prediction-ensembled:

  * legacy bcss5 KAT-UNet            (constructor defaults reproduce it bit-exactly)
  * Model 1  (A1 normalised skips + KA cross-attention bridge, A2 4-level UPer)
  * Model 2  (F0 rational Kolmogorov-Arnold FFN edges, B2 prototype-mixture head)
             -> convert_ffn_to_rational(), KAPrototypeHead  (applied per instance)

Nothing here changes the numerics of Model 1: the verified operators are carried
over unchanged. The UC1 training script adds the inference-side methods
(ROI-level Gaussian-blended sliding windows at native resolution, D4 test-time
augmentation, multi-model ensembling, greedy weight soups) and the fp16 fix.

LABEL CONTRACT: raw 0 -> IGNORE(255); raw 1..5 -> 0..4 (Tumor, Stroma,
Inflammatory, Necrosis, Other); 5 output channels.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GR_KAN_Conv", "WindowAttention", "SwkatBlockOpt", "PatchEmbed", "PatchMerging",
    "SwkatGRKAN_Opt", "STANDARD_VARIANTS", "PatchExpand", "FinalPatchExpandX4",
    "UpStage", "PyramidPooling", "DeKAHTUNet", "KATUNet", "dekaht_unet", "kat_unet", "FPNFuse",
    "BoundaryHead", "DSHead",
    "remap_swin_to_dekaht", "load_swin_pretrained", "set_prefer_kat",
    "RationalResidualAct", "KACrossAttention", "rational_residual",
    "convert_ffn_to_rational", "count_rational_ffn", "KAPrototypeHead",
]

PREFER_KAT = False


def set_prefer_kat(flag: bool):
    """Legacy no-op kept for API compatibility with the UC2 scripts."""
    global PREFER_KAT
    PREFER_KAT = bool(flag)


# ==============================================================================
# DropPath
# ==============================================================================
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    rand.floor_()
    return x.div(keep_prob) * rand


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


# ==============================================================================
# GR-KAN FFN  (M3)
# ==============================================================================
class GR_KAN_Conv(nn.Module):
    """
    Grouped-rational KAN FFN used in place of Swin's MLP.

        fc1(1x1) -> group_scale -> GELU -> depthwise(3x3) -> gate -> fc2(1x1)

    ffn_type ladder (unchanged semantics from UC2):
        'mlp'          fc1 -> GELU -> fc2                     (vanilla Swin MLP)
        'mlp_dw'       + depthwise 3x3
        'mlp_dw_gate'  + gating
        'kan'          + per-group learnable scale            (full GR-KAN)

    gate_mode:
        'residual_tanh' (default, NEW) : y = h * (1 + tanh(Wh + b)),  W=b=0 at init
                                         -> gate is EXACTLY 1 at step 0.
        'sigmoid'       (legacy UC2)   : y = h * sigmoid(Wh + b)
    """

    def __init__(self,
                 channels: int,
                 hidden_channels: Optional[int] = None,
                 groups: int = 8,
                 drop: float = 0.0,
                 spatial_kernel: int = 3,
                 use_gating: bool = True,
                 ffn_type: str = "kan",
                 gate_mode: str = "residual_tanh",
                 kernel_sizes: Optional[Sequence[int]] = None):
        super().__init__()
        hidden_channels = hidden_channels or (channels * 4)
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.groups = max(1, groups)
        self.ffn_type = ffn_type
        self.gate_mode = gate_mode
        self.drop = nn.Dropout2d(drop) if drop > 0.0 else nn.Identity()

        _use_dw = ffn_type in ("mlp_dw", "mlp_dw_gate", "kan")
        _use_gate = ffn_type in ("mlp_dw_gate", "kan")
        self.use_group_scale = (ffn_type == "kan")
        self.use_gating = bool(use_gating and _use_gate)
        self.spatial_kernel = (max(1, spatial_kernel) if _use_dw else 1)
        self.is_kan = (ffn_type == "kan")

        self.fc1 = nn.Conv2d(channels, hidden_channels, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1, bias=True)

        # ---- N1: MULTI-KERNEL GROUPED GR-KAN ----------------------------------
        # The Kolmogorov-Arnold decomposition already partitions the hidden width
        # into `groups` independent sub-spaces (that is what `group_scale` scales).
        # We give each sub-space its OWN spatial support, so a single FFN sees
        # 3x3, 5x5 and 7x7 receptive fields in parallel. Because every branch is
        # DEPTHWISE, channel count in == channel count out for each slice, and the
        # concatenation reconstructs exactly `hidden_channels`. fc1/fc2 shapes are
        # untouched, so the Swin checkpoint still loads at 100% coverage.
        self.kernel_spec: List[int] = []
        self.slices: List[Tuple[int, int]] = []
        self.dw_branches = nn.ModuleList()
        if _use_dw and self.spatial_kernel > 1:
            ks = sorted({int(k) for k in (kernel_sizes or [self.spatial_kernel])})
            ks = [k if k % 2 == 1 else k + 1 for k in ks]          # odd kernels only
            nb = len(ks)
            assert hidden_channels % nb == 0, \
                f"hidden_channels({hidden_channels}) must divide by #kernels({nb})"
            per = hidden_channels // nb
            start = 0
            for i, k in enumerate(ks):
                end = hidden_channels if i == nb - 1 else start + per
                w = end - start
                self.dw_branches.append(
                    nn.Conv2d(w, w, k, padding=(k - 1) // 2, groups=w, bias=True))
                self.kernel_spec.append(k)
                self.slices.append((start, end))
                start = end
            self.depthwise_spatial = self.dw_branches[0] if nb == 1 else None
        else:
            self.depthwise_spatial = nn.Identity()

        if self.use_group_scale:
            assert hidden_channels % self.groups == 0, "hidden_channels must be divisible by groups"
            self.group_scale = nn.Parameter(torch.ones(self.groups, hidden_channels // self.groups))
        else:
            self.group_scale = None

        self.act = nn.GELU()
        self.gate = nn.Conv2d(hidden_channels, hidden_channels, 1, bias=True) if self.use_gating else None

        self.reset_parameters()

    # ---- M3 / M5: canonical init. Call this LAST, after any module.apply(). ----
    def reset_parameters(self, swin_style_linear: bool = True):
        if swin_style_linear:
            nn.init.trunc_normal_(self.fc1.weight, std=0.02)
            nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        self.reset_kan_specific()

    def reset_kan_specific(self):
        """Put every KAN-only tensor at its identity value.

        After this call the module computes exactly  fc2(GELU(fc1(x))) — i.e.
        Swin's MLP — so Swin weights loaded into fc1/fc2 are numerically exact.
        """
        # Delta (identity) init for EVERY depthwise branch, whatever its kernel
        # size. This is what preserves GR_KAN_Conv(x) == SwinMLP(x) exactly at
        # step 0, which is what makes the pretrained transfer numerically exact.
        with torch.no_grad():
            for conv in self.dw_branches:
                conv.weight.zero_()
                k = conv.kernel_size[0]
                conv.weight[:, 0, k // 2, k // 2] = 1.0
                if conv.bias is not None:
                    conv.bias.zero_()
            if isinstance(self.depthwise_spatial, nn.Conv2d) and len(self.dw_branches) == 0:
                self.depthwise_spatial.weight.zero_()
                k = self.depthwise_spatial.kernel_size[0]
                self.depthwise_spatial.weight[:, 0, k // 2, k // 2] = 1.0
                if self.depthwise_spatial.bias is not None:
                    self.depthwise_spatial.bias.zero_()
        if self.group_scale is not None:
            with torch.no_grad():
                self.group_scale.fill_(1.0)
        if self.gate is not None:
            with torch.no_grad():
                self.gate.weight.zero_()
                if self.gate.bias is not None:
                    # residual_tanh -> tanh(0)=0 -> multiplier exactly 1
                    # sigmoid       -> sigmoid(4)=0.982 (closest safe identity)
                    self.gate.bias.fill_(0.0 if self.gate_mode == "residual_tanh" else 4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.fc1(x)
        if self.group_scale is not None and self.groups > 1:
            cpg = self.hidden_channels // self.groups
            h = h.view(B, self.groups, cpg, H, W)
            h = h * self.group_scale.view(1, self.groups, cpg, 1, 1)
            h = h.view(B, self.hidden_channels, H, W)
        h = self.act(h)
        if len(self.dw_branches) == 1:
            h = self.dw_branches[0](h)
        elif len(self.dw_branches) > 1:
            h = torch.cat([br(h[:, a:b]) for br, (a, b) in
                           zip(self.dw_branches, self.slices)], dim=1)
        if self.gate is not None:
            g = self.gate(h)
            h = h * (1.0 + torch.tanh(g)) if self.gate_mode == "residual_tanh" else h * torch.sigmoid(g)
        h = self.drop(h)
        out = self.fc2(h)
        return self.drop(out)

    def extra_repr(self) -> str:
        return (f"channels={self.channels}, hidden={self.hidden_channels}, groups={self.groups}, "
                f"kernels={self.kernel_spec or [self.spatial_kernel]}, "
                f"ffn={self.ffn_type}, gate={self.gate_mode}")


# ==============================================================================
# Window helpers
# ==============================================================================
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B,H,W,C) -> (nW*B, ws, ws, C); pads bottom/right when needed."""
    B, H, W, C = x.shape
    ws = min(window_size, H, W)
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1).contiguous()
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    ws = min(window_size, H, W)
    Hp = ((H + ws - 1) // ws) * ws
    Wp = ((W + ws - 1) // ws) * ws
    nW = (Hp // ws) * (Wp // ws)
    B = int(windows.shape[0] // nW)
    x = windows.view(B, Hp // ws, Wp // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous()


class WindowAttention(nn.Module):
    """Window MHSA with a learnable relative position bias (Swin-compatible)."""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ws = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij"))
        cf = coords.reshape(2, -1)
        rel = (cf[:, :, None] - cf[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        M = self.window_size * self.window_size
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(M, M, -1)
        bias = bias[:N, :N, :].permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0).to(attn.dtype)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(out))


# ==============================================================================
# Swkat block  (M2: Swin shift guard)
# ==============================================================================
class SwkatBlockOpt(nn.Module):
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int,
                 window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop_path: float = 0.0, groups: int = 8, ffn_type: str = "kan",
                 gate_mode: str = "residual_tanh",
                 kernel_sizes: Optional[Sequence[int]] = None):
        super().__init__()
        self.dim = dim
        self.ffn_type = ffn_type
        self.input_resolution = tuple(input_resolution)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        # ---- M2: official Swin behaviour. A single window cannot be shifted. ----
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if self.shift_size >= self.window_size:
            self.shift_size = 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_conv = GR_KAN_Conv(channels=dim, hidden_channels=int(dim * mlp_ratio),
                                    groups=groups, drop=0.0, ffn_type=ffn_type,
                                    gate_mode=gate_mode, kernel_sizes=kernel_sizes)

        self.register_buffer("attn_mask", None, persistent=False)
        self._mask_built_for: Optional[Tuple[int, int]] = None

    def _build_attn_mask(self, H: int, W: int, device=None):
        if self.shift_size <= 0 or min(H, W) <= self.window_size:
            self.attn_mask = None
            self._mask_built_for = (H, W)
            return
        img_mask = torch.zeros((1, H, W, 1), device=device)
        slices = (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mw = window_partition(img_mask, self.window_size)
        ws = mw.shape[1]
        mw = mw.view(-1, ws * ws)
        am = mw.unsqueeze(1) - mw.unsqueeze(2)
        am = am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)
        self.attn_mask = am
        self._mask_built_for = (H, W)

    def _ensure_attn_mask(self, H: int, W: int, device=None):
        if self._mask_built_for != (H, W):
            self._build_attn_mask(H, W, device=device)

    # kept for API compatibility with UC2 callers
    def _recompute_attn_mask(self, input_resolution, device=None):
        self.input_resolution = tuple(input_resolution)
        self._mask_built_for = None
        self._ensure_attn_mask(*self.input_resolution, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        Hh, Wh = self.input_resolution
        if L == Hh * Wh:
            H, W = Hh, Wh
        else:
            s = int(round(math.sqrt(L)))
            if s * s == L:
                H, W = s, s
            elif Wh > 0 and L % Wh == 0:
                H, W = L // Wh, Wh
            else:
                raise RuntimeError(f"SwkatBlockOpt: cannot infer (H,W) from L={L} (hint {self.input_resolution})")
        self.input_resolution = (H, W)

        # dynamic guard: never shift when the feature map is one window wide
        shift = self.shift_size if min(H, W) > self.window_size else 0
        if shift > 0:
            self._ensure_attn_mask(H, W, device=x.device)
            mask = self.attn_mask
        else:
            mask = None

        shortcut = x
        x_ = self.norm1(x).view(B, H, W, C)
        if shift > 0:
            x_ = torch.roll(x_, shifts=(-shift, -shift), dims=(1, 2))
        xw = window_partition(x_, self.window_size)
        ws_eff = xw.shape[1]
        xw = xw.reshape(xw.shape[0], -1, C)

        aw = self.attn(xw, mask=mask)
        aw = aw.view(-1, ws_eff, ws_eff, C)
        x_ = window_reverse(aw, self.window_size, H, W)
        if shift > 0:
            x_ = torch.roll(x_, shifts=(shift, shift), dims=(1, 2))

        x = shortcut + self.drop_path(x_.reshape(B, H * W, C))
        y = self.norm2(x).transpose(1, 2).reshape(B, C, H, W)
        y = self.mlp_conv(y).flatten(2).transpose(1, 2)
        return x + self.drop_path(y)

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, res={self.input_resolution}, "
                f"ws={self.window_size}, shift={self.shift_size}")


# ==============================================================================
# Patch embed / merge   (M1)
# ==============================================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96, use_norm: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # M1: official Swin has this LayerNorm; UC2 did not -> key was dropped.
        self.norm = nn.LayerNorm(embed_dim) if use_norm else nn.Identity()
        self.grid_size = (img_size // patch_size, img_size // patch_size)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), (H, W)


class PatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if (H % 2) or (W % 2):
            x = F.pad(x.permute(0, 3, 1, 2), (0, W % 2, 0, H % 2)).permute(0, 2, 3, 1).contiguous()
            H, W = x.shape[1], x.shape[2]
        x0, x1 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :]
        x2, x3 = x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x)), (H // 2, W // 2)


# ==============================================================================
# Encoder
# ==============================================================================
STANDARD_VARIANTS: Dict[str, Dict[str, Any]] = {
    "swkat-tiny":  {"embed_dim": 96,  "depths": (2, 2, 6, 2),  "num_heads": (3, 6, 12, 24)},
    "swkat-small": {"embed_dim": 96,  "depths": (2, 2, 18, 2), "num_heads": (3, 6, 12, 24)},
    "swkat-base":  {"embed_dim": 128, "depths": (2, 2, 18, 2), "num_heads": (4, 8, 16, 32)},
    "swkat-large": {"embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48)},
}


class SwkatGRKAN_Opt(nn.Module):
    """layers = [Seq0, PM0, Seq1, PM1, Seq2, PM2, Seq3]  (UC2 layout preserved)."""

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 num_classes: int = 0, embed_dim: int = 96,
                 depths: Tuple[int, ...] = (2, 2, 6, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24), window_size: int = 7,
                 mlp_ratio: float = 4.0, drop_path_rate: float = 0.1, groups: int = 8,
                 ffn_type: str = "kan", gate_mode: str = "residual_tanh",
                 kernel_sizes: Optional[Sequence[int]] = None):
        super().__init__()
        self.ffn_type = ffn_type
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, use_norm=True)
        self.pos_drop = nn.Dropout(0.0)
        self.layers = nn.ModuleList()

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur, dim = 0, embed_dim
        res = self.patch_embed.grid_size
        for s, depth in enumerate(depths):
            blocks = [
                SwkatBlockOpt(dim=dim, input_resolution=res, num_heads=num_heads[s],
                              window_size=window_size,
                              shift_size=0 if (i % 2 == 0) else window_size // 2,
                              mlp_ratio=mlp_ratio, drop_path=dpr[cur + i], groups=groups,
                              ffn_type=ffn_type, gate_mode=gate_mode,
                              kernel_sizes=kernel_sizes)
                for i in range(depth)
            ]
            self.layers.append(nn.Sequential(*blocks))
            cur += depth
            if s < len(depths) - 1:
                self.layers.append(PatchMerging(dim, dim * 2))
                dim *= 2
                res = (max(1, (res[0] + 1) // 2), max(1, (res[1] + 1) // 2))

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.img_size, self.patch_size, self.window_size = img_size, patch_size, window_size

        self.apply(self._init_weights)
        # M5: KAN tensors must be reset AFTER .apply(), otherwise the generic
        # Conv2d branch overwrites the identity depthwise / zero gate.
        for m in self.modules():
            if isinstance(m, GR_KAN_Conv):
                m.reset_kan_specific()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        x = self.pos_drop(x)
        for m in self.layers:
            if isinstance(m, PatchMerging):
                x, (H, W) = m(x, H, W)
            else:
                x = m(x)
        x = self.norm(x)
        return self.head(x.mean(1))


# ==============================================================================
# Swin -> DeKAHT remapping   (M4)
# ==============================================================================
def remap_swin_to_dekaht(swin_state: Dict[str, torch.Tensor],
                         model_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Translate an official Swin (or timm-Swin) checkpoint into DeKAHT encoder keys.

    Returns (remapped_state, report). `report` carries the coverage numbers that
    the training script hard-checks against --min-pretrained-coverage.
    """
    out: Dict[str, torch.Tensor] = {}
    dropped: List[str] = []
    shape_mismatch: List[str] = []

    for k, v in swin_state.items():
        key = k
        for pre in ("module.", "backbone.", "encoder."):
            if key.startswith(pre):
                key = key[len(pre):]
        # things that have no counterpart / are rebuilt at runtime
        if key.startswith("head.") or key.endswith("attn_mask"):
            dropped.append(k)
            continue

        nk: Optional[str] = None
        if key.startswith("patch_embed.") or key == "norm.weight" or key == "norm.bias":
            nk = key
        elif key.startswith("layers."):
            parts = key.split(".")
            s = int(parts[1])
            if parts[2] == "blocks":
                j = int(parts[3])
                tail = ".".join(parts[4:])
                # mlp.fc1 / mlp.fc2  ->  mlp_conv.fc1 / mlp_conv.fc2   (M3/M4)
                if tail.startswith("mlp.fc"):
                    tail = tail.replace("mlp.fc", "mlp_conv.fc", 1)
                nk = f"layers.{2 * s}.{j}.{tail}"
            elif parts[2] == "downsample":
                nk = f"layers.{2 * s + 1}." + ".".join(parts[3:])
        if nk is None:
            dropped.append(k)
            continue
        if nk not in model_state:
            dropped.append(k)
            continue

        tgt = model_state[nk]
        if v.shape != tgt.shape:
            # Linear (Cout,Cin) -> 1x1 Conv (Cout,Cin,1,1)
            if v.ndim == 2 and tgt.ndim == 4 and tgt.shape[2:] == (1, 1) and v.shape == tgt.shape[:2]:
                v = v.view(*tgt.shape)
            elif v.ndim == 4 and tgt.ndim == 2 and v.shape[2:] == (1, 1) and v.shape[:2] == tgt.shape:
                v = v.view(*tgt.shape)
            else:
                shape_mismatch.append(f"{k} {tuple(v.shape)} -> {nk} {tuple(tgt.shape)}")
                continue
        out[nk] = v

    # KAN-only tensors that legitimately have no Swin counterpart
    # KAN-only tensors have no Swin counterpart by construction. `dw_branches`
    # holds the multi-kernel depthwise convolutions (N1); omitting it here would
    # make the coverage report understate the true transfer and trip the 90% gate.
    kan_only = [k for k in model_state
                if ("group_scale" in k or "depthwise_spatial" in k
                    or "dw_branches" in k or ".gate." in k)]
    transferable = [k for k in model_state if k not in kan_only and "relative_position_index" not in k]
    covered = [k for k in transferable if k in out]

    n_par = sum(model_state[k].numel() for k in transferable)
    n_cov = sum(model_state[k].numel() for k in covered)

    report = {
        "matched_tensors": len(covered),
        "transferable_tensors": len(transferable),
        "tensor_coverage": len(covered) / max(1, len(transferable)),
        "matched_params": n_cov,
        "transferable_params": n_par,
        "param_coverage": n_cov / max(1, n_par),
        "kan_only_tensors": len(kan_only),
        "unmatched_model_keys": sorted(set(transferable) - set(covered))[:20],
        "dropped_ckpt_keys": len(dropped),
        "shape_mismatch": shape_mismatch,
    }
    return out, report


def load_swin_pretrained(model: nn.Module, ckpt_path_or_url: str,
                         min_coverage: float = 0.90, logger=None) -> Dict[str, Any]:
    """Load Swin weights into `model.encoder`; raise if coverage < min_coverage."""
    if str(ckpt_path_or_url).startswith("http"):
        ck = torch.hub.load_state_dict_from_url(ckpt_path_or_url, map_location="cpu")
    else:
        ck = torch.load(ckpt_path_or_url, map_location="cpu", weights_only=False)
    state = ck.get("model", ck.get("state_dict", ck))

    enc = model.encoder if hasattr(model, "encoder") else model
    remapped, rep = remap_swin_to_dekaht(state, enc.state_dict())
    missing, unexpected = enc.load_state_dict(remapped, strict=False)
    rep["missing_after_load"] = len(missing)
    rep["unexpected_after_load"] = len(unexpected)

    # KAN parts must be back at identity after the load (they were not in the ckpt,
    # but re-assert it so the block is provably == Swin MLP at step 0).
    for m in enc.modules():
        if isinstance(m, GR_KAN_Conv):
            m.reset_kan_specific()

    msg = (f"[pretrained] tensors {rep['matched_tensors']}/{rep['transferable_tensors']} "
           f"({100 * rep['tensor_coverage']:.1f}%) | params "
           f"{rep['matched_params'] / 1e6:.2f}M/{rep['transferable_params'] / 1e6:.2f}M "
           f"({100 * rep['param_coverage']:.1f}%)")
    (logger.info if logger else print)(msg)
    if rep["shape_mismatch"]:
        (logger.warning if logger else print)(f"[pretrained] shape mismatches: {rep['shape_mismatch'][:5]}")

    if rep["param_coverage"] < min_coverage:
        raise RuntimeError(
            f"Pretrained coverage {100 * rep['param_coverage']:.2f}% < required "
            f"{100 * min_coverage:.0f}%. Refusing to burn GPU-days on a cold encoder. "
            f"First unmatched keys: {rep['unmatched_model_keys'][:10]}")
    return rep


# ==============================================================================
# Decoder primitives
# ==============================================================================
class PatchExpand(nn.Module):
    """Spatial x2, channels /2 — exact inverse of PatchMerging."""

    def __init__(self, in_dim: int, out_dim: Optional[int] = None, norm_layer=nn.LayerNorm):
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim // 2
        self.in_dim, self.out_dim = in_dim, out_dim
        self.expand = nn.Linear(in_dim, 2 * in_dim, bias=False)
        self.norm = norm_layer(in_dim // 2)
        self.proj = nn.Linear(in_dim // 2, out_dim) if out_dim != in_dim // 2 else nn.Identity()

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"PatchExpand: L({L}) != H*W({H * W})"
        x = self.expand(x).view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * 2, W * 2, C // 2)
        x = x.view(B, -1, C // 2)
        return self.proj(self.norm(x)), (H * 2, W * 2)


class FinalPatchExpandX4(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim, self.dim_scale = dim, dim_scale
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        s = self.dim_scale
        assert L == H * W, f"FinalPatchExpandX4: L({L}) != H*W({H * W})"
        x = self.expand(x).view(B, H, W, s, s, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * s, W * s, C)
        return self.norm(x.view(B, -1, C)), (H * s, W * s)


def align_tokens(x: torch.Tensor, H: int, W: int, tH: int, tW: int) -> torch.Tensor:
    """Crop / zero-pad a (B, H*W, C) token map to (B, tH*tW, C)."""
    B, L, C = x.shape
    f = x.transpose(1, 2).reshape(B, C, H, W)
    f = f[:, :, :min(H, tH), :min(W, tW)]
    if f.shape[-2] != tH or f.shape[-1] != tW:
        f = F.pad(f, (0, tW - f.shape[-1], 0, tH - f.shape[-2]))
    return f.flatten(2).transpose(1, 2)


# =============================================================================
# Group-rational residual operator (shared by Model 1's bridge and Model 2's KAN)
# =============================================================================
class _RationalResidualFn(torch.autograd.Function):
    """rho_g(x) = P_g(x) / (1 + |R_g(x)|),  P of degree m, R of degree n (no constant).

    Input is viewed as (B, G, M): coefficients are shared inside each channel group g.
    Forward saves ONLY (x, a, b); backward recomputes every power of x. A naive
    autograd graph of the Horner evaluation would keep ~15 full-size intermediates per
    call -- several GB across a 30-block U-Net on a 16 GB card.

    Always evaluated in fp32 (fp64 when the input is fp64, for gradcheck), whatever
    the autocast dtype: x**5 in fp16/bf16 loses the precision the fit depends on.
    Denominator >= 1 by construction, so no pole and no epsilon are required.
    """

    @staticmethod
    def _cd(x):
        return torch.float64 if x.dtype == torch.float64 else torch.float32

    @staticmethod
    def forward(ctx, x3, a, b):
        cd = _RationalResidualFn._cd(x3)
        x = x3.to(cd)
        af, bf = a.to(cd), b.to(cd)
        m, n = af.shape[1] - 1, bf.shape[1]
        P = af[:, m].view(1, -1, 1).expand_as(x).clone()
        for i in range(m - 1, -1, -1):
            P = P * x + af[:, i].view(1, -1, 1)
        R = bf[:, n - 1].view(1, -1, 1) * x
        for j in range(n - 2, -1, -1):
            R = (R + bf[:, j].view(1, -1, 1)) * x
        y = P / (1.0 + R.abs())
        ctx.save_for_backward(x3, a, b)
        return y.to(x3.dtype)

    @staticmethod
    def backward(ctx, gy):
        x3, a, b = ctx.saved_tensors
        cd = _RationalResidualFn._cd(x3)
        x = x3.to(cd)
        af, bf = a.to(cd), b.to(cd)
        g = gy.to(cd)
        m, n = af.shape[1] - 1, bf.shape[1]
        P = af[:, m].view(1, -1, 1).expand_as(x).clone()
        dP = torch.zeros_like(x)
        for i in range(m - 1, -1, -1):
            dP = dP * x + P
            P = P * x + af[:, i].view(1, -1, 1)
        R = bf[:, n - 1].view(1, -1, 1) * x
        for j in range(n - 2, -1, -1):
            R = (R + bf[:, j].view(1, -1, 1)) * x
        dR = n * bf[:, n - 1].view(1, -1, 1).expand_as(x).clone()
        for j in range(n - 2, -1, -1):
            dR = dR * x + (j + 1) * bf[:, j].view(1, -1, 1)
        sR = torch.sign(R)
        Q = 1.0 + R.abs()
        invQ = 1.0 / Q
        gx = ga = gb = None
        if ctx.needs_input_grad[0]:
            gx = (g * (dP * Q - P * sR * dR) * invQ * invQ).to(x3.dtype)
        if ctx.needs_input_grad[1]:
            base = g * invQ
            cols, pw = [], torch.ones_like(x)
            for _ in range(m + 1):
                cols.append((base * pw).sum(dim=(0, 2)))
                pw = pw * x
            ga = torch.stack(cols, 1).to(a.dtype)
        if ctx.needs_input_grad[2]:
            base = -g * P * sR * invQ * invQ
            cols, pw = [], x.clone()
            for _ in range(n):
                cols.append((base * pw).sum(dim=(0, 2)))
                pw = pw * x
            gb = torch.stack(cols, 1).to(b.dtype)
        return gx, ga, gb


def rational_residual(x3: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _RationalResidualFn.apply(x3, a, b)


class RationalResidualAct(nn.Module):
    """phi_g(x) = base(x) + P_g(x) / (1 + |R_g(x)|)   -- a Kolmogorov-Arnold edge function.

    This is the KAN parametrisation of Liu et al. (base function + learnable
    univariate correction), with the B-spline replaced by a safe Pade rational
    (KAT, Yang & Wang).  base in {"gelu", "identity"}.

    Initialisation: a = 0 (residual EXACTLY zero -> phi == base, bit-exact) and
    b = (0, ..., 0, 1) so Q = 1 + x^n from the first step. With b = 0 the residual
    would start as a raw degree-m polynomial whose x^5 coefficient receives
    gradient ~x^5 (3125x at |x|=5); with Q = 1 + x^4 that gradient is x^5/(1+x^4),
    i.e. ~linear. deg_num = deg_den + 1 keeps phi = Theta(|x|) asymptotically.

    Parameters are named `rat_a` / `rat_b` so the optimiser can give them their own
    group (lower LR, no weight decay).
    """

    def __init__(self, channels: int, groups: int = 8, deg_num: int = 5, deg_den: int = 4,
                 base: str = "gelu", channel_dim: int = 1):
        super().__init__()
        assert channels % groups == 0, f"channels {channels} not divisible by groups {groups}"
        assert deg_num >= 1 and deg_den >= 1
        assert base in ("gelu", "identity")
        self.channels, self.groups = channels, groups
        self.base, self.channel_dim = base, channel_dim
        self.rat_a = nn.Parameter(torch.zeros(groups, deg_num + 1))
        b0 = torch.zeros(groups, deg_den)
        b0[:, -1] = 1.0
        self.rat_b = nn.Parameter(b0)

    def reset_parameters(self):
        with torch.no_grad():
            self.rat_a.zero_()
            self.rat_b.zero_()
            self.rat_b[:, -1] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.gelu(x) if self.base == "gelu" else x
        cd = self.channel_dim if self.channel_dim >= 0 else x.dim() + self.channel_dim
        xm = x.movedim(cd, 1) if cd != 1 else x
        shp = xm.shape
        x3 = xm.reshape(shp[0], self.groups, -1)
        r = rational_residual(x3, self.rat_a, self.rat_b).reshape(shp)
        if cd != 1:
            r = r.movedim(1, cd)
        return base + r

    def extra_repr(self):
        return (f"channels={self.channels}, groups={self.groups}, base={self.base}, "
                f"deg=({self.rat_a.shape[1] - 1},{self.rat_b.shape[1]})")


# =============================================================================
# A1 -- direction-corrected, windowed Kolmogorov-Arnold cross-attention bridge
# =============================================================================
class KACrossAttention(nn.Module):
    """Decoder queries, encoder-skip keys/values, inside non-overlapping windows.

        q = phi(W_q LN(D)),  k = psi(W_k LN(S)),  v = W_v LN(S)
        A = softmax_j( q_i.k_j / sqrt(d_h) + B_ij ) over valid keys in the window
        out = gamma * W_o (A v)                              (gamma = 0 at init)

    phi/psi are per-head RationalResidualAct(base="identity") -> exactly standard
    attention at init. Padded key positions (grids not divisible by the window,
    e.g. 18x18 under 1.25x TTA) are masked with -1e4 in fp32; every window always
    holds at least one valid key because padding < window.
    Returns ONLY the residual delta (B, L, C); the caller adds it.
    """

    def __init__(self, dim: int, num_heads: int, window: int = 7, rational: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.heads, self.window = dim, num_heads, int(window)
        self.dh = dim // num_heads
        self.scale = self.dh ** -0.5
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.phi = (RationalResidualAct(dim, groups=num_heads, base="identity", channel_dim=-1)
                    if rational else nn.Identity())
        self.psi = (RationalResidualAct(dim, groups=num_heads, base="identity", channel_dim=-1)
                    if rational else nn.Identity())
        ws = self.window
        self.rpb_table = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij"))
        cf = coords.reshape(2, -1)
        rel = (cf[:, :, None] - cf[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("rpb_index", rel.sum(-1), persistent=False)
        self.gamma = nn.Parameter(torch.zeros(dim))
        for lin in (self.q, self.kv, self.proj):
            nn.init.trunc_normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    @staticmethod
    def _partition(x, ws):
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(-1, ws * ws, C)

    def forward(self, xq: torch.Tensor, xkv: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = xq.shape
        assert L == H * W and xkv.shape == xq.shape, \
            f"KACA shape mismatch q{tuple(xq.shape)} kv{tuple(xkv.shape)} HW=({H},{W})"
        ws = self.window
        ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
        Hp, Wp = H + ph, W + pw
        q_in = self.ln_q(xq).view(B, H, W, C)
        kv_in = self.ln_kv(xkv).view(B, H, W, C)
        if ph or pw:
            q_in = F.pad(q_in, (0, 0, 0, pw, 0, ph))
            kv_in = F.pad(kv_in, (0, 0, 0, pw, 0, ph))
        qw = self._partition(q_in, ws)                         # (B*nW, N, C)
        kvw = self._partition(kv_in, ws)
        nW = (Hp // ws) * (Wp // ws)
        N = ws * ws
        q = self.phi(self.q(qw))
        k, v = self.kv(kvw).chunk(2, dim=-1)
        k = self.psi(k)
        q = q.view(-1, N, self.heads, self.dh).transpose(1, 2)
        k = k.view(-1, N, self.heads, self.dh).transpose(1, 2)
        v = v.view(-1, N, self.heads, self.dh).transpose(1, 2)
        attn = (q.float() * self.scale) @ k.float().transpose(-2, -1)   # fp32 logits
        bias = self.rpb_table[self.rpb_index.reshape(-1)].view(N, N, self.heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0).float()
        if ph or pw:
            valid = torch.zeros(1, Hp, Wp, 1, device=xq.device)
            valid[:, :H, :W, :] = 1.0
            vw = self._partition(valid, ws).view(nW, N)               # (nW, N)
            neg = (1.0 - vw).view(1, nW, 1, 1, N) * (-1e4)
            attn = (attn.view(B, nW, self.heads, N, N) + neg).view(-1, self.heads, N, N)
        attn = attn.clamp(min=-1e4, max=50.0).softmax(dim=-1)
        out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(-1, N, C)
        out = self.proj(out)
        out = out.view(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, Hp, Wp, C)[:, :H, :W, :].reshape(B, L, C)
        return out * self.gamma


def _delta_dwconv(dim: int) -> nn.Conv2d:
    """Depthwise 3x3 initialised to the identity kernel (exact no-op at init)."""
    c = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
    with torch.no_grad():
        c.weight.zero_()
        c.weight[:, 0, 1, 1] = 1.0
        c.bias.zero_()
    return c


def _run_ckpt(fn, x):
    """Gradient checkpointing that works on torch 1.10 through 2.x."""
    from torch.utils.checkpoint import checkpoint
    try:
        return checkpoint(fn, x, use_reentrant=False)
    except TypeError:                                   # torch < 1.11: no use_reentrant
        return checkpoint(fn, x)


class UpStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, depth, input_resolution,
                 window_size, mlp_ratio, groups, dpr, ffn_type="kan",
                 gate_mode="residual_tanh", kernel_sizes=None,
                 skip_norm: bool = False, skip_bridge: str = "linear",
                 kaca_window: int = 7, kaca_rational: bool = True):
        super().__init__()
        assert skip_bridge in ("linear", "kaca"), skip_bridge
        self.expand = PatchExpand(in_dim, out_dim)
        self.fuse = nn.Linear(2 * out_dim, out_dim)
        # A1: the raw encoder residual stream (5.8x the decoder RMS at the deepest
        # fusion) is layer-normalised before it meets the LayerNorm-ed decoder path.
        self.skip_norm = nn.LayerNorm(out_dim) if skip_norm else None
        self.bridge = (KACrossAttention(out_dim, num_heads, window=kaca_window,
                                        rational=kaca_rational)
                       if skip_bridge == "kaca" else None)
        self.grad_ckpt = False
        self.blocks = nn.ModuleList([
            SwkatBlockOpt(dim=out_dim, input_resolution=input_resolution, num_heads=num_heads,
                          window_size=window_size,
                          shift_size=0 if (i % 2 == 0) else window_size // 2,
                          mlp_ratio=mlp_ratio, drop_path=float(dpr[i]), groups=groups,
                          ffn_type=ffn_type, gate_mode=gate_mode,
                          kernel_sizes=kernel_sizes)
            for i in range(depth)])

    def forward(self, x, H, W, skip, skip_hw=None):
        x, (H, W) = self.expand(x, H, W)
        if skip_hw is not None and (H, W) != tuple(skip_hw):
            # PatchMerging pads odd grids (e.g. 7 -> 8 -> 4), so the x2 expansion can
            # overshoot the skip. Align by centre-crop / zero-pad instead of asserting.
            x = align_tokens(x, H, W, int(skip_hw[0]), int(skip_hw[1]))
            H, W = int(skip_hw[0]), int(skip_hw[1])
        assert x.shape[1] == skip.shape[1], f"skip len mismatch {tuple(x.shape)} vs {tuple(skip.shape)}"
        assert x.shape[2] == skip.shape[2], f"skip ch mismatch {tuple(x.shape)} vs {tuple(skip.shape)}"
        s_in = self.skip_norm(skip) if self.skip_norm is not None else skip
        y = self.fuse(torch.cat([x, s_in], -1))
        if self.bridge is not None:
            y = y + self.bridge(x, skip, H, W)     # decoder asks, encoder answers
        for blk in self.blocks:
            y = _run_ckpt(blk, y) if (self.grad_ckpt and self.training) else blk(y)
        return y, (H, W)


class PyramidPooling(nn.Module):
    """
    PSPNet/UPerNet pyramid pooling on the 7x7 bottleneck, applied RESIDUALLY with a
    zero-initialised output conv, so the module is an exact identity at step 0 and
    cannot perturb the freshly-loaded pretrained features.
    """

    def __init__(self, dim: int, bins: Sequence[int] = (1, 2, 3, 6), reduction: int = 8):
        super().__init__()
        mid = max(32, dim // reduction)
        self.bins = tuple(bins)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(dim, mid, 1, bias=False),
                          nn.GroupNorm(8, mid), nn.GELU())
            for b in self.bins])
        self.out = nn.Sequential(
            nn.Conv2d(dim + mid * len(self.bins), dim, 1, bias=False),      # cheap fuse
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),      # depthwise context
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=True))                              # zero-init gate
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, H, W):
        B, L, C = x.shape
        f = x.transpose(1, 2).reshape(B, C, H, W)
        feats = [f] + [F.interpolate(s(f), size=(H, W), mode="bilinear", align_corners=False)
                       for s in self.stages]
        r = self.out(torch.cat(feats, 1))
        return x + r.flatten(2).transpose(1, 2)


class FPNFuse(nn.Module):
    """
    UC5-A: UPerNet-style multi-level fusion.

    UC3 sent ONLY the last decoder stage to the classifier; the 14x14 and 28x28
    decoder features reached the head only through the up-sampling chain. Here every
    decoder level is projected to `dim`, resized to the finest grid, and concatenated.

    Applied as a ZERO-INIT RESIDUAL on top of the finest decoder level:

        out = finest + gate(GELU(GN(fuse(concat(all levels)))))

    with `gate` zero-initialised. At step 0 `out == finest`, i.e. EXACTLY the UC3
    path, so the new branch cannot destabilise a run that was previously working;
    it earns its contribution through training (ReZero / LayerScale recipe).
    """

    def __init__(self, in_dims: Sequence[int], dim: int, top_dim: Optional[int] = None,
                 smooth: bool = False):
        super().__init__()
        # identity projection on the finest level so the residual is EXACT at step 0
        self.projs = nn.ModuleList([nn.Linear(d, dim) if d != dim else nn.Identity()
                                    for d in in_dims])
        # A2: delta-initialised depthwise 3x3 after every bilinear upsample (identity
        # at init) and the PPM-refined bottleneck as a 4th, zero-gated level.
        self.smooth = (nn.ModuleList([_delta_dwconv(dim) for _ in range(len(in_dims) - 1)])
                       if smooth else None)
        self.top_proj = nn.Linear(top_dim, dim) if top_dim else None
        self.top_smooth = _delta_dwconv(dim) if top_dim else None
        self.top_gamma = nn.Parameter(torch.zeros(dim)) if top_dim else None
        self.fuse = nn.Conv2d(dim * len(in_dims), dim, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, dim)
        self.act = nn.GELU()
        self.gate = nn.Conv2d(dim, dim, 1, bias=True)
        self.n = len(in_dims)
        self.dim = dim
        self._init_identity()

    def _init_identity(self):
        with torch.no_grad():
            nn.init.trunc_normal_(self.fuse.weight, std=0.02)
            self.norm.weight.fill_(1.0)
            self.norm.bias.zero_()
            self.gate.weight.zero_()          # <-- residual starts at exactly zero
            self.gate.bias.zero_()
            if self.smooth is not None:
                for c in self.smooth:
                    c.weight.zero_(); c.weight[:, 0, 1, 1] = 1.0; c.bias.zero_()
            if self.top_proj is not None:
                nn.init.trunc_normal_(self.top_proj.weight, std=0.02)
                nn.init.zeros_(self.top_proj.bias)
                self.top_smooth.weight.zero_(); self.top_smooth.weight[:, 0, 1, 1] = 1.0
                self.top_smooth.bias.zero_()
                self.top_gamma.zero_()

    def forward(self, feats, hws, out_hw, top=None, top_hw=None):
        H, W = out_hw
        outs = []
        for i, (t, (h, w), proj) in enumerate(zip(feats, hws, self.projs)):
            B = t.shape[0]
            f = proj(t).transpose(1, 2).reshape(B, self.dim, h, w)
            if (h, w) != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
                if self.smooth is not None and i < len(self.smooth):
                    f = self.smooth[i](f)
            outs.append(f)
        finest = outs[-1]
        out = finest + self.gate(self.act(self.norm(self.fuse(torch.cat(outs, 1)))))
        if self.top_proj is not None and top is not None:
            th, tw = top_hw
            B = top.shape[0]
            f = self.top_proj(top).transpose(1, 2).reshape(B, self.dim, th, tw)
            f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            out = out + self.top_gamma.view(1, -1, 1, 1) * self.top_smooth(f)
        return out


class BoundaryHead(nn.Module):
    """
    UC5-B: explicit boundary stream (Gated-SCNN / BASeg family).

    Produces (a) a supervised 1-channel boundary logit and (b) a boundary feature that
    is concatenated back into the segmentation feature. `post` is initialised to the
    identity on the segmentation half and zero on the boundary half, so at step 0 the
    branch is a no-op.
    """

    def __init__(self, dim: int, mid: int = 32):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, mid, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, mid), nn.GELU())
        self.cls = nn.Conv2d(mid, 1, 1)
        self.post = nn.Conv2d(dim + mid, dim, 1, bias=True)
        self.dim, self.mid = dim, mid
        with torch.no_grad():
            self.post.weight.zero_()
            self.post.bias.zero_()
            for c in range(dim):
                self.post.weight[c, c, 0, 0] = 1.0
            nn.init.zeros_(self.cls.bias)

    def forward(self, f):
        b = self.body(f)
        logit = self.cls(b)
        return self.post(torch.cat([f, b], 1)), logit


class DSHead(nn.Module):
    """
    Deep-supervision head — the UC6 post-mortem fix (POST-MORTEM ITEM 2).

    UC6 used a BARE `nn.Conv2d(dim, C, 1)` on the raw, un-normalised decoder
    residual stream. A transformer residual stream grows in magnitude during
    training and cross-entropy grows LINEARLY with logit magnitude, so there was
    no saturation point: larger residual -> larger loss -> larger gradient ->
    larger residual. The auxiliary term reached 1.06e9 while the main objective
    stayed at 1.5, and the run lost 4.9 mDice.

    Three structural guarantees here make that impossible:
      1. GroupNorm + GELU BEFORE the classifier bounds the input magnitude, so
         the logits cannot run away with the residual stream.
      2. The classifier is ZERO-initialised: at step 0 every DS head emits exactly
         0 logits (uniform posterior), contributing a constant loss with zero
         gradient w.r.t. the decoder. The branch has to EARN its influence.
      3. `forward` returns logits only; the loss weighting/budget is enforced in
         the trainer (see train_bcss5.py::ds_weights, total <= 0.35).
    """

    def __init__(self, in_dim: int, num_classes: int, mid: Optional[int] = None,
                 drop: float = 0.0):
        super().__init__()
        mid = mid or max(32, min(256, in_dim // 2))
        groups = max(1, min(32, in_dim // 4))
        while in_dim % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, in_dim)        # (1) bound the magnitude
        self.act = nn.GELU()
        self.proj = nn.Conv2d(in_dim, mid, 3, padding=1, bias=False)
        gm = max(1, min(32, mid // 4))
        while mid % gm != 0 and gm > 1:
            gm -= 1
        self.norm2 = nn.GroupNorm(gm, mid)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.classifier = nn.Conv2d(mid, num_classes, 1, bias=True)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.classifier.weight)          # (2) zero-init
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens: torch.Tensor, H: int, W: int,
                out_hw: Tuple[int, int]) -> torch.Tensor:
        B, L, C = tokens.shape
        f = tokens.transpose(1, 2).reshape(B, C, H, W)
        f = self.act(self.norm(f))
        f = self.drop(self.act2(self.norm2(self.proj(f))))
        f = self.classifier(f)
        return F.interpolate(f, size=out_hw, mode="bilinear", align_corners=False)


class AuxHead(nn.Module):
    """FCN auxiliary head for deep supervision (dropped at inference)."""

    def __init__(self, in_dim: int, num_classes: int, mid: int = 256, drop: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, mid, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid), nn.GELU(), nn.Dropout2d(drop))
        self.cls = nn.Conv2d(mid, num_classes, 1)
        nn.init.zeros_(self.cls.weight)      # neutral at step 0, same rule as DSHead
        nn.init.zeros_(self.cls.bias)

    def forward(self, tokens, H, W, out_hw):
        B, L, C = tokens.shape
        f = tokens.transpose(1, 2).reshape(B, C, H, W)
        f = self.cls(self.body(f))
        return F.interpolate(f, size=out_hw, mode="bilinear", align_corners=False)


# ==============================================================================
# DeKAHT-UNet
# ==============================================================================
class DeKAHTUNet(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 num_classes: int = 3, embed_dim: int = 96,
                 depths: Tuple[int, ...] = (2, 2, 6, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24),
                 decoder_depths: Tuple[int, ...] = (2, 2, 2),
                 bottleneck_depth: int = 2, window_size: int = 7, mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.1, groups: int = 8, ffn_type: str = "kan",
                 gate_mode: str = "residual_tanh", use_ppm: bool = True,
                 use_aux: bool = True, head_dropout: float = 0.1,
                 use_fpn: bool = True, use_boundary: bool = True,
                 deep_supervision: bool = True,
                 kernel_sizes: Optional[Sequence[int]] = (3, 5, 7),
                 skip_norm: bool = False, skip_bridge: str = "linear",
                 kaca_window: int = 7, kaca_rational: bool = True,
                 uper_levels: int = 3, fpn_smooth: bool = False,
                 grad_ckpt: bool = False):
        super().__init__()
        assert len(depths) == 4 and len(num_heads) == 4
        assert uper_levels in (3, 4), uper_levels
        assert len(decoder_depths) == 3
        self.ffn_type = ffn_type
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.encoder = SwkatGRKAN_Opt(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=0,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate, groups=groups,
            ffn_type=ffn_type, gate_mode=gate_mode, kernel_sizes=kernel_sizes)

        self.stage_dims = [embed_dim * 2 ** i for i in range(4)]
        grid = img_size // patch_size
        self.stage_res = [(grid // 2 ** i, grid // 2 ** i) for i in range(4)]
        deepest = self.stage_dims[-1]

        self.ppm = PyramidPooling(deepest) if use_ppm else None

        b_dpr = torch.linspace(0, drop_path_rate, max(bottleneck_depth, 1)).tolist()
        self.bottleneck = nn.ModuleList([
            SwkatBlockOpt(dim=deepest, input_resolution=self.stage_res[-1],
                          num_heads=num_heads[-1], window_size=window_size,
                          shift_size=0 if (i % 2 == 0) else window_size // 2,
                          mlp_ratio=mlp_ratio, drop_path=float(b_dpr[i]), groups=groups,
                          ffn_type=ffn_type, gate_mode=gate_mode,
                          kernel_sizes=kernel_sizes)
            for i in range(bottleneck_depth)])

        dec_dpr = torch.linspace(0, drop_path_rate, max(1, sum(decoder_depths))).tolist()
        self.up_stages = nn.ModuleList()
        cur = 0
        for k in range(3):
            d = decoder_depths[k]
            self.up_stages.append(UpStage(
                in_dim=self.stage_dims[3 - k], out_dim=self.stage_dims[2 - k],
                num_heads=num_heads[2 - k], depth=d, input_resolution=self.stage_res[2 - k],
                window_size=window_size, mlp_ratio=mlp_ratio, groups=groups,
                dpr=dec_dpr[cur:cur + d], ffn_type=ffn_type, gate_mode=gate_mode,
                kernel_sizes=kernel_sizes, skip_norm=skip_norm, skip_bridge=skip_bridge,
                kaca_window=kaca_window, kaca_rational=kaca_rational))
            cur += d

        # UC5-A / UC5-B / Model-1 A2 (4th level + smoothing)
        self.fpn = FPNFuse([self.stage_dims[2], self.stage_dims[1], self.stage_dims[0]],
                           self.stage_dims[0],
                           top_dim=deepest if uper_levels == 4 else None,
                           smooth=fpn_smooth) if use_fpn else None
        self.arch_flags = dict(skip_norm=skip_norm, skip_bridge=skip_bridge,
                               kaca_window=kaca_window, kaca_rational=kaca_rational,
                               uper_levels=uper_levels, fpn_smooth=fpn_smooth)
        self.grad_ckpt = False
        self.boundary = BoundaryHead(self.stage_dims[0]) if use_boundary else None

        self.norm_up = nn.LayerNorm(self.stage_dims[0])
        self.final_expand = FinalPatchExpandX4(dim=self.stage_dims[0], dim_scale=patch_size)
        # M6: 3x3 refine before the classifier (the bare 1x1 gave blocky borders)
        self.refine = nn.Sequential(
            nn.Conv2d(self.stage_dims[0], self.stage_dims[0], 3, padding=1, bias=False),
            nn.GroupNorm(8, self.stage_dims[0]), nn.GELU(),
            nn.Dropout2d(head_dropout))
        self.seg_head = nn.Conv2d(self.stage_dims[0], num_classes, 1, bias=True)

        self.aux_head = AuxHead(self.stage_dims[2], num_classes) if use_aux else None
        # UC6-A: deep supervision on EVERY decoder stage (14x14, 28x28, 56x56).
        # UC5 supervised only one encoder-side branch; the decoder itself got
        # gradient only through the final head.
        # Deep supervision on every decoder stage. Ordered DEEPEST -> SHALLOWEST
        # (stage_dims[2]=384 @14x14, [1]=192 @28x28, [0]=96 @56x56) so the trainer
        # can decay the loss weight towards the shallow end (POST-MORTEM ITEM 1).
        self.ds_heads = nn.ModuleList([
            DSHead(self.stage_dims[2 - k], num_classes, drop=head_dropout)
            for k in range(3)
        ]) if deep_supervision else None

        self._init_decoder_weights()
        self.set_grad_checkpointing(grad_ckpt)

    def set_grad_checkpointing(self, flag: bool = True):
        """Recompute Swin/KAT blocks in backward: large activation-memory saving."""
        self.grad_ckpt = bool(flag)
        for up in self.up_stages:
            up.grad_ckpt = bool(flag)

    def _init_decoder_weights(self):
        def fn(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for mod in (self.bottleneck, self.up_stages, self.norm_up, self.final_expand):
            mod.apply(fn)
        for m in self.modules():
            if isinstance(m, GR_KAN_Conv):
                m.reset_kan_specific()          # M5, decoder side
        if self.ppm is not None:                # keep the zero-init identity
            nn.init.zeros_(self.ppm.out[-1].weight)
            nn.init.zeros_(self.ppm.out[-1].bias)
        nn.init.trunc_normal_(self.seg_head.weight, std=0.02)
        nn.init.zeros_(self.seg_head.bias)
        if self.fpn is not None:                 # re-assert after .apply()
            self.fpn._init_identity()
        for up in self.up_stages:
            if up.bridge is not None:
                nn.init.zeros_(up.bridge.gamma)
        if self.boundary is not None:
            with torch.no_grad():
                self.boundary.post.weight.zero_(); self.boundary.post.bias.zero_()
                for c in range(self.boundary.dim):
                    self.boundary.post.weight[c, c, 0, 0] = 1.0

    # ---- encoder pass, harvesting pre-merge skips ----
    def forward_encoder(self, x):
        enc = self.encoder
        x, (H, W) = enc.patch_embed(x)
        x = enc.pos_drop(x)
        skips: List[torch.Tensor] = []
        skip_hw: List[Tuple[int, int]] = []
        stage_out, stage_hw = x, (H, W)
        use_ckpt = self.grad_ckpt and self.training
        for m in enc.layers:
            if isinstance(m, PatchMerging):
                skips.append(stage_out)
                skip_hw.append(stage_hw)
                x, (H, W) = m(x, H, W)
            else:
                if use_ckpt and isinstance(m, nn.Sequential):
                    for blk in m:
                        x = _run_ckpt(blk, x)
                else:
                    x = m(x)
                stage_out, stage_hw = x, (H, W)
        self._skip_hw = skip_hw
        return enc.norm(x), (H, W), skips

    def forward_decoder(self, x, H, W, skips):
        if self.ppm is not None:
            x = self.ppm(x, H, W)
        use_ckpt = self.grad_ckpt and self.training
        for blk in self.bottleneck:
            x = _run_ckpt(blk, x) if use_ckpt else blk(x)
        top, top_hw = x, (H, W)                                 # A2: 4th UPer level
        hw = getattr(self, "_skip_hw", None)
        levels, level_hw = [], []
        for k, up in enumerate(self.up_stages):
            x, (H, W) = up(x, H, W, skips[-(k + 1)],
                           skip_hw=hw[-(k + 1)] if hw else None)
            levels.append(x)
            level_hw.append((H, W))

        self._ds_levels = list(zip(levels, level_hw))
        bnd_logit = None
        if self.fpn is not None:
            f = self.fpn(levels, level_hw, (H, W), top=top, top_hw=top_hw)              # (B, C, H, W) @ 1/4 res
            if self.boundary is not None:
                f, bnd_logit = self.boundary(f)
            x = f.flatten(2).transpose(1, 2)                    # back to tokens
        elif self.boundary is not None:
            B, L, C = x.shape
            f = x.transpose(1, 2).reshape(B, C, H, W)
            f, bnd_logit = self.boundary(f)
            x = f.flatten(2).transpose(1, 2)

        x = self.norm_up(x)
        x, (H, W) = self.final_expand(x, H, W)
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        logits = self.seg_head(self.refine(x))
        if bnd_logit is not None:
            bnd_logit = F.interpolate(bnd_logit, size=logits.shape[-2:],
                                      mode="bilinear", align_corners=False)
        self._last_boundary = bnd_logit
        return logits

    def forward(self, x):
        """Inference / TTA path — main logits only."""
        deep, (H, W), skips = self.forward_encoder(x)
        return self.forward_decoder(deep, H, W, skips)

    def forward_train(self, x):
        """Training path — (main_logits, aux_logits|None, boundary_logits|None)."""
        deep, (H, W), skips = self.forward_encoder(x)
        main = self.forward_decoder(deep, H, W, skips)
        bnd = self._last_boundary
        aux_list: List[torch.Tensor] = []
        if self.ds_heads is not None:
            for head, (t, (h, w)) in zip(self.ds_heads, self._ds_levels):
                aux_list.append(head(t, h, w, main.shape[-2:]))
        aux = None
        if self.aux_head is not None:
            s2 = skips[2]                               # encoder stage-2 tokens @ 14x14
            hw = getattr(self, "_skip_hw", None)
            h2, w2 = (hw[2] if hw else self.stage_res[2])
            if s2.shape[1] != h2 * w2:
                h2 = w2 = int(round(math.sqrt(s2.shape[1])))
            aux = self.aux_head(s2, h2, w2, main.shape[-2:])
            aux_list.append(aux)
        return main, aux_list, bnd


def dekaht_unet(variant: str = "swkat-tiny", img_size: int = 224, patch_size: int = 4,
                in_chans: int = 3, num_classes: int = 5, window_size: int = 7,
                mlp_ratio: float = 4.0, drop_path_rate: float = 0.1, groups: int = 8,
                decoder_depths: Tuple[int, ...] = (2, 2, 2), bottleneck_depth: int = 2,
                ffn_type: str = "kan", gate_mode: str = "residual_tanh",
                use_ppm: bool = True, use_aux: bool = True,
                head_dropout: float = 0.1, use_fpn: bool = True,
                use_boundary: bool = True,
                deep_supervision: bool = True,
                kernel_sizes: Optional[Sequence[int]] = (3, 5, 7),
                skip_norm: bool = False, skip_bridge: str = "linear",
                kaca_window: int = 7, kaca_rational: bool = True,
                uper_levels: int = 3, fpn_smooth: bool = False,
                grad_ckpt: bool = False) -> DeKAHTUNet:
    v = STANDARD_VARIANTS.get(variant)
    if v is None:
        raise ValueError(f"Unknown variant {variant}; choose from {list(STANDARD_VARIANTS)}")
    return DeKAHTUNet(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
        embed_dim=v["embed_dim"], depths=v["depths"], num_heads=v["num_heads"],
        decoder_depths=decoder_depths, bottleneck_depth=bottleneck_depth,
        window_size=window_size, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate,
        groups=groups, ffn_type=ffn_type, gate_mode=gate_mode, use_ppm=use_ppm,
        use_aux=use_aux, head_dropout=head_dropout, use_fpn=use_fpn,
        use_boundary=use_boundary, deep_supervision=deep_supervision,
        kernel_sizes=kernel_sizes, skip_norm=skip_norm, skip_bridge=skip_bridge,
        kaca_window=kaca_window, kaca_rational=kaca_rational,
        uper_levels=uper_levels, fpn_smooth=fpn_smooth, grad_ckpt=grad_ckpt)


KATUNet = DeKAHTUNet
kat_unet = dekaht_unet



# =============================================================================
# F0 -- turn every FFN activation into a learnable KA edge function
# =============================================================================
def convert_ffn_to_rational(model: nn.Module, scope: str = "all", deg_num: int = 5,
                            deg_den: int = 4) -> int:
    """Replace the fixed GELU inside every GR_KAN_Conv with GELU + rational residual.

    Operates on THIS instance only (module replacement, no class patching), so
    Model 1 objects built elsewhere are untouched. Returns the number converted.
    scope: "encoder" (Swin stages only) or "all" (encoder + bottleneck + decoder).
    """
    assert scope in ("encoder", "all"), scope
    root = model.encoder if scope == "encoder" else model
    n = 0
    for m in root.modules():
        if isinstance(m, GR_KAN_Conv) and isinstance(m.act, nn.GELU):
            m.act = RationalResidualAct(m.hidden_channels, groups=m.groups,
                                           deg_num=deg_num, deg_den=deg_den,
                                           base="gelu", channel_dim=1)
            n += 1
    return n


def count_rational_ffn(model: nn.Module) -> int:
    return sum(1 for m in model.modules()
               if isinstance(m, GR_KAN_Conv) and isinstance(m.act, RationalResidualAct))


# =============================================================================
# B2 -- Kolmogorov-Arnold prototype-mixture head
# =============================================================================
class KAPrototypeHead(nn.Module):
    """Drop-in replacement for `seg_head` (Conv2d(96, 5, 1)): feat (B,C,H,W) -> logits.

    Training protocol (driven by the trainer hooks):
      1. forward()                 caches (z, sims) when self.training
      2. prototype_losses(target)  Sinkhorn assignment on detached similarities,
                                   returns (L_ppc, L_ppd) with gradient through z,
                                   stages per-prototype embedding sums
      3. apply_update()            after the optimiser step: EMA-move prototypes,
                                   update usage, re-seed dead prototypes
    Prototypes are buffers (saved in the state dict, never touched by the optimiser).
    Everything inside runs in fp32 with autocast disabled.
    """

    def __init__(self, in_dim: int = 96, emb_dim: int = 64,
                 K: Sequence[int] = (3, 3, 3, 3, 8), tau: float = 0.1,
                 sinkhorn_iters: int = 3, sinkhorn_eps: float = 0.05, ema: float = 0.999,
                 max_pix: int = 4096, keep_frac: float = 0.8, reseed_after: int = 200,
                 usage_floor: float = 0.1, warp_groups: int = 8, reservoir: int = 64):
        super().__init__()
        assert emb_dim % warp_groups == 0
        self.K = tuple(int(k) for k in K)
        assert all(k >= 1 for k in self.K)
        self.num_classes = len(self.K)
        self.P = sum(self.K)
        self.emb_dim = emb_dim
        self.tau, self.iters, self.eps = float(tau), int(sinkhorn_iters), float(sinkhorn_eps)
        self.ema, self.max_pix, self.keep_frac = float(ema), int(max_pix), float(keep_frac)
        self.reseed_after, self.usage_floor = int(reseed_after), float(usage_floor)
        self.reservoir_size = int(reservoir)
        self.offsets = [0]
        for k in self.K:
            self.offsets.append(self.offsets[-1] + k)

        self.embed = nn.Conv2d(in_dim, emb_dim, 1, bias=True)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.embed.bias)
        self.ln = nn.LayerNorm(emb_dim)
        # KA metric warp: identity at init, learnable univariate reshaping per group
        self.warp = RationalResidualAct(emb_dim, groups=warp_groups, base="identity",
                                           channel_dim=1)
        owner = []
        for c, k in enumerate(self.K):
            owner += [c] * k
        self.register_buffer("owner", torch.tensor(owner, dtype=torch.long), persistent=False)
        self.register_buffer("prototypes", F.normalize(torch.randn(self.P, emb_dim), dim=1))
        self.register_buffer("initialized", torch.zeros(self.num_classes, dtype=torch.bool))
        usage = torch.cat([torch.full((k,), 1.0 / k) for k in self.K])
        self.register_buffer("usage", usage)
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("_stage_sum", torch.zeros(self.P, emb_dim), persistent=False)
        self.register_buffer("_stage_cnt", torch.zeros(self.P), persistent=False)
        self._cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._res: Dict[int, torch.Tensor] = {}
        self.last_reseeded = 0

    # ------------------------------------------------------------------ forward
    def class_logits(self, sims: torch.Tensor) -> torch.Tensor:
        """(B, P, H, W) cosine similarities -> (B, C, H, W) log-mean-exp class logits."""
        s = sims / self.tau
        outs = []
        for c in range(self.num_classes):
            a, b = self.offsets[c], self.offsets[c + 1]
            outs.append(torch.logsumexp(s[:, a:b], dim=1) - math.log(b - a))
        return torch.stack(outs, 1)

    def embed_pixels(self, feat: torch.Tensor) -> torch.Tensor:
        e = self.embed(feat.float())
        e = self.ln(e.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return F.normalize(self.warp(e), dim=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=feat.device.type, enabled=False):
            z = self.embed_pixels(feat)                               # (B, E, H, W)
            mu = F.normalize(self.prototypes.float(), dim=1)          # (P, E)
            sims = torch.einsum("behw,pe->bphw", z, mu)               # (B, P, H, W)
            logits = self.class_logits(sims)
        self._cache = (z, sims) if self.training else None
        return logits

    # ------------------------------------------------------------ assignment
    @torch.no_grad()
    def sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        """Entropic OT with equipartition, log domain, fp32.
        scores (n, K) -> soft assignment (n, K) whose rows sum to 1 and whose columns
        sum to ~n/K, so no prototype can capture every pixel (collapse prevention)."""
        n, K = scores.shape
        L = scores.float() / self.eps
        for _ in range(self.iters):
            L = L - torch.logsumexp(L, dim=0, keepdim=True) - math.log(K)
            L = L - torch.logsumexp(L, dim=1, keepdim=True) - math.log(n)
        return (L + math.log(n)).exp()

    @torch.no_grad()
    def _init_class(self, c: int, zc: torch.Tensor):
        a, b = self.offsets[c], self.offsets[c + 1]
        k = b - a
        idx = torch.randperm(zc.shape[0], device=zc.device)[:k]
        seeds = zc[idx]
        if seeds.shape[0] < k:                          # fewer pixels than prototypes
            reps = math.ceil(k / seeds.shape[0])
            seeds = seeds.repeat(reps, 1)[:k]
        seeds = seeds + 0.01 * torch.randn_like(seeds)  # break ties between duplicates
        self.prototypes[a:b] = F.normalize(seeds, dim=1)
        self.usage[a:b] = 1.0 / k
        self.initialized[c] = True

    def prototype_losses(self, target: torch.Tensor):
        """(L_ppc, L_ppd) on the cached forward, or (None, None) if nothing is cached.

        L_ppc: cross-entropy over ALL P prototypes, target = the OT-assigned prototype
               of the pixel's own class (pixel-prototype contrast).
        L_ppd: (1 - <z, mu_assigned>)^2 (compactness).
        Assignment uses only the best-matching keep_frac of each class's pixels
        (small-loss trick against crowd-label noise).
        """
        if self._cache is None:
            return None, None
        z, sims = self._cache
        self._cache = None
        B, E, H, W = z.shape
        t = target
        if t.shape[-2:] != (H, W):
            t = F.interpolate(t.unsqueeze(1).float(), size=(H, W), mode="nearest")
            t = t.squeeze(1).long()
        zf = z.permute(0, 2, 3, 1).reshape(-1, E)
        sf = sims.permute(0, 2, 3, 1).reshape(-1, self.P)
        tf = t.reshape(-1)
        ppc_sum = zf.new_zeros(())
        ppd_sum = zf.new_zeros(())
        count = 0
        for c in range(self.num_classes):
            idx = (tf == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            if idx.numel() > self.max_pix:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:self.max_pix]]
            zc, sc = zf[idx], sf[idx]
            with torch.no_grad():
                zd = zc.detach()
                if not bool(torch.isfinite(zd).all()):
                    continue                             # fp16-overflow step: skip class
                if not bool(self.initialized[c]):
                    self._init_class(c, zd)
                a, b = self.offsets[c], self.offsets[c + 1]
                s_own = zd @ F.normalize(self.prototypes[a:b], dim=1).t()      # (n, K_c)
                n_keep = max(1, int(round(self.keep_frac * s_own.shape[0])))
                keep = s_own.max(1).values.topk(n_keep).indices
                k_local = self.sinkhorn(s_own[keep]).argmax(1)
                glob = k_local + a
                self._stage_sum.index_add_(0, glob, zd[keep])
                self._stage_cnt.index_add_(0, glob, torch.ones_like(glob, dtype=torch.float))
                self._res[c] = zd[keep][:self.reservoir_size].clone()
            logits_all = sc[keep] / self.tau
            ppc_sum = ppc_sum + F.cross_entropy(logits_all, glob, reduction="sum")
            s_asg = sc[keep].gather(1, glob.unsqueeze(1)).squeeze(1)
            ppd_sum = ppd_sum + ((1.0 - s_asg) ** 2).sum()
            count += int(keep.numel())
        if count == 0:
            return None, None
        return ppc_sum / count, ppd_sum / count

    @torch.no_grad()
    def apply_update(self):
        """EMA prototype update from the staged assignments, then dead-prototype re-seeding."""
        cnt = self._stage_cnt
        if float(cnt.sum()) == 0:
            return
        upd = cnt > 0
        mean = self._stage_sum[upd] / cnt[upd].unsqueeze(1)
        ok = torch.isfinite(mean).all(1)
        rows = upd.nonzero(as_tuple=True)[0][ok]
        if rows.numel():
            new = self.ema * self.prototypes[rows] + (1.0 - self.ema) * F.normalize(mean[ok], dim=1)
            self.prototypes[rows] = F.normalize(new, dim=1)
        for c in range(self.num_classes):
            a, b = self.offsets[c], self.offsets[c + 1]
            tot = float(cnt[a:b].sum())
            if tot > 0:
                self.usage[a:b] = 0.99 * self.usage[a:b] + 0.01 * (cnt[a:b] / tot)
        self.n_updates += 1
        self.last_reseeded = 0
        if int(self.n_updates) > self.reseed_after:
            for c in range(self.num_classes):
                a, b = self.offsets[c], self.offsets[c + 1]
                k = b - a
                pool = self._res.get(c)
                if k < 2 or pool is None or pool.shape[0] == 0:
                    continue
                dead = (self.usage[a:b] < self.usage_floor / k).nonzero(as_tuple=True)[0]
                for d in dead.tolist():
                    j = int(torch.randint(pool.shape[0], (1,)))
                    v = pool[j] + 0.01 * torch.randn_like(pool[j])
                    self.prototypes[a + d] = F.normalize(v, dim=0)
                    self.usage[a + d] = 1.0 / k
                    self.last_reseeded += 1
        self._stage_sum.zero_()
        self._stage_cnt.zero_()

    def extra_repr(self):
        return (f"K={self.K}, P={self.P}, emb={self.emb_dim}, tau={self.tau}, "
                f"sinkhorn(iters={self.iters}, eps={self.eps}), ema={self.ema}")


if __name__ == "__main__":
    m = kat_unet("swkat-tiny", num_classes=5, skip_norm=True, skip_bridge="kaca",
                 uper_levels=4, fpn_smooth=True, use_boundary=False)
    x = torch.randn(2, 3, 224, 224)
    m.eval()
    with torch.no_grad():
        y = m(x)
    m.train()
    y2, a, b = m.forward_train(x)
    a = a[-1] if a else torch.zeros(1)
    print(f"[Model 1] {tuple(x.shape)} -> main {tuple(y.shape)} aux {tuple(a.shape)} | "
          f"bnd {None if b is None else tuple(b.shape)} | "
          f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params")
