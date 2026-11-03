#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dekaht_unet_UC3.py — DeKAHT-UNet (Swkat-GR-KAN) for BCSS semantic segmentation.

CHANGES vs dekaht_unet_unified_UC2.py  (each one is verified by verify_UC3.py)
------------------------------------------------------------------------------
M1  PatchEmbed gains the LayerNorm that official Swin has  -> checkpoint key
    `patch_embed.norm.*` is now loadable (it was silently dropped before).
M2  Swin shift-size guard: shift_size is forced to 0 when min(resolution) <=
    window_size (stage-3 @ 7x7 and the 7x7 bottleneck).  UC2 used shift=3 there,
    which is NOT what the pretrained Swin-T checkpoint was trained with.
M3  GR_KAN_Conv is now *exactly* Swin's MLP at initialisation:
      - group_scale     = 1                        (already was)
      - depthwise conv  = identity (delta kernel)  (was kaiming-random)
      - gate            = h * (1 + tanh(W h + b)), W=0,b=0 -> factor exactly 1
                          (was h * sigmoid(random) ~ 0.5 * h  + noise)
    -> fc1/fc2 can receive Swin's mlp.fc1/fc2 weights and the block computes
       bit-identical outputs to Swin at step 0. Capacity is unchanged; the KAN
       parts simply start from the identity, which is the standard
       zero-init-residual recipe (ReZero / LayerScale).
M4  `remap_swin_to_dekaht()` — full key remapping
       layers.{s}.blocks.{j}.*   -> layers.{2s}.{j}.*
       layers.{s}.downsample.*   -> layers.{2s+1}.*
       ...blocks.{j}.mlp.fcN.weight (C_out,C_in) -> mlp_conv.fcN.weight (C_out,C_in,1,1)
    plus a coverage report and a hard failure below --min-coverage.
M5  `_init_weights` no longer clobbers the GR-KAN convs with kaiming(fan_out);
    KAN-specific tensors are (re)initialised last, after `.apply()`.
M6  Decoder additions (encoder untouched, so pretrained weights still load):
      - PPM (pyramid pooling, PSPNet/UPerNet) at the 7x7 bottleneck, added
        residually with a ZERO-initialised output conv -> identity at step 0.
      - 3x3 refine conv (GroupNorm+GELU) before the 1x1 classifier; the old
        head was a bare 1x1 on pixel-shuffled tokens -> blocky boundaries.
      - seg_head now has a bias (it can learn the class prior).
      - auxiliary FCN head on encoder stage-2 (14x14) for deep supervision,
        returned only by forward_train() and dropped at inference.
M7  forward() returns the main logits only (TTA/eval unchanged);
    forward_train() returns (main, aux).
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
    "UpStage", "PyramidPooling", "DeKAHTUNet", "dekaht_unet", "FPNFuse", "BoundaryHead",
    "remap_swin_to_dekaht", "load_swin_pretrained", "set_prefer_kat",
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
                 gate_mode: str = "residual_tanh"):
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

        if self.spatial_kernel > 1:
            pad = (self.spatial_kernel - 1) // 2
            self.depthwise_spatial = nn.Conv2d(hidden_channels, hidden_channels,
                                               self.spatial_kernel, padding=pad,
                                               groups=hidden_channels, bias=True)
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
        if isinstance(self.depthwise_spatial, nn.Conv2d):
            with torch.no_grad():
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
        if not isinstance(self.depthwise_spatial, nn.Identity):
            h = self.depthwise_spatial(h)
        if self.gate is not None:
            g = self.gate(h)
            h = h * (1.0 + torch.tanh(g)) if self.gate_mode == "residual_tanh" else h * torch.sigmoid(g)
        h = self.drop(h)
        out = self.fc2(h)
        return self.drop(out)

    def extra_repr(self) -> str:
        return (f"channels={self.channels}, hidden={self.hidden_channels}, groups={self.groups}, "
                f"k={self.spatial_kernel}, ffn={self.ffn_type}, gate={self.gate_mode}")


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
                 gate_mode: str = "residual_tanh"):
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
                                    gate_mode=gate_mode)

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
                 ffn_type: str = "kan", gate_mode: str = "residual_tanh"):
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
                              ffn_type=ffn_type, gate_mode=gate_mode)
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
    kan_only = [k for k in model_state
                if ("group_scale" in k or "depthwise_spatial" in k or ".gate." in k)]
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


class UpStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, depth, input_resolution,
                 window_size, mlp_ratio, groups, dpr, ffn_type="kan",
                 gate_mode="residual_tanh"):
        super().__init__()
        self.expand = PatchExpand(in_dim, out_dim)
        self.fuse = nn.Linear(2 * out_dim, out_dim)
        self.blocks = nn.ModuleList([
            SwkatBlockOpt(dim=out_dim, input_resolution=input_resolution, num_heads=num_heads,
                          window_size=window_size,
                          shift_size=0 if (i % 2 == 0) else window_size // 2,
                          mlp_ratio=mlp_ratio, drop_path=float(dpr[i]), groups=groups,
                          ffn_type=ffn_type, gate_mode=gate_mode)
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
        x = self.fuse(torch.cat([x, skip], -1))
        for blk in self.blocks:
            x = blk(x)
        return x, (H, W)


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

    def __init__(self, in_dims: Sequence[int], dim: int):
        super().__init__()
        # identity projection on the finest level so the residual is EXACT at step 0
        self.projs = nn.ModuleList([nn.Linear(d, dim) if d != dim else nn.Identity()
                                    for d in in_dims])
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

    def forward(self, feats, hws, out_hw):
        H, W = out_hw
        outs = []
        for t, (h, w), proj in zip(feats, hws, self.projs):
            B = t.shape[0]
            f = proj(t).transpose(1, 2).reshape(B, self.dim, h, w)
            if (h, w) != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(f)
        finest = outs[-1]
        return finest + self.gate(self.act(self.norm(self.fuse(torch.cat(outs, 1)))))


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


class AuxHead(nn.Module):
    """FCN auxiliary head for deep supervision (dropped at inference)."""

    def __init__(self, in_dim: int, num_classes: int, mid: int = 256, drop: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, mid, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid), nn.GELU(), nn.Dropout2d(drop))
        self.cls = nn.Conv2d(mid, num_classes, 1)

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
                 deep_supervision: bool = True):
        super().__init__()
        assert len(depths) == 4 and len(num_heads) == 4
        assert len(decoder_depths) == 3
        self.ffn_type = ffn_type
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.encoder = SwkatGRKAN_Opt(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=0,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate, groups=groups,
            ffn_type=ffn_type, gate_mode=gate_mode)

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
                          ffn_type=ffn_type, gate_mode=gate_mode)
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
                dpr=dec_dpr[cur:cur + d], ffn_type=ffn_type, gate_mode=gate_mode))
            cur += d

        # UC5-A / UC5-B
        self.fpn = FPNFuse([self.stage_dims[2], self.stage_dims[1], self.stage_dims[0]],
                           self.stage_dims[0]) if use_fpn else None
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
        self.ds_heads = nn.ModuleList([
            nn.Conv2d(self.stage_dims[2 - k], num_classes, 1) for k in range(3)
        ]) if deep_supervision else None
        if self.ds_heads is not None:
            for h in self.ds_heads:
                nn.init.trunc_normal_(h.weight, std=0.02); nn.init.zeros_(h.bias)

        self._init_decoder_weights()

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
        for m in enc.layers:
            if isinstance(m, PatchMerging):
                skips.append(stage_out)
                skip_hw.append(stage_hw)
                x, (H, W) = m(x, H, W)
            else:
                x = m(x)
                stage_out, stage_hw = x, (H, W)
        self._skip_hw = skip_hw
        return enc.norm(x), (H, W), skips

    def forward_decoder(self, x, H, W, skips):
        if self.ppm is not None:
            x = self.ppm(x, H, W)
        for blk in self.bottleneck:
            x = blk(x)
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
            f = self.fpn(levels, level_hw, (H, W))              # (B, C, H, W) @ 1/4 res
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
        if self.ds_heads is not None:                       # UC6-A
            for head, (t, (h, w)) in zip(self.ds_heads, self._ds_levels):
                B, L, C = t.shape
                f = head(t.transpose(1, 2).reshape(B, C, h, w))
                aux_list.append(F.interpolate(f, size=main.shape[-2:],
                                              mode="bilinear", align_corners=False))
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
                in_chans: int = 3, num_classes: int = 3, window_size: int = 7,
                mlp_ratio: float = 4.0, drop_path_rate: float = 0.1, groups: int = 8,
                decoder_depths: Tuple[int, ...] = (2, 2, 2), bottleneck_depth: int = 2,
                ffn_type: str = "kan", gate_mode: str = "residual_tanh",
                use_ppm: bool = True, use_aux: bool = True,
                head_dropout: float = 0.1, use_fpn: bool = True,
                use_boundary: bool = True,
                deep_supervision: bool = True) -> DeKAHTUNet:
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
        use_boundary=use_boundary, deep_supervision=deep_supervision)


if __name__ == "__main__":
    m = dekaht_unet("swkat-tiny", num_classes=3)
    x = torch.randn(2, 3, 224, 224)
    m.eval()
    with torch.no_grad():
        y = m(x)
    m.train()
    y2, a, b = m.forward_train(x)
    a = a[-1] if a else torch.zeros(1)
    print(f"[DeKAHT-UNet] {tuple(x.shape)} -> main {tuple(y.shape)} aux {tuple(a.shape)} | "
          f"bnd {tuple(b.shape)} | {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params")
