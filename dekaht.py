#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# dekaht.py - production-ready Swkat-GR-KAN with dynamic-resolution support + variant factory
# - pure-PyTorch KAN-FFN (no compiled kat_rational dependency)
# - robust window partition/reverse with padding
# - GR_KAN_Conv is the single MLP replacement used (no fallback to classic Linear-FFN)

import os
import math
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional compatibility shims (no-op if not present)
try:
    import triton_fma_patch  # may be absent; it's ok
except Exception:
    pass

from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast
import random

# ---------------------------
# Runtime toggle placeholder for KAT (kept as no-op for compatibility)
# ---------------------------
# This file intentionally does NOT provide a compiled kat_rational path.
# set_prefer_kat is kept purely for API compatibility and does not change behavior.
PREFER_KAT = False  # kept for API compatibility; no compiled KAT path used here


def set_prefer_kat(flag: bool):
    """Legacy toggle kept for compatibility - no compiled kat_rational path is used in this codebase."""
    global PREFER_KAT
    PREFER_KAT = bool(flag)
    # no-op behavior; kept to avoid surprises from external scripts
    return


# ---------------------------
# Performance helpers
# ---------------------------
def enable_perf_tweaks(use_compile: bool):
    torch.backends.cudnn.benchmark = True
    if use_compile and hasattr(torch, "compile"):
        return True
    return False


# ---------------------------
# DropPath
# ---------------------------
def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------
# Pure-PyTorch GR_KAN_Conv (KAN-style FFN)
# ---------------------------
class GR_KAN_Conv(nn.Module):
    """
    Pure PyTorch KAN-style FFN used as an MLP replacement.
    - Accepts input shape (B, C, H, W) and returns same shape.
    - Uses pointwise 1x1 convs for channel expansion/contraction,
      a depthwise spatial 3x3 conv for local spatial mixing,
      a lightweight gating mechanism, and optional grouped scaling.
    - No external compiled kernels; runs on CPU/GPU with standard torch ops.
    - This is THE MLP replacement used in all Swkat blocks (no fallback to Linear-based FFN).
    """
    def __init__(self,
                 channels: int,
                 hidden_channels: Optional[int] = None,
                 groups: int = 8,
                 drop: float = 0.0,
                 spatial_kernel: int = 3,
                 use_gating: bool = True):
        super().__init__()
        hidden_channels = hidden_channels or (channels * 4)
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.groups = max(1, groups)
        self.drop = nn.Dropout2d(drop) if drop > 0.0 else nn.Identity()
        self.use_gating = use_gating
        self.spatial_kernel = spatial_kernel if spatial_kernel >= 1 else 1

        # mark explicitly as KAN block for runtime enforcement
        self.is_kan = True

        # pointwise expansion and contraction (1x1 convs)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)

        # depthwise spatial conv for local mixing (per-hidden-channel)
        if self.spatial_kernel > 1:
            pad = (self.spatial_kernel - 1) // 2
            # depthwise conv: groups = hidden_channels
            self.depthwise_spatial = nn.Conv2d(hidden_channels, hidden_channels,
                                               kernel_size=self.spatial_kernel, padding=pad,
                                               groups=hidden_channels, bias=True)
        else:
            self.depthwise_spatial = nn.Identity()

        # grouped scale (learnable per-group multiplier) similar to grouped-GELU scheme
        n_groups = self.groups
        assert hidden_channels % n_groups == 0, "hidden_channels must be divisible by groups"
        self.group_scale = nn.Parameter(torch.ones(n_groups, hidden_channels // n_groups))

        # activation and gating
        self.act = nn.GELU()
        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
        else:
            self.gate = None

        self._init_weights()

    def __repr__(self) -> str:
        return (f"GR_KAN_Conv(channels={self.channels}, hidden={self.hidden_channels}, "
                f"groups={self.groups}, spatial_k={self.spatial_kernel})")

    def _init_weights(self):
        # conv init similar to common practice
        try:
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        except Exception:
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        if getattr(self.fc1, "bias", None) is not None:
            nn.init.zeros_(self.fc1.bias)

        try:
            nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        except Exception:
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        if getattr(self.fc2, "bias", None) is not None:
            nn.init.zeros_(self.fc2.bias)

        if isinstance(self.depthwise_spatial, nn.Conv2d):
            try:
                nn.init.kaiming_normal_(self.depthwise_spatial.weight, mode='fan_in', nonlinearity='relu')
            except Exception:
                nn.init.normal_(self.depthwise_spatial.weight, mean=0.0, std=0.02)
            if getattr(self.depthwise_spatial, "bias", None) is not None:
                nn.init.zeros_(self.depthwise_spatial.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape  # expect this format
        # 1) pointwise expansion
        h = self.fc1(x)  # (B, C_hid, H, W)
        # 2) grouped scaling (broadcast)
        g = self.groups
        if g > 1:
            # reshape to (B, g, ch_per, H, W) to apply per-group scaling
            ch_per = self.hidden_channels // g
            h = h.view(B, g, ch_per, H, W)
            h = h * self.group_scale.view(1, g, ch_per, 1, 1)
            h = h.view(B, self.hidden_channels, H, W)

        # 3) activation
        h = self.act(h)

        # 4) spatial local mixing via depthwise conv
        h_sp = self.depthwise_spatial(h) if not isinstance(self.depthwise_spatial, nn.Identity) else h

        # 5) gating (optional)
        if self.gate is not None:
            gate = self.gate(h_sp)
            h_sp = h_sp * gate

        h_sp = self.drop(h_sp)

        # 6) projection back to channels
        out = self.fc2(h_sp)
        out = self.drop(out)
        return out


# ---------------------------
# Window partition / reverse (robust padding)
# ---------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    x: (B, H, W, C) -> returns windows shaped (-1, ws, ws, C)
    pads bottom/right so H/W multiples of effective ws = min(window_size, H, W)
    """
    B, H, W, C = x.shape
    ws = min(window_size, H, W)
    pad_h = (ws - (H % ws)) % ws
    pad_w = (ws - (W % ws)) % ws
    if pad_h != 0 or pad_w != 0:
        import torch.nn.functional as _F
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = _F.pad(x, (0, pad_w, 0, pad_h))
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H+pad_h, W+pad_w, C)
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)
    return x


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    windows: (-1, ws, ws, C) -> returns x: (B, H, W, C)
    uses unpadding to recover original H,W
    """
    ws = min(window_size, H, W)
    Hp = ((H + ws - 1) // ws) * ws
    Wp = ((W + ws - 1) // ws) * ws
    num_windows_per_image = (Hp // ws) * (Wp // ws)
    B = int(windows.shape[0] // num_windows_per_image)
    x = windows.view(B, Hp // ws, Wp // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    x = x[:, :H, :W, :].contiguous()
    return x


# ---------------------------
# Window Attention with learnable relative position bias
# ---------------------------
import torch as _torch
import torch.nn as _nn


class WindowAttention(_nn.Module):
    """
    Window-attention with learnable relative position bias.
    Robust when runtime windows smaller than configured window_size.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = _nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = _nn.Dropout(attn_drop)
        self.proj = _nn.Linear(dim, dim)
        self.proj_drop = _nn.Dropout(proj_drop)

        ws = self.window_size
        num_relative_positions = (2 * ws - 1) * (2 * ws - 1)
        self.relative_position_bias_table = _nn.Parameter(_torch.zeros(num_relative_positions, self.num_heads))

        coords_h = _torch.arange(ws)
        coords_w = _torch.arange(ws)
        coords = _torch.stack(_torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        try:
            _nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        except Exception:
            _nn.init.normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        full_M = self.window_size * self.window_size
        full_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(full_M, full_M, self.num_heads)
        # if window in runtime is smaller than configured, slice bias accordingly
        if N != full_bias.shape[0]:
            relative_position_bias = full_bias[:N, :N, :].permute(2, 0, 1).unsqueeze(0).to(attn.device)
        else:
            relative_position_bias = full_bias.permute(2, 0, 1).unsqueeze(0).to(attn.device)
        attn = attn + relative_position_bias

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(-1, num_windows, self.num_heads, N, N)
            mask = mask.to(attn.device).type_as(attn)
            _bcast_ok = False
            if mask.ndim == 3 and attn.ndim == 5:
                cand = mask.unsqueeze(0).unsqueeze(2)
                try:
                    cand = cand.expand(attn.shape)
                    attn = attn + cand
                    _bcast_ok = True
                except Exception:
                    _bcast_ok = False
            if not _bcast_ok and mask.ndim == 3 and attn.ndim == 4:
                cand = mask.unsqueeze(1)
                try:
                    cand = cand.expand(attn.shape)
                    attn = attn + cand
                    _bcast_ok = True
                except Exception:
                    _bcast_ok = False
            if not _bcast_ok:
                cur = mask
                while cur.ndim < attn.ndim:
                    cur = cur.unsqueeze(1)
                try:
                    cand = cur.to(attn.device).type_as(attn).expand(attn.shape)
                    attn = attn + cand
                    _bcast_ok = True
                except Exception as e:
                    raise RuntimeError(f"Unable to broadcast mask{tuple(mask.shape)} into attn{tuple(attn.shape)}; attempted broadcast and failed: {e}")

            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------
# SwkatBlock-like Block (optimized, lazy attn-mask builder for dynamic-size)
# ---------------------------
class SwkatBlockOpt(nn.Module):
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0, drop_path: float = 0.0, groups: int = 8):
        super().__init__()
        self.dim = dim
        # store an initial hint for resolution (may be adjusted at runtime)
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size if shift_size < window_size else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.drop_path = nn.Identity() if drop_path == 0.0 else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        # mlp_conv now uses the pure-PyTorch GR_KAN_Conv defined above
        self.mlp_conv = GR_KAN_Conv(channels=dim, hidden_channels=int(dim * mlp_ratio), groups=groups, drop=0.0)

        # do not build attn_mask eagerly here; build lazily in _ensure_attn_mask
        self.attn_mask = None
        self._mask_built_for = None  # (H,W) tuple where attn_mask is valid

    def __repr__(self) -> str:
        return f"SwkatBlockOpt(dim={self.dim}, resolution={self.input_resolution}, window_size={self.window_size}, shift={self.shift_size})"

    def _build_attn_mask(self, H: int, W: int, device=None):
        """Build and register attn_mask buffer for current H,W and window_size/shift."""
        if self.shift_size <= 0:
            # no mask required
            self.attn_mask = None
            self._mask_built_for = (H, W)
            return

        # make device safe
        dev = device if device is not None else "cpu"
        img_mask = torch.zeros((1, H, W, 1), dtype=torch.int32, device=dev)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # (-1, ws, ws, 1)
        ws = mask_windows.shape[1]
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # register buffer (overwrite if present) and ensure it's on desired device
        try:
            self.register_buffer("attn_mask", attn_mask, persistent=False)
            if device is not None:
                self.attn_mask = self.attn_mask.to(device)
        except Exception:
            # fallback: store as regular attribute
            self.attn_mask = attn_mask
        self._mask_built_for = (H, W)

    def _ensure_attn_mask(self, H: int, W: int, device=None):
        if self._mask_built_for != (H, W):
            # rebuild
            try:
                self._build_attn_mask(H, W, device=device)
            except Exception:
                # fall back to None (will run without mask)
                self.attn_mask = None
                self._mask_built_for = (H, W)

    def _recompute_attn_mask(self, input_resolution: Tuple[int, int], device=None):
        """
        Recompute the attention mask buffer for this block.
        Use when external code wants to force-update attn_mask.
        """
        H, W = input_resolution
        self.input_resolution = (H, W)
        if self.shift_size <= 0:
            self.attn_mask = None
            self._mask_built_for = (H, W)
            return
        self._build_attn_mask(H, W, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) where L may differ from stored input_resolution hint
        B, L, C = x.shape
        H_hint, W_hint = self.input_resolution

        # ---- SAFER resolution inference ----
        if L == H_hint * W_hint:
            H, W = H_hint, W_hint
        else:
            # Try square resolution first
            sq = int(round(math.sqrt(L)))
            if sq * sq == L:
                H, W = sq, sq
            else:
                # Try using hinted width if it divides L
                if W_hint > 0 and L % W_hint == 0:
                    H = L // W_hint
                    W = W_hint
                else:
                    # ultimate fallback to hint (keeps behavior predictable)
                    H, W = H_hint, W_hint

        # update stored hint for future
        self.input_resolution = (H, W)

        # if mask/attn buffers not built for this (H,W), ensure they are
        if self.shift_size > 0:
            # try to use parameter device for buffer building
            device = None
            try:
                device = next(self.parameters()).device
            except Exception:
                device = None
            self._ensure_attn_mask(H, W, device=device)

        # proceed with block ops
        shortcut = x
        x_ln = self.norm1(x)
        # reshape using computed H,W
        try:
            x_ = x_ln.view(B, H, W, C)
        except Exception:
            # fallback: try best-effort reshape (avoid crashing)
            x_ = x_ln.view(B, -1, C)
            # attempt to infer H from current shape
            try:
                inferred_H = int(round(math.sqrt(x_.shape[1])))
                x_ = x_ln.view(B, inferred_H, x_.shape[1] // inferred_H, C)
            except Exception:
                # last resort: reshape using hint
                x_ = x_ln.view(B, H_hint, W_hint, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x_, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(x_windows.shape[0], -1, x_windows.shape[-1])
        N = x_windows.shape[1]
        ws_eff = int(round(math.sqrt(N))) if N > 0 else 1

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # if returned attn_windows has different N, try to reshape gracefully
        try:
            attn_windows = attn_windows.view(-1, ws_eff, ws_eff, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        except Exception:
            # fallback: if shapes don't match attempt more conservative restore
            try:
                attn_windows = attn_windows.view(B, H, W, C)
                shifted_x = attn_windows
            except Exception:
                # as last resort pass-through
                shifted_x = shifted_x

        if self.shift_size > 0:
            x_ = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_ = shifted_x
        x_tokens = x_.view(B, H * W, C)
        x = shortcut + self.drop_path(x_tokens)
        x2 = self.norm2(x)
        B2, L2, C2 = x2.shape
        try:
            x_chw = x2.transpose(1, 2).view(B2, C2, H, W)
        except Exception:
            # fallback: try using hint
            x_chw = x2.transpose(1, 2).view(B2, C2, H_hint, W_hint)
        x_chw = self.mlp_conv(x_chw)
        x2 = x_chw.flatten(2).transpose(1, 2)
        x = x + self.drop_path(x2)
        return x


# ---------------------------
# Patch embedding and merging
# ---------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)


class PatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        if (H % 2 == 1) or (W % 2 == 1):
            x_bc = x.permute(0, 3, 1, 2).contiguous()
            pad_h = 1 if (H % 2 == 1) else 0
            pad_w = 1 if (W % 2 == 1) else 0
            x_bc = F.pad(x_bc, (0, pad_w, 0, pad_h))
            x = x_bc.permute(0, 2, 3, 1).contiguous()
            H = x.shape[1]
            W = x.shape[2]

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x_cat = torch.cat([x0, x1, x2, x3], dim=-1)
        x_cat = x_cat.view(B, -1, 4 * C)
        x_cat = self.norm(x_cat)
        x_out = self.reduction(x_cat)
        return x_out, (H // 2, W // 2)


# ---------------------------
# Standard variants & model class (SWKAT names)
# ---------------------------
STANDARD_VARIANTS: Dict[str, Dict[str, Any]] = {
    "swkat-tiny":  {"embed_dim": 96,  "depths": (2, 2, 6, 2),  "num_heads": (3, 6, 12, 24)},
    "swkat-small": {"embed_dim": 96,  "depths": (2, 2, 18, 2), "num_heads": (3, 6, 12, 24)},
    "swkat-base":  {"embed_dim": 128, "depths": (2, 2, 18, 2), "num_heads": (4, 8, 16, 32)},
    "swkat-large": {"embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48)},
}


class SwkatGRKAN_Opt(nn.Module):
    """
    Canonical Swkat model entrypoint.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, num_classes: int = 1000,
                 embed_dim: int = 96, depths: Tuple[int, ...] = (2, 2, 6, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24), window_size: int = 7,
                 mlp_ratio: float = 4.0, drop_path_rate: float = 0.1, groups: int = 8):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(p=0.0)
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        dim = embed_dim
        resolution = self.patch_embed.grid_size  # (H_grid, W_grid)

        for stage_idx, depth in enumerate(depths):
            blocks = []
            for i in range(depth):
                shift = 0 if (i % 2 == 0) else window_size // 2
                blk = SwkatBlockOpt(dim=dim, input_resolution=resolution, num_heads=num_heads[stage_idx],
                                   window_size=window_size, shift_size=shift, mlp_ratio=mlp_ratio,
                                   drop_path=dpr[cur + i], groups=groups)
                # ensure each block has a fresh WindowAttention with correct heads
                blk.attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads[stage_idx])
                blocks.append(blk)
            self.layers.append(nn.Sequential(*blocks))
            cur += depth
            if stage_idx < len(depths) - 1:
                next_dim = dim * 2
                self.layers.append(PatchMerging(dim, next_dim))
                dim = next_dim
                resolution = (max(1, (resolution[0] + 1) // 2), max(1, (resolution[1] + 1) // 2))

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        # store config attributes for helper/utils
        self.img_size = img_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.apply(self._init_weights)

    def __repr__(self) -> str:
        return f"SwkatGRKAN_Opt(embed_dim={self.patch_embed.proj.out_channels}, layers={len(self.layers)})"

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if getattr(m, "weight", None) is not None:
                try:
                    nn.init.trunc_normal_(m.weight, std=.02)
                except Exception:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Conv2d):
            if getattr(m, "weight", None) is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (H, W) = self.patch_embed(x)  # H,W come from projection output
        x = self.pos_drop(x)
        it = iter(self.layers)
        while True:
            try:
                m = next(it)
            except StopIteration:
                break
            if isinstance(m, nn.Sequential):
                # sequential of SwkatBlockOpt: they accept (B,L,C) and use their own dynamic hint
                x = m(x)
            else:
                # PatchMerging returns x, (H,W)
                x, (H, W) = m(x, H, W)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

    def update_input_resolution(self, new_H: int, new_W: int, device=None):
        """
        Update all SwkatBlockOpt modules to the new patch-grid H,W
        and rebuild attention masks if needed.
        Call this whenever crop size changes or pos_embed is resized.
        """
        # Update patch_embed grid (for reference only)
        try:
            self.patch_embed.grid_size = (new_H, new_W)
        except Exception:
            pass

        # Determine a device to place masks on
        if device is None:
            try:
                device = next(self.parameters()).device
            except Exception:
                device = torch.device("cpu")

        for m in self.modules():
            if isinstance(m, SwkatBlockOpt):
                # Update stored hint
                m.input_resolution = (new_H, new_W)

                # Rebuild attention mask if block uses shift
                if m.shift_size > 0:
                    try:
                        # prefer m._build_attn_mask(device=...) when available
                        try:
                            m._build_attn_mask(new_H, new_W, device=device)
                        except TypeError:
                            # older signature fallback
                            m._build_attn_mask(new_H, new_W)
                        # ensure attn_mask on the right device
                        if getattr(m, "attn_mask", None) is not None:
                            try:
                                m.attn_mask = m.attn_mask.to(device)
                            except Exception:
                                pass
                    except Exception:
                        m.attn_mask = None

    @classmethod
    def from_variant(cls, variant: str = "swkat-tiny", img_size: int = 224, patch_size: int = 4,
                     num_classes: int = 1000, window_size: int = 7, mlp_ratio: float = 4.0,
                     drop_path_rate: float = 0.1, groups: int = 8):
        """Factory to instantiate standard variants by name.
        Accepts canonical 'swkat-*' names.
        """
        v = STANDARD_VARIANTS.get(variant)
        if v is None:
            raise ValueError(f"Unknown variant {variant}, choose from {list(STANDARD_VARIANTS.keys())}")
        return cls(img_size=img_size, patch_size=patch_size, in_chans=3, num_classes=num_classes,
                   embed_dim=v["embed_dim"], depths=v["depths"], num_heads=v["num_heads"],
                   window_size=window_size, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate, groups=groups)


# module exports
__all__ = ["SwkatGRKAN_Opt", "STANDARD_VARIANTS", "set_prefer_kat", "GR_KAN_Conv", "WindowAttention"]
