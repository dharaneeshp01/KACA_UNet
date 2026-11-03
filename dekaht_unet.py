#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dekaht_unet.py  —  U-Net encoder-decoder on the DeKAHT (Swkat-GR-KAN) backbone
==============================================================================
Reviewed & verified build.

Design principle
----------------
The DeKAHT novelty is the *channel transform* (GN-KAN / GR_KAN_Conv), not the
spatial hierarchy, which is plain Swin. So this U-Net:

  1. Reuses the DeKAHT encoder unchanged and taps the pre-merge output of each
     stage as a skip connection.
  2. Mirrors the encoder's PatchMerging (concat-2x2 -> Linear : HW/2, C*2) with a
     PatchExpand (Linear -> pixel-shuffle-2x2 : HW*2, C/2). This is verified to be
     a true *spatial* inverse: the fine token at (2h,2w) originates from deep
     token (h,w), so skip concatenations fuse spatially-corresponding tokens.
  3. Carries the GN-KAN FFN into the decoder by reusing SwkatBlockOpt as the
     decoder block — this is what makes it a DeKAHT-Unet, not a Swin-Unet with a
     swapped encoder.
  4. Fuses skips by channel-concat + Linear, exactly inverting the merge.

Scope / assumptions
-------------------
Built for the paper's square 224x224 @ patch-4 pipeline. The encoder blocks infer
(H,W) from sequence length via sqrt(L), so non-square inputs are out of scope for
this decoder (the expand layers thread explicit H,W, but the blocks would need the
dynamic-resolution path). Resize inputs to a square multiple of 32 before use.

Usage
-----
    from dekaht_unet import dekaht_unet
    model = dekaht_unet("swkat-tiny", num_classes=9)
    mask  = model(torch.randn(2, 3, 224, 224))     # -> (2, 9, 224, 224)
"""

from typing import Tuple, List, Sequence, Optional

import torch
import torch.nn as nn

# --- user's encoder building blocks (defined in dekaht.py) --------------------
from dekaht import (
    SwkatGRKAN_Opt,   # full DeKAHT encoder: PatchEmbed + stages + PatchMerging
    SwkatBlockOpt,    # attention + GN-KAN block, operates on (B, L, C)
    PatchMerging,     # encoder downsampler (used here only for type-safe harvesting)
    STANDARD_VARIANTS,
    GR_KAN_Conv
)


# =============================================================================
# Up-sampling primitives (verified spatial inverse of PatchMerging)
# =============================================================================
class PatchExpand(nn.Module):
    """
    Inverse of PatchMerging.

      Merge :  concat 2x2 -> LN(4C) -> Linear(4C -> 2C)      (HW/2, C*2)
      Expand:  Linear(C -> 2C) -> pixel-shuffle 2x2 -> C/2    (HW*2, C/2)

    I/O follows the encoder convention: takes (x, H, W), returns (x, (2H, 2W)).
    Pure-torch pixel shuffle (view/permute/view) to match dekaht.py's no-einops style.
    """
    def __init__(self, in_dim: int, out_dim: Optional[int] = None,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim // 2
        self.in_dim = in_dim
        self.out_dim = out_dim
        # channel x2 so the 2x2 spatial shuffle lands on in_dim/2 channels
        self.expand = nn.Linear(in_dim, 2 * in_dim, bias=False)
        self.norm = norm_layer(in_dim // 2)
        # optional projection if the caller wants out_dim != in_dim//2
        self.proj = nn.Linear(in_dim // 2, out_dim) if out_dim != in_dim // 2 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, L, C = x.shape
        assert L == H * W, f"PatchExpand: L({L}) != H*W({H * W})"
        x = self.expand(x)                              # (B, L, 2C)
        # pixel shuffle: (b h w (p1 p2 c)) -> (b (h p1) (w p2) c), p1=p2=2
        x = x.view(B, H, W, 2, 2, C // 2)               # split 2C -> p1,p2,c=C/2
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()    # (B, H, p1, W, p2, c)
        x = x.view(B, H * 2, W * 2, C // 2)             # (B, 2H, 2W, C/2)
        x = x.view(B, -1, C // 2)                       # (B, 4L, C/2)
        x = self.norm(x)
        x = self.proj(x)
        return x, (H * 2, W * 2)


class FinalPatchExpandX4(nn.Module):
    """
    Undo the patch embedding in one step.
    Linear(C -> s^2 * C) -> pixel-shuffle s x s -> C  (resolution *s, channels kept).
    s = patch_size (4 for the default pipeline).
    """
    def __init__(self, dim: int, dim_scale: int = 4, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, L, C = x.shape
        s = self.dim_scale
        assert L == H * W, f"FinalPatchExpandX4: L({L}) != H*W({H * W})"
        x = self.expand(x)                              # (B, L, s^2 C)
        x = x.view(B, H, W, s, s, C)                    # (B, H, W, p1, p2, c=C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()    # (B, H, p1, W, p2, c)
        x = x.view(B, H * s, W * s, C)                  # (B, sH, sW, C)
        x = x.view(B, -1, C)                            # (B, s^2 L, C)
        x = self.norm(x)
        return x, (H * s, W * s)


# =============================================================================
# Decoder up-stage: expand -> [assert + concat skip] -> fuse -> GN-KAN blocks
# =============================================================================
class _UpStage(nn.Module):
    """
    One decoder level, mirroring exactly one encoder (stage + merge) pair.

      expand:  in_dim @ (H,W)  ->  out_dim @ (2H,2W)   (= skip resolution)
      concat:  [up, skip]      ->  2*out_dim channels   (inverse of merge concat)
      fuse:    Linear(2*out_dim -> out_dim)             (inverse of merge reduction)
      blocks:  SwkatBlockOpt x depth   (the SAME GN-KAN block used in the encoder)
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, depth: int,
                 input_resolution: Tuple[int, int], window_size: int, mlp_ratio: float,
                 groups: int, dpr: Sequence[float]):
        super().__init__()
        self.expand = PatchExpand(in_dim, out_dim)
        self.fuse = nn.Linear(2 * out_dim, out_dim)
        self.blocks = nn.ModuleList([
            SwkatBlockOpt(
                dim=out_dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, drop_path=float(dpr[i]), groups=groups,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, H: int, W: int, skip: torch.Tensor
                ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # ---- upsample: spatial *2, channels /2 (inverse of PatchMerging) ----
        x, (H, W) = self.expand(x, H, W)
        # ---- SKIP FUSION: assert alignment, then concat + linear-reduce ------
        #   x, skip : (B, H*W, out_dim)  at the SAME resolution and channels.
        assert x.shape[1] == skip.shape[1], (
            f"skip seq-len mismatch: up {tuple(x.shape)} vs skip {tuple(skip.shape)}")
        assert x.shape[2] == skip.shape[2], (
            f"skip channel mismatch: up {tuple(x.shape)} vs skip {tuple(skip.shape)}")
        x = torch.cat([x, skip], dim=-1)                # (B, L, 2*out_dim)
        x = self.fuse(x)                                # (B, L, out_dim)
        # ---- GN-KAN transformer blocks (attention + grouped KAN FFN) --------
        for blk in self.blocks:
            x = blk(x)
        return x, (H, W)


# =============================================================================
# Full model
# =============================================================================
class DeKAHTUNet(nn.Module):
    """
    U-Net encoder-decoder with the DeKAHT (Swkat-GR-KAN) backbone.

    Args:
        img_size, patch_size, in_chans : input geometry (must match encoder)
        num_classes                    : segmentation output channels
        embed_dim, depths, num_heads   : encoder config (defaults = swkat-tiny)
        decoder_depths                 : blocks per decoder up-stage, ordered
                                         deep->shallow i.e. (level@14, level@28, level@56).
                                         Lean default (2,2,2); use (6,2,2) to mirror the
                                         encoder's stage-2 depth symmetrically.
        bottleneck_depth               : extra GN-KAN blocks at the deepest (7x7) scale.
        window_size, mlp_ratio, groups : forwarded to the GN-KAN blocks.
    """
    def __init__(self,
                 img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 num_classes: int = 9,
                 embed_dim: int = 96, depths: Tuple[int, ...] = (2, 2, 6, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24),
                 decoder_depths: Tuple[int, ...] = (2, 2, 2),
                 bottleneck_depth: int = 2,
                 window_size: int = 7, mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.1, groups: int = 8):
        super().__init__()
        assert len(depths) == 4 and len(num_heads) == 4, "this U-Net assumes a 4-stage backbone"
        assert len(decoder_depths) == 3, "decoder has 3 up-stages (mirrors 3 patch-merges)"

        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # ---------------------------------------------------------------------
        # ENCODER: reuse DeKAHT as-is. num_classes=0 -> head=Identity; we use
        # only patch_embed / pos_drop / layers / norm. Pretrained classification
        # weights load straight into `self.encoder`.
        # ---------------------------------------------------------------------
        self.encoder = SwkatGRKAN_Opt(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            num_classes=0, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate, groups=groups,
        )

        self.stage_dims = [int(embed_dim * 2 ** i) for i in range(self.num_stages)]  # 96,192,384,768
        deepest_dim = self.stage_dims[-1]
        grid = img_size // patch_size                                                 # 56
        self.stage_res = [(grid // (2 ** i), grid // (2 ** i)) for i in range(self.num_stages)]  # 56,28,14,7

        # ---------------------------------------------------------------------
        # BOTTLENECK: refine the deepest encoder output (7x7x768) with extra
        # GN-KAN blocks (residual, dim-preserving -> no information loss). Kept
        # shallow: deep transformers converge poorly.
        # ---------------------------------------------------------------------
        b_dpr = torch.linspace(0, drop_path_rate, max(bottleneck_depth, 1)).tolist()
        self.bottleneck = nn.ModuleList([
            SwkatBlockOpt(
                dim=deepest_dim, input_resolution=self.stage_res[-1], num_heads=num_heads[-1],
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, drop_path=float(b_dpr[i]), groups=groups,
            )
            for i in range(bottleneck_depth)
        ])

        # ---------------------------------------------------------------------
        # DECODER up-stages. Level k (k=0,1,2):
        #   in_dim  = stage_dims[3-k]   (768,384,192)
        #   out_dim = stage_dims[2-k]   (384,192, 96)  == skip channels
        #   skip    = encoder stage (2-k) output @ stage_res[2-k]  (14,28,56)
        # ---------------------------------------------------------------------
        dec_dpr = torch.linspace(0, drop_path_rate, sum(decoder_depths)).tolist()
        self.up_stages = nn.ModuleList()
        cur = 0
        for k in range(3):
            in_dim = self.stage_dims[3 - k]
            out_dim = self.stage_dims[2 - k]
            res = self.stage_res[2 - k]
            heads = num_heads[2 - k]
            depth = decoder_depths[k]
            self.up_stages.append(_UpStage(
                in_dim=in_dim, out_dim=out_dim, num_heads=heads, depth=depth,
                input_resolution=res, window_size=window_size, mlp_ratio=mlp_ratio,
                groups=groups, dpr=dec_dpr[cur:cur + depth],
            ))
            cur += depth

        # ---------------------------------------------------------------------
        # HEAD: 4x expand (undo patch embedding) -> 1x1 conv to class logits.
        # ---------------------------------------------------------------------
        self.norm_up = nn.LayerNorm(self.stage_dims[0])
        self.final_expand = FinalPatchExpandX4(dim=self.stage_dims[0], dim_scale=patch_size)
        self.seg_head = nn.Conv2d(self.stage_dims[0], num_classes, kernel_size=1, bias=False)

        # explicit, encoder-consistent init for the DECODER ONLY (never touches
        # self.encoder, so pretrained weights and the GN-KAN conv init survive).
        self._init_decoder_weights()

    # -------------------------------------------------------------------------
    def _init_decoder_weights(self):
        """trunc_normal Linears + standard LayerNorm/Conv init on decoder modules.
        Intentionally skips self.encoder and skips Conv2d inside GR_KAN_Conv
        (those keep their own carefully chosen init)."""
        def fn(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for module in (self.bottleneck, self.up_stages, self.norm_up, self.final_expand):
            module.apply(fn)
        nn.init.kaiming_normal_(self.seg_head.weight, mode="fan_out", nonlinearity="relu")

    # -------------------------------------------------------------------------
    def forward_encoder(self, x: torch.Tensor):
        """Run the DeKAHT layers manually, harvesting the pre-merge output of each
        stage as a skip. Returns (deepest_feature, (H,W), [skip0, skip1, skip2])."""
        enc = self.encoder
        x, (H, W) = enc.patch_embed(x)                  # (B, 3136, 96) @ 56x56
        x = enc.pos_drop(x)

        skips: List[torch.Tensor] = []
        stage_out = x
        for m in enc.layers:                            # [Seq0, PM0, Seq1, PM1, Seq2, PM2, Seq3]
            if isinstance(m, PatchMerging):             # type-safe (not "is it Sequential")
                skips.append(stage_out)                 # the pre-merge stage output is a skip
                x, (H, W) = m(x, H, W)
            else:                                       # a stage's block sequence
                x = m(x)
                stage_out = x
        x = enc.norm(x)                                 # LayerNorm(768) on deepest feature
        return x, (H, W), skips                         # skips = [s0@56, s1@28, s2@14]

    # -------------------------------------------------------------------------
    def forward_decoder(self, x: torch.Tensor, H: int, W: int,
                        skips: List[torch.Tensor]) -> torch.Tensor:
        # ---- BOTTLENECK: deepest-scale GN-KAN refinement (7x7x768) ----------
        for blk in self.bottleneck:
            x = blk(x)
        # ---- UP path: consume skips deep -> shallow (s2@14, s1@28, s0@56) ---
        for k, up in enumerate(self.up_stages):
            skip = skips[-(k + 1)]                       # k=0->s2, 1->s1, 2->s0
            x, (H, W) = up(x, H, W, skip)
        # ---- HEAD: 4x expand to input resolution, then 1x1 conv -------------
        x = self.norm_up(x)
        x, (H, W) = self.final_expand(x, H, W)          # (B, 224*224, 96)
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()   # (B, 96, 224, 224)
        x = self.seg_head(x)                            # (B, num_classes, 224, 224)
        return x

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_deep, (H, W), skips = self.forward_encoder(x)
        return self.forward_decoder(x_deep, H, W, skips)


# =============================================================================
# Variant factory (mirrors dekaht.STANDARD_VARIANTS)
# =============================================================================
def dekaht_unet(variant: str = "swkat-tiny", img_size: int = 224, patch_size: int = 4,
                in_chans: int = 3, num_classes: int = 9, window_size: int = 7,
                mlp_ratio: float = 4.0, drop_path_rate: float = 0.1, groups: int = 8,
                decoder_depths: Tuple[int, ...] = (2, 2, 2),
                bottleneck_depth: int = 2) -> DeKAHTUNet:
    v = STANDARD_VARIANTS.get(variant)
    if v is None:
        raise ValueError(f"Unknown variant {variant}, choose from {list(STANDARD_VARIANTS.keys())}")
    return DeKAHTUNet(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
        embed_dim=v["embed_dim"], depths=v["depths"], num_heads=v["num_heads"],
        decoder_depths=decoder_depths, bottleneck_depth=bottleneck_depth,
        window_size=window_size, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate, groups=groups,
    )


if __name__ == "__main__":
    m = dekaht_unet("swkat-tiny", num_classes=9)
    xb = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        yb = m(xb)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"input  : {tuple(xb.shape)}")
    print(f"output : {tuple(yb.shape)}")
    print(f"params : {n:.2f} M")
