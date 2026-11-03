#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m1_train.py -- MODEL 1 trainer: KAT-UNet incremental upgrade, official 5-class BCSS.

WHAT THE DIAGNOSTIC RUN (diagnostics_report.json) CHANGED
---------------------------------------------------------
1. albumentations 1.4.18 on the Quadro machine silently ignored `border_mode` in
   A.Affine, falling back to BORDER_CONSTANT with mask fill 0 -- and train label 0
   is TUMOR. Measured: 10.0% of each affected patch, 3.0% of ALL training pixels,
   were black squares labelled Tumor. Every albumentations call now goes through
   `_albu()`, which maps argument names to whichever API version is installed.
2. Quadro RTX 5000 = Turing (compute capability 7.5): no hardware bf16, yet
   torch>=2.1 `is_bf16_supported()` returns True via emulation. `--amp-dtype auto`
   now picks bf16 only on cc>=8.0 and fp16+GradScaler otherwise.
3. The context-demand test peaked at scale 1.0 (every class lost Dice when
   downscaled). This model is trained with ZOOM-IN-only augmentation (RandomResized
   Crop, magnification 1.0-1.7x), so test-time TTA must use in-distribution scales
   (>= 1.0). `--tta-select` measures candidate scale sets on val and picks one.
4. No test/ split exists on this machine: test evaluation is now optional and
   reported clearly instead of assumed.
5. Python-3.8-safe (no dict union), torch-1.10-safe DataLoader / torch.load.

MODEL-1 CHANGES (A1-A4, see m1_kat_unet.py for A1/A2)
-----------------------------------------------------
 A1 --skip-norm, --skip-bridge kaca    normalised skip + KA cross-attention bridge
 A2 --uper-levels 4 --fpn-smooth       4-level UPer aggregation
 A3 --tta-mode conf                    per-pixel confidence-weighted TTA (SEUNet P-Ens)
 A4 loss parsimony                     CE + PresenceDice + annealed Lovasz (+ DS).
                                       Boundary CE, boundary head, Focal-Tversky off.
Also: non-finite losses under fp16 skip the step instead of killing the run;
val CE is logged separately from the Lovasz-ramped compound loss; checkpoint
selection uses a moving average of val mDice to reduce winner's-curse bias.

LABEL CONTRACT: raw 0 -> IGNORE(255); raw 1..5 -> 0..4 = Tumor, Stroma,
Inflammatory, Necrosis, Other; 5 output channels; all 5 classes in every metric.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import glob
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    ALBU_OK = True
except Exception:                                                    # pragma: no cover
    ALBU_OK = False

import inspect

from m1_kat_unet import (DSHead, GR_KAN_Conv, RationalResidualAct, SwkatBlockOpt,
                         kat_unet, load_swin_pretrained, remap_swin_to_dekaht)

# =============================================================================
# Constants
# =============================================================================
IGNORE = 255                      # train-space ignore label
RAW_IGNORE = 0                    # raw BCSS "outside_roi"
NUM_CLASSES = 5
CLASS_NAMES = ["Tumor", "Stroma", "Inflammatory", "Necrosis", "Other"]
# Official BCSS test institutions (Grand Challenge baseline page)
TEST_INSTITUTIONS = {"OL", "LL", "E2", "EW", "GM", "S3"}


def safe_torch_load(path: str):
    """torch.load that works before and after the `weights_only` keyword existed."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_amp_dtype(requested: str, device: torch.device) -> str:
    """'auto' -> bf16 only on compute capability >= 8.0 (Ampere+), else fp16.

    torch >= 2.1 reports is_bf16_supported() == True on Turing because bf16 tensors
    can be *emulated*; emulated bf16 matmuls are far slower than fp16 tensor cores.
    """
    if device.type != "cuda" or requested == "off":
        return "off"
    major = torch.cuda.get_device_capability(device)[0]
    if requested == "auto":
        return "bf16" if major >= 8 else "fp16"
    if requested == "bf16" and major < 8:
        return "fp16"
    return requested


# cv2 border codes, without importing cv2 (it is optional)
_BORDER_CONSTANT, _BORDER_REFLECT_101 = 0, 4


def _albu(cls, **kw):
    """Instantiate an albumentations transform under BOTH the 1.4.x and 2.x APIs.

    Callers use 2.x names; they are translated to whatever the installed version's
    constructor accepts:
        border_mode <-> mode      fill <-> value / cval     fill_mask <-> mask_value / cval_mask
        size=(h, w) <-> height=h, width=w
    Unknown keywords are dropped (never passed through to trigger 'is not valid and
    will be ignored', which is exactly how the 1.4.18 Affine silently used a constant
    black border with mask value 0 = Tumor).
    """
    params = set(inspect.signature(cls.__init__).parameters)
    out = {}
    for k, v in kw.items():
        if k in params:
            out[k] = v
            continue
        if k == "border_mode" and "mode" in params:
            out["mode"] = v
        elif k == "fill":
            for alt in ("value", "cval"):
                if alt in params:
                    out[alt] = v
                    break
        elif k == "fill_mask":
            for alt in ("mask_value", "cval_mask"):
                if alt in params:
                    out[alt] = v
                    break
        elif k == "size" and "height" in params and "width" in params:
            out["height"], out["width"] = int(v[0]), int(v[1])
    return cls(**out)


def remap_raw_label(raw: np.ndarray, num_classes: int = NUM_CLASSES) -> Tuple[np.ndarray, int]:
    """raw BCSS values -> train labels.

        raw 0            -> IGNORE (255)
        raw 1..C         -> 0..C-1
        anything else    -> IGNORE (counted, never clamped)

    Returns (train_labels, n_out_of_range). Never clamps: in UC2 a clamp turned
    'Necrosis' into 'Stroma' silently, which is the worst class of data bug
    because the model trains happily on corrupted ground truth.
    """
    raw = raw.astype(np.int64, copy=False)
    out = np.full(raw.shape, IGNORE, dtype=np.int64)
    valid = (raw >= 1) & (raw <= num_classes)
    out[valid] = raw[valid] - 1
    n_bad = int(((raw != RAW_IGNORE) & (~valid)).sum())
    return out, n_bad


# =============================================================================
# Args
# =============================================================================
def parse_args(argv=None, hooks=None):
    p = argparse.ArgumentParser("KAT-UNet Model 1 / BCSS 5-class")

    # ---- data ----
    p.add_argument("--data-root", default=None,
                   help="dir containing train/{images,masks}, val/..., test/...")
    p.add_argument("--img-subdir", default="images")
    p.add_argument("--mask-subdir", default="masks")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split-name", default="val")
    p.add_argument("--test-split-name", default="test")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--eval-size", type=int, default=224)
    p.add_argument("--in-chans", type=int, default=3)
    p.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    p.add_argument("--max-ignore-frac", type=float, default=0.90,
                   help="skip training patches whose don't-care fraction exceeds this")
    p.add_argument("--scan-limit", type=int, default=500)

    # ---- model ----
    p.add_argument("--model-variant", default="swkat-tiny",
                   choices=["swkat-tiny", "swkat-small", "swkat-base", "swkat-large"])
    p.add_argument("--decoder-depths", default="2,2,2")
    p.add_argument("--bottleneck-depth", type=int, default=2)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--window-size", type=int, default=7)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--ffn-type", default="kan",
                   choices=["mlp", "mlp_dw", "mlp_dw_gate", "kan"])
    p.add_argument("--gate-mode", default="residual_tanh",
                   choices=["residual_tanh", "sigmoid"])
    p.add_argument("--kernel-sizes", default="3,5,7",
                   help="N1: per-group depthwise kernels inside the GR-KAN FFN")
    p.add_argument("--drop-path-rate", type=float, default=0.15)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--no-ppm", dest="use_ppm", action="store_false", default=True)
    p.add_argument("--no-fpn", dest="use_fpn", action="store_false", default=True)
    p.add_argument("--no-boundary", dest="use_boundary", action="store_false", default=False)
    p.add_argument("--boundary", dest="use_boundary", action="store_true",
                   help="A4: the boundary stream is OFF by default (polygon-label noise)")
    p.add_argument("--no-deep-supervision", dest="deep_supervision",
                   action="store_false", default=True)
    p.add_argument("--no-aux", dest="use_aux", action="store_false", default=True)
    p.add_argument("--skip-norm", dest="skip_norm", action="store_true", default=True,
                   help="A1: LayerNorm the encoder skip before fusion (measured 5.8x imbalance)")
    p.add_argument("--no-skip-norm", dest="skip_norm", action="store_false")
    p.add_argument("--skip-bridge", choices=["linear", "kaca"], default="kaca",
                   help="A1: add the decoder-query / encoder-value KA cross-attention bridge")
    p.add_argument("--kaca-window", type=int, default=7)
    p.add_argument("--no-kaca-rational", dest="kaca_rational", action="store_false",
                   default=True, help="use plain linear q/k maps in the bridge")
    p.add_argument("--uper-levels", type=int, choices=[3, 4], default=4,
                   help="A2: 4 = add the PPM-refined bottleneck to the FPN fusion")
    p.add_argument("--no-fpn-smooth", dest="fpn_smooth", action="store_false", default=True)
    p.add_argument("--grad-ckpt", action="store_true",
                   help="gradient checkpointing: use if batch 8 does not fit in 16 GB")
    p.add_argument("--rational-lr-scale", type=float, default=0.1,
                   help="LR multiplier for rational coefficients (rat_a/rat_b); WD is 0")
    p.add_argument("--enforce-kan", action="store_true", default=True)
    p.add_argument("--no-enforce-kan", dest="enforce_kan", action="store_false")

    # ---- pretrained ----
    p.add_argument("--encoder-pretrained", default=None)
    p.add_argument("--min-pretrained-coverage", type=float, default=0.90)
    p.add_argument("--freeze-encoder-epochs", type=int, default=0)

    # ---- optimisation ----
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--iters-per-epoch", type=int, default=0,
                   help="F3c: cap the iterations in one 'epoch' so validation, "
                        "checkpointing and SWA happen on a sane cadence. With "
                        "104,438 patches a full pass took 5.3 h on run 103167, so a "
                        "crash at iteration 12,718 lost the entire epoch. 0 = full pass.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--opt", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder-lr-scale", type=float, default=0.4)
    p.add_argument("--layer-decay", type=float, default=0.90)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--wd-on-all", action="store_true")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--sched", choices=["cosine", "poly"], default="cosine")
    p.add_argument("--warmup-iters", type=int, default=1500)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--ema-every", type=int, default=1)

    # ---- P3: hard-ceiling gradient clipping ----
    p.add_argument("--grad-clip-mode", choices=["fixed", "calibrated"],
                   default="calibrated",
                   help="'fixed' uses --grad-clip verbatim; 'calibrated' measures the "
                        "grad-norm distribution during --clip-calib-iters, FREEZES a "
                        "threshold, and can never exceed --grad-clip-ceiling")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--grad-clip-ceiling", type=float, default=25.0,
                   help="ABSOLUTE, NON-NEGOTIABLE upper bound on the clip threshold. "
                        "Run 103167 used 10.0, but its steady-state grad norm was "
                        "8.2-10.0, so the p90 calibration CLAMPED against the ceiling "
                        "and still clipped ~50%% of steps. The ceiling must sit well "
                        "above the operating point; it exists to bound explosions "
                        "(1e12), not to shape normal training.")
    p.add_argument("--grad-clip-floor", type=float, default=1.0)
    p.add_argument("--clip-calib-iters", type=int, default=300)
    p.add_argument("--clip-calib-start", type=int, default=-1,
                   help="F2: step at which calibration BEGINS. -1 = --warmup-iters, so "
                        "the LR is at target and the gradients are representative. "
                        "Run 103167 calibrated during warmup and froze a threshold "
                        "1.3x too low, clipping ~50%% of steps.")
    p.add_argument("--clip-percentile", type=float, default=90.0)

    # ---- losses ----
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--tversky-weight", type=float, default=0.0,
                   help="A4: Focal-Tversky is an ablation arm (0.5 = previous runs)")
    p.add_argument("--tversky-alpha", type=float, default=0.3)
    p.add_argument("--tversky-beta", type=float, default=0.7)
    p.add_argument("--tversky-gamma", type=float, default=0.75)
    p.add_argument("--lovasz-weight", type=float, default=0.5)
    p.add_argument("--lovasz-start-frac", type=float, default=0.4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--boundary-weight", type=float, default=0.0,
                   help="extra CE weight on class boundaries (pixel weighting)")
    p.add_argument("--boundary-head-weight", type=float, default=0.0)
    p.add_argument("--focal-gamma", type=float, default=0.0)
    p.add_argument("--gce-q", type=float, default=0.0,
                   help="Generalized CE (noise-robust). 0.7 recommended for crowd labels")
    p.add_argument("--class-weights", default="auto",
                   help="'auto' | 'none' | comma list of 5 floats")
    p.add_argument("--class-weight-power", type=float, default=0.5,
                   help="w ~ (1/freq)^power; 0.5 = inverse sqrt (gentler than 1.0)")
    p.add_argument("--class-weight-clip", type=float, default=4.0)
    p.add_argument("--dice-include-all", action="store_true", default=True,
                   help="all 5 classes are real foreground; none is excluded")

    # ---- P1: auxiliary loss budget ----
    p.add_argument("--aux-budget", type=float, default=0.30,
                   help="TOTAL weight of all deep-supervision heads (hard cap 0.35)")
    p.add_argument("--aux-decay", type=float, default=0.6,
                   help="geometric decay of DS weight towards SHALLOW stages")
    p.add_argument("--ds-loss-size", type=int, default=56,
                   help="F3a: resolution at which deep-supervision heads are scored. "
                        "0 = full resolution (16x slower, no measured benefit).")
    p.add_argument("--max-aux-ratio", type=float, default=5.0,
                   help="abort if aux_total/main exceeds this (UC6 reached 7e8)")

    # ---- augmentation ----
    p.add_argument("--aug-mode", default="rrc", choices=["rrc", "scale_pad"])
    p.add_argument("--rrc-min-scale", type=float, default=0.35)
    p.add_argument("--scale-min", type=float, default=0.75)
    p.add_argument("--scale-max", type=float, default=1.5)
    p.add_argument("--hed-sigma", type=float, default=0.05)
    p.add_argument("--clahe-p", type=float, default=0.25)
    p.add_argument("--elastic-p", type=float, default=0.15)
    p.add_argument("--stain-norm", default=None,
                   help="path to a reference RGB patch -> Macenko normalisation")
    p.add_argument("--stain-norm-p", type=float, default=1.0)

    # ---- runtime ----
    p.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16", "off"], default="auto",
                   help="auto = bf16 on compute capability >= 8.0, else fp16 + GradScaler")
    p.add_argument("--max-consecutive-nonfinite", type=int, default=50,
                   help="skip non-finite steps; abort only after this many in a row")
    p.add_argument("--tta", action="store_true", default=True)
    p.add_argument("--no-tta", dest="tta", action="store_false")
    p.add_argument("--tta-scales", default="1.0")
    p.add_argument("--tta-mode", choices=["mean", "conf"], default="conf",
                   help="A3: conf = per-pixel margin-softmax weighting across scales")
    p.add_argument("--tta-tau", type=float, default=1.0)
    p.add_argument("--tta-select", action="store_true",
                   help="with --eval-only: score every candidate scale set x mode on the eval "
                        "split in ONE pass and write tta_selection.json")
    p.add_argument("--tta-candidates", default="1.0;1.0,1.25;1.0,1.25,1.5;0.84,1.0,1.25",
                   help="';'-separated scale sets for --tta-select")
    p.add_argument("--tta-config", default=None,
                   help="tta_selection.json from --tta-select; overrides --tta-scales/--tta-mode")
    p.add_argument("--select-smooth", type=int, default=3,
                   help="best checkpoint chosen by the moving average of the last k val mDice")
    p.add_argument("--arch-from-ckpt", dest="arch_from_ckpt", action="store_true", default=True)
    p.add_argument("--no-arch-from-ckpt", dest="arch_from_ckpt", action="store_false")
    p.add_argument("--init-from", default=None,
                   help="warm-start weights (non-strict; legacy bcss5 checkpoints accepted)")
    p.add_argument("--skip-test", action="store_true",
                   help="do not evaluate the test split at the end even if it exists")
    p.add_argument("--swa-last", type=int, default=6)
    p.add_argument("--output-dir", default="./out_m1")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--resume", default=None)
    p.add_argument("--eval-only", default=None)
    p.add_argument("--eval-weights", choices=["ema", "model"], default="ema")
    p.add_argument("--eval-on", default=None,
                   help="split name to evaluate (default: --val-split-name)")
    p.add_argument("--swa-only", default=None)
    p.add_argument("--self-test", action="store_true")
    if hooks is not None:
        hooks.add_args(p)
    return p.parse_args(argv)


# =============================================================================
# Logging / CSV
# =============================================================================
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    lg = logging.getLogger("bcss5")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(sh)
    return lg


CSV_FIELDS = (["epoch", "train_loss", "ce", "dice", "tversky", "lovasz", "bnd",
               "aux", "aux_ratio", "nonfinite", "val_loss", "val_ce", "val_smooth",
               "val_mdice_agg",
               "val_mdice_img", "val_miou", "val_acc"]
              + [f"dice_{n}" for n in CLASS_NAMES]
              + ["lr_head", "lr_enc", "gnorm_mean", "gnorm_p95", "clip_thr",
                 "clip_rate", "sec"])


def init_csv(p):
    if not os.path.exists(p):
        with open(p, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv(p, row):
    with open(p, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(
            {k: row.get(k, "") for k in CSV_FIELDS})


# =============================================================================
# Mask I/O and data audits
# =============================================================================
def read_mask(path: str) -> np.ndarray:
    """The ONLY correct reader: raw palette indices, no .convert(), no OpenCV.

    `.convert("L")` maps a mode-P index through the palette to its luminance, so
    class 3 silently becomes 87. `cv2.IMREAD_UNCHANGED` expands mode-P to RGB.
    Both corrupt labels with no error.
    """
    im = Image.open(path)
    a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.int64)


def check_mask_readers(paths: Sequence[str], logger=None) -> Dict[str, object]:
    out = {"n": 0, "modes": Counter(), "convert_L_differs": 0,
           "cv2_differs": 0, "cv2_available": False}
    try:
        import cv2
        out["cv2_available"] = True
    except Exception:
        cv2 = None
    for p in paths:
        im = Image.open(p)
        out["modes"][im.mode] += 1
        raw = np.array(im)
        if raw.ndim == 3:
            raw = raw[..., 0]
        conv = np.array(Image.open(p).convert("L"))
        if conv.shape != raw.shape or not np.array_equal(conv, raw):
            out["convert_L_differs"] += 1
        if cv2 is not None:
            c = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if c is not None:
                if c.ndim == 3:
                    c = c[..., 0]
                if c.shape != raw.shape or not np.array_equal(c, raw):
                    out["cv2_differs"] += 1
        out["n"] += 1
    if logger:
        logger.info(f"[mask-audit] {out['n']} files | PIL modes {dict(out['modes'])} | "
                    f".convert('L') would corrupt {out['convert_L_differs']} | "
                    f"cv2 would corrupt {out['cv2_differs']}"
                    f"{'' if out['cv2_available'] else ' (cv2 absent)'}")
    return out


def audit_raw_labels(paths: Sequence[str], num_classes: int, logger=None):
    """Histogram of RAW label values. This is what proves the 22->5 remap ran."""
    hist: Counter = Counter()
    for p in paths:
        v, c = np.unique(read_mask(p), return_counts=True)
        for vi, ci in zip(v.tolist(), c.tolist()):
            hist[int(vi)] += int(ci)
    total = max(1, sum(hist.values()))
    if logger:
        names = {0: "IGNORE"}
        names.update({i + 1: n for i, n in enumerate(CLASS_NAMES)})
        desc = ", ".join(f"{k}({names.get(k, '?')}):{100 * v / total:.2f}%"
                         for k, v in sorted(hist.items()))
        logger.info(f"[label-audit] raw values over {len(paths)} masks -> {desc}")
        oob = {k: v for k, v in hist.items() if k > num_classes}
        if oob:
            logger.error(
                f"[label-audit] *** raw values ABOVE {num_classes} present: {sorted(oob)}. "
                f"Your masks are NOT remapped to the official 5 classes. Run the "
                f"guide's remap_masks.py first — training now would learn garbage.")
            raise RuntimeError("masks are not 5-class remapped")
        if RAW_IGNORE not in hist:
            logger.warning(
                "[label-audit] raw value 0 never appears in a random sample of the "
                "training set. Your patch extractor discarded don't-care regions "
                "entirely, so ignore_index is inert. That is valid, but it means "
                "'Other' must NOT be silently absorbing outside_roi -- confirm your "
                "remap sent outside_roi to 0 and not to 5.")
        present = {k for k in hist if 1 <= k <= num_classes}
        missing = set(range(1, num_classes + 1)) - present
        if missing:
            logger.warning(f"[label-audit] classes absent from the scan: "
                           f"{[names[m] for m in sorted(missing)]} — if these are "
                           f"absent dataset-wide their Dice will be undefined.")
    return dict(hist)


def institution_of(filename: str) -> str:
    """TCGA-A1-7723-01Z-...  ->  'A1'. Empty string when unparseable."""
    m = re.match(r"^TCGA-([A-Z0-9]{2})-", Path(filename).name, flags=re.I)
    return m.group(1).upper() if m else ""


def audit_institution_split(splits: Dict[str, List[Tuple[str, str]]], logger=None):
    """The BCSS protocol is INSTITUTION-level. Verify it, loudly."""
    inst = {k: {institution_of(ip) for ip, _ in v} - {""} for k, v in splits.items()}
    for k, v in inst.items():
        logger.info(f"[split] {k}: {len(splits[k])} patches from {len(v)} institutions "
                    f"{sorted(v)}")
    keys = list(splits)
    leaked = False
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = inst[keys[i]] & inst[keys[j]]
            if overlap:
                leaked = True
                logger.error(f"[split] *** LEAK: institutions {sorted(overlap)} appear in "
                             f"BOTH '{keys[i]}' and '{keys[j]}'. The BCSS protocol is "
                             f"institution-level; these numbers are not publishable.")
    if not leaked:
        logger.info("[split] verified: institution-disjoint across all splits")
    test_i = inst.get("test", set())
    if test_i and test_i != TEST_INSTITUTIONS:
        logger.warning(f"[split] test institutions {sorted(test_i)} != official "
                       f"{sorted(TEST_INSTITUTIONS)} — say so explicitly in the paper.")
    elif test_i:
        logger.info("[split] test set matches the official Grand Challenge institutions")
    return not leaked


# =============================================================================
# Macenko stain normalisation
# =============================================================================
class MacenkoNormalizer:
    """Macenko et al. 2009 stain normalisation, dependency-free (numpy only).

    Highest-value preprocessing step for multi-institution TCGA: the six test
    institutions were stained and scanned elsewhere, so the colour distribution
    shift between train and test is a genuine domain gap, not noise.
    """

    def __init__(self, beta: float = 0.15, alpha: float = 1.0, io: float = 240.0):
        self.beta, self.alpha, self.io = beta, alpha, io
        self.target_stains: Optional[np.ndarray] = None
        self.target_conc: Optional[np.ndarray] = None

    @staticmethod
    def _od(img: np.ndarray, io: float) -> np.ndarray:
        x = img.reshape(-1, 3).astype(np.float64)
        # NATURAL log: the inverse in transform() is io*exp(-OD). Using log10 here
        # while reconstructing with exp() is a base mismatch of ln(10)=2.303 and
        # silently destroys the image (verified: luminance corr went NEGATIVE).
        return -np.log(np.clip(x, 1.0, 255.0) / io)

    def _stain_matrix(self, img: np.ndarray):
        od = self._od(img, self.io)
        od_h = od[~np.any(od < self.beta, axis=1)]
        if od_h.shape[0] < 50:
            raise ValueError("too few tissue pixels for Macenko")
        _, _, vt = np.linalg.svd(od_h - od_h.mean(0), full_matrices=False)
        v = vt[:2].T
        proj = od_h @ v
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        a, b = np.percentile(phi, self.alpha), np.percentile(phi, 100 - self.alpha)
        v1 = v @ np.array([np.cos(a), np.sin(a)])
        v2 = v @ np.array([np.cos(b), np.sin(b)])
        he = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
        he = he / (np.linalg.norm(he, axis=0, keepdims=True) + 1e-12)
        return he

    def _concentrations(self, img: np.ndarray, he: np.ndarray) -> np.ndarray:
        od = self._od(img, self.io)
        return np.linalg.lstsq(he, od.T, rcond=None)[0]

    def fit(self, target_img: np.ndarray) -> "MacenkoNormalizer":
        he = self._stain_matrix(target_img)
        conc = self._concentrations(target_img, he)
        self.target_stains = he
        self.target_conc = np.percentile(conc, 99, axis=1).reshape(-1, 1)
        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        if self.target_stains is None:
            return img
        he = self._stain_matrix(img)
        conc = self._concentrations(img, he)
        maxc = np.percentile(conc, 99, axis=1).reshape(-1, 1)
        conc = conc * (self.target_conc / np.maximum(maxc, 1e-6))
        out = self.io * np.exp(-self.target_stains @ conc)
        return np.clip(out.T.reshape(img.shape), 0, 255).astype(np.uint8)

    def safe_transform(self, img: np.ndarray) -> np.ndarray:
        """Degenerate patches (near-white, blurred) make the SVD fail. Those are
        rare and the raw patch is a fine fallback, so never let it kill the job."""
        try:
            return self.transform(img)
        except Exception:
            return img


# =============================================================================
# HED stain jitter (augmentation, complements normalisation)
# =============================================================================
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]], dtype=np.float64)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)
_RGB_FROM_HED_F32 = _RGB_FROM_HED.astype(np.float32)
_HED_FROM_RGB_F32 = _HED_FROM_RGB.astype(np.float32)


def hed_jitter(img_u8: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb haematoxylin/eosin concentrations independently (Tellez 2019)."""
    if sigma <= 0:
        return img_u8
    # F3b: float32 throughout. float64 cost 3.76 ms/patch vs 2.16 ms in float32 and
    # bought nothing -- the input is 8-bit and the output is re-quantised to 8-bit.
    x = img_u8.astype(np.float32) / np.float32(255.0)
    od = -np.log(np.clip(x, 1e-6, 1.0))
    hed = od.reshape(-1, 3) @ _HED_FROM_RGB_F32
    hed = (hed * rng.uniform(1 - sigma, 1 + sigma, 3).astype(np.float32)
           + rng.uniform(-sigma, sigma, 3).astype(np.float32))
    out = np.exp(-(hed @ _RGB_FROM_HED_F32)).reshape(img_u8.shape)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


if ALBU_OK:
    class HEDStainJitter(A.ImageOnlyTransform):
        def __init__(self, sigma: float = 0.05, p: float = 0.7):
            super().__init__(p=p)
            self.sigma = sigma
            self._rng = np.random.default_rng()

        def apply(self, img, **params):
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                return img
            return hed_jitter(img, self.sigma, self._rng)

        def get_transform_init_args_names(self):
            return ("sigma",)
else:                                                                # pragma: no cover
    HEDStainJitter = None


# =============================================================================
# Augmentation  (uint8 in, uint8 out — albumentations treats float32 as [0,1])
# =============================================================================
def build_train_aug(args):
    """uint8 in, uint8 out. Every geometric op uses REFLECT borders, so no transform
    can fabricate labels. Built through _albu() -> identical behaviour on
    albumentations 1.4.x and 2.x."""
    assert ALBU_OK, "albumentations is required"
    crop = args.img_size
    if args.aug_mode == "scale_pad":
        geo = [_albu(A.RandomScale, scale_limit=(args.scale_min - 1.0, args.scale_max - 1.0),
                     p=1.0),
               _albu(A.PadIfNeeded, min_height=crop, min_width=crop,
                     border_mode=_BORDER_CONSTANT, fill=0, fill_mask=IGNORE),
               _albu(A.RandomCrop, height=crop, width=crop)]
    else:
        # Patches are natively 224: zoom-IN only (no padding, no fabricated labels).
        geo = [_albu(A.RandomResizedCrop, size=(crop, crop),
                     scale=(args.rrc_min_scale, 1.0), ratio=(0.8, 1.25), p=1.0)]
    tfs = geo + [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        _albu(A.Affine, scale=(0.95, 1.05), translate_percent=(-0.03, 0.03),
              rotate=(-30, 30), border_mode=_BORDER_REFLECT_101, p=0.3),
        _albu(A.ColorJitter, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        _albu(A.GaussianBlur, blur_limit=(3, 5), p=0.15),
    ]
    if args.clahe_p > 0:
        tfs.append(_albu(A.CLAHE, clip_limit=4.0, tile_grid_size=(8, 8), p=args.clahe_p))
    if HEDStainJitter is not None and args.hed_sigma > 0:
        tfs.append(HEDStainJitter(sigma=args.hed_sigma, p=0.7))
    if args.elastic_p > 0:
        tfs.append(_albu(A.ElasticTransform, alpha=30, sigma=6,
                         border_mode=_BORDER_REFLECT_101, p=args.elastic_p))
    return A.Compose(tfs)


def build_val_aug(args):
    assert ALBU_OK
    s = args.eval_size
    return A.Compose([_albu(A.PadIfNeeded, min_height=s, min_width=s,
                            border_mode=_BORDER_CONSTANT, fill=0, fill_mask=IGNORE),
                      _albu(A.CenterCrop, height=s, width=s)])


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img_u8: np.ndarray) -> torch.Tensor:
    x = img_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


# =============================================================================
# Dataset
# =============================================================================
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_pairs(img_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    imgs = [p for p in sorted(glob.glob(os.path.join(img_dir, "*")))
            if p.lower().endswith(_IMG_EXTS)]
    mmap = {Path(p).stem: p for p in glob.glob(os.path.join(mask_dir, "*"))
            if p.lower().endswith(_IMG_EXTS)}
    return [(ip, mmap[Path(ip).stem]) for ip in imgs if Path(ip).stem in mmap]


class BCSS5Dataset(Dataset):
    """Official 5-class BCSS patches.

    __getitem__ returns:
        image : float32 (3, H, W)  ImageNet-normalised
        label : int64   (H, W)     values in {0..4} U {255}
    """

    def __init__(self, pairs: List[Tuple[str, str]], args, train: bool,
                 normalizer: Optional[MacenkoNormalizer] = None):
        self.pairs = pairs
        self.args = args
        self.train = train
        self.num_classes = args.num_classes
        self.aug = build_train_aug(args) if train else build_val_aug(args)
        self.normalizer = normalizer
        self.stain_p = args.stain_norm_p
        self.oob_pixels = 0
        self.seen_pixels = 0
        self._rng = np.random.default_rng(args.seed + (0 if train else 1))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
        raw = read_mask(mp)
        if raw.shape[:2] != img.shape[:2]:
            raw = np.array(Image.fromarray(raw.astype(np.int32)).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST), dtype=np.int64)

        # raw -> train labels (0 -> IGNORE, 1..5 -> 0..4)
        lab, n_bad = remap_raw_label(raw, self.num_classes)
        self.oob_pixels += n_bad
        self.seen_pixels += lab.size

        if self.normalizer is not None and (self.stain_p >= 1.0
                                            or self._rng.random() < self.stain_p):
            img = self.normalizer.safe_transform(img)

        out = self.aug(image=img, mask=lab.astype(np.int32))
        img, lab = out["image"], out["mask"].astype(np.int64)
        return {"image": normalize(img),
                "label": torch.from_numpy(np.ascontiguousarray(lab)).long()}


def build_loaders(args, logger):
    root = args.data_root
    splits = {}
    for name in (args.train_split, args.val_split_name, args.test_split_name):
        d = os.path.join(root, name)
        if os.path.isdir(d):
            pr = list_pairs(os.path.join(d, args.img_subdir),
                            os.path.join(d, args.mask_subdir))
            if pr:
                splits[name] = pr
    if args.train_split not in splits:
        raise RuntimeError(
            f"No image/mask pairs under {os.path.join(root, args.train_split)}. "
            f"Expected {root}/{args.train_split}/{{{args.img_subdir},{args.mask_subdir}}}")
    if args.val_split_name not in splits:
        raise RuntimeError(f"No validation split at {root}/{args.val_split_name}")

    audit_institution_split(splits, logger)

    tr_pairs = splits[args.train_split]
    va_pairs = splits[args.val_split_name]

    # F4: RANDOM sample, not the first N. `tr_pairs` is sorted by filename, so
    # tr_pairs[:500] is ~one ROI of ONE slide. On run 103167 that biased scan
    # reported Inflammatory 0.07% / Necrosis 31.41% and "raw value 0 never
    # appears", while the true shuffled distribution was Inflammatory 13.28% /
    # Necrosis 6.84%. The audit has to sample the dataset, not its first slide.
    _rs = random.Random(args.seed)
    scan_pairs = (tr_pairs if len(tr_pairs) <= args.scan_limit
                  else _rs.sample(tr_pairs, args.scan_limit))
    scan = [m for _, m in scan_pairs]
    logger.info(f"[label-audit] scanning {len(scan)} RANDOMLY sampled masks "
                f"out of {len(tr_pairs)} training patches")
    check_mask_readers(scan[:min(64, len(scan))], logger)
    audit_raw_labels(scan, args.num_classes, logger)

    normalizer = None
    if args.stain_norm:
        ref = np.array(Image.open(args.stain_norm).convert("RGB"), dtype=np.uint8)
        normalizer = MacenkoNormalizer().fit(ref)
        logger.info(f"[stain] Macenko normalisation fitted on {args.stain_norm} "
                    f"(applied with p={args.stain_norm_p})")

    tr = BCSS5Dataset(tr_pairs, args, True, normalizer)
    va = BCSS5Dataset(va_pairs, args, False, normalizer)
    common = dict(num_workers=args.workers, pin_memory=True)
    if args.workers > 0:                 # torch<2.0 rejects prefetch_factor when workers=0
        common.update(persistent_workers=True, prefetch_factor=4)
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                     drop_last=True, **common)
    val = DataLoader(va, batch_size=max(1, args.batch_size), shuffle=False, **common)
    logger.info(f"train {len(tr)} patches | val {len(va)} patches")
    return trl, val


# =============================================================================
# Losses
# =============================================================================
def boundary_weight_map(target: torch.Tensor, ignore_index: int, k: int = 3,
                        extra: float = 1.0) -> torch.Tensor:
    """1 + extra on pixels adjacent to a class change (morphological gradient)."""
    valid = target != ignore_index
    t = torch.where(valid, target, torch.zeros_like(target)).float().unsqueeze(1)
    mx = F.max_pool2d(t, k, 1, k // 2)
    mn = -F.max_pool2d(-t, k, 1, k // 2)
    return 1.0 + extra * (mx != mn).float().squeeze(1)


def boundary_target(target: torch.Tensor, ignore_index: int, k: int = 3):
    valid = target != ignore_index
    t = torch.where(valid, target, torch.zeros_like(target)).float().unsqueeze(1)
    mx = F.max_pool2d(t, k, 1, k // 2)
    mn = -F.max_pool2d(-t, k, 1, k // 2)
    edge = (mx != mn).float()
    vm = F.avg_pool2d(valid.float().unsqueeze(1), k, 1, k // 2)
    return edge, (vm > 0.999).float()


def boundary_loss(bnd_logit, target, ignore_index=IGNORE):
    edge, vm = boundary_target(target, ignore_index)
    logit = bnd_logit.float()
    pos = edge.sum().clamp(min=1.0)
    neg = (vm.sum() - pos).clamp(min=1.0)
    pw = (neg / pos).clamp(max=50.0)
    bce = F.binary_cross_entropy_with_logits(logit, edge, reduction="none", pos_weight=pw)
    return (bce * vm).sum() / vm.sum().clamp(min=1.0)


class PresenceDiceLoss(nn.Module):
    """Per-image soft Dice averaged ONLY over (image, class) pairs present in GT.

    Averaging over ABSENT classes is what froze the UC2 Dice term: a 224x224 BCSS
    patch typically contains 2-3 of the 5 classes, so the absent ones each
    contributed ~1.0 regardless of prediction quality and the loss could not move.
    """

    def __init__(self, num_classes, ignore_index=IGNORE, eps=1e-6):
        super().__init__()
        self.n, self.ig, self.eps = num_classes, ignore_index, eps

    def forward(self, logits, target):
        probs = torch.softmax(logits.float(), 1)
        valid = (target != self.ig).unsqueeze(1).float()
        t = torch.where(target == self.ig, torch.zeros_like(target), target)
        oh = F.one_hot(t.clamp(0, self.n - 1), self.n).permute(0, 3, 1, 2).float() * valid
        probs = probs * valid
        dims = (2, 3)
        inter = (probs * oh).sum(dims)
        dice = (2 * inter + self.eps) / (probs.sum(dims) + oh.sum(dims) + self.eps)
        present = oh.sum(dims) > 0
        if present.sum() == 0:
            return logits.sum() * 0.0
        return (1.0 - dice)[present].mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky (Abraham & Khan 2019), presence-masked and ignore-aware.

    alpha < beta penalises FALSE NEGATIVES harder than false positives, which is
    what small, rare classes need. Necrosis is ~9% of BCSS pixels and is the
    class every published model scores worst on; this term exists for it.
    """

    def __init__(self, num_classes, alpha=0.3, beta=0.7, gamma=0.75,
                 ignore_index=IGNORE, eps=1e-6, floor=1e-6):
        super().__init__()
        self.n, self.a, self.b = num_classes, alpha, beta
        self.g, self.ig, self.eps = gamma, ignore_index, eps
        self.floor = floor        # F1: keeps d/dx of base**gamma finite at base=0

    def forward(self, logits, target):
        probs = torch.softmax(logits.float(), 1)
        valid = (target != self.ig).unsqueeze(1).float()
        t = torch.where(target == self.ig, torch.zeros_like(target), target)
        oh = F.one_hot(t.clamp(0, self.n - 1), self.n).permute(0, 3, 1, 2).float() * valid
        probs = probs * valid
        dims = (2, 3)
        tp = (probs * oh).sum(dims)
        fp = (probs * (1 - oh)).sum(dims)
        fn = ((1 - probs) * oh).sum(dims)
        tv = (tp + self.eps) / (tp + self.a * fp + self.b * fn + self.eps)
        present = oh.sum(dims) > 0
        if present.sum() == 0:
            return logits.sum() * 0.0
        # ---- F1: THE CRASH FIX ------------------------------------------------
        # `base ** gamma` with 0 < gamma < 1 has derivative  gamma * base**(gamma-1),
        # i.e. gamma * base**(-0.25) for gamma=0.75, which is +INF at base == 0.
        # base reaches EXACTLY 0 whenever a patch contains a single class that the
        # model predicts near-perfectly (tp>0, fp==fn==0 -> tv==1.0). The forward
        # pass returns a clean 0.0; the BACKWARD pass returns inf, which becomes
        # NaN on the next multiply and poisons every parameter through AdamW.
        # Reproduced exactly: grad_norm=nan, NaN_in_grad=True.
        # Clamping the base away from 0 makes the derivative finite (max
        # gamma*eps**(gamma-1)) while changing the loss value by <1e-4.
        base = (1.0 - tv).clamp(min=self.floor, max=1.0)
        return (base ** self.g)[present].mean()


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jacc = 1.0 - inter / union
    if p > 1:
        jacc[1:p] = jacc[1:p] - jacc[0:-1]
    return jacc


class LovaszSoftmax(nn.Module):
    """Berman et al. CVPR'18 — convex surrogate of the Jaccard index."""

    def __init__(self, num_classes, ignore_index=IGNORE):
        super().__init__()
        self.n, self.ig = num_classes, ignore_index

    def forward(self, logits, target):
        probs = torch.softmax(logits.float(), 1)
        B, C, H, W = probs.shape
        probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
        target = target.reshape(-1)
        keep = target != self.ig
        if keep.sum() == 0:
            return probs.sum() * 0.0
        probs, target = probs[keep], target[keep]
        losses = []
        for c in range(C):
            fg = (target == c).float()
            if fg.sum() == 0:
                continue
            err = (fg - probs[:, c]).abs()
            err_sorted, perm = torch.sort(err, 0, descending=True)
            losses.append(torch.dot(err_sorted, _lovasz_grad(fg[perm])))
        if not losses:
            return probs.sum() * 0.0
        out = torch.stack(losses).mean()
        # F1b: Lovasz sorts and accumulates; a degenerate batch can yield non-finite.
        # Returning a zero-grad tensor is always safer than poisoning the optimizer.
        return out if torch.isfinite(out) else probs.sum() * 0.0


class CompoundLoss(nn.Module):
    """Weighted CE (+boundary pixel weighting, optional GCE/focal)
       + Presence Dice + Focal Tversky + annealed Lovasz.

    All terms honour ignore_index=255, so the don't-care region contributes
    exactly zero gradient to every one of them.
    """

    def __init__(self, num_classes, ce_w=1.0, dice_w=1.0, tversky_w=0.5,
                 lovasz_w=0.5, ignore_index=IGNORE, class_weights=None,
                 label_smoothing=0.05, boundary_w=0.5, focal_gamma=0.0, gce_q=0.0,
                 tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=0.75,
                 lovasz_start=0.4, tversky_floor=1e-6):
        super().__init__()
        self.n, self.ig = num_classes, ignore_index
        self.ce_w, self.dice_w = ce_w, dice_w
        self.tversky_w, self.lovasz_w = tversky_w, lovasz_w
        self.ls, self.bw = label_smoothing, boundary_w
        self.gamma, self.gce_q = focal_gamma, gce_q
        self.lovasz_start = lovasz_start
        self._progress = 1.0
        self.register_buffer(
            "cw", torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None)
        self.dice = PresenceDiceLoss(num_classes, ignore_index)
        self.tversky = FocalTverskyLoss(num_classes, tversky_alpha, tversky_beta,
                                        tversky_gamma, ignore_index,
                                        floor=tversky_floor) if tversky_w > 0 else None
        self.lovasz = LovaszSoftmax(num_classes, ignore_index) if lovasz_w > 0 else None

    def set_progress(self, frac: float):
        self._progress = float(frac)

    @property
    def lovasz_scale(self) -> float:
        if self.lovasz_start <= 0:
            return 1.0
        p = self._progress
        if p <= self.lovasz_start:
            return 0.0
        return float(min(1.0, (p - self.lovasz_start) / 0.2))

    def _ce(self, logits, target):
        logits = logits.float()
        if self.gce_q > 0:
            prob = torch.softmax(logits, 1)
            t = torch.where(target == self.ig, torch.zeros_like(target), target)
            pt = prob.gather(1, t.unsqueeze(1)).squeeze(1).clamp(min=1e-7)
            ce = (1.0 - pt.pow(self.gce_q)) / self.gce_q
            if self.cw is not None:
                ce = ce * self.cw[t]
        else:
            ce = F.cross_entropy(logits, target, weight=self.cw,
                                 ignore_index=self.ig, reduction="none",
                                 label_smoothing=self.ls)
        if self.gamma > 0:
            with torch.no_grad():
                pt = torch.exp(-ce.clamp(max=30.0))
            ce = ((1 - pt) ** self.gamma) * ce
        w = (target != self.ig).float()
        if self.bw > 0:
            w = w * boundary_weight_map(target, self.ig, extra=self.bw)
        return (ce * w).sum() / w.sum().clamp(min=1.0)

    def forward(self, logits, target):
        target = target.long()
        l_ce = self._ce(logits, target)
        l_di = self.dice(logits, target) if self.dice_w > 0 else logits.sum() * 0.0
        l_tv = self.tversky(logits, target) if self.tversky is not None else logits.sum() * 0.0
        ls = self.lovasz_scale
        l_lo = (self.lovasz(logits, target) if (self.lovasz is not None and ls > 0)
                else logits.sum() * 0.0)
        # F1c: name the offending TERM instead of reporting a bare "main=nan".
        # UC-BCSS5 run 103167 died with `main=nan aux=nan` and no way to tell which
        # of the four terms produced it; that cost a 5-hour epoch to diagnose.
        # Model 1: under fp16 a rare overflow must SKIP a step, not kill a 12-hour run.
        # The offending term is recorded so the trainer can name it in the log.
        self.last_nonfinite = None
        for nm, v in (("ce", l_ce), ("dice", l_di), ("tversky", l_tv), ("lovasz", l_lo)):
            if not torch.isfinite(v):
                self.last_nonfinite = nm
                break
        total = (self.ce_w * l_ce + self.dice_w * l_di
                 + self.tversky_w * l_tv + self.lovasz_w * ls * l_lo)
        return total, l_ce.detach(), l_di.detach(), l_tv.detach(), l_lo.detach()


# =============================================================================
# P1 — auxiliary loss budget
# =============================================================================
def ds_weights(n_heads: int, budget: float, decay: float,
               hard_cap: float = 0.35) -> List[float]:
    """Geometrically decaying DS weights whose SUM equals `budget` exactly.

    `budget` is clamped to `hard_cap` (0.35) because UC6 proved what happens when
    the auxiliary objective is allowed to rival the primary one: total aux weight
    1.2 vs main 1.0, and the run lost 4.9 mDice.

    Heads arrive deepest-first, so weight decays towards the SHALLOW end: a
    14x14 head carries useful semantics, a 56x56 head barely more than an edge
    detector, and over-weighting the shallow one is precisely what pulls the
    network towards low-level features at the cost of global delineation.
    """
    budget = float(min(max(budget, 0.0), hard_cap))
    if n_heads <= 0 or budget <= 0:
        return []
    raw = [decay ** i for i in range(n_heads)]
    s = sum(raw)
    return [budget * r / s for r in raw]


def assert_aux_sane(main_loss: torch.Tensor, aux_total: torch.Tensor,
                    max_ratio: float, step: int, logger=None):
    """P4 — abort rather than burn GPU-days on an exploding auxiliary branch."""
    m = float(main_loss.detach())
    a = float(aux_total.detach()) if torch.is_tensor(aux_total) else float(aux_total)
    if not math.isfinite(a) or not math.isfinite(m):
        return None                      # caller skips the step (fp16 overflow)
    ratio = a / (abs(m) + 1e-8)
    if ratio > max_ratio:
        raise RuntimeError(
            f"[aux-guard] AUX EXPLOSION at step {step}: aux/main = {ratio:.2f} > "
            f"{max_ratio}. main={m:.4f} aux={a:.4f}. This is the UC6 failure mode. "
            f"Check DSHead normalisation and --aux-budget.")
    return ratio


# =============================================================================
# P3 — hard-ceiling gradient clipper
# =============================================================================
class HardCeilingClipper:
    """Absolute-ceiling gradient clipping.

    UC4 used a fixed clip of 1.0 while grad norms were 10-22: ~100% of steps were
    scaled down ~20x, silently cutting the effective LR by the same factor.
    UC6 then used a ROLLING MEDIAN, which grew in lockstep with an exploding
    gradient and therefore never bound at all.

    This class avoids both: it measures the grad-norm distribution for
    `calib_iters` steps, FREEZES a threshold at the requested percentile, and
    clamps that threshold into [floor, ceiling]. After calibration the threshold
    NEVER changes again, so no pathology can drag it upward. `ceiling` is the
    non-negotiable absolute maximum.

    F2 -- CALIBRATION MUST NOT RUN DURING WARMUP.
    Run 103167 calibrated over optimizer steps 0-300, when the LR was still ramping
    from 2e-05 to 6e-05 and the gradient norm was 1.06 -> 3.36. It froze a threshold
    of 6.70. Once the LR reached its target of 3e-04 the steady-state gradient norm
    was 8.2-10.0, so the frozen threshold sat ~1.3x BELOW the operating point and
    the observed clip rate climbed to 61% and settled near 50% -- a silent ~26% cut
    to the effective learning rate on every clipped step. That is a milder rerun of
    the UC4 throttle.

    Calibration therefore starts only AFTER `calib_start` steps (defaults to the
    warmup length, so the LR is at its target and the gradients are representative).
    """

    def __init__(self, mode="calibrated", fixed=5.0, ceiling=10.0, floor=1.0,
                 calib_iters=300, percentile=90.0, calib_start=0):
        self.mode = mode
        self.ceiling = float(ceiling)
        self.floor = float(floor)
        self.calib_iters = int(calib_iters)
        self.calib_start = int(calib_start)
        self.percentile = float(percentile)
        self._samples: List[float] = []
        self._threshold: Optional[float] = (min(float(fixed), self.ceiling)
                                            if mode == "fixed" else None)
        self.n_steps = 0
        self.n_clipped = 0
        self.n_seen = 0
        self.frozen = (mode == "fixed")

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold

    def __call__(self, params) -> Tuple[float, bool]:
        if self._threshold is None:
            # Pre-calibration and calibration both clip at the CEILING, so the run is
            # never unprotected -- an explosion is still bounded at `ceiling`.
            gn = float(torch.nn.utils.clip_grad_norm_(params, self.ceiling))
            self.n_seen += 1
            if self.n_seen > self.calib_start and math.isfinite(gn):
                self._samples.append(gn)            # F2: only post-warmup samples
            if len(self._samples) >= self.calib_iters:
                t = float(np.percentile(self._samples, self.percentile))
                self._threshold = float(min(max(t, self.floor), self.ceiling))
                self.frozen = True
            self.n_steps += 1
            return gn, bool(math.isfinite(gn) and gn > self.ceiling)
        gn = float(torch.nn.utils.clip_grad_norm_(params, self._threshold))
        clipped = math.isfinite(gn) and gn > self._threshold
        self.n_steps += 1
        self.n_clipped += int(clipped)
        return gn, clipped

    @property
    def clip_rate(self) -> float:
        return self.n_clipped / max(1, self.n_steps)

    def reset_epoch(self):
        self.n_steps = self.n_clipped = 0

    def describe(self) -> str:
        if self._threshold is None:
            if self.n_seen <= self.calib_start:
                return (f"ceiling {self.ceiling:.1f} "
                        f"(warmup {self.n_seen}/{self.calib_start})")
            return (f"ceiling {self.ceiling:.1f} "
                    f"(calibrating {len(self._samples)}/{self.calib_iters})")
        return f"{self._threshold:.2f} (frozen, ceiling {self.ceiling:.1f})"


# =============================================================================
# Metric — 5 classes, all real, none excluded
# =============================================================================
class ConfusionMeter:
    """GPU confusion matrix + per-image Dice, both ignore-aware.

    Reports BOTH conventions because papers disagree and the numbers differ by
    several points on BCSS:
        mdice_agg : dataset-level (one confusion matrix over all pixels)
        mdice_img : mean over images of the per-image mean class Dice
    """

    def __init__(self, num_classes, device, ignore_index=IGNORE):
        self.n = num_classes
        self.ig = ignore_index
        self.cm = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)
        self.img_dice_sum = 0.0
        self.img_dice_cnt = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        k = target != self.ig
        if k.any():
            idx = target[k] * self.n + pred[k]
            self.cm += torch.bincount(idx.reshape(-1), minlength=self.n * self.n)
        for b in range(pred.shape[0]):
            kb = target[b] != self.ig
            if not kb.any():
                continue
            p, t = pred[b][kb], target[b][kb]
            vals = []
            for c in range(self.n):
                g = (t == c)
                if not g.any():
                    continue
                q = (p == c)
                vals.append(float(2.0 * (g & q).sum().float()
                                  / (g.sum() + q.sum()).clamp(min=1)))
            if vals:
                self.img_dice_sum += float(np.mean(vals))
                self.img_dice_cnt += 1

    @torch.no_grad()
    def compute(self):
        cm = self.cm.reshape(self.n, self.n).double()
        tp = cm.diag()
        gt, pr = cm.sum(1), cm.sum(0)
        dice_c = (2 * tp) / (gt + pr).clamp(min=1e-9)
        iou_c = tp / (gt + pr - tp).clamp(min=1e-9)
        valid = [c for c in range(self.n) if (gt[c] + pr[c]) > 0]
        return dict(
            mdice_agg=float(dice_c[valid].mean()) if valid else 0.0,
            mdice_img=self.img_dice_sum / max(1, self.img_dice_cnt),
            miou=float(iou_c[valid].mean()) if valid else 0.0,
            acc=float(tp.sum() / cm.sum().clamp(min=1)),
            dice_c=dice_c.cpu().numpy(), iou_c=iou_c.cpu().numpy(),
            support=gt.cpu().numpy())


# =============================================================================
# EMA
# =============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.9998, warmup=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.warmup, self.step = decay, warmup, 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (self.warmup + self.step))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            s = msd[k]
            if v.dtype.is_floating_point:
                if not torch.isfinite(s).all():
                    return
                v.mul_(d).add_(s.detach(), alpha=1.0 - d)
            else:
                v.copy_(s)


# =============================================================================
# SWA
# =============================================================================
def swa_init():
    return {"n": 0, "sd": None}


@torch.no_grad()
def swa_add(state, model):
    sd = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
    if state["sd"] is None:
        state["sd"], state["n"] = sd, 1
        return
    n = state["n"]
    for k, v in sd.items():
        if state["sd"][k].dtype.is_floating_point:
            state["sd"][k].mul_(n / (n + 1)).add_(v, alpha=1.0 / (n + 1))
        else:
            state["sd"][k] = v
    state["n"] = n + 1


@torch.no_grad()
def swa_apply(state, model):
    if state["sd"] is None or state["n"] == 0:
        return False
    tgt = model.state_dict()
    model.load_state_dict({k: state["sd"][k].to(tgt[k].dtype) for k in tgt}, strict=True)
    return True


# =============================================================================
# Optimizer with layer-wise LR decay
# =============================================================================
def _no_decay(name: str, p: torch.Tensor) -> bool:
    return (p.ndim <= 1 or name.endswith(".bias")
            or "relative_position_bias_table" in name
            or "group_scale" in name or "norm" in name.lower())


def encoder_layer_id(name: str, num_stages: int = 4) -> int:
    if name.startswith("encoder.patch_embed"):
        return 0
    if name.startswith("encoder.layers."):
        return int(name.split(".")[2]) // 2 + 1
    if name.startswith("encoder.norm"):
        return num_stages
    return num_stages + 1


def build_optimizer(args, model, logger=None):
    n_layers, max_id = 4, 5
    buckets: Dict[Tuple[int, bool], List[nn.Parameter]] = {}
    rational: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("rat_a") or name.endswith("rat_b"):
            rational.append(p)           # KA edge coefficients: own LR, never decayed
            continue
        buckets.setdefault((encoder_layer_id(name, n_layers), _no_decay(name, p)),
                           []).append(p)
    groups = []
    for (lid, nod), params in sorted(buckets.items()):
        lr = (args.lr if lid == max_id
              else args.lr * args.encoder_lr_scale * (args.layer_decay ** (max_id - 1 - lid)))
        groups.append({"params": params, "lr": lr, "initial_lr": lr,
                       "weight_decay": 0.0 if (nod and not args.wd_on_all) else args.weight_decay,
                       "layer_id": lid, "no_decay": nod})
    if rational:
        rlr = args.lr * getattr(args, "rational_lr_scale", 0.1)
        groups.append({"params": rational, "lr": rlr, "initial_lr": rlr,
                       "weight_decay": 0.0, "layer_id": "R", "no_decay": True})
    opt = (torch.optim.AdamW(groups, betas=(0.9, 0.999)) if args.opt == "adamw"
           else torch.optim.SGD(groups, momentum=args.momentum, nesterov=True))
    if logger:
        logger.info("[optim] " + ", ".join(
            f"L{g['layer_id']}{'/nd' if g['no_decay'] else ''}:{g['lr']:.2e}" for g in groups))
    return opt


# =============================================================================
# Per-iteration schedule
# =============================================================================
def lr_scale_at(it, total_iters, warmup_iters, sched, floor):
    if it < warmup_iters:
        return (it + 1) / max(1, warmup_iters)
    prog = min(1.0, (it - warmup_iters) / max(1, total_iters - warmup_iters))
    if sched == "cosine":
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    return floor + (1 - floor) * (1 - prog) ** 0.9


def apply_lr(optimizer, scale):
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * scale


# =============================================================================
# Model construction
# =============================================================================
def enforce_kan_only(model, logger):
    bad = [n for n, m in model.named_modules()
           if isinstance(m, SwkatBlockOpt)
           and not getattr(getattr(m, "mlp_conv", None), "is_kan", False)]
    if bad:
        raise RuntimeError("KAN-only enforcement failed:\n" + "\n".join(bad))
    logger.info("[KAN] all blocks verified full GR-KAN (is_kan=True).")


def verify_ds_heads(model, logger):
    """P2 guard — refuse to train if any auxiliary classifier lacks normalisation."""
    bad = []
    for name, m in model.named_modules():
        if isinstance(m, DSHead):
            if not isinstance(m.norm, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
                bad.append(f"{name}: missing normalisation")
            if float(m.classifier.weight.abs().max()) != 0.0:
                bad.append(f"{name}: classifier not zero-initialised")
    if bad:
        raise RuntimeError("[ds-guard] UC6 failure mode present:\n" + "\n".join(bad))
    n = sum(1 for _, m in model.named_modules() if isinstance(m, DSHead))
    logger.info(f"[ds-guard] {n} deep-supervision heads verified: "
                f"GroupNorm+GELU present, classifiers zero-initialised.")


def build_model(args, logger):
    dec = tuple(int(x) for x in args.decoder_depths.split(","))
    ks = tuple(int(x) for x in args.kernel_sizes.split(",")) if args.kernel_sizes else None
    model = kat_unet(variant=args.model_variant, img_size=args.img_size,
                     patch_size=args.patch_size, in_chans=args.in_chans,
                     num_classes=args.num_classes, window_size=args.window_size,
                     mlp_ratio=args.mlp_ratio, drop_path_rate=args.drop_path_rate,
                     groups=args.groups, decoder_depths=dec,
                     bottleneck_depth=args.bottleneck_depth, ffn_type=args.ffn_type,
                     gate_mode=args.gate_mode, use_ppm=args.use_ppm,
                     use_aux=args.use_aux, head_dropout=args.head_dropout,
                     use_fpn=args.use_fpn, use_boundary=args.use_boundary,
                     deep_supervision=args.deep_supervision, kernel_sizes=ks,
                     skip_norm=args.skip_norm, skip_bridge=args.skip_bridge,
                     kaca_window=args.kaca_window, kaca_rational=args.kaca_rational,
                     uper_levels=args.uper_levels, fpn_smooth=args.fpn_smooth,
                     grad_ckpt=args.grad_ckpt)
    if args.encoder_pretrained:
        load_swin_pretrained(model, args.encoder_pretrained,
                             min_coverage=args.min_pretrained_coverage, logger=logger)
    else:
        logger.warning("[pretrained] none supplied — the encoder starts COLD.")
    enc = sum(p.numel() for p in model.encoder.parameters())
    tot = sum(p.numel() for p in model.parameters())
    logger.info(f"[params] enc {enc / 1e6:.2f}M | dec+heads {(tot - enc) / 1e6:.2f}M "
                f"| total {tot / 1e6:.2f}M")
    logger.info(f"[kan] FFN depthwise kernels per group: {ks}")
    logger.info(f"[model-1] skip_norm={args.skip_norm} skip_bridge={args.skip_bridge} "
                f"(window {args.kaca_window}, rational={args.kaca_rational}) "
                f"uper_levels={args.uper_levels} fpn_smooth={args.fpn_smooth} "
                f"boundary={args.use_boundary} grad_ckpt={args.grad_ckpt}")
    if args.enforce_kan:
        enforce_kan_only(model, logger)
    verify_ds_heads(model, logger)
    return model


# =============================================================================
# Extension hooks (Model 2 plugs in here instead of copying this file)
# =============================================================================
class TrainHooks:
    """Default hooks = Model 1. A subclass may override any method.

    add_args(parser)                     extra CLI flags
    build_model(args, logger) -> model   construct (and convert) the network
    extra_loss(model, target) -> (tensor|None, dict)   added to the main loss
    after_step(model)                    called after every successful optimizer step
    """
    name = "model1"

    def add_args(self, p):
        return None

    def build_model(self, args, logger):
        return build_model(args, logger)

    def extra_loss(self, model, target):
        return None, {}

    def after_step(self, model):
        return None


ARCH_KEYS = ("model_variant", "img_size", "patch_size", "window_size", "mlp_ratio",
             "groups", "decoder_depths", "bottleneck_depth", "ffn_type", "gate_mode",
             "kernel_sizes", "use_ppm", "use_fpn", "use_boundary", "deep_supervision",
             "use_aux", "num_classes")
M1_ARCH_KEYS = ("skip_norm", "skip_bridge", "kaca_window", "kaca_rational",
                "uper_levels", "fpn_smooth")
# a checkpoint written before Model 1 has none of M1_ARCH_KEYS -> legacy-identical values
LEGACY_M1 = dict(skip_norm=False, skip_bridge="linear", kaca_window=7,
                 kaca_rational=True, uper_levels=3, fpn_smooth=False)
NEW_MODULE_TOKENS = ("bridge", "smooth", "top_proj", "top_smooth", "top_gamma", "skip_norm")


def apply_ckpt_arch(args, ck, logger, extra_keys: Sequence[str] = ()):
    """Rebuild the architecture recorded in a checkpoint (eval-only / warm start)."""
    saved = ck.get("args") if isinstance(ck, dict) else None
    if not saved:
        logger.warning("[arch] checkpoint has no saved args; using CLI architecture flags")
        return args
    legacy = not any(k in saved for k in M1_ARCH_KEYS)
    for k in ARCH_KEYS + tuple(extra_keys):
        if k in saved:
            setattr(args, k, saved[k])
    for k in M1_ARCH_KEYS:
        setattr(args, k, saved.get(k, LEGACY_M1[k]))
    logger.info(f"[arch] rebuilt from checkpoint ({'LEGACY bcss5' if legacy else 'Model 1+'}): "
                + ", ".join(f"{k}={getattr(args, k)}" for k in M1_ARCH_KEYS))
    return args


def load_weights_checked(model, sd, logger, allowed_missing_tokens=NEW_MODULE_TOKENS):
    """Non-strict load that REFUSES to silently leave real weights random.

    Missing keys are allowed only for modules that are identity-at-init (the new
    Model 1 branches); any other missing or unexpected key raises.
    """
    tgt = model.state_dict()
    res = model.load_state_dict(dict(sd), strict=False)   # shape mismatches still raise
    bad_missing = [k for k in res.missing_keys
                   if not any(t in k for t in allowed_missing_tokens)]
    if res.unexpected_keys or bad_missing:
        raise RuntimeError(
            f"[load] refusing partial load: {len(bad_missing)} real weights missing "
            f"(e.g. {bad_missing[:3]}), {len(res.unexpected_keys)} unexpected "
            f"(e.g. {res.unexpected_keys[:3]})")
    logger.info(f"[load] ok: {len(tgt) - len(res.missing_keys)} tensors loaded, "
                f"{len(res.missing_keys)} new identity-at-init tensors left at init")
    return res


# =============================================================================
# Sanity gate
# =============================================================================
def sanity_check_batch(loader, args, logger):
    b = next(iter(loader))
    x, y = b["image"], b["label"]
    per_img_std = x.reshape(x.shape[0], -1).std(dim=1)
    dead = int((per_img_std < 1e-3).sum())
    uniq = sorted(torch.unique(y).tolist())
    ig_frac = float((y == IGNORE).float().mean())
    logger.info(f"[sanity] image {tuple(x.shape)} mask {tuple(y.shape)} | "
                f"img mean {x.mean():.3f} std {x.std():.3f} | "
                f"per-image std min {per_img_std.min():.3f} | "
                f"train labels {uniq} | ignore {100 * ig_frac:.1f}%")
    if dead > 0:
        raise RuntimeError(f"[sanity] {dead}/{x.shape[0]} augmented images are constant. "
                           f"Augmentation is destroying the input.")
    bad = [v for v in uniq if v != IGNORE and not (0 <= v < args.num_classes)]
    if bad:
        raise RuntimeError(f"[sanity] train labels outside [0,{args.num_classes - 1}] "
                           f"and != {IGNORE}: {bad}. The raw->train remap is broken.")
    if not torch.isfinite(x).all():
        raise RuntimeError("[sanity] non-finite values in the normalised image tensor")
    if ig_frac > 0.6:
        logger.warning(f"[sanity] {100 * ig_frac:.0f}% of training pixels are don't-care. "
                       f"Lower --max-ignore-frac during patch extraction.")
    logger.info("[sanity] PASSED — real signal, labels in range, ignore handled.")


def compute_class_weights(loader, args, logger, max_batches=120):
    """Inverse-frequency^power, normalised to MEAN 1.

    The guide normalises weights to SUM 1, which makes every weight ~0.2 and
    silently scales the whole CE term down ~5x relative to Dice/Tversky — a loss
    re-balance disguised as a class re-balance. Mean-1 keeps CE's magnitude and
    changes only the RELATIVE emphasis, which is the intent.
    """
    cnt = torch.zeros(args.num_classes, dtype=torch.double)
    for i, b in enumerate(loader):
        y = b["label"]
        y = y[y != IGNORE]
        if y.numel():
            cnt += torch.bincount(y.reshape(-1), minlength=args.num_classes).double()
        if i + 1 >= max_batches:
            break
    freq = cnt / cnt.sum().clamp(min=1)
    w = (1.0 / freq.clamp(min=1e-6)) ** args.class_weight_power
    w = w / w.mean()
    w = w.clamp(max=args.class_weight_clip)
    w = w / w.mean()
    logger.info("[loss] class pixel freq: " + ", ".join(
        f"{CLASS_NAMES[i]} {100 * float(freq[i]):.2f}%" for i in range(args.num_classes)))
    logger.info("[loss] class weights  : " + ", ".join(
        f"{CLASS_NAMES[i]} {float(w[i]):.3f}" for i in range(args.num_classes)))
    return w.float().tolist()


# =============================================================================
# Train / validate
# =============================================================================
def amp_ctx(args, device):
    if args.amp_dtype == "off" or device.type != "cuda":
        return contextlib.nullcontext(), None
    if args.amp_dtype == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16), None
    scaler = (torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler())
    return torch.autocast("cuda", dtype=torch.float16), scaler


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, args,
                    scaler, ema, logger, sched_state, clipper, aux_w, hooks=None):
    model.train()
    run = dict(loss=0.0, ce=0.0, dice=0.0, tv=0.0, lov=0.0, bnd=0.0, aux=0.0, ratio=0.0)
    extra_run: Dict[str, float] = {}
    n_ok = 0
    gnorms: List[float] = []
    nonfinite = 0
    cap = int(getattr(args, "iters_per_epoch", 0) or 0)
    n = min(len(loader), cap) if cap else len(loader)
    accum = max(1, args.accum_steps)
    t0 = time.time()
    clipper.reset_epoch()
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        if cap and i >= cap:                                   # F3c
            break
        apply_lr(optimizer, lr_scale_at(sched_state["it"], sched_state["total"],
                                        args.warmup_iters, args.sched, args.min_lr_ratio))
        criterion.set_progress(sched_state["it"] / max(1, sched_state["total"]))
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        ctx, _ = amp_ctx(args, device)
        with ctx:
            main, aux_list, bnd = model.forward_train(x)

        main_loss, l_ce, l_di, l_tv, l_lo = criterion(main, y)
        extra, extra_logs = (hooks.extra_loss(model, y) if hooks is not None else (None, {}))

        # ---- P1: bounded, decaying auxiliary budget ----
        # F3a THROUGHPUT: supervise the auxiliary heads at REDUCED resolution.
        # Previously every one of the 4 heads ran the full compound loss
        # (CE + PresenceDice + FocalTversky + Lovasz) at 224x224x5, so the loss
        # stack executed 5x per step at full resolution. Downsampling the TARGET
        # with nearest-neighbour instead of upsampling the LOGITS cuts that work by
        # (224/ds)^2 -- 16x at ds=56 -- and changes nothing about what the heads
        # learn, because a 14x14 head cannot resolve 224x224 detail anyway.
        aux_total = torch.zeros((), device=x.device, dtype=main_loss.dtype)
        if aux_list and aux_w:
            ds = args.ds_loss_size
            if ds and ds < y.shape[-1]:
                y_aux = F.interpolate(y.unsqueeze(1).float(), size=(ds, ds),
                                      mode="nearest").squeeze(1).long()
            else:
                y_aux = y
            for w, a in zip(aux_w, aux_list):
                if a.shape[-2:] != y_aux.shape[-2:]:
                    a = F.interpolate(a, size=y_aux.shape[-2:], mode="bilinear",
                                      align_corners=False)
                a_loss, _, _, _, _ = criterion(a, y_aux)
                aux_total = aux_total + w * a_loss
        l_bn = torch.zeros((), device=x.device)
        if bnd is not None and args.boundary_head_weight > 0:
            l_bn = boundary_loss(bnd, y, IGNORE)
            aux_total = aux_total + args.boundary_head_weight * l_bn

        # ---- P4: abort on explosion instead of limping for 80 epochs ----
        ratio = assert_aux_sane(main_loss, aux_total, args.max_aux_ratio,
                                sched_state["it"], logger)
        loss = main_loss + aux_total
        if extra is not None:
            loss = loss + extra

        if ratio is None or not torch.isfinite(loss):
            # fp16 overflow: skip this micro-batch (and the partial accumulation window).
            nonfinite += 1
            streak = sched_state.get("nf_streak", 0) + 1
            sched_state["nf_streak"] = streak
            optimizer.zero_grad(set_to_none=True)
            logger.warning(f"[nonfinite] e{epoch} step {i}: skipped "
                           f"(term={getattr(criterion, 'last_nonfinite', None)}, streak {streak})")
            if streak >= args.max_consecutive_nonfinite:
                raise RuntimeError(
                    f"[nonfinite] {streak} consecutive non-finite steps -- the run has "
                    f"diverged, this is not an occasional fp16 overflow. Last term: "
                    f"{getattr(criterion, 'last_nonfinite', None)}")
            continue
        sched_state["nf_streak"] = 0

        (scaler.scale(loss / accum) if scaler is not None else (loss / accum)).backward()

        if (i + 1) % accum == 0 or (i + 1) == n:
            if scaler is not None:
                scaler.unscale_(optimizer)
            g, _ = clipper(model.parameters())              # P3
            if math.isfinite(g):
                gnorms.append(g)
            else:
                nonfinite += 1
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sched_state["it"] += 1
            if hooks is not None:
                hooks.after_step(model)
            if ema is not None and sched_state["it"] % args.ema_every == 0:
                ema.update(model)

        n_ok += 1
        run["loss"] += float(loss); run["ce"] += float(l_ce); run["dice"] += float(l_di)
        run["tv"] += float(l_tv); run["lov"] += float(l_lo); run["bnd"] += float(l_bn)
        run["aux"] += float(aux_total); run["ratio"] += ratio
        for kk, vv in extra_logs.items():
            extra_run[kk] = extra_run.get(kk, 0.0) + float(vv)

        if (i + 1) % args.log_interval == 0 or (i + 1) == n:
            gm = float(np.mean(gnorms)) if gnorms else float("nan")
            head_lr = max(g_["lr"] for g_ in optimizer.param_groups)
            k = max(1, n_ok)
            xtra = "".join(f" {kk} {vv / k:.4f}" for kk, vv in extra_run.items())
            logger.info(
                f"Train e{epoch} [{k}/{n}] loss {run['loss'] / k:.4f} "
                f"(ce {run['ce'] / k:.4f} dice {run['dice'] / k:.4f} "
                f"tv {run['tv'] / k:.4f} lov {run['lov'] / k:.4f} "
                f"bnd {run['bnd'] / k:.4f} aux {run['aux'] / k:.4f} "
                f"a/m {run['ratio'] / k:.3f}{xtra}) lr {head_lr:.3e} gnorm {gm:.2f} "
                f"clip {clipper.describe()}@{100 * clipper.clip_rate:.0f}% "
                f"nf {nonfinite} {(time.time() - t0) / (i + 1):.2f}s/it")

    m = max(1, n_ok)
    return {"loss": run["loss"] / m, "ce": run["ce"] / m, "dice": run["dice"] / m,
            "tv": run["tv"] / m, "lov": run["lov"] / m, "bnd": run["bnd"] / m,
            "aux": run["aux"] / m, "ratio": run["ratio"] / m, "nonfinite": nonfinite,
            "gnorm_mean": float(np.mean(gnorms)) if gnorms else float("nan"),
            "gnorm_p95": float(np.percentile(gnorms, 95)) if gnorms else float("nan"),
            "clip_thr": clipper.threshold if clipper.threshold is not None else float("nan"),
            "clip_rate": clipper.clip_rate}


_FLIPS = ((lambda t: t, lambda t: t),
          (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
          (lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),
          (lambda t: torch.flip(t, [2, 3]), lambda t: torch.flip(t, [2, 3])))


@torch.no_grad()
def scale_probs(model, x, s: float) -> torch.Tensor:
    """Softmax at test scale s, averaged over 4 flips, resized back to the input grid."""
    H, W = x.shape[-2:]
    xi = x if s == 1.0 else F.interpolate(
        x, size=(max(4, int(round(H * s / 4)) * 4), max(4, int(round(W * s / 4)) * 4)),
        mode="bilinear", align_corners=False)
    acc = None
    for fwd, inv in _FLIPS:
        o = inv(model(fwd(xi)))
        if o.shape[-2:] != (H, W):
            o = F.interpolate(o, size=(H, W), mode="bilinear", align_corners=False)
        p = torch.softmax(o.float(), 1)
        acc = p if acc is None else acc + p
    return acc / len(_FLIPS)


def fuse_scale_probs(probs: Sequence[torch.Tensor], mode: str = "mean",
                     tau: float = 1.0) -> torch.Tensor:
    """A3 -- combine per-scale probability maps.

    mean : arithmetic average (SEUNet's worst strategy on BCSS).
    conf : SEUNet P-Ens. Per pixel u and scale s, margin
           delta_s(u) = p_(1)(u) - p_(2)(u); weights w_s(u) = softmax_s(delta_s(u)/tau);
           fused p(u) = sum_s w_s(u) p_s(u).
    """
    if len(probs) == 1:
        return probs[0]
    P = torch.stack(list(probs), 0)                           # (S, B, C, H, W)
    if mode == "mean":
        return P.mean(0)
    top2 = P.topk(2, dim=2).values                            # (S, B, 2, H, W)
    margin = top2[:, :, 0] - top2[:, :, 1]                    # (S, B, H, W)
    w = torch.softmax(margin / max(tau, 1e-6), dim=0).unsqueeze(2)
    return (w * P).sum(0)


@torch.no_grad()
def tta_probs(model, x, scales: Sequence[float], mode: str = "mean", tau: float = 1.0):
    return fuse_scale_probs([scale_probs(model, x, float(s)) for s in scales], mode, tau)


@torch.no_grad()
def validate(model, loader, criterion, device, args):
    """Returns metrics + BOTH losses.

    `loss` is the training criterion (includes the Lovasz term once it ramps in, so it
    is not comparable across that boundary); `ce` is plain, unweighted CE and is the
    quantity to watch for calibration drift.
    """
    model.eval()
    meter = ConfusionMeter(args.num_classes, device, IGNORE)
    scales = [float(s) for s in args.tta_scales.split(",")] if args.tta else [1.0]
    mode = getattr(args, "tta_mode", "mean")
    tau = getattr(args, "tta_tau", 1.0)
    tot, n, ce_tot = 0.0, 0, 0.0
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        y = b["label"].to(device, non_blocking=True)
        logits = model(x)
        loss, *_ = criterion(logits, y)
        ce = F.cross_entropy(logits.float(), y, ignore_index=IGNORE)
        if torch.isfinite(loss) and torch.isfinite(ce):
            tot += float(loss)
            ce_tot += float(ce)
            n += 1
        pred = tta_probs(model, x, scales, mode, tau).argmax(1) if args.tta else logits.argmax(1)
        meter.update(pred, y)
    r = meter.compute()
    r["loss"] = tot / max(1, n)
    r["ce"] = ce_tot / max(1, n)
    return r


def parse_tta_candidates(spec: str) -> List[Tuple[float, ...]]:
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if part:
            out.append(tuple(sorted(float(v) for v in part.split(","))))
    if not out:
        raise ValueError("empty --tta-candidates")
    return out


@torch.no_grad()
def evaluate_tta_grid(model, loader, device, args, candidates, modes=("mean", "conf")):
    """Score every (scale set, fusion mode) in ONE pass: each distinct scale is run once
    per batch and re-used by every candidate that contains it."""
    model.eval()
    scales = sorted({s for c in candidates for s in c})
    meters = {(c, m): ConfusionMeter(args.num_classes, device, IGNORE)
              for c in candidates for m in modes}
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        y = b["label"].to(device, non_blocking=True)
        per = {s: scale_probs(model, x, s) for s in scales}
        for c in candidates:
            pl = [per[s] for s in c]
            for m in modes:
                meters[(c, m)].update(fuse_scale_probs(pl, m, args.tta_tau).argmax(1), y)
    results = []
    for (c, m), mt in meters.items():
        r = mt.compute()
        results.append({"scales": list(c), "mode": m, "mdice_agg": float(r["mdice_agg"]),
                        "mdice_img": float(r["mdice_img"]), "miou": float(r["miou"]),
                        "dice_c": [float(v) for v in r["dice_c"]]})
    results.sort(key=lambda d: -d["mdice_agg"])
    return results


def build_split_loader(args, split: str, logger):
    """Evaluation loader for any split directory (val / test), or None if absent."""
    d = os.path.join(args.data_root, split)
    pairs = list_pairs(os.path.join(d, args.img_subdir), os.path.join(d, args.mask_subdir))
    if not pairs:
        return None
    normalizer = None
    if args.stain_norm:
        ref = np.array(Image.open(args.stain_norm).convert("RGB"), dtype=np.uint8)
        normalizer = MacenkoNormalizer().fit(ref)
    ds = BCSS5Dataset(pairs, args, False, normalizer)
    kw = dict(num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        kw.update(persistent_workers=False, prefetch_factor=4)
    logger.info(f"[{split}] {len(ds)} patches")
    return DataLoader(ds, batch_size=max(1, args.batch_size), shuffle=False, **kw)


def log_per_class(r, logger, prefix=""):
    logger.info(f"{prefix}{'Class':<14}{'Dice':>9}{'IoU':>9}{'support%':>11}")
    sup = r["support"]
    tot = max(1.0, float(sup.sum()))
    for i, name in enumerate(CLASS_NAMES[:len(r["dice_c"])]):
        logger.info(f"{prefix}{name:<14}{r['dice_c'][i]:>9.4f}{r['iou_c'][i]:>9.4f}"
                    f"{100 * float(sup[i]) / tot:>10.2f}%")
    logger.info(f"{prefix}{'MEAN':<14}{r['mdice_agg']:>9.4f}{r['miou']:>9.4f}")


def save_ckpt(state, out, name):
    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(out, name))


# =============================================================================
# Main
# =============================================================================
def main(argv=None, hooks=None):
    hooks = hooks if hooks is not None else TrainHooks()
    args = parse_args(argv, hooks)
    if args.self_test:
        runner = getattr(hooks, "self_test", None)
        if runner is None:
            from m1_verify import run_all as runner
        sys.exit(0 if runner(args) else 1)
    if not args.data_root:
        raise SystemExit("--data-root is required (or pass --self-test)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger = setup_logging(args.output_dir)
    logger.info("=" * 78)
    logger.info(f"KAT-UNet [{hooks.name}] | BCSS 5-class | "
                f"{datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"args: {vars(args)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = args.amp_dtype
    args.amp_dtype = resolve_amp_dtype(requested, device)
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        logger.info(f"[amp] {torch.cuda.get_device_name(device)} cc {cap[0]}.{cap[1]} | "
                    f"requested {requested} -> using {args.amp_dtype}")
    logger.info(f"device {device} | amp {args.amp_dtype} | "
                f"eff batch {args.batch_size * args.accum_steps}")
    logger.info(f"[labels] raw 0 -> IGNORE({IGNORE}); raw 1..{args.num_classes} -> "
                f"0..{args.num_classes - 1} = {CLASS_NAMES[:args.num_classes]}")

    if args.tta_config:
        with open(args.tta_config) as f:
            tc = json.load(f)
        best = tc.get("best", tc)
        args.tta_scales = ",".join(str(v) for v in best["scales"])
        args.tta_mode = best["mode"]
        args.tta = True
        logger.info(f"[tta] from {args.tta_config}: scales={args.tta_scales} mode={args.tta_mode}")

    ck_eval = None
    if args.swa_only or args.eval_only:
        ck_eval = safe_torch_load(args.swa_only or args.eval_only)
        if args.arch_from_ckpt and not args.swa_only:
            args = apply_ckpt_arch(args, ck_eval, logger,
                                   extra_keys=getattr(hooks, "arch_keys", ()))
        args.encoder_pretrained = None          # weights come from the checkpoint

    if args.eval_on:
        args.val_split_name = args.eval_on
    train_loader, val_loader = build_loaders(args, logger)
    sanity_check_batch(train_loader, args, logger)

    model = hooks.build_model(args, logger).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    n_aux = ((3 if args.deep_supervision else 0) + (1 if args.use_aux else 0))
    aux_w = ds_weights(n_aux, args.aux_budget, args.aux_decay)
    logger.info(f"[aux] {n_aux} deep-supervision heads | weights "
                f"{[round(w, 4) for w in aux_w]} | TOTAL {sum(aux_w):.4f} "
                f"(main = 1.0, hard cap 0.35)  [UC6 used 1.20]")

    if args.swa_only or args.eval_only:
        ck = ck_eval
        if args.swa_only:
            sd = ck["sd"] if "sd" in ck else ck["model"]
            tgt = model.state_dict()
            model.load_state_dict({k: sd[k].to(tgt[k].dtype) for k in tgt}, strict=True)
            tag = f"swa(n={ck.get('n', '?')})"
        else:
            sd = ck.get("ema") if (args.eval_weights == "ema" and ck.get("ema")) else ck["model"]
            load_weights_checked(model, sd, logger,
                                 getattr(hooks, "allowed_missing_tokens", NEW_MODULE_TOKENS))
            tag = f"{args.eval_weights}@e{ck.get('epoch', '?')}"
        if args.tta_select:
            cands = parse_tta_candidates(args.tta_candidates)
            logger.info(f"[tta-select] {len(cands)} scale sets x 2 modes on "
                        f"'{args.val_split_name}' in one pass: {cands}")
            res = evaluate_tta_grid(model, val_loader, device, args, cands)
            for r_ in res:
                logger.info(f"[tta-select] scales={r_['scales']} mode={r_['mode']:4s} "
                            f"mDice(agg) {r_['mdice_agg']:.4f} mIoU {r_['miou']:.4f} | "
                            + " ".join(f"{CLASS_NAMES[i][:5]} {v:.3f}"
                                       for i, v in enumerate(r_["dice_c"])))
            out = {"best": {"scales": res[0]["scales"], "mode": res[0]["mode"]},
                   "split": args.val_split_name, "weights": tag, "all": res}
            with open(os.path.join(args.output_dir, "tta_selection.json"), "w") as f:
                json.dump(out, f, indent=2)
            logger.info(f"[tta-select] BEST scales={res[0]['scales']} mode={res[0]['mode']} "
                        f"mDice {res[0]['mdice_agg']:.4f} -> tta_selection.json")
            return
        crit = CompoundLoss(args.num_classes, ignore_index=IGNORE).to(device)
        r = validate(model, val_loader, crit, device, args)
        logger.info(f"[eval] weights={tag} split={args.val_split_name} "
                    f"tta_scales={args.tta_scales} mode={args.tta_mode}")
        logger.info(f"[eval] mDice(agg) {r['mdice_agg']:.4f} | mDice(img) "
                    f"{r['mdice_img']:.4f} | mIoU {r['miou']:.4f} | acc {r['acc']:.4f}")
        log_per_class(r, logger, prefix="[eval] ")
        with open(os.path.join(args.output_dir,
                               f"eval_{args.val_split_name}.json"), "w") as f:
            json.dump({k: (v.tolist() if hasattr(v, "tolist") else v)
                       for k, v in r.items()}, f, indent=2)
        return

    cw = None
    if args.class_weights == "auto":
        cw = compute_class_weights(train_loader, args, logger)
    elif args.class_weights and args.class_weights != "none":
        cw = [float(v) for v in args.class_weights.split(",")]
        assert len(cw) == args.num_classes
        logger.info(f"[loss] class_weights={cw}")

    criterion = CompoundLoss(
        args.num_classes, ce_w=args.ce_weight, dice_w=args.dice_weight,
        tversky_w=args.tversky_weight, lovasz_w=args.lovasz_weight,
        ignore_index=IGNORE, class_weights=cw, label_smoothing=args.label_smoothing,
        boundary_w=args.boundary_weight, focal_gamma=args.focal_gamma,
        gce_q=args.gce_q, tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta, tversky_gamma=args.tversky_gamma,
        lovasz_start=args.lovasz_start_frac).to(device)
    logger.info(f"[loss] {args.ce_weight}*CE(ls={args.label_smoothing},bnd+{args.boundary_weight}) "
                f"+ {args.dice_weight}*PresenceDice "
                f"+ {args.tversky_weight}*FocalTversky(a={args.tversky_alpha},"
                f"b={args.tversky_beta},g={args.tversky_gamma}) "
                f"+ {args.lovasz_weight}*Lovasz(from {args.lovasz_start_frac:.0%}) "
                f"| all with ignore_index={IGNORE}")

    optimizer = build_optimizer(args, model, logger)
    calib_start = (args.warmup_iters if args.clip_calib_start < 0
                   else args.clip_calib_start)
    clipper = HardCeilingClipper(args.grad_clip_mode, args.grad_clip,
                                 args.grad_clip_ceiling, args.grad_clip_floor,
                                 args.clip_calib_iters, args.clip_percentile,
                                 calib_start)
    logger.info(f"[clip] mode={args.grad_clip_mode} ceiling={args.grad_clip_ceiling} "
                f"floor={args.grad_clip_floor} | calibration starts at step "
                f"{calib_start} (after warmup), samples {args.clip_calib_iters} steps "
                f"@p{args.clip_percentile:.0f}, then FREEZES")
    _, scaler = amp_ctx(args, device)
    if args.init_from:
        ck0 = safe_torch_load(args.init_from)
        sd0 = ck0.get("ema") or ck0.get("model") or ck0
        load_weights_checked(model, sd0, logger,
                             getattr(hooks, "allowed_missing_tokens", NEW_MODULE_TOKENS)
                             + ("skip_norm",))
        logger.info(f"[init] warm-started from {args.init_from}")
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    cap = int(args.iters_per_epoch or 0)
    iters_ep = min(len(train_loader), cap) if cap else len(train_loader)
    total_iters = max(1, (iters_ep // max(1, args.accum_steps)) * args.epochs)
    if cap:
        logger.info(f"[sched] --iters-per-epoch {cap}: one epoch covers "
                    f"{100 * iters_ep / len(train_loader):.1f}% of the {len(train_loader)} "
                    f"available batches; the sampler reshuffles each epoch so all data "
                    f"is still seen across the run")
    sched_state = {"it": 0, "total": total_iters}
    logger.info(f"[sched] {total_iters} optimizer steps over {args.epochs} epochs | "
                f"warmup {args.warmup_iters} ({100 * args.warmup_iters / total_iters:.1f}%)")

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    init_csv(csv_path)
    start_epoch, best, best_ep = 0, -1.0, -1
    best_raw = -1.0
    val_hist: List[float] = []
    swa_state = swa_init()

    if args.resume and os.path.isfile(args.resume):
        ck = safe_torch_load(args.resume)
        model.load_state_dict(ck["model"])
        if ema and ck.get("ema"):
            ema.ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        sched_state["it"] = ck.get("it", 0)
        start_epoch = ck.get("epoch", -1) + 1
        best = ck.get("best", best)
        logger.info(f"resumed {args.resume} @ e{start_epoch} (best {best:.4f})")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if args.freeze_encoder_epochs > 0:
            frozen = epoch < args.freeze_encoder_epochs
            for p in model.encoder.parameters():
                p.requires_grad_(not frozen)
            if epoch in (0, args.freeze_encoder_epochs):
                logger.info(f"[encoder] {'FROZEN' if frozen else 'trainable'} at e{epoch}")

        tr = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             epoch, args, scaler, ema, logger, sched_state,
                             clipper, aux_w, hooks)

        row = {"epoch": epoch, "train_loss": f"{tr['loss']:.6f}", "ce": f"{tr['ce']:.6f}",
               "dice": f"{tr['dice']:.6f}", "tversky": f"{tr['tv']:.6f}",
               "lovasz": f"{tr['lov']:.6f}", "bnd": f"{tr['bnd']:.6f}",
               "aux": f"{tr['aux']:.6f}", "aux_ratio": f"{tr['ratio']:.4f}",
               "nonfinite": tr["nonfinite"],
               "lr_head": f"{max(g['lr'] for g in optimizer.param_groups):.3e}",
               "lr_enc": f"{min(g['lr'] for g in optimizer.param_groups):.3e}",
               "gnorm_mean": f"{tr['gnorm_mean']:.4f}",
               "gnorm_p95": f"{tr['gnorm_p95']:.4f}",
               "clip_thr": f"{tr['clip_thr']:.4f}", "clip_rate": f"{tr['clip_rate']:.4f}"}

        if (epoch + 1) % args.eval_interval == 0:
            em = ema.ema if ema else model
            r = validate(em, val_loader, criterion, device, args)
            val_hist.append(float(r["mdice_agg"]))
            k_s = max(1, int(args.select_smooth))
            smooth = float(np.mean(val_hist[-k_s:]))
            logger.info(f"Val e{epoch} loss {r['loss']:.4f} ce {r['ce']:.4f} | "
                        f"mDice(agg) {r['mdice_agg']:.4f} (avg{k_s} {smooth:.4f}) "
                        f"| mDice(img) {r['mdice_img']:.4f} | mIoU {r['miou']:.4f} "
                        f"| acc {r['acc']:.4f}{' [ema]' if ema else ''}"
                        f"{' [tta]' if args.tta else ''}")
            log_per_class(r, logger, prefix="  ")
            row.update({"val_loss": f"{r['loss']:.6f}", "val_ce": f"{r['ce']:.6f}",
                        "val_smooth": f"{smooth:.6f}",
                        "val_mdice_agg": f"{r['mdice_agg']:.6f}",
                        "val_mdice_img": f"{r['mdice_img']:.6f}",
                        "val_miou": f"{r['miou']:.6f}", "val_acc": f"{r['acc']:.6f}"})
            for i, nm in enumerate(CLASS_NAMES[:args.num_classes]):
                row[f"dice_{nm}"] = f"{r['dice_c'][i]:.6f}"
            state = {"epoch": epoch, "it": sched_state["it"], "model": model.state_dict(),
                     "ema": ema.ema.state_dict() if ema else None,
                     "optimizer": optimizer.state_dict(),
                     "metrics": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                 for k, v in r.items()},
                     "args": vars(args)}
            # Selection on a moving average: with a 2-institution val set, the single
            # best epoch is a noisy maximum (winner's curse).
            if smooth > best:
                best, best_ep = smooth, epoch
                state["best"] = best
                save_ckpt(state, args.output_dir, "best_model.pth")
                logger.info(f"  * new best avg{k_s} mDice {best:.4f} (e{epoch}, raw "
                            f"{r['mdice_agg']:.4f}) -> best_model.pth")
            if r["mdice_agg"] > best_raw:
                best_raw = float(r["mdice_agg"])
                state["best"] = best_raw
                save_ckpt(state, args.output_dir, "best_raw_model.pth")

        save_ckpt({"epoch": epoch, "it": sched_state["it"], "model": model.state_dict(),
                   "ema": ema.ema.state_dict() if ema else None,
                   "optimizer": optimizer.state_dict(), "best": best,
                   "args": vars(args)}, args.output_dir, "last_model.pth")

        if args.swa_last > 0 and epoch >= args.epochs - args.swa_last:
            swa_add(swa_state, ema.ema if ema else model)
            torch.save({"n": swa_state["n"], "sd": swa_state["sd"], "args": vars(args)},
                       os.path.join(args.output_dir, "swa_model.pth"))
            logger.info(f"[swa] accumulated {swa_state['n']} -> swa_model.pth")

        row["sec"] = f"{time.time() - t0:.1f}"
        append_csv(csv_path, row)
        logger.info(f"epoch {epoch} done {time.time() - t0:.1f}s (loss {tr['loss']:.4f} "
                    f"gnorm {tr['gnorm_mean']:.2f} clip {tr['clip_thr']:.2f}"
                    f"@{100 * tr['clip_rate']:.1f}% aux/main {tr['ratio']:.3f} "
                    f"nf {tr['nonfinite']})")
        if tr["clip_rate"] > 0.5:
            logger.warning(f"[clip] {100 * tr['clip_rate']:.0f}% of steps clipped — the "
                           f"threshold is throttling the effective LR. Raise "
                           f"--grad-clip-ceiling / --clip-percentile.")

    if args.swa_last > 0 and swa_state["n"] > 1:
        swa_model = copy.deepcopy(model)
        swa_apply(swa_state, swa_model)
        r = validate(swa_model, val_loader, criterion, device, args)
        logger.info(f"[swa] averaged {swa_state['n']} checkpoints -> mDice(agg) "
                    f"{r['mdice_agg']:.4f} | mIoU {r['miou']:.4f} (best single {best:.4f})")
        log_per_class(r, logger, prefix="[swa] ")
        if r["mdice_agg"] > best:
            best, best_ep = r["mdice_agg"], "swa"
            save_ckpt({"epoch": args.epochs - 1, "model": swa_model.state_dict(),
                       "ema": swa_model.state_dict(), "best": best,
                       "args": vars(args)}, args.output_dir, "best_model.pth")
            logger.info("  * SWA beat the best single checkpoint -> best_model.pth replaced")

    logger.info(f"Finished. Best avg{args.select_smooth} mDice(agg) {best:.4f} @ epoch "
                f"{best_ep} | best single-epoch {best_raw:.4f}.")

    # ---------------- held-out test split (only if present) ----------------
    test_dir = os.path.join(args.data_root, args.test_split_name)
    if args.skip_test:
        logger.info("[test] skipped (--skip-test)")
    elif not os.path.isdir(test_dir):
        logger.warning(f"[test] no '{args.test_split_name}/' under {args.data_root}. The "
                       f"official test institutions (OL, LL, E2, EW, GM, S3) are required for "
                       f"any benchmark claim -- copy that split here and run --eval-only.")
    else:
        best_path = os.path.join(args.output_dir, "best_model.pth")
        test_loader = build_split_loader(args, args.test_split_name, logger)
        if test_loader is not None and os.path.isfile(best_path):
            ckb = safe_torch_load(best_path)
            sdb = ckb.get("ema") or ckb["model"]
            model.load_state_dict(sdb)
            r = validate(model, test_loader, criterion, device, args)
            logger.info(f"[test] mDice(agg) {r['mdice_agg']:.4f} | mDice(img) "
                        f"{r['mdice_img']:.4f} | mIoU {r['miou']:.4f} | acc {r['acc']:.4f} "
                        f"(tta {args.tta_scales} {args.tta_mode})")
            log_per_class(r, logger, prefix="[test] ")
            with open(os.path.join(args.output_dir, "eval_test.json"), "w") as f:
                json.dump({k: (v.tolist() if hasattr(v, "tolist") else v)
                           for k, v in r.items()}, f, indent=2)


if __name__ == "__main__":
    main()
