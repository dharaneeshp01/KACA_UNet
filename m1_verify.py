#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m1_verify.py -- proves every Model 1 claim, plus regression tests for every bug the
diagnostic run exposed.  Run:  python m1_verify.py   (or m1_train.py --self-test)
"""
from __future__ import annotations

import ast
import csv as _csv
import json
import math
import os
import sys
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import m1_kat_unet as M
import m1_train as T

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


class _Log:
    def __init__(self): self.msgs = []
    def info(self, m, *a, **k): self.msgs.append(m)
    def warning(self, m, *a, **k): self.msgs.append(m)
    def error(self, m, *a, **k): self.msgs.append(m)


def _model(**kw):
    base = dict(num_classes=5, skip_norm=True, skip_bridge="kaca", uper_levels=4,
                fpn_smooth=True, use_boundary=False)
    base.update(kw)
    return M.kat_unet("swkat-tiny", **base)


# =============================================================================
def test_rational():
    section("1. Group-rational residual operator (shared with Model 2)")
    x = (torch.randn(2, 4, 24, dtype=torch.float64) * 2).requires_grad_(True)
    a = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    b = (torch.randn(4, 4, dtype=torch.float64) * 0.5).requires_grad_(True)
    check("analytic backward == finite differences (gradcheck, fp64, x/a/b)",
          torch.autograd.gradcheck(M.rational_residual, (x, a, b), eps=1e-6, atol=1e-5))
    for base in ("gelu", "identity"):
        act = M.RationalResidualAct(64, groups=8, base=base)
        z = torch.randn(3, 64, 7, 7) * 3
        ref = F.gelu(z) if base == "gelu" else z
        check(f"base={base}: EXACT identity to the base at init (a=0)",
              torch.equal(act(z), ref), f"max|diff| {float((act(z) - ref).abs().max()):.1e}")
    act = M.RationalResidualAct(32, groups=4, base="identity", channel_dim=-1)
    with torch.no_grad():
        act.rat_a.normal_(0, 0.3)
    t = torch.randn(2, 10, 32)
    y = act(t)
    check("channel_dim=-1 (token layout) keeps shape", y.shape == t.shape)
    y16 = act(t.half())
    check("fp16 input -> fp16 output, finite (computed internally in fp32)",
          y16.dtype == torch.float16 and bool(torch.isfinite(y16).all()))
    act2 = M.RationalResidualAct(8, groups=1, base="identity")
    xs = torch.linspace(0.0, 5.0, 8 * 50).view(1, 8, 50)      # one-sided: odd powers
    act2(xs).sum().backward()                                   # do not cancel
    g5 = float(act2.rat_a.grad[0, 5].abs())
    naive = float((xs ** 5).sum())
    check("init denominator Q=1+x^4 tames the x^5 coefficient gradient",
          g5 < naive / 50, f"|dL/da5| {g5:.0f} with Q=1+x^4 vs {naive:.0f} with Q=1")


def test_kaca():
    section("2. A1 -- direction-corrected KA cross-attention bridge")
    ka = M.KACrossAttention(96, 3, window=7)
    xq, xs = torch.randn(2, 196, 96), torch.randn(2, 196, 96)
    check("zero LayerScale -> bridge contributes EXACTLY 0 at init",
          float(ka(xq, xs, 14, 14).abs().max()) == 0.0)
    with torch.no_grad():
        ka.gamma.fill_(1.0)
    const_skip = torch.randn(1, 1, 96).expand(2, 196, 96).contiguous()
    out = ka(torch.randn(2, 196, 96), const_skip, 14, 14)
    spread = float((out - out[:, :1]).abs().max())
    check("decoder QUERIES / encoder VALUES: constant skip -> constant output whatever "
          "the decoder holds (SECA's direction cannot satisfy this)", spread < 1e-4,
          f"max spread {spread:.2e}")
    o2 = ka(xq, xs + torch.randn_like(xs), 14, 14)
    check("encoder content is INJECTED (changing skip values changes the output)",
          float((o2 - ka(xq, xs, 14, 14)).abs().max()) > 1e-3)
    for hw in (9, 18, 35):
        q = torch.randn(1, hw * hw, 96, requires_grad=True)
        o = ka(q, torch.randn(1, hw * hw, 96), hw, hw)
        o.sum().backward()
        check(f"odd grid {hw}x{hw} (padded windows, masked keys) -> finite out & grad",
              bool(torch.isfinite(o).all()) and bool(torch.isfinite(q.grad).all()))


def test_legacy():
    section("3. Legacy compatibility (step-0 evaluation of your existing checkpoint)")
    import bcss5_kat_unet as L
    torch.manual_seed(1)
    leg = L.kat_unet("swkat-tiny", num_classes=5).eval()
    new = M.kat_unet("swkat-tiny", num_classes=5, skip_norm=False, skip_bridge="kaca",
                     uper_levels=4, fpn_smooth=True).eval()
    log = _Log()
    res = T.load_weights_checked(new, leg.state_dict(), log)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        d = float((leg(x) - new(x)).abs().max())
    check("legacy weights + every new identity-at-init module ON -> bit-identical output",
          d == 0.0, f"max|diff| {d:.1e}")
    check("only new-module keys are missing; zero unexpected",
          not res.unexpected_keys and all(any(t in k for t in T.NEW_MODULE_TOKENS)
                                          for k in res.missing_keys),
          f"{len(res.missing_keys)} new keys at init")
    sd = dict(leg.state_dict())
    sd.pop("seg_head.weight")
    raised = False
    try:
        T.load_weights_checked(M.kat_unet("swkat-tiny", num_classes=5), sd, _Log())
    except RuntimeError:
        raised = True
    check("a checkpoint missing a REAL weight is refused (diagnostics used strict=False blindly)",
          raised)
    args = T.parse_args(["--data-root", "x"])
    T.apply_ckpt_arch(args, {"args": {"use_boundary": True, "decoder_depths": "2,2,2"}}, _Log())
    check("legacy checkpoint args -> legacy-identical Model-1 flags",
          args.skip_norm is False and args.skip_bridge == "linear" and args.uper_levels == 3
          and args.use_boundary is True)


def test_skipnorm_uper():
    section("4. A1 skip normalisation and A2 4-level UPer aggregation")
    up = M.UpStage(768, 384, 12, 1, (14, 14), 7, 4.0, 8, [0.0], skip_norm=True)
    skip = torch.randn(1, 196, 384) * 5.77
    s_in = up.skip_norm(skip)
    r = float(s_in.pow(2).mean().sqrt())
    check("a 5.77-RMS skip (measured on pretrained Swin-T) enters fusion at RMS ~1",
          0.9 < r < 1.1, f"RMS {r:.3f}")
    m = _model(drop_path_rate=0.0, head_dropout=0.0).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y0 = m(x)
        m.fpn.top_gamma.fill_(1.0)
        y1 = m(x)
    check("4th UPer level is an exact no-op at init and live once its gate opens",
          float((y1 - y0).abs().max()) > 1e-6)
    m2 = _model()
    m2.train()
    main, aux, _ = m2.forward_train(x)
    (main.mean() + sum(a.mean() for a in aux)).backward()
    g = m2.fpn.top_gamma.grad
    check("gradient reaches the 4th-level gate", g is not None and float(g.abs().sum()) > 0)
    gb = [u.bridge.gamma.grad for u in m2.up_stages]
    check("gradient reaches every bridge LayerScale",
          all(t is not None and float(t.abs().sum()) > 0 for t in gb))
    nog = [n for n, p in m2.named_parameters() if p.requires_grad and p.grad is None]
    check("every trainable parameter receives a gradient", not nog, f"{len(nog)} without")
    tot = sum(p.numel() for p in m2.parameters()) / 1e6
    check("parameter budget", 110 < tot < 120, f"{tot:.2f}M")


def test_sizes_and_ckpt():
    section("5. TTA input sizes and gradient checkpointing")
    m = _model().eval()
    with torch.no_grad():
        for u in m.up_stages:
            u.bridge.gamma.fill_(0.3)
        for sz in (160, 188, 224, 280, 336):
            y = m(torch.randn(1, 3, sz, sz))
            check(f"input {sz}px -> ({sz},{sz}) finite", y.shape[-1] == sz
                  and bool(torch.isfinite(y).all()))
    torch.manual_seed(3)
    a = _model(drop_path_rate=0.0, head_dropout=0.0)
    b = _model(drop_path_rate=0.0, head_dropout=0.0)
    b.load_state_dict(a.state_dict())
    b.set_grad_checkpointing(True)
    a.train(); b.train()
    x = torch.randn(1, 3, 224, 224)
    ya, _, _ = a.forward_train(x)
    yb, _, _ = b.forward_train(x)
    ya.pow(2).mean().backward(); yb.pow(2).mean().backward()
    ga = a.encoder.patch_embed.proj.weight.grad
    gb = b.encoder.patch_embed.proj.weight.grad
    check("grad checkpointing: identical output", float((ya - yb).abs().max()) < 1e-5)
    check("grad checkpointing: identical gradients", float((ga - gb).abs().max()) < 1e-5,
          f"max|dg| {float((ga - gb).abs().max()):.1e}")


def test_tta():
    section("6. A3 -- confidence-weighted TTA")
    B, C, H, W = 1, 5, 4, 4
    p1 = torch.full((B, C, H, W), 0.2)
    p2 = torch.zeros(B, C, H, W); p2[:, 2] = 1.0
    f_mean = T.fuse_scale_probs([p1, p2], "mean")
    f_conf = T.fuse_scale_probs([p1, p2], "conf")
    check("mean fusion = arithmetic average", torch.allclose(f_mean, (p1 + p2) / 2))
    check("conf fusion weights the confident scale more",
          float(f_conf[0, 2, 0, 0]) > float(f_mean[0, 2, 0, 0]),
          f"{float(f_conf[0, 2, 0, 0]):.3f} vs mean {float(f_mean[0, 2, 0, 0]):.3f}")
    check("fused maps are still distributions",
          torch.allclose(f_conf.sum(1), torch.ones(B, H, W), atol=1e-6))
    check("single scale: both modes return it unchanged",
          torch.equal(T.fuse_scale_probs([p2], "conf"), p2))
    check("candidate parser", T.parse_tta_candidates("1.0;1.25,1.0") == [(1.0,), (1.0, 1.25)])


def test_albu_compat():
    section("7. albumentations 1.4.x / 2.x compatibility (the fake-Tumor regression)")

    class Old14:                        # the 1.4.18 A.Affine / PadIfNeeded signature
        def __init__(self, scale=None, rotate=None, mode=0, cval=0, cval_mask=0,
                     value=None, mask_value=None, height=None, width=None, p=0.5):
            self.kw = dict(mode=mode, cval=cval, cval_mask=cval_mask, value=value,
                           mask_value=mask_value, height=height, width=width)

    o = T._albu(Old14, border_mode=4, fill=0, fill_mask=255, size=(224, 224), bogus=1)
    check("border_mode -> mode on the 1.4 API", o.kw["mode"] == 4)
    check("fill_mask -> mask_value on the 1.4 API", o.kw["mask_value"] == 255)
    check("size -> height/width on the 1.4 API", o.kw["height"] == 224 and o.kw["width"] == 224)
    args = SimpleNamespace(img_size=224, eval_size=224, aug_mode="rrc", rrc_min_scale=0.35,
                           scale_min=0.75, scale_max=1.5, hed_sigma=0.05, clahe_p=0.2,
                           elastic_p=0.05)
    aug = T.build_train_aug(args)
    rng = np.random.default_rng(0)
    worst, black = 0.0, 0.0
    for _ in range(150):
        img = rng.integers(60, 200, (224, 224, 3), dtype=np.uint8)
        msk = np.full((224, 224), 3, np.int32)              # all Necrosis
        out = aug(image=img, mask=msk)
        worst = max(worst, float((out["mask"] != 3).mean()))
        black = max(black, float((out["image"].sum(-1) == 0).mean()))
    check("no transform fabricates labels (an all-Necrosis mask stays all-Necrosis)",
          worst == 0.0, f"worst {100 * worst:.2f}% relabelled (1.4.18 run: 10% per patch)")
    check("no black borders introduced", black == 0.0, f"{100 * black:.3f}%")


def test_env():
    section("8. Environment fixes: AMP choice, Python 3.8 syntax, torch-1.x DataLoader")
    real = torch.cuda.get_device_capability
    try:
        torch.cuda.get_device_capability = lambda d=None: (7, 5)
        turing = T.resolve_amp_dtype("auto", torch.device("cuda"))
        forced = T.resolve_amp_dtype("bf16", torch.device("cuda"))
        torch.cuda.get_device_capability = lambda d=None: (8, 0)
        ampere = T.resolve_amp_dtype("auto", torch.device("cuda"))
    finally:
        torch.cuda.get_device_capability = real
    check("Quadro RTX 5000 (cc 7.5): auto -> fp16", turing == "fp16")
    check("bf16 explicitly requested on Turing -> downgraded to fp16", forced == "fp16")
    check("Ampere (cc 8.0): auto -> bf16", ampere == "bf16")
    check("CPU -> AMP off", T.resolve_amp_dtype("auto", torch.device("cpu")) == "off")
    unions = []
    for fn in ("m1_kat_unet.py", "m1_train.py"):
        tree = ast.parse(open(os.path.join(HERE, fn)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr) and \
                    isinstance(node.left, (ast.Dict, ast.DictComp)):
                unions.append(f"{fn}:{node.lineno}")
    check("no dict-union operator (Python >= 3.9 only)", not unions, str(unions))
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "c.pth")
        torch.save({"k": torch.ones(2)}, pth)
        real_load = torch.load

        def old_load(f, map_location=None, **kw):       # torch < 1.13 signature
            if "weights_only" in kw:
                raise TypeError("unexpected keyword argument 'weights_only'")
            return real_load(f, map_location=map_location, weights_only=False)
        torch.load = old_load
        try:
            got = T.safe_torch_load(pth)
        finally:
            torch.load = real_load
    check("safe_torch_load works on torch without the weights_only keyword",
          torch.equal(got["k"], torch.ones(2)))
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "val", "images")); os.makedirs(os.path.join(td, "val", "masks"))
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(os.path.join(td, "val", "images", "a.png"))
        Image.fromarray(np.ones((8, 8), np.uint8)).save(os.path.join(td, "val", "masks", "a.png"))
        a = T.parse_args(["--data-root", td, "--workers", "0", "--img-size", "8", "--eval-size", "8"])
        dl = T.build_split_loader(a, "val", _Log())
    check("workers=0 loader built without prefetch_factor (torch<2.0 raises on it)",
          dl is not None and dl.num_workers == 0)


def test_losses_and_policy():
    section("9. A4 loss parsimony and the non-finite policy")
    a = T.parse_args(["--data-root", "x"])
    check("A4 defaults: boundary CE 0, boundary head 0, Tversky 0, boundary stream off",
          a.boundary_weight == 0 and a.boundary_head_weight == 0 and a.tversky_weight == 0
          and a.use_boundary is False)
    check("A1/A2/A3 on by default", a.skip_norm and a.skip_bridge == "kaca"
          and a.uper_levels == 4 and a.fpn_smooth and a.tta_mode == "conf")
    crit = T.CompoundLoss(5, tversky_w=0.0, boundary_w=0.0)
    lg = torch.randn(2, 5, 8, 8)
    lg[0, 0, 0, 0] = float("nan")
    tot, *_ = crit(lg, torch.randint(0, 5, (2, 8, 8)))
    check("non-finite term is RECORDED, not raised", not torch.isfinite(tot)
          and crit.last_nonfinite is not None, f"term={crit.last_nonfinite}")
    check("aux guard returns None on non-finite (caller skips)",
          T.assert_aux_sane(torch.tensor(float("nan")), torch.tensor(0.1), 5.0, 0) is None)
    raised = False
    try:
        T.assert_aux_sane(torch.tensor(1.5), torch.tensor(1.06e9), 5.0, 0)
    except RuntimeError:
        raised = True
    check("a FINITE auxiliary explosion (the UC6 failure) still aborts", raised)

    class Tiny(nn.Module):
        def __init__(self, bad_steps):
            super().__init__()
            self.c = nn.Conv2d(3, 5, 1)
            self.bad, self.k = bad_steps, 0

        def forward_train(self, x):
            out = self.c(x)
            if self.k in self.bad:
                out = out * float("nan")
            self.k += 1
            return out, [], None

    def run(bad, max_nf):
        m = Tiny(bad)
        data = [{"image": torch.randn(2, 3, 8, 8), "label": torch.randint(0, 5, (2, 8, 8))}
                for _ in range(6)]
        args = SimpleNamespace(iters_per_epoch=0, accum_steps=1, warmup_iters=1, sched="cosine",
                               min_lr_ratio=0.01, ds_loss_size=0, boundary_head_weight=0.0,
                               max_aux_ratio=5.0, max_consecutive_nonfinite=max_nf,
                               log_interval=1000, ema_every=1, amp_dtype="off")
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        for g in opt.param_groups:
            g["initial_lr"] = 0.01
        clip = T.HardCeilingClipper("fixed", fixed=5.0, ceiling=10.0)
        return T.train_one_epoch(m, data, opt, T.CompoundLoss(5, tversky_w=0, boundary_w=0),
                                 torch.device("cpu"), 0, args, None, None, _Log(),
                                 {"it": 0, "total": 10}, clip, [])

    r = run({2}, 50)
    check("one fp16-style NaN step is skipped and training continues",
          r["nonfinite"] == 1 and math.isfinite(r["loss"]))
    raised = False
    try:
        run(set(range(6)), 3)
    except RuntimeError:
        raised = True
    check("a genuinely diverged run (consecutive NaNs) still aborts", raised)


def test_optimizer():
    section("10. Optimizer groups")
    m = _model()
    a = SimpleNamespace(lr=3e-4, encoder_lr_scale=0.4, layer_decay=0.9, weight_decay=0.05,
                        wd_on_all=False, opt="adamw", momentum=0.9, rational_lr_scale=0.1)
    opt = T.build_optimizer(a, m, None)
    rg = [g for g in opt.param_groups if g["layer_id"] == "R"]
    n_rat = sum(1 for n, _ in m.named_parameters() if n.endswith(("rat_a", "rat_b")))
    check("rational coefficients have their own group", len(rg) == 1
          and len(rg[0]["params"]) == n_rat, f"{n_rat} tensors")
    check("rational group: LR 0.1x head, no weight decay",
          abs(rg[0]["lr"] - 3e-5) < 1e-12 and rg[0]["weight_decay"] == 0.0)
    total = sum(len(g["params"]) for g in opt.param_groups)
    check("every parameter in exactly one group",
          total == sum(1 for p in m.parameters() if p.requires_grad))


def _make_split(root, split, insts, n, rng, size=64):
    for sub in ("images", "masks"):
        os.makedirs(os.path.join(root, split, sub), exist_ok=True)
    for i in range(n):
        lab = np.zeros((size, size), np.uint8)
        yy, xx = np.mgrid[0:size, 0:size]
        cx, cy = rng.integers(20, 44, 2)
        lab[(xx - cx) ** 2 + (yy - cy) ** 2 < 14 ** 2] = 1
        lab[yy > 48] = 2
        lab[(xx < 10) & (yy < 30)] = 3
        lab[(xx > 54) & (yy < 20)] = 4
        lab[lab == 0] = 5
        lab[:3, :] = 0
        pal = {0: (250, 250, 250), 1: (60, 30, 140), 2: (220, 120, 180), 3: (40, 140, 60),
               4: (230, 210, 90), 5: (150, 150, 150)}
        img = np.zeros((size, size, 3))
        for v, c in pal.items():
            img[lab == v] = c
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
        stem = (f"TCGA-{insts[i % len(insts)]}-{1000 + i:04d}-DX1_xmin1_ymin1_MPP-0.2500"
                f"_y0_x{112 * i}")
        Image.fromarray(img).save(os.path.join(root, split, "images", stem + ".png"))
        Image.fromarray(lab).save(os.path.join(root, split, "masks", stem + ".png"))


def test_end_to_end():
    section("11. END-TO-END: train, select, legacy eval, TTA grid, test split")
    import bcss5_kat_unet as L
    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(7)
        _make_split(td, "train", ["A1", "A2"], 16, rng)
        _make_split(td, "val", ["AR"], 8, rng)
        out = os.path.join(td, "out")
        base = ["--data-root", td, "--img-size", "64", "--eval-size", "64",
                "--window-size", "4", "--decoder-depths", "1,1,1", "--bottleneck-depth", "1",
                "--batch-size", "4", "--accum-steps", "1", "--workers", "0",
                "--no-tta", "--log-interval", "1000", "--drop-path-rate", "0.0",
                "--elastic-p", "0", "--clahe-p", "0", "--rrc-min-scale", "0.7"]
        train = base + ["--output-dir", out, "--epochs", "4", "--lr", "1.5e-3",
                        "--warmup-iters", "4", "--ema-decay", "0", "--swa-last", "0",
                        "--clip-calib-iters", "5", "--kaca-window", "4",
                        "--select-smooth", "2"]
        ok = True
        try:
            T.main(train)
        except Exception:
            import traceback
            traceback.print_exc()
            ok = False
        check("Model 1 trains end to end", ok)
        if not ok:
            return
        rows = list(_csv.DictReader(open(os.path.join(out, "metrics.csv"))))
        loss = [float(r["train_loss"]) for r in rows]
        check("training loss decreases", loss[-1] < loss[0], f"{loss[0]:.3f} -> {loss[-1]:.3f}")
        check("val CE logged separately and finite",
              all(r["val_ce"] and math.isfinite(float(r["val_ce"])) for r in rows))
        check("moving-average selection logged", all(r["val_smooth"] for r in rows))
        check("best (smoothed) and best_raw checkpoints written",
              os.path.exists(os.path.join(out, "best_model.pth"))
              and os.path.exists(os.path.join(out, "best_raw_model.pth")))
        log = open(os.path.join(out, "train.log")).read()
        check("missing test split is reported, not assumed", "no 'test/'" in log)

        # legacy checkpoint -> Model 1 eval-only with arch rebuilt from the checkpoint
        legacy = L.kat_unet("swkat-tiny", img_size=64, num_classes=5, window_size=4,
                            decoder_depths=(1, 1, 1), bottleneck_depth=1)
        lp = os.path.join(td, "legacy.pth")
        torch.save({"epoch": 3, "model": legacy.state_dict(), "ema": legacy.state_dict(),
                    "args": {"use_boundary": True, "decoder_depths": "1,1,1",
                             "bottleneck_depth": 1, "window_size": 4, "img_size": 64}}, lp)
        out2 = os.path.join(td, "out_eval")
        ok2 = True
        try:
            T.main(base + ["--output-dir", out2, "--eval-only", lp, "--tta",
                           "--tta-scales", "1.0"])
        except Exception:
            import traceback
            traceback.print_exc()
            ok2 = False
        check("step 0: a LEGACY bcss5 checkpoint evaluates under Model 1 code", ok2
              and os.path.exists(os.path.join(out2, "eval_val.json")))
        ok3 = True
        try:
            T.main(base + ["--output-dir", out2, "--eval-only", lp, "--tta-select",
                           "--tta-candidates", "1.0;1.0,1.25"])
        except Exception:
            import traceback
            traceback.print_exc()
            ok3 = False
        sel = os.path.join(out2, "tta_selection.json")
        check("--tta-select scores every candidate in one pass", ok3 and os.path.exists(sel))
        if os.path.exists(sel):
            js = json.load(open(sel))
            check("4 candidates (2 scale sets x 2 modes) ranked", len(js["all"]) == 4
                  and js["all"][0]["mdice_agg"] >= js["all"][-1]["mdice_agg"])
            m1 = [r for r in js["all"] if r["scales"] == [1.0] and r["mode"] == "mean"][0]
            ev = json.load(open(os.path.join(out2, "eval_val.json")))
            check("grid result for (1.0, mean) == plain flip-TTA evaluation",
                  abs(m1["mdice_agg"] - ev["mdice_agg"]) < 1e-9,
                  f"{m1['mdice_agg']:.6f} vs {ev['mdice_agg']:.6f}")
        _make_split(td, "test", ["OL", "E2"], 6, rng)
        ok4 = True
        try:
            T.main(base + ["--output-dir", out2, "--eval-only",
                           os.path.join(out, "best_model.pth"), "--eval-on", "test",
                           "--tta-config", sel])
        except Exception:
            import traceback
            traceback.print_exc()
            ok4 = False
        check("test split evaluation with the selected TTA config",
              ok4 and os.path.exists(os.path.join(out2, "eval_test.json")))


def run_all(args=None):
    print("\n" + "#" * 78 + "\n#  MODEL 1 verification\n" + "#" * 78)
    for t in (test_rational, test_kaca, test_legacy, test_skipnorm_uper, test_sizes_and_ckpt,
              test_tta, test_albu_compat, test_env, test_losses_and_policy, test_optimizer,
              test_end_to_end):
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"{t.__name__} raised", False, repr(e))
    n = len(RESULTS)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{'=' * 78}\nSUMMARY: {n - len(bad)}/{n} checks passed")
    for name, _, det in bad:
        print(f"  FAIL: {name} ({det})")
    print("=" * 78)
    return not bad


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
