"""The demo's solves.  Every mode keeps the solver state (a Session), so it can be continued or extended with new photos
(solved with the scene, or against the scene held fixed), and records snapshots of the state for the viewer to play the
solve back."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch

from solv3r import (PRESETS, Solv3R, StreamingSolver, cameras_from_pose_enc, cameras_from_prediction,
                    load_images, recover_shared_intrinsics)
from solv3r.export import write_ply
from solv3r.images import _ffmpeg, model_size
from solv3r.model import Context, autocast
from solv3r.network.block import _attn_kv, _attn_with_cache

from solv3r.viewer.pack import frame_of, pack_solve, pack_stream, solve_export, stream_export

DEV = "cuda" if torch.cuda.is_available() or os.environ.get("SPACES_ZERO_GPU") else "cpu"
SNAPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
OUT = os.environ.get("SOLV3R_DEMO_TMP", os.path.join(tempfile.gettempdir(), "solv3r_demo"))
os.makedirs(OUT, exist_ok=True)

_model = None
CACHE = os.environ.get("SOLV3R_DEMO_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "solv3r"))


def sky_of(images, grow=6):
    """(V, H, W) bool, True on sky (the sky segmentation of solv3r.export, model cached in CACHE), grown by `grow`
    pixels so that the band along the horizon goes too."""
    import cv2
    import solv3r.network.skyseg as S
    from solv3r.export import sky_mask
    os.makedirs(CACHE, exist_ok=True)
    S.SKYSEG_ONNX_PATH = os.environ.get("SOLV3R_SKYSEG", os.path.join(CACHE, "skyseg.onnx"))
    images = images.float() / 255 if images.dtype == torch.uint8 else images.float()
    m = np.concatenate([sky_mask(images[a:a + 16]) for a in range(0, len(images), 16)])
    if grow:
        k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
        m = np.stack([cv2.dilate(x.astype(np.uint8), k) > 0 for x in m])
    return m


def model():
    global _model
    if _model is None:
        _model = Solv3R.from_pretrained(os.environ.get("SOLV3R_CHECKPOINT"), device=DEV)
    return _model


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def new_dir(tag):
    d = tempfile.mkdtemp(prefix=f"{tag}_", dir=OUT)
    return d


# ---------------------------------------------------------------------- inputs
def video_duration(path):
    r = subprocess.run([_ffmpeg(), "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", r.stderr)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None


def video_info(path):
    r = subprocess.run([_ffmpeg(), "-i", path], capture_output=True, text=True)
    dur = video_duration(path)
    fps = re.search(r"(\d+(?:\.\d+)?) fps", r.stderr)
    size = re.search(r"Video: [^\n]*?(\d{2,5})x(\d{2,5})", r.stderr)
    return dict(duration=dur, fps=float(fps[1]) if fps else None, size=(int(size[1]), int(size[2])) if size else None)


def extract_frames(video, n, long_side=1008, max_secs=None):
    """n frames spread evenly over the video (or its first max_secs), as .jpg files (long side at most long_side)."""
    d = new_dir("frames")
    dur = video_duration(video)                       # unknown for browser recordings (webm): take every frame
    if dur and max_secs:
        dur = min(dur, max_secs)
    scale = f"scale='if(gt(iw,ih),min({long_side},iw),-2)':'if(gt(iw,ih),-2,min({long_side},ih))'"
    vf = f"fps={max(n / max(dur, 1e-3), 0.05):.5f},{scale}" if dur else scale
    cut = ["-t", str(max_secs)] if max_secs else []
    subprocess.run([_ffmpeg(), "-loglevel", "error", "-i", video, *cut, "-vf", vf, "-q:v", "2",
                    os.path.join(d, "%05d.jpg")], check=True)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))
    if len(paths) > n:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, n).round().astype(int)]
    return paths


def depth_edges(d, thr=0.04):
    """(S, H, W) depth -> bool mask of depth discontinuities (relative central difference above thr): their pixels
    land between the surfaces ("flying pixels")."""
    g = torch.zeros_like(d)
    gx = (d[:, :, 2:] - d[:, :, :-2]).abs() / 2
    gy = (d[:, 2:, :] - d[:, :-2, :]).abs() / 2
    g[:, :, 1:-1] = gx
    g[:, 1:-1, :] = torch.maximum(g[:, 1:-1, :], gy)
    return g / d.clamp_min(1e-6) > thr


def to_u8(images):
    return images if images.dtype == torch.uint8 else (images.clamp(0, 1) * 255).round().byte()


def thumbs_of(images, h=96):
    """(S, 3, H, W) in [0, 1] -> (S, h, w, 3) uint8."""
    S, _, H, W = images.shape
    w = max(1, round(h * W / H))
    images = images.float() / 255 if images.dtype == torch.uint8 else images.float()
    t = torch.nn.functional.interpolate(images, size=(h, w), mode="area")
    return (t.permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().numpy()


# ---------------------------------------------------------------------- the kept solve
@dataclass
class Session:
    """Everything a solve leaves behind, on the CPU.  Views are in arrival order (row i of the state)."""
    images: torch.Tensor                  # (V, 3, H, W) uint8
    labels: list
    paths: list
    x0: torch.Tensor | None = None        # (V, P, C) encodings
    pos: torch.Tensor | None = None
    z: torch.Tensor | None = None         # (V, P, C) state
    first_call: list = field(default_factory=list)   # call at which each view joined (0 = from the start)
    calls: int = 0                        # calls made so far
    residual: list = field(default_factory=list)     # (call, relative change) of every call
    ms_per_call: list = field(default_factory=list)
    kind: str = "solve"                   # "solve" | "stream"
    final: dict | None = None             # last decode: c2w, K, depth, conf, world points (numpy)
    stream: dict | None = None            # streaming result
    frame: tuple | None = None            # viewer frame (centre, radius, rotation, aerial), kept across updates
    log: list = field(default_factory=list)
    peak_gb: float = 0.0
    gpu: str = ""                                   # the device that ran it (ZeroGPU assigns one per run)
    mode: str = ""
    sky: bool = False                     # leave the sky out of the points (outdoor scenes)
    source: str = ""                      # what was solved (an example's name)
    skymask: np.ndarray | None = None     # (V, H, W) True on sky, computed once
    view_keep: float = 1.0                # share of each view's points the viewer shows (its point budget): exports keep it
    hist: object = None                   # (Recorder, first_snap) of the last solve shown: a continuation plays after it
    shared_intrinsics: bool = True        # complete image collections; streaming keeps its original camera protocol

    @property
    def V(self):
        return len(self.labels)

    @property
    def imf(self):
        return self.images.float() / 255

    @property
    def hw(self):
        return tuple(self.images.shape[-2:])

    def ctx(self):
        return Context(self.x0.to(DEV), self.pos.to(DEV), *self.hw)


# ---------------------------------------------------------------------- recording snapshots
class Recorder:
    """Snapshots of the state for the viewer: dense points of a pixel subset of every view at some calls, cameras at
    every call (for the camera motion colour)."""

    def __init__(self, images, budget=2_000_000):
        V, _, H, W = images.shape
        self.H, self.W = H, W
        s = 1
        while V * (H // s) * (W // s) > budget and s < 8:
            s += 1
        self.stride = s
        ys, xs = np.mgrid[s // 2:H:s, s // 2:W:s]
        self.ys, self.xs = ys.ravel(), xs.ravel()
        self.images = images
        self.snaps = []                   # dict(call, P (V, n, 3) with NaN for absent views, conf, depth)
        self.cams = {}                    # call -> (V_present, 4, 4)

    def cameras(self, m, z, ctx, call):
        with torch.no_grad():
            c2w, K = cameras_from_pose_enc(m.decode_cameras(z, ctx), (ctx.H, ctx.W))
        self.cams[call] = c2w.numpy()
        self.K = K.numpy()
        return self.cams[call]

    def dense(self, m, z, ctx, call, chunk=16):
        V = z.shape[0]
        n = len(self.ys)
        P = np.full((V, n, 3), np.nan, np.float32)
        conf = np.zeros((V, n), np.float32)
        depth = np.zeros((V, n), np.float32)
        edge = np.zeros((V, n), bool)
        yi, xi = torch.as_tensor(self.ys), torch.as_tensor(self.xs)
        with torch.no_grad():
            for a in range(0, V, chunk):
                idx = list(range(a, min(V, a + chunk)))
                pred = m.decode(z[idx], ctx.subset(idx))
                wp = pred["world_points"].float()[:, yi, xi].cpu().numpy()
                P[a:a + len(idx)] = wp
                conf[a:a + len(idx)] = pred["depth_conf"].float()[:, yi, xi].cpu().numpy()
                d = pred["depth"][..., 0].float()
                depth[a:a + len(idx)] = d[:, yi, xi].cpu().numpy()
                edge[a:a + len(idx)] = depth_edges(d)[:, yi, xi].cpu().numpy()
        c2w = self.cameras(m, z, ctx, call)
        self.snaps.append(dict(call=call, P=P, conf=conf, depth=depth, c2w=c2w, K=self.K.copy(), edge=edge))

    def motion(self, call, V):
        """Rotation change (deg) of every view's camera over the last call."""
        a, b = self.cams.get(call - 1), self.cams.get(call)
        out = np.full(V, np.nan, np.float32)
        if a is None or b is None:
            return out
        n = min(len(a), len(b))
        Ra, Rb = a[:n, :3, :3], b[:n, :3, :3]
        Ra, Rb = Ra[0].T[None] @ Ra, Rb[0].T[None] @ Rb
        c = (np.einsum("nij,nij->n", Ra, Rb) - 1) / 2
        out[:n] = np.degrees(np.arccos(np.clip(c, -1, 1)))
        return out

    def pack(self, path, labels, first_snap, title="", frame=None, n_pts=None, sky=False, exposure=1.0, horizon=None,
             final=None, mv=None):
        V = len(labels)
        if n_pts is None:                           # about 40 MB of positions in all, at most a million points
            n_pts = int(np.clip(40e6 / (6 * len(self.snaps)), 200_000, 1_000_000))
        S = len(self.snaps)
        n = len(self.ys)
        P = np.full((S, V, n, 3), np.nan, np.float32)
        c2w = np.tile(np.eye(4, dtype=np.float32), (S, V, 1, 1))
        motion = np.full((S, V), np.nan, np.float32)
        for k, s in enumerate(self.snaps):
            v = s["P"].shape[0]
            P[k, :v] = s["P"]
            c2w[k, :v] = s["c2w"]
            for j in range(v, V):                  # views not yet present: their first camera (hidden by the viewer)
                c2w[k, j] = np.nan
            motion[k] = self.motion(s["call"], V)
        for j in range(V):                          # absent views take their first known camera
            known = [k for k in range(S) if np.isfinite(c2w[k, j]).all()]
            for k in range(S):
                if not np.isfinite(c2w[k, j]).all():
                    c2w[k, j] = c2w[known[0], j] if known else np.eye(4)
        last = self.snaps[-1]
        K = last["K"]
        if final is not None:
            c2w[-1] = final["c2w"]
            K = final["K"]
            motion[-1] = np.nan       # changing camera decoders is not a solver update
        vid = np.repeat(np.arange(V), n)
        img = to_u8(self.images).permute(0, 2, 3, 1).numpy()
        rgb = lift(img[:, self.ys, self.xs].reshape(-1, 3), exposure)
        conf = last["conf"].reshape(-1).astype(np.float32)   # raw: ranked over the whole scene in pack_solve
        foot = (last["depth"] / K[:, 0, 0, None] * self.stride * 1.1).reshape(-1)
        calls = [s["call"] for s in self.snaps]
        P = P.reshape(S, V * n, 3)
        keep = ~last.get("edge", np.zeros((V, n), bool)).reshape(-1)    # depth edges (flying pixels) leave
        if sky:                                     # and the sky
            keep &= ~sky_of(self.images)[:, self.ys, self.xs].reshape(-1)
        if horizon is not None:                     # aerial examples: nothing seen near or above the horizon
            keep &= below_horizon(P[-1], c2w[-1][vid, :3, 3], horizon)
        if mv and final is not None:                # floaters leave: a point stays when enough other views see its depth
            keep &= agreeing_views(P[-1], vid, final) >= mv
        P, vid, rgb, foot, conf = P[:, keep], vid[keep], rgb[keep], foot[keep], conf[keep]
        meta = pack_solve(path, P, vid, rgb, foot, conf, c2w, calls, first_snap, K,
                          (self.H, self.W), thumbs_of(self.images), labels, motion=motion, n_pts=n_pts, title=title,
                          frame=frame)
        return meta


def snap_calls(total, start=0):
    """Calls after `start` at which to keep dense snapshots, up to `total` (always the last)."""
    out = [start + c for c in SNAPS if c < total - start]
    if total not in out:
        out.append(total)
    return out


def finish(sess, m, z, ctx, keep_final=True):
    """Final decode kept for the exports and the views panel."""
    sess.z = z.float().cpu()
    if not keep_final:
        return
    with torch.no_grad():
        depth, conf, wp, cameras, intrinsics = [], [], [], [], []
        for a in range(0, z.shape[0], 16):
            idx = list(range(a, min(z.shape[0], a + 16)))
            pred = m.decode(z[idx], ctx.subset(idx))
            c2w, K, camera_info = cameras_from_prediction(pred, shared_intrinsics=False)
            cameras.append(c2w.numpy())
            intrinsics.append(K.numpy())
            depth.append(pred["depth"][..., 0].float().cpu().numpy())
            conf.append(pred["depth_conf"].float().cpu().numpy())
            wp.append(pred["world_points"].float().cpu().numpy())
    c2w, K = np.concatenate(cameras), np.concatenate(intrinsics)
    depth, conf, wp = np.concatenate(depth), np.concatenate(conf), np.concatenate(wp)
    if sess.shared_intrinsics:
        K, camera_info = recover_shared_intrinsics(wp, c2w, depth, conf)
        camera_info["pose_source"] = "equal_weight_ray_fitting"
    sess.final = dict(c2w=c2w, K=K, depth=depth.astype(np.float16), conf=conf.astype(np.float16),
                      points=wp.astype(np.float16), camera_metadata=camera_info)


def _peak(sess):
    if DEV == "cuda":
        sess.peak_gb = max(sess.peak_gb, torch.cuda.max_memory_allocated() / 2 ** 30)
        sess.gpu = torch.cuda.get_device_name(0).replace("NVIDIA ", "")


# ---------------------------------------------------------------------- calls with bookkeeping
def run_calls(sess, m, z, ctx, n, rec=None, dense_at=(), adaptive=None, min_calls=4, progress=None, offset=0):
    """n calls from z (or fewer with adaptive), recording cameras at every call and dense snapshots at dense_at."""
    done = 0
    for i in range(n):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            zn = m.solve(ctx, 1, z)
        sync()
        sess.ms_per_call.append((time.perf_counter() - t0) * 1000)
        change = float((zn.float() - z.float()).norm() / z.float().norm().clamp_min(1e-9))
        z = zn
        done += 1
        sess.calls += 1
        sess.residual.append((sess.calls, change))
        if rec is not None:
            if sess.calls in dense_at:
                rec.dense(m, z, ctx, sess.calls)
            else:
                rec.cameras(m, z, ctx, sess.calls)
        if progress:
            progress((offset + done) / max(offset + n, 1), f"call {sess.calls}")
        if adaptive and done >= min_calls and change < adaptive:
            break
    if rec is not None and (not rec.snaps or rec.snaps[-1]["call"] != sess.calls):
        rec.dense(m, z, ctx, sess.calls)
    return z


# ---------------------------------------------------------------------- modes
def solve_offline(images, labels, paths, calls=16, adaptive=None, max_calls=64, progress=None, shared_intrinsics=True):
    m = model()
    sess = Session(images=to_u8(images), labels=labels, paths=paths, mode="offline", shared_intrinsics=shared_intrinsics)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ctx = m.encode(images)
    sess.x0, sess.pos = ctx.x0.cpu(), ctx.pos.cpu()
    sess.first_call = [0] * len(labels)
    z = m.start(ctx)
    rec = Recorder(images)
    rec.cameras(m, z, ctx, 0)
    total = max_calls if adaptive else calls          # adaptive stopping decides when to stop, up to max_calls
    z = run_calls(sess, m, z, ctx, total, rec, set(snap_calls(total)), adaptive, progress=progress)
    finish(sess, m, z, ctx)
    _peak(sess)
    sess.log.append(f"Offline: {sess.V} views, {sess.calls} calls, {(time.perf_counter() - t0):.1f} s")
    return sess, rec


def stages_of(n, initial, group, order="file", seed=0):
    """Stages of view indices: the first `initial`, then groups; order 'file' or 'random' (any order)."""
    idx = np.arange(n)
    if order == "random":
        rng = np.random.default_rng(seed)
        first = np.linspace(0, n - 1, min(initial, n)).round().astype(int)       # spread over the capture
        rest = rng.permutation(np.setdiff1d(idx, first))
        idx = np.concatenate([first, rest])
    stages = [idx[:initial].tolist()]
    for a in range(initial, n, group):
        stages.append(idx[a:a + group].tolist())
    return stages


def solve_stages(images, labels, paths, stages, calls=8, progress=None, mode="incremental", shared_intrinsics=True):
    """Views join in stages; stage i gets calls[i] calls (or `calls` each), which also update the views solved before."""
    cs = list(calls) if isinstance(calls, (list, tuple)) else [calls] * len(stages)
    m = model()
    order = [i for s in stages for i in s]
    images, labels, paths = images[order], [labels[i] for i in order], [paths[i] for i in order]
    sess = Session(images=to_u8(images), labels=labels, paths=paths, mode=mode, shared_intrinsics=shared_intrinsics)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    rec = Recorder(images)
    first_snap, n_done = [], 0
    total = sum(cs)
    for i, st in enumerate(stages):
        rows = list(range(n_done, n_done + len(st)))
        if i == 0:
            ctx = m.encode(images[rows])
            z = m.start(ctx)
            rec.cameras(m, z, ctx, 0)
        else:
            z, ctx = m.add_views(z, ctx, images[rows])
        sess.first_call += [sess.calls] * len(st)
        first_snap += [len(rec.snaps)] * len(st)
        n_done += len(st)
        dense = {sess.calls + 1, sess.calls + cs[i]}
        z = run_calls(sess, m, z, ctx, cs[i], rec, dense, progress=progress, offset=sum(cs[:i]))
        if progress:
            progress(sum(cs[:i + 1]) / total, f"{n_done} of {len(order)} views")
    sess.x0, sess.pos = ctx.x0.cpu(), ctx.pos.cpu()
    finish(sess, m, z, ctx)
    _peak(sess)
    sess.log.append(f"{mode.capitalize()}: {sess.V} views in {len(stages)} stages, {sess.calls} calls, "
                    f"{(time.perf_counter() - t0):.1f} s")
    return sess, rec, first_snap


HIST_BYTES = 600e6                        # snapshots kept for a continuation to play after, at most
HIST_SNAPS = 2                            # of which a continuation's timeline shows about this many (each costs points)


def with_history(hist, rec, first):
    """A continuation's snapshots after those of the solve it continues (hist: its Recorder and first_snap), so its
    timeline shows the whole solve (views that join appear at their call).  rec's first snapshot is the state the kept
    solve ended in: it is dropped."""
    prev, pfirst = hist
    size = sum(v.nbytes for s in prev.snaps + rec.snaps for v in s.values() if isinstance(v, np.ndarray))
    if size > HIST_BYTES or (prev.H, prev.W) != (rec.H, rec.W):
        return rec, first
    grid = {(y, x): i for i, (y, x) in enumerate(zip(prev.ys.tolist(), prev.xs.tolist()))}
    idx = [grid.get(p) for p in zip(rec.ys.tolist(), rec.xs.tolist())]   # rec's pixels on prev's grid (a coarser
    if any(i is None for i in idx):                                       # stride: a subset of a finer one)
        return rec, first
    idx = np.array(idx)
    S = len(prev.snaps)                   # a few of them (the snapshots cost the browser points): ends and joins kept
    keep = sorted({0, S - 1, *pfirst, *np.linspace(0, S - 1, HIST_SNAPS).round().astype(int).tolist()})
    at = {j: i for i, j in enumerate(keep)}
    k, n_old = len(keep), len(pfirst)
    pick = lambda d: {k: (v[:, idx] if isinstance(v, np.ndarray) and v.ndim >= 2 and k != "c2w" and k != "K" else v)
                      for k, v in d.items()}
    rec.snaps = [pick(prev.snaps[j]) for j in keep] + rec.snaps[1:]
    rec.cams = {**prev.cams, **rec.cams}
    return rec, [at[f] for f in pfirst] + [k - 1 + f for f in first[n_old:]]


def continue_solve(sess, calls, progress=None):
    """More calls on the kept state (all views)."""
    m = model()
    ctx = sess.ctx()
    z = sess.z.to(DEV)
    rec = Recorder(sess.imf)
    start = sess.calls
    rec.cameras(m, z, ctx, start)
    rec.dense(m, z, ctx, start)
    dense = set(snap_calls(start + calls, start))
    z = run_calls(sess, m, z, ctx, calls, rec, dense, progress=progress)
    finish(sess, m, z, ctx)
    _peak(sess)
    sess.log.append(f"Continued: +{calls} calls (now {sess.calls})")
    return sess, rec


def add_photos(sess, images, labels, paths, calls=8, progress=None):
    """New views join the kept solve; all views are updated."""
    m = model()
    images = load_resized(images, sess.hw)
    ctx = sess.ctx()
    z = sess.z.to(DEV)
    rec = Recorder(torch.cat([sess.imf, images]))
    start = sess.calls
    rec.cameras(m, z, ctx, start)
    rec.dense(m, z, ctx, start)
    z, ctx = m.add_views(z, ctx, images)
    n_old = sess.V
    sess.images = torch.cat([sess.images, to_u8(images)])
    sess.labels += labels
    sess.paths += paths
    sess.first_call += [start] * len(labels)
    sess.x0, sess.pos = ctx.x0.cpu(), ctx.pos.cpu()
    dense = set(snap_calls(start + calls, start))
    z = run_calls(sess, m, z, ctx, calls, rec, dense, progress=progress)
    finish(sess, m, z, ctx)
    _peak(sess)
    first_snap = [0] * n_old + [1] * len(labels)
    sess.log.append(f"Added {len(labels)} photos at call {start}, +{calls} calls")
    return sess, rec, first_snap


def load_resized(images, hw):
    if tuple(images.shape[-2:]) == tuple(hw):
        return images
    from solv3r.images import resize
    return resize(images, hw)


def add_fixed(sess, images, labels, paths, calls=16, progress=None):
    """New views join with the scene held fixed: the kept views enter every call through one key / value cache and only
    the new views are solved, together.  They then belong to the scene like any other view."""
    m = model()
    images = load_resized(images, sess.hw)
    ctx = sess.ctx()
    z = sess.z.to(DEV)
    rec = Recorder(torch.cat([sess.imf, images]))
    start, n_old = sess.calls, sess.V
    rec.cameras(m, z, ctx, start)
    rec.dense(m, z, ctx, start)
    sync()
    t0 = time.perf_counter()
    with torch.no_grad(), autocast():
        cache = m.cache(z, ctx)
    sync()
    t_cache = time.perf_counter() - t0
    new = m.encode(images, reference_first=False)
    zq = m.start(new)
    ctx_all = Context.cat([ctx, new])
    dense = set(snap_calls(start + calls, start))
    ms = []
    for i in range(calls):
        sync()
        t1 = time.perf_counter()
        with torch.no_grad(), autocast():
            zn = m.step_cached(zq, new, cache)
        sync()
        ms.append((time.perf_counter() - t1) * 1000)
        change = float((zn.float() - zq.float()).norm() / zq.float().norm().clamp_min(1e-9))
        zq = zn
        sess.calls += 1
        sess.residual.append((sess.calls, change))
        z_all = torch.cat([z, zq.to(z.dtype)])
        if sess.calls in dense:
            rec.dense(m, z_all, ctx_all, sess.calls)
        else:
            rec.cameras(m, z_all, ctx_all, sess.calls)
        if progress:
            progress((i + 1) / calls, f"call {sess.calls}")
    z_all = torch.cat([z, zq.to(z.dtype)])
    if not rec.snaps or rec.snaps[-1]["call"] != sess.calls:
        rec.dense(m, z_all, ctx_all, sess.calls)
    sess.images = torch.cat([sess.images, to_u8(images)])
    sess.labels += labels
    sess.paths += paths
    sess.first_call += [start] * len(labels)
    sess.x0, sess.pos = ctx_all.x0.cpu(), ctx_all.pos.cpu()
    finish(sess, m, z_all, ctx_all)
    _peak(sess)
    first_snap = [0] * n_old + [1] * len(labels)
    sess.log.append(f"Added {len(labels)} photos with the scene frozen: one cache of {n_old} views ({t_cache * 1000:.0f} ms), "
                    f"{calls} calls on the new views ({np.median(ms):.1f} ms per call)")
    return sess, rec, first_snap


def solve_stream(paths, labels, config="L8", progress=None, keep=0.5, per_frame=2500):
    m = model()
    size = load_images(paths[:1]).shape[-2:]
    cfg = PRESETS[config]
    import dataclasses
    cfg = dataclasses.replace(cfg, memory_stride=min(cfg.memory_stride, max(1, -(-len(paths) // cfg.memory))))
    solver = StreamingSolver(m, cfg, depth=True)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    frames, live = [], []
    rng = np.random.default_rng(0)
    for i, p in enumerate(paths):
        im = load_images([p], size)[0]
        frames.append(to_u8(im))
        for f in solver.push(im):                    # each frame answered now: its camera and a few of its points
            live.append(live_frame(solver, f, frames[f], rng))
        if progress and (i % 2 == 0 or i == len(paths) - 1):
            progress((i + 1) / len(paths), f"frame {i + 1} of {len(paths)} answered", live)
            live = []
    res = solver.result()
    images = torch.stack(frames)
    sess = Session(images=images, labels=labels, paths=paths, kind="stream", mode="streaming", shared_intrinsics=False)
    sess.calls = len(paths)
    sess.stream = dict(c2w=res.c2w, c2w_exit=res.c2w_exit, K=res.intrinsics, depth=res.depth, conf=res.depth_conf,
                       ms=res.ms_per_frame, refreshed=list(res.refreshed), config=config)
    sess.final = dict(c2w=res.c2w_exit, K=res.intrinsics, depth=res.depth.astype(np.float16),
                      conf=res.depth_conf.astype(np.float16), points=None)
    _peak(sess)
    sess.log.append(f"Streaming ({config}): {len(paths)} frames, {res.ms_per_frame:.1f} ms per frame (solver), "
                    f"memory refreshed at {sum(res.refreshed)} frames")
    return sess


def live_frame(solver, f, image, rng, n=1500, stride=3):
    """What the live view draws for a frame on arrival: camera centre, forward and up (world), intrinsics (fractions of
    the image), n of its confident points (float16 xyz + uint8 rgb + float16 footprint, base64) and a small JPEG.  The
    points and their footprints are chosen as in pack_stream_session, so the live cloud looks like the final one."""
    import base64

    import cv2
    R, C, K = solver.R[f].float().numpy(), solver.C[f].float().numpy(), solver.K[f].float().numpy()
    img = image.permute(1, 2, 0).numpy()
    H, W = img.shape[:2]
    out = [int(f)] + np.round(np.concatenate([C, R[:, 2], -R[:, 1]]), 5).tolist()
    out += [round(float(K[0, 0] / W), 5), round(float(K[1, 1] / H), 5), round(float(K[0, 2] / W), 5), round(float(K[1, 2] / H), 5)]
    pts, rgb, foot = b"", b"", b""
    if f in solver.depths:
        d, c = solver.depths[f], solver.confs[f]
        h, w = d.shape
        ys, xs = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
        ys, xs = ys.ravel(), xs.ravel()
        z, cf = d[ys, xs], c[ys, xs]
        ok = np.isfinite(z) & (z > 0) & ~depth_edges(torch.as_tensor(d[None]).float())[0].numpy()[ys, xs]
        if ok.any():
            ok &= cf >= np.quantile(cf[ok], 0.5)
        idx = np.nonzero(ok)[0]
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        if len(idx):
            Kd = K * np.array([[w / W], [h / H], [1]])          # intrinsics at the depth resolution
            zz = z[idx]
            X = np.stack([(xs[idx] + 0.5 - Kd[0, 2]) / Kd[0, 0] * zz, (ys[idx] + 0.5 - Kd[1, 2]) / Kd[1, 1] * zz, zz], 1)
            pts = (X @ R.T + C).astype(np.float16).tobytes()
            rgb = img[(ys[idx] * H) // h, (xs[idx] * W) // w].astype(np.uint8).tobytes()
            dens = min(1.4, np.sqrt(np.count_nonzero(ok) / len(idx)))
            foot = (zz / Kd[0, 0] * stride * 1.1 * dens).astype(np.float16).tobytes()
    th = cv2.resize(img, (96, max(1, round(96 * H / W))), interpolation=cv2.INTER_AREA)
    ok_, jpg = cv2.imencode(".jpg", cv2.cvtColor(th, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = lambda b: base64.b64encode(b).decode()
    return out + [b64(pts), b64(rgb), b64(foot), b64(jpg.tobytes())]


def consistent(P, f, S, tol=0.04):
    """Points of frame f that agree with the depth of frames f +- 2 and f +- 4 (exit states) where those frames see them."""
    F_, H, W = S["depth"].shape
    ok, seen = np.zeros(len(P), int), np.zeros(len(P), int)
    for g in (f - 4, f - 2, f + 2, f + 4):
        if g < 0 or g >= F_:
            continue
        w2c = np.linalg.inv(S["c2w_exit"][g]); Q = P @ w2c[:3, :3].T + w2c[:3, 3]; zq = Q[:, 2]
        K = S["K"][g]
        u = K[0, 0] * Q[:, 0] / np.maximum(zq, 1e-6) + K[0, 2]; v = K[1, 1] * Q[:, 1] / np.maximum(zq, 1e-6) + K[1, 2]
        i = np.nonzero((zq > 1e-4) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1))[0]
        dz = np.abs(S["depth"][g][v[i].astype(int), u[i].astype(int)] - zq[i]) / zq[i]
        seen[i] += 1; ok[i] += dz < tol
    return (seen == 0) | (ok * 2 >= seen)


def agreeing_views(P, vid, final, tol=0.03):
    """For each point (N, 3) of view vid, the number of other views whose final depth map agrees with it (relative tol)."""
    D, K, W2C = final["depth"], final["K"], np.linalg.inv(final["c2w"].astype(np.float64))
    V, H, W = D.shape
    cnt = np.zeros(len(P), np.int16)
    fin = np.isfinite(P).all(1)
    for i in range(V):
        X = P @ W2C[i, :3, :3].T.astype(np.float32) + W2C[i, :3, 3].astype(np.float32)
        z = X[:, 2]
        with np.errstate(all="ignore"):
            u = K[i, 0, 0] * X[:, 0] / z + K[i, 0, 2]
            v = K[i, 1, 1] * X[:, 1] / z + K[i, 1, 2]
            idx = np.nonzero(fin & (vid != i) & (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H))[0]
        d = D[i, v[idx].astype(int), u[idx].astype(int)].astype(np.float32)
        cnt[idx] += np.abs(d - z[idx]) < tol * z[idx]
    return cnt


def below_horizon(P, C, deg=3.0):
    """Aerial scenes: points seen at least `deg` below the horizon from their camera (P points, C their cameras' centres).  Distant hills,
    city lights and the sky that the sky mask misses come back with a compressed depth and float above the map; they are seen near or
    above the horizon.  Up is the ground normal (smallest principal axis of the points, toward the cameras), not the cameras' -y, which
    the drone's pitch tilts (as the project page's demo videos)."""
    ok = np.isfinite(P).all(1)
    Q = P[ok][np.random.default_rng(0).choice(int(ok.sum()), min(int(ok.sum()), 200000), replace=False)].astype(np.float64)
    c = np.median(Q, 0)
    lo, hi = np.quantile(Q, 0.05, 0), np.quantile(Q, 0.95, 0)
    Q = Q[((Q >= lo) & (Q <= hi)).all(1)] - c
    up = np.linalg.eigh(Q.T @ Q / len(Q))[1][:, 0]
    up = up if (np.nanmean(C, 0) - c) @ up > 0 else -up
    ray = P - C
    el = np.degrees(np.arcsin(np.clip((ray @ up) / np.maximum(np.linalg.norm(ray, axis=1), 1e-9), -1, 1)))
    return ok & (el < -deg)


def lift(rgb, exposure):
    """Display brightness of point colours (night scenes): a gamma lift, dark tones raised most, nothing clipped."""
    if exposure == 1:
        return rgb
    return np.round(255 * (np.asarray(rgb, np.float32) / 255) ** (1 / exposure)).astype(np.uint8)


def facing(depth, K, ys, xs):
    """|cos| of the angle between each pixel's viewing ray and the surface it sees (normal from the depth map's neighbours):
    small where the frame sees the surface at a grazing angle, where its depth is least precise."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    X = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * depth, (v + 0.5 - K[1, 2]) / K[1, 1] * depth, depth], -1)
    du, dv = np.zeros_like(X), np.zeros_like(X)
    du[:, 1:-1] = X[:, 2:] - X[:, :-2]
    dv[1:-1] = X[2:] - X[:-2]
    n = np.cross(du, dv)[ys, xs]
    r = X[ys, xs]
    return np.abs((n * r).sum(1)) / np.maximum(np.linalg.norm(n, axis=1) * np.linalg.norm(r, axis=1), 1e-12)


def first_seen(P, fid, vox=0.006, slack=2):
    """Points in space that no earlier frame (more than `slack` frames before) already covered: a later, less precise view of
    the same surface does not blur what an earlier frame saw sharply; later frames only extend the map (voxels of `vox` x the
    scene radius)."""
    c = np.median(P, 0)
    r = float(np.quantile(np.linalg.norm(P - c, axis=1), 0.9))
    key = np.floor((P - c) / (vox * r)).astype(np.int64)
    key = (key[:, 0] * 73856093) ^ (key[:, 1] * 19349663) ^ (key[:, 2] * 83492791)
    order = np.argsort(fid, kind="stable")
    _, first_idx = np.unique(key[order], return_index=True)
    first = dict(zip(key[order][first_idx].tolist(), fid[order][first_idx].tolist()))
    return fid <= np.array([first[k] for k in key.tolist()]) + slack


def pack_stream_session(sess, path, keep=0.9, per_frame=8000, stride=3, sky=False, exposure=1.0, horizon=None, graze=None,
                        firstseen=False):   # the viewer's confidence slider does the rest
    S = sess.stream
    skym = sky_of(sess.images) if sky else None
    F_, H, W = S["depth"].shape
    ys, xs = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
    ys, xs = ys.ravel(), xs.ravel()
    img = sess.images.permute(0, 2, 3, 1).numpy()
    rng = np.random.default_rng(0)
    Pn, Pl, col, fid, foot, cf = [], [], [], [], [], []
    cand = 0
    for f in range(F_):
        d, c = S["depth"][f][ys, xs], S["conf"][f][ys, xs]
        ok = np.isfinite(d) & (d > 0) & ~depth_edges(torch.as_tensor(S["depth"][f][None]).float())[0].numpy()[ys, xs]
        if skym is not None:
            ok &= ~skym[f][ys, xs]
        thr = np.quantile(c[ok], 1 - keep) if ok.any() else 0
        K = S["K"][f]
        if graze is not None:                       # nothing seen at a grazing angle: it would blur what frontal frames saw sharply
            ok &= facing(S["depth"][f], K, ys, xs) > graze
        idx = np.nonzero(ok & (c >= thr))[0]
        z = d[idx]
        X = np.stack([(xs[idx] + 0.5 - K[0, 2]) / K[0, 0] * z, (ys[idx] + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
        good = consistent(X @ S["c2w_exit"][f][:3, :3].T + S["c2w_exit"][f][:3, 3], f, S)   # as the project page's streams
        idx, z, X = idx[good], z[good], X[good]
        cand += len(idx)
        if len(idx) > per_frame:
            sub = np.sort(rng.choice(len(idx), per_frame, replace=False))
            idx, z, X = idx[sub], z[sub], X[sub]
        for out, C in ((Pn, S["c2w"][f]), (Pl, S["c2w_exit"][f])):
            out.append(X @ C[:3, :3].T + C[:3, 3])
        col.append(img[f, ys[idx], xs[idx]])
        fid.append(np.full(len(idx), f))
        foot.append(z / K[0, 0] * stride * 1.1)
        cf.append(c[idx])
    cat = np.concatenate
    Pn, Pl, fid, col, foot, cf = cat(Pn), cat(Pl), cat(fid), cat(col), cat(foot), cat(cf)
    if firstseen:                                   # later frames only extend the map
        k = first_seen(Pl, fid)
        Pn, Pl, fid, col, foot, cf = Pn[k], Pl[k], fid[k], col[k], foot[k], cf[k]
    if horizon is not None:                         # aerial examples: nothing seen near or above the horizon
        k = below_horizon(Pl, S["c2w_exit"][fid][:, :3, 3], horizon)
        Pn, Pl, fid, col, foot, cf = Pn[k], Pl[k], fid[k], col[k], foot[k], cf[k]
    return pack_stream(path, Pn, Pl, fid, lift(col, exposure), foot, S["c2w"], S["c2w_exit"], S["K"],
                       sess.hw, thumbs_of(sess.images, 64), labels=sess.labels,
                       dens=min(2.0, np.sqrt(cand / max(len(fid), 1))), conf=cf)


# ---------------------------------------------------------------------- large scenes: a random subset of views per call
class Stochastic:
    """Each call updates b random views; the others join as constraints through their cached keys / values.

    Per view we keep the global-attention keys / values of its current state and the energy of its latest update
    (for the state scale), so a call costs b views of frame attention plus their attention over the cache.  Peak
    memory grows with the number of views only through the cache (about 2.7 MB per view)."""

    def __init__(self, m, ctx, chunk=16):
        self.m, self.ctx, self.blk = m, ctx, m.net.block
        self.N, self.P, self.C = ctx.x0.shape
        attn = self.blk.global_attn.attn
        self.Hh = attn.num_heads
        self.z = m.start(ctx).float()
        self.K = torch.empty(self.N, self.Hh, self.P, self.C // self.Hh, dtype=torch.bfloat16, device=DEV)
        self.Vv = torch.empty_like(self.K)
        tgt = float(self.blk.target_rms[3])
        self.e = torch.full((self.N,), tgt ** 2 * self.P * self.C, device=DEV)
        self.updates = torch.zeros(self.N, dtype=torch.long)
        for a in range(0, self.N, chunk):
            self._kv(list(range(a, min(self.N, a + chunk))))

    @torch.no_grad()
    def _kv(self, S):
        b = len(S)
        with autocast():
            y = self.blk._frame(self.z[S], self.ctx.x0[S], self.ctx.pos[S]).reshape(1, b * self.P, self.C)
            k, v = _attn_kv(self.blk.global_attn.attn, self.blk.global_attn.norm1(y))
        self.K[S] = k[0].reshape(self.Hh, b, self.P, -1).permute(1, 0, 2, 3).to(torch.bfloat16)
        self.Vv[S] = v[0].reshape(self.Hh, b, self.P, -1).permute(1, 0, 2, 3).to(torch.bfloat16)

    @torch.no_grad()
    def update(self, S):
        b = len(S)
        held = torch.ones(self.N, dtype=torch.bool, device=DEV)
        held[S] = False
        nh = int(held.sum())
        Kh = self.K[held].permute(1, 0, 2, 3).reshape(1, self.Hh, nh * self.P, -1)
        Vh = self.Vv[held].permute(1, 0, 2, 3).reshape(1, self.Hh, nh * self.P, -1)
        attn = self.blk.global_attn.attn
        with autocast():
            y = self.blk._frame(self.z[S], self.ctx.x0[S], self.ctx.pos[S]).reshape(1, b * self.P, self.C)
            y = self.blk.global_attn(y, self.blk.g_global,
                                     attn_fn=lambda x: _attn_with_cache(attn, x, Kh, Vh))
            y = y.reshape(b, self.P, self.C)
            eq = y.float().pow(2).sum((1, 2))
            rms = ((self.e[held].sum() + eq.sum()) / (self.N * self.P * self.C)).clamp_min(1e-12).sqrt()
            zq = self.blk._rescale(y, rms).float()
        self.e[S] = eq
        self.z[S] = zq
        self.updates[S] += 1
        self._kv(S)


def solve_stochastic(images, labels, paths, per_call=8, updates=32, seed=0, progress=None, shared_intrinsics=True):
    m = model()
    sess = Session(images=to_u8(images), labels=labels, paths=paths, mode="large", shared_intrinsics=shared_intrinsics)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ctx = m.encode(images)
    sess.x0, sess.pos = ctx.x0.cpu(), ctx.pos.cpu()
    sess.first_call = [0] * len(labels)
    N = len(labels)
    st = Stochastic(m, ctx)
    rec = Recorder(images)
    rec.cameras(m, st.z, ctx, 0)
    g = torch.Generator().manual_seed(seed)
    total = math.ceil(updates * N / per_call)
    per_epoch = max(1, N // per_call)
    dense_epochs = {1, 2, 4, 8, 16, 32, 64, updates}
    z_prev = st.z.clone()
    for c in range(1, total + 1):
        if (c - 1) % per_epoch == 0:
            torch.randperm(N, generator=g)
        S = torch.randperm(N, generator=g)[:per_call].tolist()
        sync()
        t1 = time.perf_counter()
        st.update(S)
        sync()
        sess.ms_per_call.append((time.perf_counter() - t1) * 1000)
        sess.calls = c
        if c % per_epoch == 0 or c == total:
            ep = c // per_epoch
            sess.residual.append((c, float((st.z - z_prev).norm() / z_prev.norm().clamp_min(1e-9))))   # per pass
            z_prev = st.z.clone()
            if ep in dense_epochs or c == total:
                rec.dense(m, st.z, ctx, c)
            else:
                rec.cameras(m, st.z, ctx, c)
            if progress:
                progress(c / total, f"{c} calls, {ep} updates per view")
    finish(sess, m, st.z, ctx)
    _peak(sess)
    sess.log.append(f"Large scene: {N} views, {per_call} per call, {total} calls (~{updates} updates per view), "
                    f"{(time.perf_counter() - t0):.1f} s, peak {sess.peak_gb:.2f} GB")
    return sess, rec


# ---------------------------------------------------------------------- exports
def confident(sess, keep=1.0):
    """The points the viewer shows, at every pixel (solv3r.viewer.pack), and their colours: an export is what one sees.
    keep: the viewer's confidence filtering (1: all)."""
    f = sess.final
    if f.get("points") is None:              # stream: the depth on arrival placed with the exit cameras
        S = sess.stream
        res = SimpleNamespace(c2w=S["c2w"], c2w_exit=S["c2w_exit"], intrinsics=S["K"], depth=S["depth"], depth_conf=S["conf"])
        return stream_export(sess.images, res, keep, sess.sky)
    return solve_export(f["points"], f["conf"], f["depth"], sess.images, sess.view_keep, keep, sess.sky)


def export_ply(sess, keep=1.0, max_points=8_000_000):
    xyz, rgb = confident(sess, keep)
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    path = os.path.join(new_dir("export"), "solv3r_points.ply")
    write_ply(path, xyz, rgb)
    return path


def export_npz(sess):
    f = sess.final
    path = os.path.join(new_dir("export"), "solv3r_cameras_depth.npz")
    np.savez_compressed(path, c2w=f["c2w"], K=f["K"], depth=f["depth"], depth_conf=f["conf"],
                        names=np.array(sess.labels), image_size=np.array(sess.hw),
                        **({"camera_metadata": np.array(json.dumps(f["camera_metadata"]))} if "camera_metadata" in f else {}))
    return path


def _qvec(R):
    """Rotation matrix -> COLMAP quaternion (w, x, y, z)."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return w, x, y, z


def export_colmap(sess, keep=1.0, max_points=300_000, with_images=True):
    """COLMAP text model (sparse/0: cameras, images, points3D) at the original image sizes, plus the images; for
    Gaussian splatting and other tools that start from a COLMAP model."""
    from PIL import Image

    f = sess.final
    d = new_dir("colmap")
    sp = os.path.join(d, "sparse", "0")
    os.makedirs(sp)
    H, W = sess.hw
    cams, ims = [], []
    camera_ids = {}
    shared = f.get("camera_metadata", {}).get("shared_intrinsics", False)
    names = []
    for i in range(sess.V):
        src = sess.paths[i]
        try:
            w0, h0 = Image.open(src).size if src and os.path.isfile(src) else (W, H)
        except OSError:
            w0, h0 = W, H
        K = f["K"][i]
        sx, sy = w0 / W, h0 / H
        name = f"{i:05d}_{os.path.basename(src) if src else sess.labels[i] + '.jpg'}"
        names.append(name)
        params = (K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy)
        key = (w0, h0, *params) if shared else (i,)
        if key not in camera_ids:
            camera_ids[key] = len(camera_ids) + 1
            cams.append(f"{camera_ids[key]} PINHOLE {w0} {h0} " + " ".join(f"{v:.6f}" for v in params))
        camera_id = camera_ids[key]
        w2c = np.linalg.inv(f["c2w"][i].astype(np.float64))
        q = _qvec(w2c[:3, :3])
        t = w2c[:3, 3]
        ims.append(f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {camera_id} {name}\n")
    open(os.path.join(sp, "cameras.txt"), "w").write("# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + "\n".join(cams) + "\n")
    open(os.path.join(sp, "images.txt"), "w").write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + "\n".join(ims))
    if "camera_metadata" in f:
        with open(os.path.join(d, "camera_metadata.json"), "w") as fh:
            json.dump(dict(f["camera_metadata"], colmap_camera_count=len(cams)), fh, indent=2)
    xyz, rgb = confident(sess, keep)
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    with open(os.path.join(sp, "points3D.txt"), "w") as fh:
        fh.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for j, (p, c) in enumerate(zip(xyz, rgb)):
            fh.write(f"{j + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0\n")
    if with_images:
        os.makedirs(os.path.join(d, "images"))
        for i, name in enumerate(names):
            src = sess.paths[i]
            if src and os.path.isfile(src):
                shutil.copy(src, os.path.join(d, "images", name))
            else:
                Image.fromarray(sess.images[i].permute(1, 2, 0).numpy()).save(os.path.join(d, "images", name))
    zp = os.path.join(new_dir("export"), "solv3r_colmap.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(d):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, d))
    shutil.rmtree(d, ignore_errors=True)
    return zp
