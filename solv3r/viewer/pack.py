"""Packing a solve for the browser viewer, in the format of the demo and the project page (moved here from the demo's
scene_pack.py and engine.py, so the examples and the demo write the same viewer).

A pack is b"S3R1", then three length-prefixed parts (uint32 little endian): the scene metadata (JSON), the point data
(gzip) and the view thumbnails (one JPEG atlas).

Solve (kind "solve"): snapshots of the solver state at given calls.  Every point keeps one pixel of one view; its
position is stored for every snapshot (int16, delta-coded), so the viewer interpolates between the saved states.
Views can join at any snapshot ("first"); points are sorted by it, so the points present at a snapshot are a prefix.
Stream (kind "stream"): every frame has two states, on arrival and when it left the window of recent frames.

Coordinates are rotated so that the scene's up is +y and the mean viewing direction is -z, in units of the scene
radius.  Points on depth edges ("flying pixels") are left out, and the sky too when asked.
"""
import gzip
import io
import json
import struct

import cv2
import numpy as np
import torch

from ..model import cameras_from_pose_enc

SNAPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)    # calls with a dense snapshot (and the last call)

# ---------------------------------------------------------------------- framing
def frame_of(points, c2w):
    """Centre, radius, rotation (rows: new axes) and whether the cameras look down on the scene (aerial).

    Up is the mean camera up, or for cameras that look steeply down, the normal of the dominant plane of the points
    (towards the cameras).  The mean viewing direction becomes -z."""
    pts = points[np.isfinite(points).all(1)]
    center = np.median(pts, 0)
    radius = float(np.quantile(np.linalg.norm(pts - center, axis=1), 0.9)) or 1.0
    Rc = c2w[:, :3, :3]
    up = -Rc[:, :, 1].mean(0)
    up /= np.linalg.norm(up) + 1e-12
    fwd = Rc[:, :, 2].mean(0)
    fwd /= np.linalg.norm(fwd) + 1e-12
    aerial = float(fwd.dot(-up)) > 0.6 or np.linalg.norm(Rc[:, :, 2].mean(0)) > 0.9 and np.linalg.norm(Rc[:, :, 1].mean(0)) < 0.5
    if aerial:                                        # plane normal of the points, on the side of the cameras
        sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 200000), replace=False)] - center
        n = np.linalg.svd(sub, full_matrices=False)[2][2]
        if n.dot(c2w[:, :3, 3].mean(0) - center) < 0:
            n = -n
        up = n
        fwd = Rc[:, :, 1].mean(0)                     # image "down" of the photos points forward when looking down
        if np.linalg.norm(fwd - fwd.dot(up) * up) < 1e-3:
            fwd = np.cross(up, [1.0, 0, 0])
    z = -(fwd - fwd.dot(up) * up)
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(up, z)
    return center, radius, np.stack([x, up, z]), bool(aerial)


def _view_dist(P, k):
    """Default viewing distance (scene units): the shown points, not all candidates, set the scale, so a long tail of
    far terrain in a few oblique photos does not put the camera inside the cloud."""
    d = np.linalg.norm(P[np.isfinite(P).all(1)], axis=1)
    r90 = float(np.quantile(d, 0.9)) if len(d) else 0.6
    return round(float(np.clip(k * r90, 1.5, 8.0)), 3)


def _conf_rank(conf, group):
    """Confidence rank of every point within its view, as uint8 (255: the most confident point of that view)."""
    out = np.zeros(len(conf), np.uint8)
    order = np.lexsort((conf, group))                 # by view, then confidence
    g = group[order]
    start = np.searchsorted(g, g, side="left")
    count = np.searchsorted(g, g, side="right") - start
    rank = (np.arange(len(order)) - start) / np.maximum(count - 1, 1)
    out[order] = np.round(rank * 255).astype(np.uint8)
    return out


def _quant(stack):
    stored = stack[np.isfinite(stack).all(1)]
    lo, hi = np.quantile(stored, 0.001, 0), np.quantile(stored, 0.999, 0)
    pad = (hi - lo) * 0.03 + 1e-6
    return lo - pad, hi + pad


def _q(P, lo, hi):
    return np.clip(np.round((P - lo) / (hi - lo) * 65535) - 32768, -32768, 32767).astype(np.int32)


def _planes(a16, buf):
    b = a16.astype(np.int16).view(np.uint8).reshape(-1, 2)
    buf.write(b[:, 0].tobytes())
    buf.write(b[:, 1].tobytes())


def _atlas(thumbs, tw=112, quality=84):
    V, H, W = thumbs.shape[:3]
    th = int(round(tw * H / W))
    cols = int(np.ceil(np.sqrt(V * th / tw)))
    rows = int(np.ceil(V / cols))
    atlas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for r in range(V):
        atlas[(r // cols) * th:(r // cols + 1) * th, (r % cols) * tw:(r % cols + 1) * tw] = cv2.resize(
            thumbs[r], (tw, th), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpg.tobytes(), dict(w=tw, h=th, cols=cols, rows=rows)


def _write(path, meta, raw, jpg):
    js = json.dumps(meta, separators=(",", ":")).encode()
    gz = gzip.compress(raw, 6)
    with open(path, "wb") as f:
        f.write(b"S3R1")
        for part in (js, gz, jpg):
            f.write(struct.pack("<I", len(part)))
            f.write(part)
    return len(js) + len(gz) + len(jpg) + 16


def _views(K, HW, labels, first):
    H, W = HW
    return [dict(cap=int(i), first=int(first[i]), stage=int(first[i] > 0) + 1, label=labels[i],
                 fx=round(float(K[i, 0, 0] / W), 5), fy=round(float(K[i, 1, 1] / H), 5),
                 cx=round(float(K[i, 0, 2] / W), 5), cy=round(float(K[i, 1, 2] / H), 5)) for i in range(len(K))]


# ---------------------------------------------------------------------- solve: snapshots
def pack_solve(path, P, vid, rgb, foot, conf, c2w, calls, first, K, HW, thumbs, labels, motion=None, n_pts=600000,
               boost=1.0, title="", frame=None, seed=0, far=2.5):
    """P (S, N, 3) world points of the candidate pixels at every snapshot; vid (N,) their view; rgb (N, 3) uint8;
    foot (N,) world size of a pixel; conf (N,) confidence at the last snapshot;
    c2w (S, V, 4, 4); calls (S,); first (V,) snapshot at which a view joins; K (V, 3, 3) at HW; thumbs (V, h, w, 3).
    motion (S, V): camera rotation change (deg per call) shown as the frustum colour.  Returns (meta, bytes)."""
    S, N = P.shape[:2]
    V = c2w.shape[1]
    first = np.asarray(first, int)
    rng = np.random.default_rng(seed)
    center, radius, R, aerial = frame or frame_of(P[-1], c2w[-1])
    P = ((P - center.astype(np.float32)) / np.float32(radius)) @ R.T.astype(np.float32)
    Rc = np.einsum("ij,svjk->svik", R, c2w[:, :, :3, :3])
    tc = np.einsum("ij,svj->svi", R, (c2w[:, :, :3, 3] - center) / radius)
    foot = foot / radius
    dist = np.linalg.norm(np.nan_to_num(P[-1], nan=1e9), axis=1)
    near = dist < far * max(1.0, float(np.quantile(dist[dist < 1e8], 0.9)))   # stray far points (sky, horizon) leave
    P, vid, rgb, foot, conf = P[:, near], vid[near], rgb[near], foot[near], conf[near]

    # points: per view, the most confident first, up to its share of n_pts
    cnt = np.bincount(vid, minlength=V)
    w = cnt * np.where(first == 0, boost, 1.0)
    quota = np.minimum(cnt, np.floor(w / max(w.sum(), 1) * n_pts)).astype(int)
    order_v = np.lexsort((np.arange(V), first))       # arrival order: joining snapshot, then capture index
    vrank = np.zeros(V, int)
    pts = []
    for r, v in enumerate(order_v):
        vrank[v] = r
        idx = np.nonzero(vid == v)[0]
        if quota[v] < len(idx):
            idx = idx[np.argsort(-conf[idx], kind="stable")[:quota[v]]]
        pts.append(np.sort(idx))
    pts = np.concatenate(pts)
    nP = len(pts)
    view_keep = nP / max(len(vid), 1)                 # the share the budget kept: exports keep the same (solve_export)
    cq = _conf_rank(conf[pts], np.zeros(nP, int))     # one ranking over the whole scene: hard views lose more
    view_dist = _view_dist(P[-1, pts][cq >= 128], 5.4 if aerial else 4.2)   # framed on the more confident half
    pf = first[vid[pts]]
    prefix = [int((pf <= k).sum()) for k in range(S)]
    lo, hi = _quant(np.concatenate([P[k, pts[:prefix[k]]] for k in range(S)]))
    raw = io.BytesIO()
    q_prev = None
    for k in range(S):
        Q = P[k, pts[:prefix[k]]].copy()
        bad = ~np.isfinite(Q).all(1)
        if bad.any():
            Q[bad] = 0
        q = _q(Q, lo, hi)
        out = q.copy()
        if q_prev is not None:
            out[:len(q_prev)] = q[:len(q_prev)] - q_prev
        _planes(out, raw)
        q_prev = q
    lf = np.log(np.maximum(foot[pts], 1e-6))
    f0, f1 = np.quantile(lf, 0.01), np.quantile(lf, 0.99)
    size = np.clip(np.round((lf - f0) / max(f1 - f0, 1e-6) * 255), 0, 255).astype(np.uint8)
    cams = np.zeros((S, V, 12), np.float32)
    for k in range(S):
        for v in range(V):
            cams[k, vrank[v], :9] = Rc[k, v].reshape(-1)
            cams[k, vrank[v], 9:] = tc[k, v]
    raw.write(rgb[pts].astype(np.uint8).tobytes())
    raw.write(vrank[vid[pts]].astype(np.uint16).tobytes())
    raw.write(size.tobytes())
    raw.write(cams.tobytes())
    raw.write(cq.tobytes())                           # 255 = the most confident shown point of the scene
    jpg, tmeta = _atlas(thumbs[order_v])
    if motion is None:
        E = [[None] * V for _ in range(S)]
    else:
        E = [[None if not np.isfinite(motion[k, v]) else round(float(motion[k, v]), 3) for v in order_v] for k in range(S)]
    views = _views(K[order_v], HW, [labels[v] for v in order_v], first[order_v])
    meta = dict(kind="solve", title=title, calls=[int(c) for c in calls], nPts=int(nP), nViews=int(V), prefix=prefix,
                dens=round(float(np.sqrt(cnt.sum() / max(nP, 1))), 4),
                quant=dict(lo=[round(float(x), 6) for x in lo], hi=[round(float(x), 6) for x in hi]),
                size=dict(f0=round(float(f0), 5), f1=round(float(f1), 5)), views=views, err=E,
                aspect=round(HW[1] / HW[0], 5), thumbs=tmeta, anchor=int(vrank[0]), aerial=aerial,
                camera=dict(yaw=0, elev=55 if aerial else 20, dist=view_dist), order=[int(v) for v in order_v], confq=1,
                frame=dict(center=[float(x) for x in center], radius=radius, R=R.tolist()), view_keep=round(view_keep, 4))
    meta["bytes"] = _write(path, meta, raw.getvalue(), jpg)
    return meta


# ---------------------------------------------------------------------- stream: on arrival and exit
def pack_stream(path, Pn, Pl, fid, rgb, foot, Cn, Cl, K, HW, thumbs, window=6, title="", labels=None, far=3.0, dens=1.0,
                conf=None):
    """Pn, Pl (N, 3) world points on arrival and at exit; fid (N,) frame; rgb (N, 3); foot (N,);
    Cn, Cl (F, 4, 4) cameras on arrival and at exit; K (F, 3, 3) at HW; thumbs (F, h, w, 3)."""
    F = len(Cn)
    center, radius, R, aerial = frame_of(Pl, Cl)
    tr = lambda X: ((X - center) / radius) @ R.T
    Pn, Pl = tr(Pn), tr(Pl)
    near = (np.linalg.norm(Pl, axis=1) < far) & np.isfinite(Pn).all(1) & np.isfinite(Pl).all(1)
    conf = np.ones(len(fid)) if conf is None else conf
    Pn, Pl, fid, rgb, foot, conf = Pn[near], Pl[near], fid[near], rgb[near], foot[near], conf[near]
    lo, hi = _quant(np.concatenate([Pn, Pl]))
    view_dist = _view_dist(Pl[_conf_rank(conf, np.zeros(len(fid), int)) >= 128], 4.6 if aerial else 5.0)
    lf = np.log(np.maximum(foot / radius, 1e-6))
    f0, f1 = np.quantile(lf, 0.01), np.quantile(lf, 0.99)
    size = np.clip(np.round((lf - f0) / max(f1 - f0, 1e-6) * 255), 0, 255).astype(np.uint8)

    def cams(C):
        out = np.zeros((F, 12), np.float32)
        out[:, :9] = np.einsum("ij,fjk->fik", R, C[:, :3, :3]).reshape(F, 9)
        out[:, 9:] = ((C[:, :3, 3] - center) / radius) @ R.T
        return out

    raw = io.BytesIO()
    for X in (Pn, Pl):
        _planes(_q(X, lo, hi), raw)
    raw.write(rgb.astype(np.uint8).tobytes())
    raw.write(fid.astype(np.uint16).tobytes())
    raw.write(size.tobytes())
    raw.write(cams(Cn).tobytes())
    raw.write(cams(Cl).tobytes())
    raw.write(_conf_rank(conf, np.zeros(len(fid), int)).tobytes())
    jpg, tmeta = _atlas(thumbs, tw=80, quality=82)
    labels = labels or [f"frame {f + 1}" for f in range(F)]
    views = _views(K, HW, labels, np.zeros(F, int))
    meta = dict(kind="stream", title=title, nPts=int(len(fid)), nViews=int(F), window=window, anchor=0, dens=round(float(dens), 4),
                quant=dict(lo=[round(float(x), 6) for x in lo], hi=[round(float(x), 6) for x in hi]),
                size=dict(f0=round(float(f0), 5), f1=round(float(f1), 5)), views=views,
                err=[[None] * F, [None] * F], aspect=round(HW[1] / HW[0], 5), thumbs=tmeta, aerial=aerial,
                camera=dict(yaw=35, elev=55 if aerial else 32, dist=view_dist), frustum=0.05, confq=1,
                frame=dict(center=[float(x) for x in center], radius=radius, R=R.tolist()))
    meta["bytes"] = _write(path, meta, raw.getvalue(), jpg)
    return meta


def sky_of(images, grow=6):
    """(V, H, W) bool, True on sky (solv3r.export.sky_mask), grown by `grow` pixels so that the band along the horizon
    goes too."""
    from ..export import sky_mask
    images = images.float() / 255 if images.dtype == torch.uint8 else images.float()
    m = np.concatenate([sky_mask(images[a:a + 16]) for a in range(0, len(images), 16)])
    if grow:
        k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
        m = np.stack([cv2.dilate(x.astype(np.uint8), k) > 0 for x in m])
    return m


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


# ---------------------------------------------------------------------- recording a solve
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

    def record(self, m, z, ctx, call, dense=None):
        """After a call: dense points at the snapshot calls (SNAPS, or `dense`), the cameras at every other call."""
        if call in (SNAPS if dense is None else dense):
            self.dense(m, z, ctx, call)
        else:
            self.cameras(m, z, ctx, call)

    def close(self, m, z, ctx, call):
        """The state the solve ended in is always a snapshot."""
        if not self.snaps or self.snaps[-1]["call"] != call:
            self.dense(m, z, ctx, call)

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
            motion[-1] = np.nan       # final camera decoding is not a solver update
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


# ---------------------------------------------------------------------- a stream (solv3r.StreamingSolver)
def pack_stream_result(path, images, res, labels=None, keep=0.9, per_frame=8000, stride=3, sky=False, exposure=1.0,
                       horizon=None, graze=None, firstseen=False, title=""):   # the viewer's confidence slider does the rest
    """images (F, 3, H, W) the frames; res a StreamResult of a StreamingSolver run with depth=True."""
    S = dict(c2w=res.c2w, c2w_exit=res.c2w_exit, K=res.intrinsics, depth=res.depth, conf=res.depth_conf)
    images = to_u8(images)
    skym = sky_of(images) if sky else None
    F_, H, W = S["depth"].shape
    ys, xs = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
    ys, xs = ys.ravel(), xs.ravel()
    img = images.permute(0, 2, 3, 1).numpy()
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
                       tuple(images.shape[-2:]), thumbs_of(images, 64), labels=labels, title=title,
                       dens=min(2.0, np.sqrt(cand / max(len(fid), 1))), conf=cf)


# ---------------------------------------------------------------------- exports: what the viewer shows, at every pixel
def solve_export(points, conf, depth, images, view_keep=1.0, keep=1.0, sky=False, far=2.5):
    """The points a solve's viewer shows (pack_solve's choice) at full resolution: no depth edges, no sky (sky=True), none
    farther than `far` scene radii, of each view its most confident `view_keep` share (the viewer's point budget,
    meta["view_keep"] of its pack), then the scene's most confident `keep` share (the viewer's confidence filtering).
    points (V, H, W, 3), conf and depth (V, H, W), images (V, 3, H, W) -> xyz (N, 3), rgb (N, 3) uint8."""
    points, conf = np.asarray(points, np.float32), np.asarray(conf, np.float32)
    images = to_u8(torch.as_tensor(images))
    rgb = images.permute(0, 2, 3, 1).numpy()
    ok = np.isfinite(points).all(-1) & ~depth_edges(torch.as_tensor(np.asarray(depth, np.float32))).numpy()
    if sky:
        ok &= ~sky_of(images)
    if ok.any():
        c = np.median(points[ok], 0)
        d = np.linalg.norm(np.where(ok[..., None], points, c) - c, axis=-1)
        ok &= d < far * float(np.quantile(d[ok], 0.9))
    if view_keep < 1:
        for v in range(len(points)):
            if ok[v].any():
                ok[v] &= conf[v] >= np.quantile(conf[v][ok[v]], 1 - view_keep)
    if keep < 1 and ok.any():
        ok &= conf >= np.quantile(conf[ok], 1 - keep)
    return points[ok], rgb[ok]


def stream_export(images, res, keep=1.0, sky=False, frame_keep=0.9, far=3.0):
    """The points a stream's viewer shows (pack_stream_result's choice) at full resolution: each frame's depth on arrival
    placed with its camera at exit, no depth edges, no sky (sky=True), the frame's most confident `frame_keep` share, only
    points the depths of the neighbouring frames agree with, none farther than `far` scene radii; then the stream's most
    confident `keep` share (the viewer's confidence filtering).  images (F, 3, H, W), res a StreamResult with depth."""
    S = dict(c2w=res.c2w, c2w_exit=res.c2w_exit, K=res.intrinsics, depth=res.depth, conf=res.depth_conf)
    images = to_u8(torch.as_tensor(images))
    skym = sky_of(images) if sky else None
    img = images.permute(0, 2, 3, 1).numpy()
    P, col, cf = [], [], []
    for f in range(len(S["depth"])):
        d, c = S["depth"][f].astype(np.float32), S["conf"][f].astype(np.float32)
        ok = np.isfinite(d) & (d > 0) & ~depth_edges(torch.as_tensor(d[None]))[0].numpy()
        if skym is not None:
            ok &= ~skym[f]
        if not ok.any():
            continue
        ok &= c >= np.quantile(c[ok], 1 - frame_keep)
        ys, xs = np.nonzero(ok)
        K, C, z = S["K"][f], S["c2w_exit"][f], d[ys, xs]
        X = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * z, (ys + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1) @ C[:3, :3].T + C[:3, 3]
        good = consistent(X, f, S)
        P.append(X[good]); col.append(img[f, ys[good], xs[good]]); cf.append(c[ys[good], xs[good]])
    P, col, cf = np.concatenate(P), np.concatenate(col), np.concatenate(cf)
    dd = np.linalg.norm(P - np.median(P, 0), axis=1)
    m = dd < far * float(np.quantile(dd, 0.9))
    if keep < 1 and m.any():
        m &= cf >= np.quantile(cf[m], 1 - keep)
    return P[m].astype(np.float32), col[m]
