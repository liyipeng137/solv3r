"""Final cameras for a same-lens image collection; the learned geometry stays unchanged."""
import numpy as np
import torch


def _numpy(value):
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def recover_shared_intrinsics(world_points, c2w, depth, depth_conf):
    """Fit one focal to raw world points in their predicted camera frames.

    Inputs have shapes (S,H,W,3), (S,4,4), (S,H,W[,1]), (S,H,W). The images must
    share a lens and pixel grid. Returns repeated float32 K (S,3,3) and a JSON
    report. fx=fy and the principal point is (W/2,H/2); there is no Z shift.
    Raw point maps must not be regenerated using the K being estimated.
    """
    import cv2

    shape = tuple(world_points.shape)
    if len(shape) != 4 or shape[-1] != 3 or min(shape[:3]) < 1:
        raise ValueError("world_points must have shape (S,H,W,3) with nonempty views")
    S, H, W, _ = shape
    if tuple(depth.shape) == (S, H, W, 1):
        depth = depth[..., 0]
    if tuple(depth.shape) != (S, H, W) or tuple(depth_conf.shape) != (S, H, W):
        raise ValueError("depth and depth_conf must match the world-point grid")
    cameras = _numpy(c2w).astype(np.float64)
    if cameras.shape != (S, 4, 4) or not np.isfinite(cameras).all():
        raise ValueError("c2w must contain finite (S,4,4) camera-to-world matrices")
    rotations = cameras[:, :3, :3]
    if (not np.allclose(cameras[:, 3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3)):
        raise ValueError("c2w must contain rigid camera-to-world transforms")

    ys = np.floor(np.arange(min(H, 64)) * H / min(H, 64)).astype(int)
    xs = np.floor(np.arange(min(W, 64)) * W / min(W, 64)).astype(int)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    uv = np.stack([gx + 0.5 - W / 2, gy + 0.5 - H / 2], -1)
    kernel = np.ones((3, 3), np.uint8)
    normalized, targets, weights, counts = [], [], [], []
    for i in range(S):
        d = _numpy(depth[i]).astype(np.float32)
        conf = _numpy(depth_conf[i])
        valid = np.isfinite(d) & (d > 0) & np.isfinite(conf)
        if not valid.any():
            counts.append(0)
            continue
        clean = np.where(valid, d, 0)
        variation = (cv2.dilate(clean, kernel) - cv2.erode(clean, kernel)) / np.maximum(clean, 1e-6)
        mask = ((variation <= 0.03) & (cv2.erode(valid.astype(np.uint8), kernel) > 0)
                & (conf >= np.median(conf[valid])))[gy, gx]
        points = _numpy(world_points[i])[gy, gx].astype(np.float64)
        local = (points - cameras[i, :3, 3]) @ rotations[i]
        mask &= np.isfinite(local).all(-1) & (local[..., 2] > 1e-6)
        n = int(mask.sum())
        counts.append(n if n >= 16 else 0)
        if n < 16:
            continue
        normalized.append(local[mask, :2] / local[mask, 2:3])
        targets.append(uv[mask])
        weights.append(np.full(n, 1 / n))             # equal initial weight per supported view
    if not normalized:
        raise ValueError("Cannot recover shared intrinsics: no view has 16 valid geometry samples")
    x, y, base_weight = np.concatenate(normalized), np.concatenate(targets), np.concatenate(weights)

    def solve(weight):
        denominator = np.sum(weight[:, None] * x * x)
        if denominator <= 1e-12:
            raise ValueError("Cannot recover shared intrinsics: degenerate camera rays")
        return float(np.sum(weight[:, None] * x * y) / denominator)

    focal = solve(base_weight)
    for _ in range(30):
        residual = np.linalg.norm(focal * x - y, axis=1)
        updated = solve(base_weight * np.minimum(1, 3 / np.maximum(residual, 1e-9)))
        change = abs(updated - focal)
        focal = updated
        if change < 1e-7:
            break
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Cannot recover shared intrinsics: fitted focal is not finite and positive")
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], np.float32)
    report = dict(shared_intrinsics=True, method="raw_pointmap_joint_focal", image_size_hw=[H, W],
                  focal_pixels=focal, K_shared=K.tolist(), samples_per_view=counts,
                  supported_views=sum(n > 0 for n in counts), total_samples=sum(counts),
                  median_fit_error_pixels=float(np.median(np.linalg.norm(focal * x - y, axis=1))),
                  sampling_grid_max=64, confidence_quantile=0.5, depth_edge_threshold=0.03,
                  huber_pixels=3.0, zero_skew=True, square_pixels=True, centered_principal_point=True,
                  depth_shift=False, raw_geometry_unchanged=True)
    return np.repeat(K[None], S, axis=0), report


@torch.no_grad()
def cameras_from_prediction(pred, *, shared_intrinsics=True):
    """Return CPU tensors (c2w, K) and a JSON report from a raw Solv3R.decode result.

    Poses always come from equal-weight ray fitting. By default all views share K
    recovered from the raw point maps. False keeps the fitted per-view K, with
    identical poses. This does not mutate pred or the solver state. Camera-head
    decoding remains available separately through cameras_from_pose_enc.
    """
    from .model import cameras_from_rays

    rays = pred["rays"]
    if len(rays.shape) != 4 or rays.shape[-1] != 6 or rays.shape[0] == 0:
        raise ValueError("rays must have shape (S,H,W,6) with nonempty views")
    cameras, intrinsics = [], []
    for start in range(0, rays.shape[0], 8):
        c2w, K = cameras_from_rays(torch.as_tensor(rays[start:start + 8]), depth_conf=None)
        cameras.append(c2w.cpu())
        intrinsics.append(K.cpu())
    c2w, K = torch.cat(cameras), torch.cat(intrinsics)
    if shared_intrinsics:
        shared_K, report = recover_shared_intrinsics(pred["world_points"], c2w, pred["depth"], pred["depth_conf"])
        K = torch.from_numpy(shared_K)
    else:
        report = dict(shared_intrinsics=False, method="per_view_ray_fit", raw_geometry_unchanged=True)
    report["pose_source"] = "equal_weight_ray_fitting"
    return c2w, K, report
