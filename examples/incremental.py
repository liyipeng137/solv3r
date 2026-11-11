"""Incremental reconstruction: new images join an ongoing solve.

The first --initial images are solved for --calls calls. The other images then join in groups of --group, in file
order: their encodings are appended to the state and the solve continues for --calls calls, which also updates the
images solved before. The final solve is saved as <out>/solve.{npz,ply,html}, and with --every-stage each stage also
as <out>/stage_NN.{npz,ply}.

usage: python examples/incremental.py --images path/to/images --out outputs/incremental --initial 8 --group 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solv3r import Solv3R, cameras_from_prediction, list_images, load_images  # noqa: E402
from solv3r.export import save_solve  # noqa: E402
from solv3r.viewer import Recorder, save_solve_viewer  # noqa: E402


def solve_in_stages(model, images, stages, calls, out, names, sky=False, every_stage=False, title="Solv3R", keep=1.0,
                    shared_intrinsics=True):
    """stages: lists of image indices; the first image of the first stage is the reference."""
    os.makedirs(out, exist_ok=True)
    rec = Recorder(images[[j for idx in stages for j in idx]])   # snapshots for the viewer, views in arrival order
    order, first, call = [], [], 0
    for i, idx in enumerate(stages):
        if i == 0:
            ctx = model.encode(images[idx])
            z = model.start(ctx)
            rec.cameras(model, z, ctx, 0)
        else:
            z, ctx = model.add_views(z, ctx, images[idx])
        order += list(idx)
        first += [len(rec.snaps)] * len(idx)                     # the snapshot at which these views appear
        for k in range(calls):
            z = model.solve(ctx, 1, z)
            call += 1
            rec.record(model, z, ctx, call, dense={call - k, call - k + calls - 1})   # first and last call of the stage
        if every_stage:
            save_solve(f"{out}/stage_{i:02d}", images[order], model.decode(z, ctx), [names[j] for j in order], keep, sky,
                       shared_intrinsics=shared_intrinsics)
        print(f"stage {i}: {len(order)} images solved")
    rec.close(model, z, ctx, call)
    pred = model.decode(z, ctx)
    camera_result = cameras_from_prediction(pred, shared_intrinsics=shared_intrinsics)
    c2w, K, _ = camera_result
    meta = save_solve_viewer(f"{out}/solve.html", rec, [names[j] for j in order], first, sky=sky, title=title,
                             stats=[f"{len(order)} views", f"{len(stages)} stages", f"{call} calls"],
                             final=dict(c2w=c2w.numpy(), K=K.numpy()))
    save_solve(f"{out}/solve", images[order], pred, [names[j] for j in order], keep, sky,
               view_keep=meta["view_keep"], shared_intrinsics=shared_intrinsics, camera_result=camera_result)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="a folder of images or a video file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="a .safetensors file or a Hugging Face repo id (default: checkpoints/ if present, else the released weights)")
    ap.add_argument("--initial", type=int, default=8)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--keep", type=float, default=1.0, help="fraction of the points in the .ply, the most confident first "
                    "(1: all; points on depth edges always leave), as the demo's confidence filtering")
    ap.add_argument("--sky-mask", action="store_true", help="leave the sky out of the point cloud (outdoor captures)")
    ap.add_argument("--every-stage", action="store_true", help="also save the solve after every stage")
    ap.add_argument("--no-shared-intrinsics", action="store_true", help="keep per-view ray-fitted K")
    a = ap.parse_args()

    model = Solv3R.from_pretrained(a.checkpoint)
    paths = list_images(a.images)
    n = len(paths)
    stages = [list(range(min(a.initial, n)))] + [list(range(s, min(s + a.group, n))) for s in range(a.initial, n, a.group)]
    solve_in_stages(model, load_images(paths), stages, a.calls, a.out, [os.path.basename(p) for p in paths], a.sky_mask,
                    a.every_stage, os.path.basename(os.path.normpath(a.images)), a.keep, not a.no_shared_intrinsics)


if __name__ == "__main__":
    main()
