import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from solv3r import cameras_from_prediction, recover_shared_intrinsics


@pytest.fixture(scope='module')
def demo_engine():
    # demo/ is a script directory; unrelated installed packages can own `demo`.
    name = '_solv3r_demo_engine_test'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'demo' / 'engine.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[name]


def scene(S=3, H=112, W=84, focal=120.0):
    yy, xx = np.mgrid[:H, :W]
    local_rays = np.stack([(xx + .5 - W / 2) / focal, (yy + .5 - H / 2) / focal, np.ones((H, W))], -1)
    cameras = np.repeat(np.eye(4)[None], S, 0)
    points, rays = [], []
    depth = np.empty((S, H, W, 1), np.float32)
    for i in range(S):
        t = i * .2
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        C = np.array([i * .3, i * .1, 0])
        cameras[i, :3, :3], cameras[i, :3, 3] = R, C
        direction = local_rays @ R.T
        depth[i] = 2 + i * .3
        points.append(direction * depth[i] + C)
        rays.append(np.concatenate([direction, np.broadcast_to(C, direction.shape)], -1))
    pred = dict(world_points=torch.tensor(np.array(points), dtype=torch.float32),
                rays=torch.tensor(np.array(rays), dtype=torch.float32), depth=torch.tensor(depth),
                depth_conf=torch.ones(S, H, W))
    return pred, cameras


def test_recovers_known_shared_camera_without_mutating_geometry():
    pred, poses = scene()
    original = {k: v.clone() for k, v in pred.items()}
    K, info = recover_shared_intrinsics(pred['world_points'], poses, pred['depth'], pred['depth_conf'])
    expected = [[120, 0, 42], [0, 120, 56], [0, 0, 1]]
    np.testing.assert_allclose(K, np.broadcast_to(expected, K.shape), rtol=1e-6)
    assert np.array_equal(K[0], K[-1])
    assert info['supported_views'] == 3
    for key in pred:
        assert torch.equal(pred[key], original[key])


def test_similarity_and_view_order_do_not_change_focal():
    pred, poses = scene()
    G = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([20, -3, 7])
    points = pred['world_points'].numpy() * 3.2 @ G.T + shift
    transformed = poses.copy()
    transformed[:, :3, :3] = G @ poses[:, :3, :3]
    transformed[:, :3, 3] = poses[:, :3, 3] * 3.2 @ G.T + shift
    idx = [2, 0, 1]
    K, _ = recover_shared_intrinsics(points[idx], transformed[idx], pred['depth'][idx] * 3.2, pred['depth_conf'][idx])
    np.testing.assert_allclose(K[:, 0, 0], 120, rtol=1e-6)


def test_invalid_pixels_and_unsupported_views_are_reported():
    pred, poses = scene()
    pred['depth'][1] = 0
    pred['world_points'][0, :20] = float('nan')
    pred['depth_conf'][0, :20] = float('nan')
    K, info = recover_shared_intrinsics(pred['world_points'], poses, pred['depth'], pred['depth_conf'])
    np.testing.assert_allclose(K[:, 0, 0], 120, rtol=1e-6)
    assert info['samples_per_view'][1] == 0
    assert info['supported_views'] == 2


def test_low_confidence_bad_geometry_is_excluded():
    pred, poses = scene()
    pred['world_points'][:, :, :20] += 5
    pred['depth_conf'][:, :, :20] = -100
    K, _ = recover_shared_intrinsics(pred['world_points'], poses, pred['depth'], pred['depth_conf'])
    np.testing.assert_allclose(K[:, 0, 0], 120, rtol=1e-6)


@pytest.mark.parametrize('bad', ['empty', 'degenerate', 'pose', 'shape'])
def test_invalid_recovery_fails_without_fallback(bad):
    pred, poses = scene()
    if bad == 'empty':
        pred['depth'].zero_()
    elif bad == 'degenerate':
        pred['world_points'] = torch.tensor(poses[:, None, None, :3, 3] + poses[:, None, None, :3, 2] * 2).expand(3, 112, 84, 3)
    elif bad == 'pose':
        poses[0, 0, 0] = 2
    else:
        pred['depth_conf'] = pred['depth_conf'][:, :-1]
    with pytest.raises(ValueError):
        recover_shared_intrinsics(pred['world_points'], poses, pred['depth'], pred['depth_conf'])


def test_default_and_disabled_share_the_same_ray_fitting_poses():
    pred, poses = scene()
    c, K, info = cameras_from_prediction(pred)
    c_off, K_off, info_off = cameras_from_prediction(pred, shared_intrinsics=False)
    torch.testing.assert_close(c, c_off, rtol=0, atol=0)
    np.testing.assert_allclose(c.numpy(), poses, atol=2e-4)
    np.testing.assert_allclose(K.numpy()[:, 0, 0], 120, atol=2e-3)
    assert info['shared_intrinsics'] and not info_off['shared_intrinsics']
    assert info['pose_source'] == info_off['pose_source'] == 'equal_weight_ray_fitting'


def test_export_uses_shared_K_and_preserves_depth(tmp_path, monkeypatch):
    import solv3r.export as export
    pred, _ = scene()
    monkeypatch.setattr(export, 'scene_points', lambda *a: (np.array([[1, 2, 3]]), np.array([[20, 30, 40]])))
    path = str(tmp_path / 'solve')
    export.save_solve(path, torch.zeros(3, 3, 112, 84), pred, ['a', 'b', 'c'])
    data = np.load(path + '.npz')
    assert np.array_equal(data['K'][0], data['K'][2])
    np.testing.assert_array_equal(data['depth'], pred['depth'][..., 0].numpy().astype(np.float16))
    meta = json.loads(str(data['camera_metadata']))
    assert meta == json.loads((tmp_path / 'solve.cameras.json').read_text())
    assert meta['shared_intrinsics']


@pytest.mark.parametrize('shared', [True, False])
def test_demo_fits_across_decode_chunks(shared, monkeypatch, demo_engine):
    pred, poses = scene(S=20)
    for i in range(16, 20):
        local = (pred['world_points'][i].numpy() - poses[i, :3, 3]) @ poses[i, :3, :3]
        local[..., :2] *= 120 / 180
        pred['world_points'][i] = torch.tensor(local @ poses[i, :3, :3].T + poses[i, :3, 3])
    calls = []
    def ray_cameras(part, *, shared_intrinsics):
        assert not shared_intrinsics            # fit once across the collection, not independently per decode batch
        idx = part['indices']
        calls.append(len(idx))
        K = np.repeat(np.eye(3)[None], len(idx), 0)
        K[:, 0, 0] = K[:, 1, 1] = [120 if i < 16 else 180 for i in idx]
        return torch.tensor(poses[idx]), torch.tensor(K), dict(shared_intrinsics=False, pose_source='equal_weight_ray_fitting')
    monkeypatch.setattr(demo_engine, 'cameras_from_prediction', ray_cameras)
    class Model:
        def decode(self, z, ctx):
            return {**{k: v[ctx.indices] for k, v in pred.items()}, 'indices': ctx.indices}
    ctx = SimpleNamespace(subset=lambda idx: SimpleNamespace(indices=idx))
    sess = demo_engine.Session(images=torch.zeros(20, 3, 112, 84, dtype=torch.uint8), labels=list(range(20)), paths=[],
                               shared_intrinsics=shared)
    demo_engine.finish(sess, Model(), torch.zeros(20, 1, 1), ctx)
    assert calls == [16, 4]
    np.testing.assert_array_equal(sess.final['c2w'], poses)
    if shared:
        assert np.array_equal(sess.final['K'][0], sess.final['K'][-1])
        assert sess.final['camera_metadata']['supported_views'] == 20
    else:
        assert sess.final['K'][0, 0, 0] == 120
        assert sess.final['K'][-1, 0, 0] == 180


@pytest.mark.parametrize('which', ['library', 'demo'])
def test_final_viewer_uses_exported_cameras(which, tmp_path, monkeypatch, demo_engine):
    import solv3r.viewer.pack as library
    module = library if which == 'library' else demo_engine
    rec = module.Recorder(torch.zeros(2, 3, 28, 28))
    n = len(rec.ys)
    old_C = np.repeat(np.eye(4)[None], 2, 0).astype(np.float32)
    old_K = np.repeat(np.eye(3)[None], 2, 0).astype(np.float32)
    for call in [1, 2]:
        rec.cams[call] = old_C.copy()
        rec.snaps.append(dict(call=call, P=np.ones((2, n, 3)), conf=np.ones((2, n)),
                              depth=np.ones((2, n)), c2w=old_C.copy(), K=old_K.copy()))
    final_C, final_K = old_C.copy(), old_K.copy()
    final_C[:, 0, 3] = 9
    final_K[:, 0, 0] = final_K[:, 1, 1] = 100
    captured = {}
    def pack(*args, **kwargs):
        captured['C'], captured['K'], captured['motion'] = args[6], args[9], kwargs['motion']
        return {}
    monkeypatch.setattr(module, 'pack_solve', pack)
    rec.pack(str(tmp_path / 'scene'), ['a', 'b'], [0, 0], final=dict(c2w=final_C, K=final_K))
    np.testing.assert_array_equal(captured['C'][-1], final_C)
    np.testing.assert_array_equal(captured['C'][0], old_C)
    np.testing.assert_array_equal(captured['K'], final_K)
    assert np.isnan(captured['motion'][-1]).all()


@pytest.mark.parametrize('shared,expected_cameras', [(True, 1), (False, 3)])
def test_demo_colmap_camera_ids_follow_shared_setting(shared, expected_cameras, tmp_path, monkeypatch, demo_engine):
    demo = demo_engine
    pred, poses = scene()
    K, _ = recover_shared_intrinsics(pred['world_points'], poses, pred['depth'], pred['depth_conf'])
    paths = []
    for i in range(3):
        path = tmp_path / f'{i}.jpg'
        Image.new('RGB', (168, 224)).save(path)
        paths.append(str(path))
    monkeypatch.setattr(demo, 'OUT', str(tmp_path))
    monkeypatch.setattr(demo, 'confident', lambda *a: (np.array([[1, 2, 3]]), np.array([[20, 30, 40]])))
    sess = demo.Session(images=torch.zeros(3, 3, 112, 84, dtype=torch.uint8), labels=['a', 'b', 'c'], paths=paths)
    sess.final = dict(c2w=poses, K=K, camera_metadata=dict(shared_intrinsics=shared))
    archive = demo.export_colmap(sess, with_images=False)
    with zipfile.ZipFile(archive) as z:
        cameras = [x for x in z.read('sparse/0/cameras.txt').decode().splitlines() if x and not x.startswith('#')]
        images = [x for x in z.read('sparse/0/images.txt').decode().splitlines() if x and not x.startswith('#')]
        assert len(cameras) == expected_cameras
        assert len({x.split()[8] for x in images}) == expected_cameras
        assert float(cameras[0].split()[4]) == pytest.approx(240)
        assert json.loads(z.read('camera_metadata.json'))['shared_intrinsics'] == shared
