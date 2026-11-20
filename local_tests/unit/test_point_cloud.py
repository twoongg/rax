"""The point-cloud skill's helper on observations encoded exactly as robodojo_observe replies.

A ray-cast scene (a table and boxes, seen by an oblique overview camera and a close wrist
camera) is encoded with the MCP server's own RGB and depth PNG encoders, so the helper is
checked against the real reply format, depth quantization and calibration conventions.
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from services.robodojo.mcp_server import _depth_png_data, _png_data

PROJECT = Path(__file__).resolve().parents[2]
HELPER = PROJECT / 'agent_skills/perception/depth-robodojo/point-cloud/point_cloud.py'
spec = importlib.util.spec_from_file_location('point_cloud', HELPER)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

WIDTH, HEIGHT = 640, 480
K = np.array([[520.0, 0.0, 319.5], [0.0, 520.0, 239.5], [0.0, 0.0, 1.0]])
ORIGIN = np.array([2.0, -1.0, 0.1])   # Environment origin in world coordinates.
TABLE_Z = 0.75                         # World height of the table top.
TABLE_RGB = (120, 120, 120)


def look_at(position, target):
    """Camera-to-world pose with USD camera axes (+X right, +Y up, -Z forward)."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, up, -forward], axis=1)
    T[:3, 3] = position
    return T


def box(centre_env_xy, size, yaw_deg, rgb, lift=0.0):
    """A box standing on the table, or with its bottom `lift` metres above it."""
    return {'centre': np.array([*centre_env_xy]) + ORIGIN[:2], 'size': np.array(size, float),
            'yaw': math.radians(yaw_deg), 'rgb': rgb, 'lift': lift}


def render(T, boxes):
    """(rgb, depth) of the scene: depth along the optical axis, 0 where nothing is hit."""
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH]
    ray_cv = np.linalg.solve(K, np.stack([u.ravel(), v.ravel(), np.ones(u.size)]))
    direction = T[:3, :3] @ np.stack([ray_cv[0], -ray_cv[1], -ray_cv[2]])  # Optical z = 1.
    origin = T[:3, 3][:, None]
    depth = np.full(u.size, np.inf)
    rgb = np.zeros((u.size, 3), np.uint8)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_table = np.where(direction[2] < 0, (TABLE_Z - origin[2]) / direction[2], np.inf)
    hit = t_table < depth
    depth[hit], rgb[hit] = t_table[hit], TABLE_RGB
    for item in boxes:
        c, s = math.cos(-item['yaw']), math.sin(-item['yaw'])
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        centre = np.array([*item['centre'], TABLE_Z + item['lift'] + item['size'][2] / 2])
        p = rotation @ (origin - centre[:, None])
        d = rotation @ direction
        half = item['size'][:, None] / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            t1, t2 = (-half - p) / d, (half - p) / d
        near = np.nanmax(np.minimum(t1, t2), axis=0)
        far = np.nanmin(np.maximum(t1, t2), axis=0)
        hit = (near <= far) & (near > 0) & (near < depth)
        depth[hit], rgb[hit] = near[hit], item['rgb']
    depth[~np.isfinite(depth)] = 0.0
    return rgb.reshape(HEIGHT, WIDTH, 3), depth.reshape(HEIGHT, WIDTH).astype(np.float32)


def reply(boxes, *, cameras=None, with_depth=True, eef=None):
    """A robodojo_observe reply: JSON text, RGB blocks, then depth blocks, indexed by attachments."""
    cameras = cameras or {
        'cam_high': look_at(ORIGIN + [0.0, -0.55, TABLE_Z + 0.55], ORIGIN + [0.0, 0.0, TABLE_Z]),
        'cam_left_wrist': look_at(ORIGIN + [-0.18, 0.12, TABLE_Z + 0.22], ORIGIN + [0.0, 0.0, TABLE_Z])}
    rendered = {name: render(T, boxes) for name, T in cameras.items()}
    content, attachments, depth_maps = [], [], {}
    for name, (rgb, _) in rendered.items():
        content.append({'type': 'image', 'mimeType': 'image/png', 'data': _png_data(rgb)})
        attachments.append({'kind': 'rgb', 'camera': name, 'content_index': len(content)})
    if with_depth:
        for name, (_, depth) in rendered.items():
            data, stats = _depth_png_data(depth)
            content.append({'type': 'image', 'mimeType': 'image/png', 'data': data})
            depth_maps[name] = dict(stats, content_index=len(content), raw_field=f'{name}_depth_m')
            attachments.append({'kind': 'depth', 'camera': name, 'content_index': len(content)})
    packet = {'step_id': 7, 'environment_origin_world_m': ORIGIN.tolist(), 'images': list(cameras),
              'depth': depth_maps, 'attachments': attachments,
              'eef_positions': eef if eef is not None else [[-0.3, 0.3, 0.9], [-0.3, -0.3, 0.9]],
              'camera_parameters': {} if not with_depth else {
                  name: {'intrinsic_matrix': K.tolist(), 'camera_to_world_usd': T.tolist(),
                         'resolution': [WIDTH, HEIGHT]} for name, T in cameras.items()}}
    return {'content': [{'type': 'text', 'text': json.dumps(packet)}, *content]}


def pixel_of(obs, camera, point_env):
    uv, z = pc.project(obs, camera, np.asarray(point_env)[None])
    assert z[0] > 0
    return int(round(uv[0, 0])), int(round(uv[0, 1]))


RED = box((0.03, 0.02), (0.06, 0.04, 0.05), 30.0, (200, 30, 30))


def test_reply_decodes_to_metric_depth_and_calibration():
    obs = pc.observation(reply([RED]))
    assert obs.cameras == ('cam_high', 'cam_left_wrist') and obs.step_id == 7
    assert obs.rgb['cam_high'].shape == (HEIGHT, WIDTH, 3)
    assert obs.depth['cam_high'].dtype == np.float32 and 0.5 < obs.depth['cam_high'].max() < 2.5
    assert pc.observation(obs) is obs


def test_cloud_puts_the_table_at_its_height_in_the_environment_frame():
    xyz, rgb, cam = pc.cloud(reply([RED]))
    assert xyz.dtype == np.float32 and len(xyz) == len(rgb) == len(cam) > 100_000
    assert set(np.unique(cam)) == {0, 1}
    table_z, count = pc.table_height(xyz)
    assert abs(table_z - (TABLE_Z - ORIGIN[2])) < 0.0015 and count > 50_000
    table = np.all(rgb == TABLE_RGB, axis=1)
    assert np.abs(xyz[table, 2] - (TABLE_Z - ORIGIN[2])).max() < 0.004


def test_cloud_takes_one_camera_by_name_and_crops_to_a_box():
    obs = pc.observation(reply([RED]))
    xyz, _, cam = pc.cloud(obs, 'cam_left_wrist', stride=2)
    assert len(xyz) and set(np.unique(cam)) == {obs.cameras.index('cam_left_wrist')} == {1}
    inside, _, _ = pc.cloud(obs, box=((0.0, 0.06), (0.0, 0.04), (0.66, 1.0)))
    assert len(inside) and inside[:, 2].min() >= 0.66 and inside[:, 0].max() <= 0.06
    with pytest.raises(ValueError, match='cam_right_wrist'):
        pc.cloud(obs, ['cam_right_wrist'])


def test_pixel_point_and_project_are_inverse():
    obs = pc.observation(reply([RED]))
    for camera, (u, v) in (('cam_high', (300, 260)), ('cam_left_wrist', (350, 200))):
        point = pc.pixel_point(obs, camera, u, v, radius=0)
        uv, z = pc.project(obs, camera, point[None])
        assert z[0] > 0 and np.abs(uv[0] - (u, v)).max() < 0.6


def test_segment_recovers_a_box_from_one_pixel():
    obs = pc.observation(reply([RED]))
    top_env = np.array([0.03, 0.02, TABLE_Z - ORIGIN[2] + 0.05])
    for cameras in (None, 'cam_high'):
        part = pc.segment(obs, 'cam_high', *pixel_of(obs, 'cam_high', top_env), cameras=cameras)
        assert np.allclose(part['top'][:2], top_env[:2], atol=0.004), part['top']
        assert abs(part['top_z'] - top_env[2]) < 0.002
        assert abs(part['table_z'] - (TABLE_Z - ORIGIN[2])) < 0.0015
        obb = part['obb']
        assert abs(obb['yaw_deg'] - 30.0) < 3.0, obb
        assert np.allclose(obb['size'][:2], (0.06, 0.04), atol=0.006), obb
        assert np.allclose(obb['center'][:2], top_env[:2], atol=0.004), obb
        assert np.all(np.all(part['rgb'] == (200, 30, 30), axis=1))
    fused = pc.segment(obs, 'cam_high', *pixel_of(obs, 'cam_high', top_env))
    alone = pc.segment(obs, 'cam_high', *pixel_of(obs, 'cam_high', top_env), cameras='cam_high')
    assert fused['n_points'] > alone['n_points'] and fused['cameras'] == ['cam_high', 'cam_left_wrist']


def test_segment_from_a_wrist_view_keeps_the_table_height_of_the_overview():
    obs = pc.observation(reply([RED]))
    top_env = np.array([0.03, 0.02, TABLE_Z - ORIGIN[2] + 0.05])
    part = pc.segment(obs, 'cam_left_wrist', *pixel_of(obs, 'cam_left_wrist', top_env),
                      cameras='cam_left_wrist')
    assert abs(part['table_z'] - (TABLE_Z - ORIGIN[2])) < 0.0015
    assert abs(part['top_z'] - top_env[2]) < 0.002


def test_color_tol_splits_touching_objects_of_different_colours():
    blue = box((0.03 + 0.06, 0.02), (0.06, 0.04, 0.05), 0.0, (30, 30, 200))
    red = box((0.03, 0.02), (0.06, 0.04, 0.05), 0.0, (200, 30, 30))
    obs = pc.observation(reply([red, blue]))
    seed = pixel_of(obs, 'cam_high', [0.03, 0.02, TABLE_Z - ORIGIN[2] + 0.05])
    merged = pc.segment(obs, 'cam_high', *seed)
    split = pc.segment(obs, 'cam_high', *seed, color_tol=0.08)
    assert merged['size'][0] > 0.11 and abs(split['size'][0] - 0.06) < 0.006
    # Voxels on the colour edge mix both colours; they fold into one side.
    assert np.mean(np.all(split['rgb'] == (200, 30, 30), axis=1)) > 0.99
    assert sorted(item['label'] for item in pc.objects(obs, color_tol=0.08)) == ['blue · 6x4x5 cm', 'red · 6x4x5 cm']


def test_segment_refuses_the_table_and_bad_arguments():
    obs = pc.observation(reply([RED]))
    table = pixel_of(obs, 'cam_high', [-0.15, -0.1, TABLE_Z - ORIGIN[2]])
    with pytest.raises(ValueError, match='table'):
        pc.segment(obs, 'cam_high', *table)
    with pytest.raises(ValueError, match='radius_mm'):
        pc.segment(obs, 'cam_high', 320, 240, radius_mm=100)
    with pytest.raises(ValueError, match='outside'):
        pc.pixel_point(obs, 'cam_high', 700, 10)


def test_overlay_paints_the_segment_where_the_camera_sees_it():
    obs = pc.observation(reply([RED]))
    seed = pixel_of(obs, 'cam_high', [0.03, 0.02, TABLE_Z - ORIGIN[2] + 0.05])
    part = pc.segment(obs, 'cam_high', *seed)
    image = pc.overlay(obs, 'cam_high', part['points'])
    painted = np.all(image == (255, 0, 255), axis=2)
    red = np.all(obs.rgb['cam_high'] == (200, 30, 30), axis=2)
    assert painted[seed[1], seed[0]] and (painted & red).sum() > 0.9 * red.sum()
    assert not np.any(obs.rgb['cam_high'] == (255, 0, 255))


def test_an_rgb_only_reply_is_refused_with_the_reason():
    with pytest.raises(ValueError, match='rgbd'):
        pc.observation(reply([RED], with_depth=False))


def test_pixel_point_matches_the_pixel_to_position_tool():
    """The helper and robodojo_pixel_to_position (server side, raw float depth) agree."""
    from services.robodojo.pixel_geometry import locate_pixel_selections
    cameras = {'cam_high': look_at(ORIGIN + [0.0, -0.55, TABLE_Z + 0.55], ORIGIN + [0.0, 0.0, TABLE_Z]),
               'cam_left_wrist': look_at(ORIGIN + [-0.18, 0.12, TABLE_Z + 0.22], ORIGIN + [0.0, 0.0, TABLE_Z])}
    obs = pc.observation(reply([RED], cameras=cameras))
    raw = {'environment_origin_world_m': ORIGIN.tolist(),
           'camera_parameters': {name: {'intrinsic_matrix': K.tolist(), 'camera_to_world_usd': T.tolist(),
                                        'resolution': [WIDTH, HEIGHT]} for name, T in cameras.items()}}
    for name, T in cameras.items():
        raw[f'{name}_depth_m'] = render(T, [RED])[1]
    for camera, pixel in (('cam_high', (300, 260)), ('cam_high', (318, 205)), ('cam_left_wrist', (350, 200))):
        tool = locate_pixel_selections(raw, [{'camera': camera, 'pixel': list(pixel)}],
                                       method='depth', depth_window_radius=0)
        mine = pc.pixel_point(obs, camera, *pixel, radius=0)
        assert np.abs(mine - tool['position_environment_m']).max() < 0.0015, (camera, pixel, mine, tool)


# The table top in the environment frame, and objects for the scene-wide tests.
TOP = TABLE_Z - ORIGIN[2]
GREEN = box((-0.12, 0.05), (0.08, 0.05, 0.03), -20.0, (30, 170, 60))
ARM = box((0.02, -0.12), (0.05, 0.05, 0.05), 0.0, (90, 90, 90), lift=0.12)   # Floats: a piece of an arm.


def test_objects_lists_what_stands_on_the_table_without_a_pixel():
    obs = pc.observation(reply([RED, GREEN, ARM]))
    found = pc.objects(obs)
    assert [item['index'] for item in found] == [0, 1]
    green, red = found  # Largest footprint first.
    assert np.allclose(green['obb']['center'][:2], (-0.12, 0.05), atol=0.004)
    assert abs(green['obb']['yaw_deg'] + 20.0) < 3.0 and abs(green['top_z'] - (TOP + 0.03)) < 0.002
    assert np.allclose(red['obb']['center'][:2], (0.03, 0.02), atol=0.004) and abs(red['obb']['yaw_deg'] - 30.0) < 3.0
    assert green['label'] == 'green · 8x5x3 cm' and red['label'] == 'red · 6x4x5 cm', (green['label'], red['label'])
    assert not green['robot'] and green['supported'] and red['table_z'] is not None
    every = pc.objects(obs, include_robot=True)
    arm = [item for item in every if not item['supported']]
    assert len(every) == 3 and len(arm) == 1 and abs(arm[0]['bottom_z'] - (TOP + 0.12)) < 0.003


def test_objects_marks_what_the_gripper_reaches_as_robot():
    gripper = [0.03, 0.02, TOP + 0.05 + 0.03]  # A link6 position 3 cm above the red box.
    obs = pc.observation(reply([RED, GREEN], eef=[gripper, [-0.3, -0.3, TOP + 0.3]]))
    assert [item['label'] for item in pc.objects(obs)] == ['green · 8x5x3 cm']
    every = pc.objects(obs, include_robot=True)
    assert [item['robot'] for item in every] == [False, True]


def test_objects_keeps_a_stacked_object_and_splits_it_by_colour():
    top = box((0.03, 0.02), (0.04, 0.03, 0.02), 30.0, (30, 30, 200), lift=0.05)
    obs = pc.observation(reply([RED, top]))
    assert len(pc.objects(obs)) == 1  # Touching parts join without color_tol.
    lower, upper = pc.objects(obs, color_tol=0.08)
    assert upper['supported'] and abs(upper['bottom_z'] - (TOP + 0.05)) < 0.004
    assert upper['label'].startswith('blue') and lower['label'].startswith('red')


def test_profile_tells_the_ends_of_an_object_apart_and_flip_turns_its_frame():
    # A body with a thin, low tab at its +x end (yaw 30 degrees).
    body = box((0.0, 0.0), (0.05, 0.032, 0.026), 30.0, (40, 40, 40))
    tab_centre = 0.035 * np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
    tab = box(tuple(tab_centre), (0.02, 0.014, 0.004), 30.0, (40, 40, 40), lift=0.011)
    obs = pc.observation(reply([body, tab]))
    (part,) = pc.objects(obs)
    rows = [row for row in pc.profile(part) if row['n']]
    tab_end = rows[-1] if part['obb']['yaw_deg'] > 0 else rows[0]
    body_end = rows[0] if part['obb']['yaw_deg'] > 0 else rows[-1]
    assert tab_end['width'] < 0.6 * body_end['width'] and tab_end['top_z'] < body_end['top_z'] - 0.005
    turned = pc.flip(part)
    assert abs(((turned['yaw_deg'] - part['obb']['yaw_deg']) % 360.0) - 180.0) < 1e-9
    assert pc.flip(turned)['yaw_deg'] == pytest.approx(part['obb']['yaw_deg'])


def test_object_frame_offsets_follow_the_object_to_a_new_layout():
    offset = np.array([[0.02, -0.01, 0.025]])  # A feature on the box, in the box's own frame.
    for centre, yaw in (((0.03, 0.02), 30.0), ((-0.05, 0.08), -60.0)):
        obs = pc.observation(reply([box(centre, (0.06, 0.04, 0.05), yaw, (200, 30, 30))]))
        (part,) = pc.objects(obs)
        expected_yaw = math.radians(part['obb']['yaw_deg'])
        assert abs(part['obb']['yaw_deg'] - yaw) < 3.0
        point = pc.world(part, offset)[0]
        c, s = math.cos(expected_yaw), math.sin(expected_yaw)
        truth = np.array([centre[0] + 0.02 * c + 0.01 * s, centre[1] + 0.02 * s - 0.01 * c])
        assert np.allclose(point[:2], truth, atol=0.004)
        assert np.allclose(pc.local(part, pc.world(part, offset)), offset)


def test_describe_summarizes_points_cut_with_a_box():
    obs = pc.observation(reply([RED, GREEN]))
    xyz, rgb, _ = pc.cloud(obs, box=((0.0, 0.07), (-0.01, 0.06), (TOP + 0.006, TOP + 0.2)))
    item = pc.describe(xyz, rgb, table_z=TOP)
    assert item['label'].startswith('red') and abs(item['top_z'] - (TOP + 0.05)) < 0.002
    with pytest.raises(ValueError, match='three points'):
        pc.describe(xyz[:2])


@pytest.mark.parametrize('yaw', [150.0, -30.0, 60.0, -100.0])
def test_profile_finds_thin_pins_at_one_end_of_a_small_object(yaw):
    """A 5 x 3.2 x 2.6 cm body with two 1.5 mm pins 1 cm long at one end, seen by cam_high only."""
    a = math.radians(yaw)
    along, across = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    centre = np.array([-0.05, 0.0])
    scene = [box(tuple(centre), (0.05, 0.032, 0.026), yaw, (235, 235, 235))] + [
        box(tuple(centre + 0.03 * along + side * across), (0.010, 0.0015, 0.006), yaw, (200, 200, 205), lift=0.010)
        for side in (-0.00625, 0.00625)]
    high = {'cam_high': look_at(ORIGIN + [0.0, -0.55, TABLE_Z + 0.55], ORIGIN + [0.0, 0.0, TABLE_Z])}
    (part,) = pc.objects(pc.observation(reply(scene, cameras=high)))
    assert abs(part['obb']['size'][0] - 0.06) < 0.004
    rows = [row for row in pc.profile(part) if row['n']]
    pins, body = (rows[-1], rows[0]) if rows[-1]['width'] < rows[0]['width'] else (rows[0], rows[-1])
    assert pins['width'] < 0.6 * body['width'] and pins['top_z'] < body['top_z'] - 0.006
    facing = pc.flip(part) if pins is rows[0] else part['obb']  # +x now points at the pins.
    assert abs(((facing['yaw_deg'] - yaw + 180.0) % 360.0) - 180.0) < 3.0
