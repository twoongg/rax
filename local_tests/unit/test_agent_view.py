"""What the agent CLI sees of a robot tool reply: depth and robot spheres reach Python only."""
import copy
import json

from services.benchmarks import BENCHMARKS
from services.controller.agent_view import PYTHON_CLIENT, agent_view


def block(tag):
    return {'type': 'image', 'mimeType': 'image/png', 'data': tag}


def reply(packet, images):
    return {'content': [{'type': 'text', 'text': json.dumps(packet)}, *images]}


def packet_of(result):
    return json.loads(result['content'][0]['text'])


CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def observe_reply():
    """An observation as robodojo_observe sends it: RGB blocks first, then depth blocks."""
    attachments = [{'kind': 'rgb', 'camera': c, 'content_index': 1 + i} for i, c in enumerate(CAMERAS)]
    attachments += [{'kind': 'depth', 'camera': c, 'content_index': 4 + i} for i, c in enumerate(CAMERAS)]
    packet = {'step_id': 7, 'images': list(CAMERAS), 'attachments': attachments,
              'depth': {c: {'content_index': 4 + i, 'unit': 'millimeters'} for i, c in enumerate(CAMERAS)},
              'camera_parameters': {'cam_high': {'intrinsic_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
              'robot_spheres': {'step_id': 7, 'frame': 'environment_origin',
                                'spheres_m': [[0.1, 0.2, 0.3, 0.01]] * 3, 'counts': {'left': 2, 'right': 1}}}
    images = [block(f'rgb:{c}') for c in CAMERAS] + [block(f'depth:{c}') for c in CAMERAS]
    return reply(packet, images)


def test_observe_drops_depth_and_counts_spheres():
    seen = agent_view(observe_reply())
    packet = packet_of(seen)
    assert [b['data'] for b in seen['content'][1:]] == [f'rgb:{c}' for c in CAMERAS]
    assert [(a['kind'], a['camera'], a['content_index']) for a in packet['attachments']] == [
        ('rgb', c, 1 + i) for i, c in enumerate(CAMERAS)]
    assert isinstance(packet['depth'], str) and 'ctx.call' in packet['depth']
    spheres = packet['robot_spheres']
    assert spheres['count'] == 3 and spheres['counts'] == {'left': 2, 'right': 1}
    assert 'spheres_m' not in spheres and 'ctx.call' in spheres['note']
    assert packet['camera_parameters'] == packet_of(observe_reply())['camera_parameters']


def test_ik_solution_spheres_become_counts():
    rows = [{'ok': True, 'why': 'ok', 'joints_rad': [0.0] * 6, 'robot_spheres': [[0.0, 0.0, 0.0, 0.01]] * 5},
            {'ok': False, 'why': 'unreachable'}]
    seen = packet_of(agent_view(reply({'results': [{'solutions': rows}]}, [])))
    first, second = seen['results'][0]['solutions']
    assert first['robot_spheres']['count'] == 5 and 'ctx.call' in first['robot_spheres']['note']
    assert first['joints_rad'] == [0.0] * 6 and second == rows[1]


def test_depth_before_rgb_is_renumbered():
    packet = {'attachments': [{'kind': 'depth', 'camera': 'cam_high', 'content_index': 1},
                              {'kind': 'rgb', 'camera': 'cam_high', 'content_index': 2},
                              {'kind': 'rgb', 'camera': 'cam_left_wrist', 'content_index': 3}]}
    seen = agent_view(reply(packet, [block('depth'), block('rgb-high'), block('rgb-left')]))
    assert [b['data'] for b in seen['content'][1:]] == ['rgb-high', 'rgb-left']
    assert [(a['camera'], a['content_index']) for a in packet_of(seen)['attachments']] == [
        ('cam_high', 1), ('cam_left_wrist', 2)]


def test_plain_and_error_replies_unchanged():
    plain = reply({'step_id': 3, 'states': [0.5]}, [block('rgb')])
    error = {'isError': True, 'content': [{'type': 'text', 'text': json.dumps({'error': {'reason': 'x'}})}]}
    not_json = {'content': [{'type': 'text', 'text': 'not json'}]}
    for result in (plain, error, not_json, {}):
        assert agent_view(result) == result


def test_input_not_mutated():
    original = observe_reply()
    kept = copy.deepcopy(original)
    agent_view(original)
    assert original == kept


def test_robodojo_flag_only():
    assert BENCHMARKS['robodojo'].depth_in_python_only
    assert not BENCHMARKS['behavior'].depth_in_python_only
    assert PYTHON_CLIENT == 'python-controller'
