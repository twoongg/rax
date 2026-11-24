"""What the agent CLI sees of a robot tool reply: depth and robot spheres reach Python only."""
import base64
import copy
import json
import os
import tempfile

from services.benchmarks import BENCHMARKS
from services.controller.agent_view import PYTHON_CLIENT, agent_view


def block(tag):
    return {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(tag.encode()).decode()}


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
    assert seen['content'][1:] == [block(f'rgb:{c}') for c in CAMERAS]
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
    assert seen['content'][1:] == [block('rgb-high'), block('rgb-left')]
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


# The frontend: which connection gets which view.
import socket
import threading

import pytest

from test_controller_backend import supervisor  # noqa: F401 (fixture)
from services.controller.frontend import ResearchFrontend


def initialize(name):
    return {'jsonrpc': '2.0', 'id': 0, 'method': 'initialize',
            'params': {'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': name}}}


OBSERVE = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
           'params': {'name': 'robodojo_observe', 'arguments': {'robot_spheres': True}}}


@pytest.fixture
def observing(supervisor, tmp_path):  # noqa: F811
    s, backend = supervisor
    observed = observe_reply()
    step = backend.call

    def call(name, arguments):
        if name == 'robodojo_observe':
            return packet_of(observed), observed['content'][1:]
        return step(name, arguments)
    backend.call = call
    f = ResearchFrontend(s, tmp_path/'agent')
    f.call('start_episode', {})
    yield f, s
    f.close()


def observed_by(frontend, client):
    return frontend.handle(OBSERVE, client)['result']


def test_agent_connection_gets_agent_view(observing):
    f, _ = observing
    client = {}
    f.handle(initialize('claude-code'), client)
    result = observed_by(f, client)
    assert len(result['content']) == 4
    assert packet_of(result)['robot_spheres']['count'] == 3


def test_python_connection_gets_full_reply(observing):
    f, _ = observing
    client = {}
    f.handle(initialize(PYTHON_CLIENT), client)
    for result in (observed_by(f, client), f.handle(OBSERVE)['result']):  # and internal callers
        assert len(result['content']) == 7
        assert len(packet_of(result)['robot_spheres']['spheres_m']) == 3


def test_uninitialized_connection_is_the_agent(observing):
    f, _ = observing
    assert len(observed_by(f, {})['content']) == 4


def test_recording_keeps_full_reply(observing):
    f, s = observing
    observed_by(f, {})
    rows = [json.loads(line) for line in (s.config.root/'mcp.jsonl').read_text().splitlines()]
    last = [r for r in rows if r.get('event') == 'mcp_result' and r.get('tool') == 'robodojo_observe'][-1]
    assert len(json.loads(last['result']['content'][0]['text'])['robot_spheres']['spheres_m']) == 3


def socket_path():
    # pytest's tmp_path is too long for AF_UNIX on macOS (about 104 bytes).
    return os.path.join(tempfile.mkdtemp(prefix='rax'), 'mcp.sock')


def exchange(path, *requests):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(10)
        connection.connect(path)
        connection.sendall(b''.join(json.dumps(r).encode() + b'\n' for r in requests))
        with connection.makefile('rb') as stream:
            return [json.loads(stream.readline()) for _ in requests]


def test_socket_connections_keep_their_own_view(observing):
    f, _ = observing
    path = socket_path()
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(path)
    listener.listen()
    thread = threading.Thread(target=lambda: pytest.raises(Exception, f.serve_socket, listener), daemon=True)
    thread.start()
    try:
        agent = exchange(path, initialize('claude-code'), OBSERVE)[1]['result']
        python = exchange(path, initialize(PYTHON_CLIENT), OBSERVE)[1]['result']
    finally:
        listener.close()
        os.unlink(path)
    assert len(agent['content']) == 4 and len(python['content']) == 7


def test_parallel_transport_passes_each_connection_state(tmp_path):
    from services.exploration.transport import serve_socket
    seen = []

    class Service:
        def parallel_request(self, request):
            return request['method'] == 'step'

        def handle(self, request, client=None):
            if request['method'] == 'stop':
                raise SystemExit()
            if request['method'] == 'initialize':
                client['name'] = request['params']['clientInfo']['name']
            seen.append((request['method'], dict(client)))
            return {'jsonrpc': '2.0', 'id': request['id'], 'result': {}}

        def interrupt(self):
            pass

    path = socket_path()
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(path)
    listener.listen()
    thread = threading.Thread(target=lambda: pytest.raises(SystemExit, serve_socket, Service(), listener, slots=2))
    thread.start()
    try:
        exchange(path, initialize('claude-code'), {'jsonrpc': '2.0', 'id': 1, 'method': 'step'})
        exchange(path, {'jsonrpc': '2.0', 'id': 2, 'method': 'step'})
        with socket.socket(socket.AF_UNIX) as stop:
            stop.connect(path)
            stop.sendall(b'{"jsonrpc":"2.0","id":3,"method":"stop"}\n')
            thread.join(10)
    finally:
        listener.close()
        os.unlink(path)
    assert seen == [('initialize', {'name': 'claude-code'}), ('step', {'name': 'claude-code'}), ('step', {})]


def test_formal_copy_endpoint_gives_python_only_the_full_reply():
    """Agent formal's live copies reach the gateway through their own endpoint."""
    from services.agent_formal.runner import FormalMCP
    from services.mcp_contract import Contract

    class Gateway:
        contract = Contract()  # RoboDojo, rgbd
        terminal = False

        def handle(self, token, request):
            return {'jsonrpc': '2.0', 'id': request.get('id'), 'result': observe_reply()}

    endpoint = FormalMCP(Gateway(), 'token', info={})
    agent, python = {}, {}
    endpoint.handle(initialize('claude-code'), agent)
    endpoint.handle(initialize(PYTHON_CLIENT), python)
    assert len(endpoint.handle(OBSERVE, agent)['result']['content']) == 4
    assert len(endpoint.handle(OBSERVE, python)['result']['content']) == 7
    assert len(endpoint.handle(OBSERVE)['result']['content']) == 7
