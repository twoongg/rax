"""The agent's own robodojo_observe lists the objects with ids; Python's replies stay whole."""
import json

import pytest

from services.benchmarks import BENCHMARKS
from services.controller.agent_objects import for_agent
from services.controller.agent_view import PYTHON_CLIENT
from services.controller.frontend import ResearchFrontend
from test_controller_backend import supervisor  # noqa: F401 (fixture)
from test_point_cloud import GREEN, RED, box, reply

FAR_SPHERE = [[-1.0, -1.0, -1.0, 0.01]]
OPEN = {'states': [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0], 'gripper_commands': [1.0, 1.0],
        'tcp_positions': [[-0.3, 0.3, 0.9], [-0.3, -0.3, 0.9]], 'episode_id': 'episode'}


def scene_reply(boxes, *, spheres, **fields):
    """A robodojo_observe reply of `boxes` from test_point_cloud, with the fields servers add."""
    built = reply(boxes, spheres=FAR_SPHERE if spheres else None, **fields)
    packet = {**json.loads(built['content'][0]['text']), **OPEN}
    return packet, built['content'][1:]


def packet_of(result):
    return json.loads(result['content'][0]['text'])


def initialize(name):
    return {'jsonrpc': '2.0', 'id': 0, 'method': 'initialize',
            'params': {'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': name}}}


def tool(name, arguments=None):
    return {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {'name': name, 'arguments': arguments or {}}}


@pytest.fixture
def scene(supervisor, tmp_path):  # noqa: F811
    """A frontend whose backend serves `state['boxes']`; it records every robodojo_observe."""
    s, backend = supervisor
    state = {'boxes': [RED, GREEN], 'calls': [], 'refuse_spheres': False}
    step = backend.call

    def call(name, arguments):
        if name != 'robodojo_observe':
            return step(name, arguments)
        state['calls'].append(dict(arguments))
        if arguments.get('robot_spheres') and state['refuse_spheres']:
            raise RuntimeError('robot spheres unavailable')
        return scene_reply(state['boxes'], spheres=arguments.get('robot_spheres', False),
                           **state.get('fields', {}))
    backend.call = call
    f = ResearchFrontend(s, tmp_path/'agent')
    f.call('start_episode', {})
    state['calls'].clear()  # Starting the episode observes too.
    yield f, state
    f.close()


def agent(frontend, name='claude-code'):
    client = {}
    frontend.handle(initialize(name), client)
    return client


def test_agent_observe_lists_objects_with_ids(scene):
    f, state = scene
    result = f.handle(tool('robodojo_observe'), agent(f))['result']
    packet = packet_of(result)
    assert state['calls'] == [{'robot_spheres': True}]
    assert [(o['id'], o['label']) for o in packet['objects']] == [('obj1', 'green · 8x5x3 cm'),
                                                                   ('obj2', 'red · 6x4x5 cm')]
    assert packet['object_notes'] == ['obj1 is new', 'obj2 is new']
    assert packet['robot_spheres']['count'] == 1 and 'link6' in packet['eef_note']
    assert len(result['content']) == 3  # Text and two RGB images; no depth.


def test_ids_survive_a_small_move(scene):
    f, state = scene
    client = agent(f)
    f.handle(tool('robodojo_observe'), client)
    state['boxes'] = [box((0.05, 0.02), (0.06, 0.04, 0.05), 30.0, (200, 30, 30)), GREEN]
    packet = packet_of(f.handle(tool('robodojo_observe'), client)['result'])
    red = next(o for o in packet['objects'] if o['label'].startswith('red'))
    assert red['id'] == 'obj2'
    assert len(packet['object_notes']) == 1 and packet['object_notes'][0].startswith('obj2 moved 2.')


def test_python_gets_no_objects(scene):
    f, state = scene
    result = f.handle(tool('robodojo_observe'), agent(f, PYTHON_CLIENT))['result']
    packet = packet_of(result)
    assert state['calls'] == [{}] and 'objects' not in packet and 'eef_note' not in packet
    assert len(result['content']) == 5


def test_failed_sphere_call_is_retried_without(scene):
    f, state = scene
    state['refuse_spheres'] = True
    packet = packet_of(f.handle(tool('robodojo_observe'), agent(f))['result'])
    assert state['calls'] == [{'robot_spheres': True}, {}]
    assert len(packet['objects']) == 2
    assert any('without robot spheres' in note for note in packet['object_notes'])


def test_perception_error_is_a_note(scene):
    f, state = scene
    state['fields'] = {'with_depth': False}
    result = f.handle(tool('robodojo_observe'), agent(f))['result']
    packet = packet_of(result)
    assert not result.get('isError') and 'objects' not in packet and packet['objects_error']


def test_motion_reply_has_the_hint(scene):
    f, _ = scene
    packet = packet_of(f.handle(tool('robodojo_step', {'actions': [[0] * 14]}), agent(f))['result'])
    assert 'robodojo_observe' in packet['objects_hint']


def test_formal_copy_gets_objects():
    from services.agent_formal.runner import FormalMCP
    from services.mcp_contract import Contract

    class Gateway:
        contract = Contract()
        terminal = False
        seen = []

        def handle(self, token, request):
            if request.get('method') != 'tools/call':
                return {'jsonrpc': '2.0', 'id': request.get('id'), 'result': {}}
            self.seen.append(request['params']['arguments'])
            packet, images = scene_reply([RED, GREEN], spheres=request['params']['arguments'].get('robot_spheres'))
            return {'jsonrpc': '2.0', 'id': request.get('id'),
                    'result': {'content': [{'type': 'text', 'text': json.dumps(packet)}, *images]}}

    endpoint = FormalMCP(Gateway(), 'token', info={})
    client = {}
    endpoint.handle(initialize('claude-code'), client)
    response = endpoint.handle(tool('robodojo_observe'), client)
    assert response['id'] == 1 and len(packet_of(response['result'])['objects']) == 2
    assert Gateway.seen[-1] == {'robot_spheres': True}


def test_two_environments_keep_two_trackers():
    client = {'name': 'claude-code'}
    scenes = {0: [RED, GREEN], 1: [GREEN]}

    def call_for(env_id):
        def call(arguments):
            packet, images = scene_reply(scenes[env_id], spheres=arguments.get('robot_spheres'))
            return {'content': [{'type': 'text', 'text': json.dumps(packet)}, *images]}
        return call
    for env_id in (0, 1):
        result = for_agent('robodojo_observe', {'env_id': env_id}, call_for(env_id), client,
                           bench=BENCHMARKS['robodojo'], geometry=True)
        assert packet_of(result)['objects'][0]['id'] == 'obj1'
    assert set(client['trackers']) == {0, 1}
