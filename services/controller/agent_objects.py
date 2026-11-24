"""The agent CLI's robot calls: its own observe lists the objects with ids, as AMP's does.

Every connection but Python's (agent_view.PYTHON_CLIENT) is the agent. On a benchmark whose
depth reaches Python only, its robodojo_observe runs with robot spheres; the full reply is
decoded with the point-cloud skill's own helper, objects() lists what is on the table, and
an ObjectTracker per connection and environment keeps their ids. agent_view then removes
depth and spheres. Python and bundles get whole replies and find objects themselves.
"""
import importlib.util
import json
from pathlib import Path

from .agent_view import PYTHON_CLIENT, agent_view
from .object_tracking import ObjectTracker

PROJECT = Path(__file__).resolve().parents[2]
POINT_CLOUD = PROJECT/'agent_skills/perception/depth-robodojo/point-cloud/point_cloud.py'
EEF_NOTE = ('eef_positions and eef_quaternions_wxyz are link6, about 15 cm behind tcp_positions, '
            'the point between the finger pads.')
OBJECTS_HINT = ('For the objects after this motion, call robodojo_observe: its objects list gives '
                "each object's id, centre, size and yaw.")
NO_SPHERES_NOTE = ('Observed without robot spheres (they were unavailable): objects within 5 cm of '
                   'a link6 position may be missing.')
_point_cloud = None


def point_cloud():
    """The point-cloud skill's helper, the same file agents copy into their projects."""
    global _point_cloud
    if _point_cloud is None:
        spec = importlib.util.spec_from_file_location('rax_point_cloud_skill', POINT_CLOUD)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _point_cloud = module
    return _point_cloud


def _packet(result):
    content = result.get('content') if isinstance(result, dict) else None
    if result.get('isError') or not content or content[0].get('type') != 'text':
        return None
    try:
        packet = json.loads(content[0]['text'])
    except (TypeError, ValueError):
        return None
    return packet if isinstance(packet, dict) else None


def _with(result, packet):
    return {**result, 'content': [{**result['content'][0], 'text': json.dumps(packet, allow_nan=False)},
                                  *result['content'][1:]]}


def with_objects(result, tracker, notes=()):
    """`result` with this observation's objects, their ids and what changed."""
    packet = _packet(result)
    if packet is None:
        return result
    try:
        pc = point_cloud()
        items = pc.objects(pc.observation(result))
        states = packet.get('states') or []
        openings = [states[6], states[13]] if len(states) >= 14 else [None, None]
        rows, changes = tracker.update(items, episode_id=packet.get('episode_id'),
                                       tcp=packet.get('tcp_positions') or [None, None],
                                       openings=openings, commands=packet.get('gripper_commands') or [None, None])
        packet['objects'], packet['object_notes'] = rows, [*notes, *changes]
    except Exception as exc:  # noqa: BLE001 - perception never fails the call
        packet['objects_error'] = f'{type(exc).__name__}: {exc}'
    return _with(result, packet)


def _note(result, **fields):
    packet = _packet(result)
    return result if packet is None else _with(result, {**packet, **fields})


def _failed(call, arguments):
    try:
        result = call(arguments)
    except Exception:  # noqa: BLE001 - the plain call follows
        return None
    return None if not isinstance(result, dict) or result.get('isError') else result


def for_agent(name, arguments, call, client, *, bench, geometry):
    """The reply to one tool call over a connection; `call(arguments)` makes the call.

    `client` is the connection's state (None for internal callers: whole replies)."""
    if client is None or client.get('name') == PYTHON_CLIENT or not bench.depth_in_python_only:
        return call(arguments)
    if name == bench.observe_tool and geometry:
        notes = []
        result = _failed(call, {**arguments, 'robot_spheres': True})
        if result is None:
            result = call(arguments)
            if not arguments.get('robot_spheres'):
                notes.append(NO_SPHERES_NOTE)
        tracker = client.setdefault('trackers', {}).setdefault(arguments.get('env_id'), ObjectTracker())
        result = with_objects(result, tracker, notes)
    else:
        result = call(arguments)
        if name in bench.mutating and geometry:
            result = _note(result, objects_hint=OBJECTS_HINT)
    packet = _packet(result)
    if packet is not None and 'eef_positions' in packet:
        result = _with(result, {**packet, 'eef_note': EEF_NOTE})
    return agent_view(result)
