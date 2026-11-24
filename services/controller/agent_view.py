"""A robot tool reply as the agent CLI sees it: depth and robot spheres reach Python only.

Depth images are uint16 millimetre PNGs, near-white to a vision model, and robot spheres
are thousands of numbers; both are for Python (`ctx.call` with the point-cloud helper),
whose connection names itself PYTHON_CLIENT. Every other connection gets the RGB images,
the JSON without depth, and a sphere count with a note saying where the numbers are.
"""
import copy
import json

PYTHON_CLIENT = 'python-controller'  # clientInfo.name of api/runtime.py's Context.

DEPTH_NOTE = ('Depth images go to Python only: point_cloud.observation(ctx.call(...)) reads them. '
              'Use the point-cloud helper for 3-D geometry.')
SPHERES_NOTE = ('Robot spheres go to Python only: call this tool with ctx.call(...) and pass the '
                'reply to the point-cloud or grasp helpers.')


def _spheres(value):
    """Robot spheres as a count; any other value walked for nested spheres."""
    if isinstance(value, dict):
        return {key: _count(item) if key == 'robot_spheres' else _spheres(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_spheres(item) for item in value]
    return value


def _count(spheres):
    if isinstance(spheres, list):
        return {'count': len(spheres), 'note': SPHERES_NOTE}
    if isinstance(spheres, dict) and isinstance(spheres.get('spheres_m'), list):
        kept = {key: item for key, item in spheres.items() if key in ('step_id', 'frame', 'counts')}
        return {'count': len(spheres['spheres_m']), **kept, 'note': SPHERES_NOTE}
    return spheres


def agent_view(result):
    """The agent CLI's copy of an MCP tool result; `result` itself is left as it is."""
    content = result.get('content') if isinstance(result, dict) else None
    if not isinstance(content, list) or not content or content[0].get('type') != 'text':
        return result
    try:
        packet = json.loads(content[0]['text'])
    except (TypeError, ValueError):
        return result
    if not isinstance(packet, dict):
        return result
    attachments = packet.get('attachments') if isinstance(packet.get('attachments'), list) else []
    depth = {a['content_index'] for a in attachments
             if isinstance(a, dict) and a.get('kind') == 'depth' and isinstance(a.get('content_index'), int)}
    if not depth and 'robot_spheres' not in json.dumps(packet):
        return result
    view = _spheres(copy.deepcopy(packet))
    blocks = [content[0]]
    moved = {}
    for index, item in enumerate(content[1:], start=1):
        if index in depth:
            continue
        moved[index] = len(blocks)
        blocks.append(item)
    if depth:
        view['attachments'] = [{**a, 'content_index': moved[a['content_index']]}
                               if isinstance(a, dict) and a.get('content_index') in moved else a
                               for a in view['attachments']
                               if not (isinstance(a, dict) and a.get('kind') == 'depth')]
        view['depth'] = DEPTH_NOTE
    blocks[0] = {**content[0], 'text': json.dumps(view, allow_nan=False)}
    return {**result, 'content': blocks}
