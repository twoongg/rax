"""Object ids across one episode's observations, for the agent's own robodojo_observe.

A port of the rules of AMP's perception identity (src/amp/perception/identity.py) and
held-object inference (held.py), with AMP's values. Each observation's objects (rows of
the point-cloud skill's objects()) are matched one to one with the previous ones by
centroid distance, colour and size. An object that vanishes while a gripper holds
something near it is held; when an object appears again, a held or released one may
match at any distance, by colour and size. Others that vanish are remembered and may come
back where they were. Ids start again at obj1 in each episode.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

MATCH_M = 0.05             # AMP match_dist: farther than this is not the same object in place.
COLOR_MAX = 0.3            # AMP match_color, in the shading-tolerant colour feature below.
INTENSITY_WEIGHT = 0.1     # AMP intensity_weight (and the point-cloud skill's).
HELD_CLAIM_M = 0.15        # A holding arm claims the vanished object this near its TCP (AMP: 0.06
                           # with every frame perceived; RAX agents look less often).
EMPTY_OPENING, OPEN_OPENING = 0.03, 0.97   # grasp.py's grip(): closed on nothing / not closed.
CLOSED_COMMAND = 0.5       # A gripper command at or below this is a close.
MOVED_M = 0.01             # Smaller moves go unmentioned.
ARMS = ('left', 'right')
IMPOSSIBLE = 1e6


def _color_feature(rgb):
    c = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0
    i = c.mean(axis=1, keepdims=True)
    return np.hstack([(c - i) / (i + 0.03), INTENSITY_WEIGHT * i])[0]


def signature(item):
    size = np.asarray(item['size'], dtype=np.float64)
    return {'c': np.asarray(item['centroid'], dtype=np.float64), 'ext': np.sort(size[:2]),
            'height': float(size[2]), 'color': _color_feature(item['color'])}


def _cost(new, old, *, anywhere):
    """AMP's match cost of two signatures, or IMPOSSIBLE."""
    dcol = float(np.linalg.norm(new['color'] - old['color']))
    de = float(np.abs(new['ext'] - old['ext']).sum() + abs(new['height'] - old['height']))
    tol = max(0.03, 0.35 * float(max(new['ext'].max(), old['ext'].max())))
    d = float(np.linalg.norm(new['c'] - old['c']))
    if dcol > COLOR_MAX or de > (3 if anywhere else 2) * tol or (not anywhere and d > MATCH_M):
        return IMPOSSIBLE
    return (0.0 if anywhere else d / MATCH_M) + dcol / COLOR_MAX + de / tol


def _match(sigs, todo, candidates):
    """Hungarian match of sigs[todo] to candidates [(id, sig, anywhere)]: {index: id}."""
    if not todo or not candidates:
        return {}
    cost = np.array([[_cost(sigs[i], sig, anywhere=anywhere) for _, sig, anywhere in candidates]
                     for i in todo])
    rows, cols = linear_sum_assignment(cost)
    return {todo[r]: candidates[c][0] for r, c in zip(rows, cols) if cost[r, c] < IMPOSSIBLE}


def _holding(openings, commands):
    """The arms whose gripper was commanded closed and stopped on something."""
    return [arm for arm, opening, command in zip(ARMS, openings or (), commands or ())
            if opening is not None and command is not None
            and float(command) <= CLOSED_COMMAND and EMPTY_OPENING < float(opening) < OPEN_OPENING]


def _cm(distance):
    return f'{100 * distance:.1f} cm'


def row(oid, item):
    obb = item['obb']
    table_z = item.get('table_z') or 0.0
    return {'id': oid, 'label': item['label'], 'centre_m': [round(float(v), 3) for v in item['top']],
            'top_z_m': round(float(item['top_z']), 3),
            'size_m': [round(float(obb['size'][0]), 3), round(float(obb['size'][1]), 3),
                       round(float(item['top_z']) - float(table_z), 3)],
            'yaw_deg': round(float(obb['yaw_deg']), 1), 'n_points': int(item['n_points'])}


class ObjectTracker:
    """The ids of one connection's view of one environment."""

    def __init__(self):
        self.reset(None)

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.visible = {}    # id -> signature, from the last observation
        self.memory = {}     # id -> {'sig', 'arm' (holding it) or None, 'anywhere'}
        self.tcp = [None, None]
        self.next_id = 1

    def update(self, items, *, episode_id, tcp, openings, commands):
        """(rows, notes) for this observation's objects."""
        if episode_id != self.episode_id:
            self.reset(episode_id)
        sigs = [signature(item) for item in items]
        notes = []
        ids = _match(sigs, list(range(len(sigs))), [(oid, sig, False) for oid, sig in self.visible.items()])
        rest = [i for i in range(len(sigs)) if i not in ids]
        found = _match(sigs, rest, [(oid, entry['sig'], entry['anywhere'])
                                    for oid, entry in self.memory.items()])
        ids.update(found)
        out = []
        for i, item in enumerate(items):
            if i in ids:
                oid = ids[i]
                if i in found:
                    distance = float(np.linalg.norm(sigs[i]['c'] - self.memory.pop(oid)['sig']['c']))
                    notes.append(f'{oid} found again {_cm(distance)} from where it was last seen')
                else:
                    distance = float(np.linalg.norm(sigs[i]['c'] - self.visible[oid]['c']))
                    if distance >= MOVED_M:
                        notes.append(f'{oid} moved {_cm(distance)}')
            else:
                oid = f'obj{self.next_id}'
                self.next_id += 1
                notes.append(f'{oid} is new')
            out.append((oid, item, sigs[i]))
        notes += self._vanished({oid for oid, _, _ in out}, tcp, _holding(openings, commands))
        self.visible = {oid: sig for oid, _, sig in out}
        self.tcp = list(tcp)
        return [row(oid, item) for oid, item, _ in out], notes

    def _vanished(self, seen, tcp, holding):
        notes = []
        gone = {oid: sig for oid, sig in self.visible.items() if oid not in seen}
        for arm in holding:
            if any(entry['arm'] == arm for entry in self.memory.values()):
                continue  # Already holding something it claimed.
            index = ARMS.index(arm)
            points = [np.asarray(p, dtype=np.float64) for p in (self.tcp[index], tcp[index]) if p is not None]
            near = sorted((min(float(np.linalg.norm(sig['c'] - p)) for p in points), oid)
                          for oid, sig in gone.items() if points)
            if near and near[0][0] <= HELD_CLAIM_M:
                oid = near[0][1]
                self.memory[oid] = {'sig': gone.pop(oid), 'arm': arm, 'anywhere': True}
        for oid, sig in gone.items():
            self.memory[oid] = {'sig': sig, 'arm': None, 'anywhere': False}
        for oid, entry in sorted(self.memory.items(), key=lambda pair: int(pair[0][3:])):
            if entry['arm'] is not None and entry['arm'] not in holding:
                entry['arm'] = None  # Released: it may turn up anywhere.
            if entry['arm'] is not None:
                notes.append(f"{oid} is held by the {entry['arm']} gripper")
            else:
                last = [round(float(v), 3) for v in entry['sig']['c']]
                notes.append(f'{oid} not seen; last at {last}')
        return notes
