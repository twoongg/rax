"""Object ids across an episode's observations: AMP's identity and held rules."""
import pytest

from services.controller.object_tracking import ObjectTracker

RED, BLUE = (200, 30, 30), (30, 60, 200)
OPEN = dict(openings=[1.0, 1.0], commands=[1.0, 1.0])
LEFT_HOLDS = dict(openings=[0.5, 1.0], commands=[0.0, 1.0])
FAR = [-0.4, -0.4, 0.4]


def item(x, y, size=(0.06, 0.04, 0.05), color=RED, yaw=0.0):
    """A point_cloud.objects() row of an upright box standing on a table at z = 0."""
    sx, sy, h = size
    long_side, short_side = max(sx, sy), min(sx, sy)
    return {'centroid': [x, y, h / 2], 'size': [sx, sy, h], 'top': [x, y, h], 'top_z': h, 'bottom_z': 0.005,
            'table_z': 0.0, 'color': list(color), 'label': f'box {sx}x{sy}', 'n_points': 400,
            'obb': {'center': [x, y, h / 2], 'size': [long_side, short_side, h], 'yaw_deg': yaw}}


def ids(rows):
    return [row['id'] for row in rows]


def test_rows_carry_the_summary_a_reader_needs():
    rows, notes = ObjectTracker().update([item(0.1, 0.2)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    assert rows == [{'id': 'obj1', 'label': 'box 0.06x0.04', 'centre_m': [0.1, 0.2, 0.05], 'top_z_m': 0.05,
                     'size_m': [0.06, 0.04, 0.05], 'yaw_deg': 0.0, 'n_points': 400}]
    assert notes == ['obj1 is new']


def test_unmoved_object_keeps_its_id():
    tracker = ObjectTracker()
    scene = [item(0.0, 0.0), item(0.2, 0.0, size=(0.08, 0.05, 0.03), color=BLUE)]
    tracker.update(scene, episode_id='e', tcp=[FAR, FAR], **OPEN)
    rows, notes = tracker.update(list(reversed(scene)), episode_id='e', tcp=[FAR, FAR], **OPEN)
    assert ids(rows) == ['obj2', 'obj1'] and notes == []


def test_small_move_keeps_id_and_notes_it():
    tracker = ObjectTracker()
    tracker.update([item(0.0, 0.0)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    rows, notes = tracker.update([item(0.03, 0.0)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    assert ids(rows) == ['obj1'] and notes == ['obj1 moved 3.0 cm']


def test_big_move_without_hold_is_a_new_object():
    tracker = ObjectTracker()
    tracker.update([item(0.0, 0.0)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    rows, notes = tracker.update([item(0.10, 0.0)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    assert ids(rows) == ['obj2']
    assert 'obj2 is new' in notes and any(note.startswith('obj1 not seen') for note in notes)


def test_held_object_found_again_far_away_and_turned():
    tracker = ObjectTracker()
    plug = (0.08, 0.03, 0.04)
    # The last look before the grasp: the TCP still 12 cm above the plug.
    tracker.update([item(0.0, 0.0, size=plug)], episode_id='e', tcp=[[0.0, 0.0, 0.16], FAR], **OPEN)
    rows, notes = tracker.update([], episode_id='e', tcp=[[0.0, 0.0, 0.2], FAR], **LEFT_HOLDS)
    assert rows == [] and notes == ['obj1 is held by the left gripper']
    turned = item(0.25, 0.1, size=(0.03, 0.04, 0.08), yaw=90.0)  # set down upright, 27 cm away
    rows, notes = tracker.update([turned], episode_id='e', tcp=[[0.25, 0.1, 0.25], FAR], **OPEN)
    assert ids(rows) == ['obj1']
    assert notes == ['obj1 found again 27.0 cm from where it was last seen']


def test_other_colour_does_not_take_an_id():
    tracker = ObjectTracker()
    tracker.update([item(0.0, 0.0)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    rows, _ = tracker.update([item(0.0, 0.0, color=BLUE)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    assert ids(rows) == ['obj2']


def test_twins_keep_their_ids():
    tracker = ObjectTracker()
    tracker.update([item(0.0, 0.0), item(0.08, 0.0)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    rows, _ = tracker.update([item(0.09, 0.01), item(0.01, 0.01)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    assert ids(rows) == ['obj2', 'obj1']


def test_occluded_object_returns_in_place():
    tracker = ObjectTracker()
    tracker.update([item(0.0, 0.0)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    _, notes = tracker.update([], episode_id='e', tcp=[FAR, FAR], **OPEN)
    assert notes == ['obj1 not seen; last at [0.0, 0.0, 0.025]']
    rows, notes = tracker.update([item(0.01, 0.0)], episode_id='e', tcp=[FAR, FAR], **OPEN)
    assert ids(rows) == ['obj1'] and notes == ['obj1 found again 1.0 cm from where it was last seen']


def test_new_episode_restarts_ids():
    tracker = ObjectTracker()
    tracker.update([item(0.0, 0.0), item(0.2, 0.0)], episode_id='e1', tcp=[FAR, FAR], **OPEN)
    rows, notes = tracker.update([item(0.3, 0.3)], episode_id='e2', tcp=[FAR, FAR], **OPEN)
    assert ids(rows) == ['obj1'] and notes == ['obj1 is new']


@pytest.mark.parametrize('openings,commands', [([0.02, 1.0], [0.0, 1.0]), ([0.5, 1.0], [1.0, 1.0])])
def test_an_empty_or_opened_gripper_holds_nothing(openings, commands):
    tracker = ObjectTracker()
    tracker.update([item(0.0, 0.0)], episode_id='e', tcp=[[0.0, 0.0, 0.1], FAR], **OPEN)
    _, notes = tracker.update([], episode_id='e', tcp=[[0.0, 0.0, 0.1], FAR], openings=openings, commands=commands)
    assert notes == ['obj1 not seen; last at [0.0, 0.0, 0.025]']
