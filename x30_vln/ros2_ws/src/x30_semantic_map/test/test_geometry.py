"""Relations, rooms, frontiers, summary, find ranking and the annotated map renderer (no ROS)."""
import math

import numpy as np
import pytest
from helpers import det, feed, make_map

from x30_semantic_map import annotated_map, finder, relations, rooms, summary
from x30_semantic_map import synthetic as syn
from x30_semantic_map.config import load_config
from x30_semantic_map.gridtools import GridView, find_frontiers
from x30_semantic_map.semantic_map import SemanticMap


@pytest.fixture(scope='module')
def world():
    sc = syn.generate(syn.ScenarioConfig(seed=0))
    m = SemanticMap(load_config())
    m.los_fn = lambda a, b: sc.grid.line_of_sight(a, b, ignore_end_m=0.45)
    for f in sc.frames:
        m.process_frame(f)
    rooms.update_rooms(m, sc.grid, m.cfg.rooms)
    return m, sc


def _scene():
    m = make_map()
    items = [('table', (2, 2, 0.375), (1.2, 0.8, 0.75)), ('bottle', (2.1, 2.0, 0.88), (0.08, 0.08, 0.26)),
             ('chair', (2.0, 3.2, 0.45), (0.45, 0.45, 0.9)), ('chair', (4.0, 2.0, 0.45), (0.45, 0.45, 0.9)),
             ('couch', (2.0, 0.2, 0.4), (1.8, 0.8, 0.8)), ('tv', (7.0, 7.0, 0.4), (0.1, 1.0, 0.6))]
    feed(m, lambda i: [det(l, p, e) for l, p, e in items], n=8)
    return m


def by_label(m, label, k=0):
    return [o for o in m.visible_objects() if m.label(o) == label][k]


def test_on_under_near_relations():
    m = _scene()
    cfg = m.cfg.relations
    bottle, table, far = by_label(m, 'bottle'), by_label(m, 'table'), by_label(m, 'tv')
    assert 'on' in relations.pair_relations(bottle, table, cfg)
    assert 'under' in relations.pair_relations(table, bottle, cfg)
    assert relations.pair_relations(far, table, cfg) == []
    chair = by_label(m, 'chair', 0)
    assert 'near' in relations.pair_relations(chair, table, cfg)


def test_left_right_in_front_behind_from_a_viewer_pose():
    m = _scene()
    cfg = m.cfg.relations
    table = by_label(m, 'table')
    viewer = (2.0, -2.0, math.pi / 2)                       # standing south of the table, looking north (+y)
    chair_north, chair_east = by_label(m, 'chair', 0), by_label(m, 'chair', 1)
    assert relations.viewer_relation(chair_north.position, table.position, viewer, cfg) == 'behind'
    assert relations.viewer_relation(chair_east.position, table.position, viewer, cfg) == 'right of'
    assert relations.viewer_relation(np.array([0.5, 2.0, 0.4]), table.position, viewer, cfg) == 'left of'
    assert relations.viewer_relation(np.array([2.0, 0.5, 0.4]), table.position, viewer, cfg) == 'in front of'
    ego = relations.egocentric((0, 0, 0), np.array([1.0, 1.0, 0]), cfg)
    assert ego['direction'] == 'in front' and ego['bearing_deg'] == 45.0
    assert relations.egocentric((0, 0, 0), np.array([-1.0, 0.1, 0]), cfg)['direction'] == 'behind'
    assert relations.egocentric((0, 0, 0), np.array([0.0, 1.0, 0]), cfg)['direction'] == 'left'


def test_between():
    m = make_map()
    feed(m, lambda i: [det('chair', (0, 0, 0.45)), det('table', (4, 0, 0.4), (1, 1, 0.8)), det('couch', (2, 0.1, 0.4), (1, 0.6, 0.8))], n=6)
    a, b, c = (by_label(m, k) for k in ('couch', 'chair', 'table'))
    assert relations.between(a, b, c, m.cfg.relations)
    assert not relations.between(b, a, c, m.cfg.relations)
    out = relations.object_relations(m, a.id, (0, -3, math.pi / 2))
    assert out['between'] and 'between' in out['text']
    assert out['from_robot']['sector'] in ('ahead', 'ahead-left', 'ahead-right')


def test_rooms_segment_the_floorplan_and_keep_ids(world):
    m, sc = world
    names = sorted(r.name for r in m.regions.values())
    assert names == ['corridor', 'kitchen', 'living room']
    ids = {r.name: r.id for r in m.regions.values()}
    rooms.update_rooms(m, sc.grid, m.cfg.rooms)                  # second pass: same ids and names
    assert {r.name: r.id for r in m.regions.values()} == ids
    fridge = next(o for o in m.visible_objects() if m.canon(o) == 'refrigerator')
    assert m.room_name_of(fridge) == 'kitchen'
    assert m.room_name_of(next(o for o in m.visible_objects() if m.canon(o) == 'couch')) == 'living room'


def test_frontiers_found_where_the_map_is_open():
    data = -np.ones((100, 100), np.int8)
    data[20:80, 20:80] = 0
    data[20:22, 20:80] = 100
    data[78:80, 20:80] = 100
    data[20:80, 20:22] = 100
    data[40:60, 78:80] = 0              # opening on the east wall
    g = GridView(data, 0.1, 0.0, 0.0)
    fr = find_frontiers(g, (4.0, 4.0), load_config().frontiers)
    assert len(fr) >= 1
    assert abs(fr[0].center[0] - 7.9) < 0.5 and 4.0 <= fr[0].center[1] <= 6.0
    assert fr[0].reachable


def test_summary_is_compact_ordered_and_respects_the_budget(world):
    m, sc = world
    pose = (1.0, 1.0, 0.0)
    out = summary.build_summary(m, pose, sc.grid)
    text = out['text']
    assert 'Rooms:' in text and 'kitchen' in text and 'Unexplored' in text
    dists = [o['range_m'] for o in out['objects']]
    assert dists == sorted(dists)
    small = summary.build_summary(m, pose, sc.grid, token_budget=200)
    assert small['tokens_est'] <= 200 and small['truncated'] and small['objects_shown'] < out['objects_shown']
    assert 'omitted' in small['text']
    assert summary.build_summary(m, None, None)['pose_known'] is False


def test_summary_of_an_empty_map():
    out = summary.build_summary(make_map(), (0, 0, 0), None)
    assert 'No objects mapped yet' in out['text'] and out['objects_total'] == 0


def test_find_ranks_by_label_colour_relation_and_room(world):
    m, sc = world
    pose = (1.0, 1.0, 0.0)
    r = finder.rank(m, 'the red chair near the table', pose)
    top = r['matches'][0]
    assert top['label'] == 'chair' and 'red' in top['attributes']
    assert 0.0 <= top['score'] <= 1.0 and top['reasons']
    in_kitchen = finder.rank(m, 'fire extinguisher in the kitchen', pose)['matches'][0]
    in_living = finder.rank(m, 'fire extinguisher in the living room', pose)['matches'][0]
    assert in_kitchen['room'] == 'kitchen' and 'in the kitchen' in in_kitchen['reasons']
    assert in_kitchen['raw_score'] > in_living['raw_score']          # the wrong room is penalised
    r = finder.rank(m, 'a sofa', pose)
    assert r['matches'][0]['label'] == 'couch'                   # synonym
    assert finder.rank(m, 'a spaceship', pose)['matches'] == []
    r = finder.rank(m, 'the nearest chair', pose)
    chairs = [o for o in m.visible_objects() if m.canon(o) == 'chair']
    nearest = min(chairs, key=lambda o: math.hypot(o.position[0] - 1.0, o.position[1] - 1.0))
    assert r['matches'][0]['id'] == nearest.id
    p = finder.parse_request('the small bottle on the table', m.vocab.known_labels())
    assert p.head.label == 'bottle' and p.head.size == 'small' and p.relations[0][0] == 'on'


def test_annotated_map_image_and_text(world, tmp_path):
    m, sc = world
    r = annotated_map.MapRenderer(m.cfg.render)
    pose = sc.robot_poses[-1]
    traj = [(p[0], p[1]) for p in sc.robot_poses]
    img = r.render(m, sc.grid, pose, traj)
    assert img.dtype == np.uint8 and img.shape[2] == 3 and max(img.shape[:2]) <= m.cfg.render.max_size_px + 2
    assert (img == annotated_map.OCC_RGB).all(axis=2).sum() > 1000         # obstacles drawn
    assert (img == annotated_map.ROBOT_RGB).all(axis=2).sum() > 20         # robot arrow drawn
    ego = r.render(m, sc.grid, pose, traj, egocentric=True)
    assert ego.shape != img.shape or not np.array_equal(ego, img)
    png = annotated_map.to_png(img)
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    (tmp_path / 'asm.png').write_bytes(png)
    txt = r.render_text(m, sc.grid, pose, traj)
    assert 'Legend:' in txt and '#' in txt and 'refrigerator' in txt and 'kitchen' in txt
    assert len(max(txt.splitlines(), key=len)) < 400
    assert r.render_text(make_map(), None, None) == '(empty map)'
    assert r.render(make_map(), None, None, None).shape[2] == 3        # empty map renders too
