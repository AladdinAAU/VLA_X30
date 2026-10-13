"""Unit tests of the semantic map core (no ROS, no network)."""
import json
import math

import numpy as np
import pytest
from helpers import det, feed, frame, make_map, red_crop

from x30_semantic_map import attributes as attr
from x30_semantic_map import synthetic as syn
from x30_semantic_map.config import config_from_dict, load_config
from x30_semantic_map.datatypes import ViewGeometry
from x30_semantic_map.semantic_map import SemanticMap


def test_config_file_loads_and_is_strict():
    cfg = load_config()
    assert 'person' in cfg.categories and cfg.categories['person'].dynamic
    assert cfg.thermal.enabled is False                    # no thermal camera on the stock X30
    with pytest.raises(ValueError, match='unknown key'):
        config_from_dict({'association': {'gate_mim_m': 1.0}})


def test_noisy_duplicates_become_one_object_with_small_error():
    m = make_map()
    rng = np.random.default_rng(1)
    truth = np.array([2.0, 1.5, 0.45])
    feed(m, lambda i: [det(pos=truth + rng.normal(0, 0.08, 3), points=int(rng.integers(30, 300)))], n=30)
    objs = m.visible_objects()
    assert len(objs) == 1
    assert np.linalg.norm(objs[0].position - truth) < 0.06
    assert m.status(objs[0]) == 'confirmed'


def test_two_chairs_close_together_stay_separate():
    m = make_map()
    rng = np.random.default_rng(2)
    a, b = np.array([1.0, 1.0, 0.45]), np.array([1.9, 1.0, 0.45])
    feed(m, lambda i: [det(pos=a + rng.normal(0, 0.03, 3)), det(pos=b + rng.normal(0, 0.03, 3))], n=20)
    assert len(m.visible_objects()) == 2


def test_association_is_one_to_one_per_frame():
    m = make_map()
    feed(m, lambda i: [det(pos=(1, 1, 0.45)), det(pos=(1.3, 1.0, 0.45))], n=12)
    assert len(m.objects) == 2


def test_synonym_labels_vote_for_one_instance():
    m = make_map()
    feed(m, lambda i: [det(label='chair' if i % 3 else 'armchair')], n=15)
    (o,) = m.visible_objects()
    assert m.label(o) == 'chair' and m.canon(o) == 'chair'


def test_cross_category_label_on_same_place_votes_not_duplicates():
    m = make_map()
    feed(m, lambda i: [det(label='bottle', pos=(1, 1, 0.9), ext=(0.08, 0.08, 0.25)),
                       det(label='cup', pos=(1.02, 1.0, 0.9), ext=(0.08, 0.08, 0.25), score=0.4)], n=10)
    assert len(m.objects) == 1


def test_per_category_confirmation_thresholds():
    m = make_map()
    feed(m, lambda i: [det(label='person', pos=(3, 3, 0.9), ext=(0.5, 0.5, 1.7)),
                       det(label='chair', pos=(1, 1, 0.45))], n=2)
    status = {m.label(o): m.status(o) for o in m.objects.values()}
    assert status == {'person': 'confirmed', 'chair': 'tentative'}      # person needs 2 observations, chair 4
    feed(m, lambda i: [det(label='chair', pos=(1, 1, 0.45))], n=6, t0=1.0)
    assert m.status(next(o for o in m.objects.values() if m.label(o) == 'chair')) == 'confirmed'


def test_low_score_object_is_never_confirmed():
    m = make_map()
    feed(m, lambda i: [det(score=0.1)], n=20)
    assert m.visible_objects() == []


def test_transient_false_positive_expires_and_is_remembered():
    m = make_map()
    m.process_frame(frame(0, [det(label='bottle', pos=(2, 2, 0.3), ext=(0.1, 0.1, 0.25), score=0.4)]))
    assert len(m.objects) == 1
    m.housekeeping(60.0)
    assert len(m.objects) == 0 and len(m.rejected_places) == 1
    # reappearing at the same place needs more observations
    feed(m, lambda i: [det(label='bottle', pos=(2.05, 2, 0.3), ext=(0.1, 0.1, 0.25), score=0.5)], n=5, t0=61.0)
    assert m.visible_objects() == []
    feed(m, lambda i: [det(label='bottle', pos=(2.05, 2, 0.3), ext=(0.1, 0.1, 0.25), score=0.5)], n=8, t0=63.0)
    assert len(m.visible_objects()) == 1


def test_degenerate_detections_are_ignored():
    m = make_map()
    res = m.process_frame(frame(0, [det(ext=(9, 9, 1)), det(label='person', ext=(0.5, 0.5, 0.3))]))
    assert res.ignored == 2 and not m.objects


def test_moving_person_is_tracked_flagged_moving_and_expires():
    m = make_map()
    feed(m, lambda i: [det(label='person', pos=(0.5 + 0.25 * i, 2.0, 0.9), ext=(0.5, 0.5, 1.7))], n=25, dt=0.3)
    persons = [o for o in m.objects.values() if o.dynamic]
    assert len(persons) == 1, 'a walking person must stay one instance'
    p = persons[0]
    assert p.moving and 0.5 < p.velocity[0] < 1.2
    assert m.attribute_dict(p)['motion'] == 'moving'
    m.housekeeping(m.now + 30.0)
    assert not [o for o in m.objects.values() if o.dynamic]


def test_standing_person_is_kept_longer_and_not_moving():
    m = make_map()
    feed(m, lambda i: [det(label='person', pos=(4, 4, 0.9), ext=(0.5, 0.5, 1.7))], n=15)
    p = next(iter(m.objects.values()))
    assert not p.moving
    m.housekeeping(m.now + 60.0)
    assert m.objects, 'standing people are kept for ttl_static_s'


def view_towards(pos, cam=(0, 0, 0.5)):
    cam = np.array(cam, float)
    f = np.array(pos, float) - cam
    return ViewGeometry(cam, f / np.linalg.norm(f), math.radians(100), math.radians(75), 7.0)


def test_object_that_is_no_longer_there_is_removed_by_misses():
    m = make_map()
    pos = (3.0, 0.0, 0.45)
    feed(m, lambda i: [det(pos=pos)], n=10)
    assert len(m.visible_objects()) == 1
    for i in range(20):                                   # camera looks at the spot and sees nothing
        m.process_frame(frame(10 + i * 0.3, [], view=view_towards(pos)))
    assert m.visible_objects() == []


def test_object_out_of_view_is_not_penalised():
    m = make_map()
    feed(m, lambda i: [det(pos=(3, 0, 0.45))], n=10)
    away = ViewGeometry(np.array([0, 0, 0.5]), np.array([-1.0, 0, 0]), math.radians(100), math.radians(75), 7.0)
    for i in range(30):
        m.process_frame(frame(10 + i * 0.3, [], view=away))
    assert len(m.visible_objects()) == 1


def test_occlusion_blocks_miss_evidence():
    m = make_map()
    m.los_fn = lambda a, b: False
    pos = (3.0, 0.0, 0.45)
    feed(m, lambda i: [det(pos=pos)], n=10)
    for i in range(30):
        m.process_frame(frame(10 + i * 0.3, [], view=view_towards(pos)))
    assert len(m.visible_objects()) == 1


# ---------------------------------------------------------------------------------------------- bearing-only
def ray(label, target, cam):
    return det(label, target, bearing_from=cam, score=0.6)


def test_two_rays_from_different_poses_triangulate():
    m = make_map()
    target = (4.0, 2.0, 0.3)
    for i, cam in enumerate([(0, 0, 0.5), (0.1, 0.0, 0.5), (0, 3, 0.5), (0, 3.1, 0.5)]):
        m.process_frame(frame(i * 0.5, [ray('fire extinguisher', target, cam)], cam=cam))
    (o,) = m.visible_objects()
    assert o.bearing_only and m.status(o) == 'bearing'
    assert np.linalg.norm(o.position - np.array(target)) < 0.3


def test_rays_from_one_pose_or_collinear_rays_do_not_triangulate():
    m = make_map()
    target = (6.0, 0.0, 0.3)
    for i in range(6):                                    # moving straight towards the object: no baseline across it
        cam = (i * 0.5, 0.0, 0.3)
        m.process_frame(frame(i * 0.5, [ray('bottle', target, cam)], cam=cam))
    assert m.visible_objects() == []
    m2 = make_map()
    for i in range(6):
        m2.process_frame(frame(i * 0.5, [ray('bottle', target, (0, 0, 0.5))], cam=(0, 0, 0.5)))
    assert m2.visible_objects() == []


def test_disagreeing_rays_do_not_triangulate():
    m = make_map()
    m.process_frame(frame(0, [ray('bottle', (4, 0, 0.3), (0, 0, 0.5))], cam=(0, 0, 0.5)))
    m.process_frame(frame(1, [ray('bottle', (4, 4, 0.3), (0, 3, 0.5))], cam=(0, 3, 0.5)))   # a different bottle
    assert m.visible_objects() == []


def test_rays_of_different_labels_do_not_pair():
    m = make_map()
    m.process_frame(frame(0, [ray('bottle', (4, 2, 0.3), (0, 0, 0.5))], cam=(0, 0, 0.5)))
    m.process_frame(frame(1, [ray('chair', (4, 2, 0.3), (0, 3, 0.5))], cam=(0, 3, 0.5)))
    assert m.visible_objects() == []


def test_depth_detection_upgrades_a_bearing_object():
    m = make_map()
    target = np.array([4.0, 2.0, 0.3])
    for i, cam in enumerate([(0, 0, 0.5), (0.1, 0.0, 0.5), (0, 3, 0.5), (0, 3.1, 0.5)]):
        m.process_frame(frame(i * 0.5, [ray('bottle', target, cam)], cam=cam))
    (b,) = m.visible_objects()
    assert b.bearing_only
    feed(m, lambda i: [det('bottle', target + 0.05, ext=(0.08, 0.08, 0.25))], n=6, t0=3.0)
    objs = m.visible_objects()
    assert len(objs) == 1 and not objs[0].bearing_only and objs[0].id == b.id


def test_ghost_bearing_object_is_absorbed_by_a_confirmed_depth_object():
    m = make_map()
    target = np.array([6.0, 0.0, 0.3])
    # two poses roughly looking along the same line create a near-degenerate, along-the-ray uncertain triangulation
    for i, cam in enumerate([(0, 0, 0.3), (0.4, 0.3, 0.3), (0.0, 2.0, 0.3), (0.0, 2.2, 0.3)]):
        m.process_frame(frame(i * 0.5, [ray('bottle', target, cam)], cam=cam))
    feed(m, lambda i: [det('bottle', target, ext=(0.08, 0.08, 0.25))], n=8, t0=3.0)
    m.housekeeping(m.now + 5.0)
    labels = [(m.label(o), o.bearing_only) for o in m.visible_objects()]
    assert labels == [('bottle', False)]


# ---------------------------------------------------------------------------------------------- attributes
def test_thermal_attributes_off_by_default_and_nan_safe():
    m = make_map()
    feed(m, lambda i: [det(thermal_mean_c=35.0, thermal_max_c=60.0)], n=6)
    assert 'warm' not in m.attribute_list(m.visible_objects()[0])
    m2 = make_map(thermal__enabled=True)
    feed(m2, lambda i: [det(thermal_mean_c=float('nan'), thermal_max_c=float('nan'))], n=6)
    assert 'warm' not in m2.attribute_list(m2.visible_objects()[0])
    m3 = make_map(thermal__enabled=True)
    feed(m3, lambda i: [det(thermal_mean_c=34.0, thermal_max_c=61.0)], n=6)
    attrs = m3.attribute_list(m3.visible_objects()[0])
    assert 'warm' in attrs and 'hot' in attrs


def test_person_like_from_label_and_from_thermal_shape():
    m = make_map(thermal__enabled=True)
    feed(m, lambda i: [det('person', pos=(2, 2, 0.9), ext=(0.5, 0.5, 1.7))], n=5)
    assert 'person-like' in m.attribute_list(m.visible_objects()[0])
    m2 = make_map(thermal__enabled=True)
    feed(m2, lambda i: [det('unknown thing', pos=(2, 2, 0.9), ext=(0.5, 0.5, 1.7), thermal_mean_c=33.0)], n=6)
    assert 'person-like' in m2.attribute_list(m2.visible_objects()[0])


def test_colour_from_crop_and_size_class():
    m = make_map()
    feed(m, lambda i: [det(crop=red_crop(np.random.default_rng(i)), box=(10, 10, 74, 74))], n=8)
    (o,) = m.visible_objects()
    assert m.attribute_dict(o)['colour'] == 'red'
    assert m.attribute_dict(o)['size'] == 'medium'
    for rgb, name in (((20, 20, 220), 'blue'), ((40, 160, 60), 'green'), ((240, 240, 240), 'white'), ((20, 20, 20), 'black')):
        img = np.full((48, 48, 3), rgb, np.uint8)
        assert attr.dominant_colour(attr.encode_jpeg(img)) == name


def test_best_views_are_distinct_and_quality_ordered():
    m = make_map()
    rng = np.random.default_rng(0)
    for i, cam in enumerate([(0, 0, 0.5), (0.05, 0, 0.5), (0, 4, 0.5), (4, 0, 0.5), (-3, 0, 0.5)]):
        m.process_frame(frame(i, [det(crop=red_crop(rng), box=(0, 0, 40 + 20 * i, 40 + 20 * i))], cam=cam))
    (o,) = m.objects.values()
    assert 1 <= len(o.views) <= 3
    assert [v.quality for v in o.views] == sorted((v.quality for v in o.views), reverse=True)


# ---------------------------------------------------------------------------------------------- persistence
def _scenario_map(seed=0, frames=250):
    sc = syn.generate(syn.ScenarioConfig(seed=seed))
    m = SemanticMap(load_config())
    m.los_fn = lambda a, b: sc.grid.line_of_sight(a, b, ignore_end_m=0.45)
    for f in sc.frames[:frames]:
        m.process_frame(f)
    return m, sc


def test_ids_and_state_are_deterministic():
    a, _ = _scenario_map()
    b, _ = _scenario_map()
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)
    ids = sorted(a.objects)
    assert ids == sorted(set(ids)) and all(i < a.next_id for i in ids)


def test_save_load_roundtrip(tmp_path):
    m, _ = _scenario_map()
    path = m.save(tmp_path / 'map.json')
    assert (tmp_path / 'map_crops').is_dir()
    m2 = SemanticMap(load_config())
    m2.load(path)
    assert [m.describe(o) for o in m.visible_objects(True)] == [m2.describe(o) for o in m2.visible_objects(True)]
    assert m2.next_id == m.next_id
    some = next(o for o in m2.objects.values() if o.views)
    assert some.views[0].jpeg[:2] == b'\xff\xd8'
    m2.save(tmp_path / 'again.json')                      # a loaded map saves identically
    assert json.loads((tmp_path / 'again.json').read_text())['objects'] == json.loads(path.read_text())['objects']


def test_feedback_reject_confirm_relabel_merge():
    m = make_map()
    feed(m, lambda i: [det(pos=(1, 1, 0.45)), det(pos=(5, 5, 0.45))], n=10)
    a, b = sorted(m.objects)
    assert m.relabel_object(a, 'table') and m.label(m.objects[a]) == 'table'
    assert m.reject_object(b) and b not in m.objects and m.rejected_places
    assert m.confirm_object(a) and m.status(m.objects[a]) == 'confirmed'
    assert not m.reject_object(999)


def test_query_filters():
    m, sc = _scenario_map(frames=640)
    chairs = m.query(label='chair', limit=20)
    assert chairs and all(m.canon(o) == 'chair' for o in chairs)
    near = m.query(label='chair', near=(3.0, 5.0), radius=2.5)
    assert all(math.hypot(o.position[0] - 3.0, o.position[1] - 5.0) <= 2.5 for o in near)
    assert m.query(label='chair', attributes=['red'])[0].colour == 'red'
    assert m.query(label='spaceship') == []


# ---------------------------------------------------------------------------------------------- scenario quality
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_synthetic_scenario_quality(seed):
    m, sc = _scenario_map(seed, frames=10 ** 6)
    ev = syn.evaluate(m, sc.gt_final)
    assert ev['recall'] >= 0.85, ev
    assert ev['mean_err_m'] < 0.12, ev
    assert ev['duplicates'] == 0, ev
    assert ev['false_positives'] <= 3, ev
    # the chair that was moved half way is at its new place only
    moved = next(g for g in sc.gt_final if g.name == 'chair_k3')
    old = np.array([3.0, 3.9])
    near_old = [o for o in m.visible_objects() if m.canon(o) == 'chair' and np.linalg.norm(o.position[:2] - old) < 0.6]
    assert not near_old, 'the old place of the moved chair must be empty'
    assert any(np.linalg.norm(o.position[:2] - moved.pos[:2]) < 0.5 for o in m.visible_objects() if m.canon(o) == 'chair')
