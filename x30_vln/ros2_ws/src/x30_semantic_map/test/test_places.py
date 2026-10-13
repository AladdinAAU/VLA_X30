import numpy as np
import pytest

from x30_semantic_map import places, rooms
from x30_semantic_map import synthetic as syn
from x30_semantic_map.config import load_config
from x30_semantic_map.semantic_map import SemanticMap
from x30_semantic_map import summary


@pytest.fixture()
def world():
    sc = syn.generate(syn.ScenarioConfig(seed=0))
    m = SemanticMap(load_config())
    m.los_fn = lambda a, b: sc.grid.line_of_sight(a, b, ignore_end_m=0.45)
    for f in sc.frames:
        m.process_frame(f)
    rooms.update_rooms(m, sc.grid, m.cfg.rooms)
    return m, sc


def centre(m, name):
    r = next(r for r in m.regions.values() if r.name == name)
    return float(r.centroid[0]), float(r.centroid[1]), r


def test_room_place_names_the_region_and_keeps_the_name_through_resegmentation(world):
    m, sc = world
    x, y, r = centre(m, 'kitchen')
    out = places.add_place(m, 'Marcus', 'room', x, y, owner='Marcus', utterance="this is Marcus's office")
    assert out['ok'] and out['action'] == 'added' and out['place']['kind'] == 'room'
    assert m.regions[r.id].name == 'Marcus' and m.regions[r.id].name_source == 'user'
    for _ in range(2):                                   # automatic naming and re-segmentation must not overwrite the taught name
        rooms.update_rooms(m, sc.grid, m.cfg.rooms)
        places.apply_places(m)
    assert sorted(rg.name for rg in m.regions.values()) == ['Marcus', 'corridor', 'living room']
    assert m.room_name_of(next(o for o in m.visible_objects() if m.canon(o) == 'refrigerator')) == 'Marcus'


def test_room_in_a_corridor_becomes_a_spot_and_duplicates_update(world):
    m, sc = world
    r = next(r for r in m.regions.values() if r.name == 'corridor')
    rm, g = m.room_map, m.room_map.grid
    cr, cc = np.nonzero(rm.labels == r.id)
    room_ids = [i for i, q in m.regions.items() if places.room_like(q)]
    rr, rcol = np.nonzero(np.isin(rm.labels, room_ids))
    rx, ry = g.origin_x + (rcol + 0.5) * g.resolution, g.origin_y + (rr + 0.5) * g.resolution
    xs, ys = g.origin_x + (cc + 0.5) * g.resolution, g.origin_y + (cr + 0.5) * g.resolution
    far = [float(np.min(np.hypot(rx - x_, ry - y_))) for x_, y_ in zip(xs[::5], ys[::5])]
    k = int(np.argmax(far)) * 5
    x, y = float(xs[k]), float(ys[k])                    # the corridor point farthest from every room
    assert max(far) > 1.5
    out = places.add_place(m, 'Delivery', 'room', x, y, function='we deliver supplies here')
    assert out['place']['kind'] == 'spot' and 'corridor' in out['note']
    assert m.regions[r.id].name == 'corridor'
    again = places.add_place(m, 'delivery', 'spot', x + 0.5, y)
    assert again['action'] == 'updated' and len(m.places) == 1


def test_remove_and_aliases(world):
    m, sc = world
    x, y, r = centre(m, 'living room')
    places.add_place(m, 'Alexander', 'room', x, y, aliases=["Alexander's lab"])
    assert places.find(m, "alexander's lab").name == 'Alexander'
    assert places.remove_place(m, 'Alexander')['ok']
    assert m.regions[r.id].name == '' or m.regions[r.id].name_source != 'user'
    rooms.update_rooms(m, sc.grid, m.cfg.rooms)
    assert any(rg.name == 'living room' for rg in m.regions.values())
    assert not places.remove_place(m, 'Alexander')['ok']


def test_save_load_roundtrip_and_summary_and_context(world, tmp_path):
    m, sc = world
    x, y, r = centre(m, 'kitchen')
    places.add_place(m, 'Marcus', 'room', x, y, owner='Marcus')
    cx, cy, _ = centre(m, 'corridor')
    places.add_place(m, 'Delivery', 'spot', cx, cy, function='deliver supplies')
    p = m.save(tmp_path / 'map.json')
    m2 = SemanticMap(load_config())
    m2.load(p)
    assert {q['name'] for q in places.list_places(m2)} == {'Marcus', 'Delivery'}
    text = summary.build_summary(m, (0.0, 0.0, 0.0), None, None)['text']
    assert 'Places (taught' in text and 'Marcus (room, owner Marcus' in text and 'Delivery (spot, deliver supplies)' in text
    ctx = places.context_at(m, x, y)
    assert ctx['region']['name'] == 'Marcus' and ctx['places'][0]['name'] == 'Marcus' and ctx['objects_nearby']


def test_a_position_next_to_a_door_still_names_the_room_and_reset_clears_places(world):
    m, sc = world
    rm = m.room_map
    kitchen = next(i for i, r in m.regions.items() if r.name == 'kitchen')
    kr, kc = np.nonzero(rm.labels == kitchen)
    g = rm.grid
    kx, ky = g.origin_x + (kc + 0.5) * g.resolution, g.origin_y + (kr + 0.5) * g.resolution
    # a cell that is NOT in a room-like region but within 1.2 m of the kitchen
    best = None
    rr, cc = np.nonzero(rm.labels >= 0)
    for r_, c_ in zip(rr[::7], cc[::7]):
        x, y = g.origin_x + (c_ + 0.5) * g.resolution, g.origin_y + (r_ + 0.5) * g.resolution
        rid = m.room_id_at(np.array([x, y, 0.0]))
        reg = m.regions.get(rid) if rid else None
        j = int(np.argmin(np.hypot(kx - x, ky - y)))
        d = float(np.hypot(kx[j] - x, ky[j] - y))
        if reg is not None and not places.room_like(reg) and 0.3 < d < 1.2 and m.los_fn(np.array([x, y, 0.0]), np.array([kx[j], ky[j], 0.0])) and (best is None or d < best[0]):
            best = (d, x, y)
    assert best is not None, 'the synthetic plan has a corridor cell within 1.2 m of the kitchen'
    out = places.add_place(m, 'Marcus', 'room', best[1], best[2])
    assert out['place']['kind'] == 'room' and m.regions[kitchen].name == 'Marcus'
    assert places.context_at(m, best[1], best[2])['region']['name'] == 'Marcus'
    m.reset()
    assert m.places == {} and m.next_place_id == 1
