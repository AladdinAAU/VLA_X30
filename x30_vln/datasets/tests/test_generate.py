from x30_datasets.episodes import io as eio
from x30_datasets.episodes import language as lang
from x30_datasets.episodes.context import load_context
from x30_datasets.episodes.generate import goal_pose_episodes, object_nav_episodes, velocity_follow_episodes


def test_generation_is_reproducible():
    ctx = load_context("apt_03")
    a = object_nav_episodes(ctx, seed=5, n=8)
    assert a == object_nav_episodes(ctx, seed=5, n=8)
    assert a != object_nav_episodes(ctx, seed=6, n=8)


def test_object_nav_episodes_are_valid_and_reachable():
    ctx = load_context("apt_01")
    eps = object_nav_episodes(ctx, seed=0, n=12)
    assert eps
    for e in eps:
        assert eio.schema_errors(e) == []
        assert eio.semantic_errors(e, ctx.by_name, {"apt_01"}) == []
        g = e["goal"]
        assert ctx.reach[ctx.cell(e["start"]["x"], e["start"]["y"])]
        assert e["difficulty"]["geodesic_m"] >= 2.5
        if not g["any_instance"]:
            assert g["other_instances"] and g["instance"] not in g["other_instances"]


def test_split_by_instance_or_class():
    ctx = load_context("apt_02")
    by_target = {}
    for e in object_nav_episodes(ctx, seed=0, n=40):
        key = e["goal"]["instance"] if not e["goal"]["any_instance"] else "class:" + e["goal"]["label"]
        by_target.setdefault(key, set()).add(e["split"])
    assert all(len(v) == 1 for v in by_target.values())


def test_goal_pose_and_velocity_follow():
    ctx = load_context("apt_04")
    for e in goal_pose_episodes(ctx, n=5):
        assert eio.schema_errors(e) == [] and ctx.reach[ctx.cell(e["goal"]["x"], e["goal"]["y"])]
    for e in velocity_follow_episodes(ctx, n=5):
        assert eio.schema_errors(e) == []
        assert e["reference_actions"][-1] == {"action": "stop"} and len(e["instruction_variants"]) >= 3


def test_path_actions():
    acts = lang.path_to_actions([(0, 0), (4, 0), (4, 3)], 0.0)
    assert acts[:3] == [{"action": "forward", "distance_m": 4.0}, {"action": "turn_left", "degrees": 90},
                        {"action": "forward", "distance_m": 3.0}]


def test_phrases():
    assert lang.dist_phrase(3.2) == "3 meters" and lang.dist_phrase(1.0) == "1 meter"
    assert lang.turn_word(90) == "turn left" and lang.turn_word(-170) == "turn around"
