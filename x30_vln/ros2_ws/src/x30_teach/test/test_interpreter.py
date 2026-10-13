from types import SimpleNamespace as NS

from x30_teach import interpreter as itp

ROOM = {"region": {"id": 3, "name": "", "name_source": "", "area_m2": 18.0, "width_m": 3.4, "length_m": 5.3}, "objects_nearby": {"chair": 4, "desk": 1}, "places": []}
CORRIDOR = {"region": {"id": 1, "name": "corridor", "name_source": "shape", "area_m2": 40.0, "width_m": 1.9, "length_m": 20.0}, "objects_nearby": {}, "places": []}


def fake(tool, **args):
    return lambda messages, tools: NS(tool_calls=[NS(name=tool, arguments=args)])


def test_prompt_contains_the_context_the_model_needs():
    msgs = itp.build_messages("this is Marcus", ROOM, "Alexander")
    user = msgs[1]["content"]
    assert "inside a room (18 m2" in user and "chair x4" in user and "Last taught place: Alexander" in user and '"this is Marcus"' in user
    assert "a corridor" in itp.build_messages("here we deliver stuff", CORRIDOR, None)[1]["content"]
    assert "unmapped space" in itp.region_phrase({"region": None})
    assert {t["function"]["name"] for t in itp.TOOLS} == {"save_place", "rename_place", "forget_place", "ignore"}


def test_save_is_parsed_and_cleaned():
    a = itp.interpret(fake("save_place", name='  "Marcus". ', kind="room", owner="Marcus", function="", aliases=["Marcus's office", " "]), "this is Marcus", ROOM)
    assert (a.kind, a.name, a.place_kind, a.owner, a.aliases) == ("save", "Marcus", "room", "Marcus", ["Marcus's office"])
    b = itp.interpret(fake("save_place", name="Delivery", kind="weird", function="we deliver stuff"), "here we deliver stuff", CORRIDOR)
    assert b.place_kind == "spot" and b.function == "we deliver stuff"
    assert itp.interpret(fake("save_place", name="x" * 100, kind="spot"), "", ROOM).name == "x" * 40


def test_rename_forget_ignore_and_errors():
    assert itp.interpret(fake("rename_place", old_name="Marcus", new_name="Marco"), "no it is Marco", ROOM).new_name == "Marco"
    assert itp.interpret(fake("forget_place", name="Pharmacy"), "forget the pharmacy", ROOM).kind == "forget"
    assert itp.interpret(fake("ignore", reason="small talk"), "nice weather", ROOM).kind == "ignore"
    assert itp.interpret(fake("save_place", name="", kind="room"), "", ROOM).kind == "error"
    assert itp.interpret(lambda m, t: NS(tool_calls=[]), "x", ROOM).kind == "error"
    assert itp.interpret(fake("dance"), "x", ROOM).kind == "error"
