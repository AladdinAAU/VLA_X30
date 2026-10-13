"""Live check of the place-teaching interpreter against the real model (host; the x30-vlm container must be up).
python scripts/teach/live_llm_check.py [--url http://127.0.0.1:8000/v1] [--model qwen3-vl] [--repeat 1]
22 utterances with a context (room / corridor / hall), expected action, name token, and kind; prints failures and the pass rate. Results -> results/teach/live_llm_check.json."""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("x30_common", "x30_teach"):
    sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", p))
from x30_common.vlm_client import VLMClient
from x30_teach import interpreter as itp

ROOM = {"region": {"id": 3, "name": "", "name_source": "", "area_m2": 18.0, "width_m": 3.4, "length_m": 5.3}, "objects_nearby": {"chair": 4, "desk": 1}, "places": []}
OFFICE = {"region": {"id": 4, "name": "", "name_source": "", "area_m2": 12.0, "width_m": 3.0, "length_m": 4.0}, "objects_nearby": {"desk": 2, "monitor": 2, "chair": 2}, "places": []}
CORR = {"region": {"id": 1, "name": "corridor", "name_source": "shape", "area_m2": 40.0, "width_m": 1.9, "length_m": 20.0}, "objects_nearby": {}, "places": []}
HALL = {"region": {"id": 2, "name": "", "name_source": "", "area_m2": 150.0, "width_m": 9.0, "length_m": 17.0}, "objects_nearby": {"bench": 6}, "places": []}
# (utterance, context, last_place, expected kind, name token (lower, '' = any), place_kind expected ('' = any), owner token ('' = any))
CASES = [
    ("This is Marcus.", ROOM, None, "save", "marcus", "room", "marcus"),
    ("this is Alexander's office", OFFICE, None, "save", "alexander", "room", "alexander"),
    ("Alexander.", ROOM, None, "save", "alexander", "room", ""),
    ("Here we deliver stuff.", CORR, None, "save", "deliver", "spot", ""),
    ("this is where the nurses take their break", ROOM, None, "save", "break", "room", ""),
    ("Room 12", ROOM, None, "save", "12", "room", ""),
    ("This is the pharmacy", ROOM, None, "save", "pharmacy", "room", ""),
    ("This is the entrance", HALL, None, "save", "entrance", "spot", ""),
    ("No, that's Marco, not Marcus.", ROOM, "Marcus", "rename", "marco", "", ""),
    ("forget this place", ROOM, "Marcus", "forget", "marcus", "", ""),
    ("delete the pharmacy", ROOM, "Alexander", "forget", "pharmacy", "", ""),
    ("nice weather today, isn't it?", ROOM, None, "ignore", "", "", ""),
    ("ok let's keep going, follow me", CORR, None, "ignore", "", "", ""),
    ("this is Sarah's desk", OFFICE, None, "save", "sarah", "spot", "sarah"),
    ("this is where we keep the supplies", ROOM, None, "save", "suppl", "room", ""),
    ("Here we do the handover.", CORR, None, "save", "handover", "spot", ""),
    ("this is the lab of Doctor Weber", ROOM, None, "save", "weber", "room", ""),
    ("let's call this room the quiet room", ROOM, None, "save", "quiet", "room", ""),
    ("this is the ice machine", CORR, None, "save", "ice", "spot", ""),
    ("oh, and this is Alexander, he works on the robots", ROOM, None, "save", "alexander", "room", "alexander"),
    ("This is the main hall.", HALL, None, "save", "hall", "spot", ""),
    ("what time is it?", ROOM, None, "ignore", "", "", ""),
]

# held-out phrasings, written before the model was run on them (no prompt tuning on these)
HELDOUT = [
    ("Okay, this one is Priya's.", OFFICE, None, "save", "priya", "", "priya"),
    ("we store the blood samples here", CORR, None, "save", "sample", "spot", ""),
    ("That is Tom, he is in charge of procurement.", ROOM, None, "save", "tom", "room", "tom"),
    ("Scratch that, it's Thomas.", ROOM, "Tom", "rename", "thomas", "", ""),
    ("remove the entrance", HALL, "Entrance", "forget", "entrance", "", ""),
    ("Mind the step, it's slippery.", CORR, None, "ignore", "", "", ""),
    ("this is the server room", ROOM, None, "save", "server", "room", ""),
    ("Reception.", HALL, None, "save", "reception", "", ""),
    ("here is where visitors sign in", CORR, None, "save", "sign", "spot", ""),
    ("How many floors does the building have?", ROOM, None, "ignore", "", "", ""),
]
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000/v1"); ap.add_argument("--model", default="qwen3-vl"); ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--heldout", action="store_true")
a = ap.parse_args()
client = VLMClient(base_url=a.url, model=a.model, max_tokens=200, temperature=0.0, timeout_s=60.0)
print("server healthy:", client.health(5.0))
rows, ok_n, lat = [], 0, []
for utt, ctx, last, kind, tok, pk, owner in (HELDOUT if a.heldout else CASES):
    for _ in range(a.repeat):
        t0 = time.time()
        try:
            act = itp.interpret(lambda m, t: client.chat(m, tools=t, tool_choice="required", max_tokens=200, temperature=0.0), utt, ctx, last)
        except Exception as ex:
            act = itp.Action("error", reason=f"{type(ex).__name__}: {ex}")
        lat.append(time.time() - t0)
        name = (act.name or act.new_name or act.old_name or "").lower() if act.kind != "rename" else (act.new_name or "").lower()
        if act.kind == "forget":
            name = act.name.lower()
        good = act.kind == kind and (not tok or tok in name or (kind == "save" and any(tok in (x or "").lower() for x in [act.function] + list(act.aliases))))
        if good and kind == "save" and pk and act.place_kind != pk:
            good = False
        if good and kind == "save" and owner and owner not in (act.owner or "").lower():
            good = False
        ok_n += good
        rows.append({"utterance": utt, "expected": [kind, tok, pk, owner], "got": vars(act), "ok": bool(good)})
        print(("PASS " if good else "FAIL ") + f"{utt!r:58s} -> {act.kind} name={act.name or act.new_name!r} kind={act.place_kind} owner={act.owner!r} fn={act.function!r}")
print(f"\n{ok_n}/{len(rows)} correct, median latency {sorted(lat)[len(lat) // 2]:.2f} s")
os.makedirs(os.path.join(ROOT, "results", "teach"), exist_ok=True)
json.dump({"model": a.model, "n": len(rows), "correct": ok_n, "rows": rows}, open(os.path.join(ROOT, "results", "teach", "live_llm_check_heldout.json" if a.heldout else "live_llm_check.json"), "w"), indent=1)
