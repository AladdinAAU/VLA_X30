"""World <-> map frame for saved semantic maps (known_map_nav puts the map frame at the robot's start pose: map = R(-yaw)(world - t)).

Saved maps are stored in the WORLD frame (results/maps/<world>.json) so any episode start can load them:
load_for_start(world_json, start_xyyaw, out_json) writes a copy shifted into the map frame of that start (crops dir copied)."""
import json
import math
import shutil
from pathlib import Path


def _tf(p, t, yaw, inverse):
    c, s = math.cos(yaw), math.sin(yaw)
    if inverse:  # world -> map
        x, y = p[0] - t[0], p[1] - t[1]
        return [round(c * x + s * y, 5), round(-s * x + c * y, 5)] + list(p[2:])
    return [round(t[0] + c * p[0] - s * p[1], 5), round(t[1] + s * p[0] + c * p[1], 5)] + list(p[2:])


def _rot(v, yaw, inverse):
    c, s = math.cos(-yaw if inverse else yaw), math.sin(-yaw if inverse else yaw)
    return [round(c * v[0] - s * v[1], 5), round(s * v[0] + c * v[1], 5)] + list(v[2:])


def transform_map(data: dict, t, yaw: float, inverse: bool) -> dict:
    """Return a copy of a saved semantic map with all positions moved (inverse=True: world -> map, else map -> world)."""
    d = json.loads(json.dumps(data))
    for o in d["objects"]:
        o["position"] = _tf(o["position"], t, yaw, inverse)
        o["velocity"] = _rot(o["velocity"], yaw, inverse)
        o["hist"] = [[h[0], _tf(h[1], t, yaw, inverse), h[2], h[3]] for h in o["hist"]]
        o["track"] = [[tt, _tf(p, t, yaw, inverse)] for tt, p in o["track"]]
        for v in o.get("views", []):
            v["camera_position"] = _tf(v["camera_position"], t, yaw, inverse)
    for r in d["rays"]:
        r["origin"] = _tf(r["origin"], t, yaw, inverse)
        r["direction"] = _rot(r["direction"], yaw, inverse)
    for rp in d["rejected_places"]:
        rp["pos"] = _tf(rp["pos"], t, yaw, inverse)
    d["regions"] = []  # recomputed by the mapper from the grid
    return d


def convert_file(src, dst, t, yaw, inverse):
    src, dst = Path(src), Path(dst)
    data = transform_map(json.loads(src.read_text(encoding="utf-8")), t, yaw, inverse)
    dst.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    cs, cd = src.with_name(src.stem + "_crops"), dst.with_name(dst.stem + "_crops")
    if cs.is_dir():
        shutil.copytree(cs, cd, dirs_exist_ok=True)
    return data
