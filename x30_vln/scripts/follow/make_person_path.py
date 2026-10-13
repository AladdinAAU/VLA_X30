"""Scripted walking path for the simulated person (host: numpy scipy pillow pyyaml). Obstacle-free by construction: A* on the world's 2D map
(widest clearance tier) between waypoints, smoothed, then timed at a constant speed with pauses.
python scripts/follow/make_person_path.py --world aws_hospital --out configs/follow/hospital_walk1.json --speed 0.8 \
   --wp 4.9,11.5 --wp 6.5,6.0 --wp 6.5,-2.0,4 --wp 6.5,-12.0 --wp 6.5,-24.0
Waypoint syntax: --wp x,y[,pause_s] (repeat; write --wp=-3,2 when x is negative). Output: {"world", "speed", "hz", "poses": [[t, x, y, yaw], ...]} (sim seconds from the start trigger)."""
import argparse, json, math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "x30_navigation"))
from x30_navigation.known_map import KnownMap, PlanParams, load_world_grid, path_length

ap = argparse.ArgumentParser()
ap.add_argument("--world", default="aws_hospital"); ap.add_argument("--out", required=True); ap.add_argument("--speed", type=float, default=0.8)
ap.add_argument("--hz", type=float, default=20.0); ap.add_argument("--wp", action="append", required=True); ap.add_argument("--clear", type=float, default=0.55)
a = ap.parse_args()
blocked, origin = load_world_grid(os.path.join(ROOT, "ros2_ws", "src", "x30_worlds", "worlds", a.world, "map.yaml"), 0.1)
km = KnownMap(blocked, origin, 0.1, PlanParams(radii=(a.clear,)))
wp = []
for w in a.wp:
    f = [float(v) for v in w.split(",")]
    wp.append((np.array(f[:2]), f[2] if len(f) > 2 else 0.0))
for p, _ in wp:
    print("waypoint", p, "clearance %.2f m" % km.clearance(p))
route, stops = [wp[0][0]], {0: wp[0][1]}
for k in range(1, len(wp)):
    r = km.plan(route[-1], wp[k][0])
    assert r is not None, f"no path to waypoint {k} {wp[k][0]}"
    seg = r[0]
    print("leg", k, "length %.1f m, tier %.2f" % (path_length(seg), r[1]))
    route.extend(list(seg[1:])); stops[len(route) - 1] = wp[k][1]
route = np.array(route)
d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
pause_at = {float(d[i]): s for i, s in stops.items() if s > 0}
poses, t, s_pos, dt = [], 0.0, 0.0, 1.0 / a.hz
heading = 0.0
while s_pos < d[-1]:
    x = float(np.interp(s_pos, d, route[:, 0])); y = float(np.interp(s_pos, d, route[:, 1]))
    ahead = min(s_pos + 0.6, d[-1]); hx, hy = float(np.interp(ahead, d, route[:, 0])) - x, float(np.interp(ahead, d, route[:, 1])) - y
    if math.hypot(hx, hy) > 1e-3:
        tgt = math.atan2(hy, hx); dh = (tgt - heading + math.pi) % (2 * math.pi) - math.pi; heading += max(-2.0 * dt, min(2.0 * dt, dh))   # turn rate <= 2 rad/s
    poses.append([round(t, 3), round(x, 3), round(y, 3), round(heading, 4)])
    stop = next((sec for sd, sec in pause_at.items() if abs(s_pos - sd) < a.speed * dt * 0.51), None)
    if stop:
        for _ in range(int(stop * a.hz)):
            t += dt; poses.append([round(t, 3), round(x, 3), round(y, 3), round(heading, 4)])
        pause_at = {k: v for k, v in pause_at.items() if v != stop or abs(s_pos - k) > a.speed * dt}
    t += dt; s_pos += a.speed * dt
os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
json.dump({"world": a.world, "speed": a.speed, "hz": a.hz, "poses": poses}, open(os.path.join(ROOT, a.out), "w"))
print("path %.1f m, %.0f s, %d poses -> %s" % (d[-1], t, len(poses), a.out))
