"""Demo video of a follow_person episode from its bag export (host; numpy, opencv, pyyaml, pillow).
python scripts/follow/follow_demo.py results/bags/follow_rec1_export results/follow/follow_rec1 --world aws_hospital
Export first (inside e2e-fw): python3 /x30/scripts/e2e/bag_export.py /x30/results/bags/follow_rec1 /x30/results/bags/follow_rec1_export
Left: map with the robot footprint + trail, the ground-truth person (magenta) and the tracker's estimate (cyan), a 2 m floor circle around the person
(red when the robot is inside it). Right: live camera, follow state, ground-truth distance / target, bearing, cmd_vel, guard.
Sim-time paced (10 fps). Writes <export>/demo.mp4."""
import argparse, json, math, os, textwrap
import cv2, numpy as np, yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("export"); ap.add_argument("episode"); ap.add_argument("--world", default="aws_hospital")
ap.add_argument("--fps", type=int, default=10); ap.add_argument("--speed", type=float, default=1.0)
a = ap.parse_args()
ix = json.load(open(os.path.join(a.export, "index.json")))
res = json.load(open(os.path.join(a.episode, "result.json")))
start = res["start"]; d_target = res["d_target"]
pose = np.array(ix["pose"]); person = np.array(ix["person"]); status = ix["status"]
walk0 = person[0, 0]                                   # first /gt/person sample = walk start
i0 = next((i for i, r in enumerate(pose) if math.hypot(r[1] - start[0], r[2] - start[1]) < 0.3), 0)
pose = pose[i0:]; t0 = max(pose[0][0], walk0 - 6.0); t1 = person[-1, 0] + 3.0
frames = ix["frames"]; ft = np.array([f[0] for f in frames])
cmd = np.array(ix["cmd"]); safety = ix["safety"]
wd = os.path.join(ROOT, "ros2_ws", "src", "x30_worlds", "worlds", a.world)
m = yaml.safe_load(open(os.path.join(wd, "map.yaml"))); rs = m["resolution"]; ox, oy = m["origin"][:2]
g = np.array(Image.open(os.path.join(wd, "map.png")).convert("L")); H = g.shape[0]
xs = list(pose[:, 1]) + list(person[:, 1]); ys = list(pose[:, 2]) + list(person[:, 2])
x0, x1, y0, y1 = min(xs) - 4, max(xs) + 4, min(ys) - 3, max(ys) + 3
S = min(720 / (y1 - y0), 560 / (x1 - x0)); MW, MH = int((x1 - x0) * S), int((y1 - y0) * S)
cols = ((x0 + (np.arange(MW) + 0.5) / S - ox) / rs).astype(int); rows_ = (H - 1 - ((y1 - (np.arange(MH) + 0.5) / S - oy) / rs)).astype(int)
okc = (cols >= 0) & (cols < g.shape[1]); okr = (rows_ >= 0) & (rows_ < H)
sub = np.full((MH, MW), 205, np.uint8); sub[np.ix_(okr, okc)] = g[np.ix_(rows_[okr], cols[okc])]
base = np.full((MH, MW, 3), 128, np.uint8); base[sub >= 230] = (255, 255, 255); base[sub <= 50] = (40, 40, 40)
P = lambda x, y: (int((x - x0) * S), int((y1 - y) * S))
CW = 720; W = MW + CW; Ht = max(MH, 720)
st_t = np.array([s["t"] for s in status]) if status else np.zeros(0)
people = ix.get("people", []); people_t = np.array([p["t"] for p in people]) if people else np.zeros(0)
OUTFIT_BGR = {"red": (40, 40, 220), "green": (60, 160, 60), "dark": (60, 60, 60), "maroon": (60, 30, 120), "purple": (190, 60, 150), "yellow": (40, 200, 220), "navy": (160, 60, 20)}


def put(img, text, org, scale=0.55, col=(20, 20, 20), w=1, wrap=60):
    y = org[1]
    for line in textwrap.wrap(text, wrap) or [""]:
        cv2.putText(img, line, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, w, cv2.LINE_AA); y += int(scale * 36)
    return y


vw = cv2.VideoWriter(os.path.join(a.export, "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, Ht))
step = a.speed / a.fps; T = t0; n = 0; cache = {}
while T <= t1:
    k = max(0, int(np.searchsorted(pose[:, 0], T, side="right")) - 1)
    _, x, y, yaw = pose[k]
    j = max(0, int(np.searchsorted(person[:, 0], T, side="right")) - 1); px_, py_ = person[j, 1], person[j, 2]
    d = math.hypot(px_ - x, py_ - y); brg = (math.degrees(math.atan2(py_ - y, px_ - x) - yaw) + 180) % 360 - 180
    img = np.full((Ht, W, 3), 245, np.uint8); mp = base.copy()
    trail = np.array([P(r[1], r[2]) for r in pose[:k + 1]], np.int32).reshape(-1, 1, 2); cv2.polylines(mp, [trail], False, (220, 120, 20), 2, cv2.LINE_AA)
    ptrail = np.array([P(r[1], r[2]) for r in person[:j + 1]], np.int32).reshape(-1, 1, 2)
    if len(ptrail): cv2.polylines(mp, [ptrail], False, (200, 80, 200), 1, cv2.LINE_AA)
    cv2.circle(mp, P(px_, py_), int(2.0 * S), (0, 0, 220) if d < 2.0 else (200, 200, 255), 2, cv2.LINE_AA)       # 2 m floor
    cv2.circle(mp, P(px_, py_), int(d_target * S), (150, 200, 150), 1, cv2.LINE_AA)                                # target distance
    cv2.circle(mp, P(px_, py_), 6, (200, 0, 200), -1)
    if people:
        pk = max(0, int(np.searchsorted(people_t, T, side="right")) - 1)
        for nm, v in people[pk]["people"].items():
            if nm == "target":
                continue
            col = OUTFIT_BGR.get(v[3].split()[0] if v[3] else "", (120, 120, 120))
            cv2.circle(mp, P(v[0], v[1]), 6, col, -1); cv2.circle(mp, P(v[0], v[1]), 7, (60, 60, 60), 1)
    sk = int(np.searchsorted(st_t, T, side="right")) - 1 if len(st_t) else -1
    s = status[sk] if sk >= 0 and T - st_t[sk] < 1.5 else None
    if s and "x" in s:
        cv2.drawMarker(mp, P(s["x"] + start[0], s["y"] + start[1]), (200, 200, 0), cv2.MARKER_CROSS, 14, 2)
    c, s_ = math.cos(yaw), math.sin(yaw)
    box = np.array([P(x + c * bx - s_ * by, y + s_ * bx + c * by) for bx, by in ((0.5, 0.23), (0.5, -0.23), (-0.5, -0.23), (-0.5, 0.23))], np.int32)
    cv2.fillPoly(mp, [box], (60, 60, 230)); cv2.polylines(mp, [box], True, (0, 0, 120), 1, cv2.LINE_AA)
    cv2.arrowedLine(mp, P(x, y), P(x + 0.9 * c, y + 0.9 * s_), (0, 0, 0), 2, tipLength=0.3)
    img[:MH, :MW] = mp
    i = max(0, int(np.searchsorted(ft, T, side="right")) - 1)
    if cache.get("i") != i:
        cache = {"i": i, "im": cv2.imread(os.path.join(a.export, "frames", frames[i][1]))}
    ci = cv2.resize(cache["im"], (CW - 20, int((CW - 20) * cache["im"].shape[0] / cache["im"].shape[1])))
    img[10:10 + ci.shape[0], MW + 10:MW + 10 + ci.shape[1]] = ci
    yy = 10 + ci.shape[0] + 28
    state = (s or {}).get("state", "-")
    ncol = (0, 0, 200) if state in ("ambiguous", "lost") else (150, 40, 0)
    extra = f"   candidates {s.get('candidates')}" if (s and state == "ambiguous") else ""
    yy = put(img, f"follow_person   state: {state}{extra}     walk t = {max(0.0, T - walk0):5.1f} s", (MW + 12, yy), 0.62, ncol, 2)
    ok = abs(d - d_target) <= 0.5
    yy = put(img, f"distance to person (ground truth): {d:4.2f} m   (target {d_target:.1f} m, floor 2.0 m)", (MW + 12, yy + 4), 0.55, (0, 110, 0) if d >= 2.0 else (0, 0, 200), 1)
    yy = put(img, f"bearing {brg:+5.1f} deg", (MW + 12, yy + 2), 0.55)
    ci_c = cmd[np.searchsorted(cmd[:, 0], T, side="right") - 1] if len(cmd) else None
    gi = int(np.searchsorted([q[0] for q in safety], T, side="right")) - 1 if safety else -1
    put(img, (f"cmd_vel vx {ci_c[1]:+.2f} wz {ci_c[3]:+.2f}   " if ci_c is not None else "") + (f"guard scale {safety[gi][1]:.2f}, nearest obstacle {safety[gi][2]:.2f} m" if gi >= 0 else ""),
        (MW + 12, yy + 2), 0.5, (90, 90, 90))
    put(img, "magenta = ground-truth person, cyan x = tracker estimate, circles: red 2 m floor, green target distance", (MW + 12, Ht - 14), 0.42, (110, 110, 110))
    vw.write(img); T += step; n += 1
vw.release()
print(n, "frames ->", os.path.join(a.export, "demo.mp4"))
