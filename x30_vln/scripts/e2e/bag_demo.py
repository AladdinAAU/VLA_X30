"""Demo video from an episode bag export (host; numpy, opencv, pyyaml, pillow). Sim-time paced (10 fps = 1 sim second per 10 frames' worth, i.e. real sim speed),
camera, robot footprint + trail + the planner's current path on the map, rooms, velocity command, guard state, task and reasoner/planner text.

python scripts/e2e/bag_demo.py results/bags/room_C_rec1_export results/e2e/room_C_rec1 --world aws_hospital \
       --rooms configs/rooms/aws_hospital.yaml --target-room "Room C" [--fps 10] [--speed 1.0]
Writes <export dir>/demo.mp4. Bag export: scripts/e2e/bag_export.py (inside e2e-fw)."""
import argparse, glob, json, math, os, textwrap
import cv2, numpy as np, yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("export"); ap.add_argument("episode")
ap.add_argument("--world", default="aws_hospital"); ap.add_argument("--rooms", default=None); ap.add_argument("--target-room", default=None)
ap.add_argument("--fps", type=int, default=10); ap.add_argument("--speed", type=float, default=1.0, help="sim seconds per video second")
a = ap.parse_args()
ix = json.load(open(os.path.join(a.export, "index.json")))
res_json = json.load(open(os.path.join(a.episode, "result.json")))
start = res_json["start"]
pose = np.array(ix["pose"])
i0 = next((i for i, r in enumerate(pose) if math.hypot(r[1] - start[0], r[2] - start[1]) < 0.3), 0)
pose = pose[i0:]; t0, t1 = pose[0][0], pose[-1][0]
frames = ix["frames"]; ft = np.array([f[0] for f in frames])
plans = ix["plans"]; cmd = np.array(ix["cmd"]) if ix["cmd"] else np.zeros((0, 4)); safety = ix["safety"]
tr = [json.loads(l) for l in open(glob.glob(os.path.join(a.episode, "traces", "*.jsonl"))[0])]
turns = [r for r in tr if r["kind"] == "turn" and r.get("tool")]
call = turns[-1] if turns else None
wd = os.path.join(ROOT, "ros2_ws", "src", "x30_worlds", "worlds", a.world)
m = yaml.safe_load(open(os.path.join(wd, "map.yaml"))); res = m["resolution"]; ox, oy = m["origin"][:2]
g = np.array(Image.open(os.path.join(wd, "map.png")).convert("L")); H = g.shape[0]
rooms = yaml.safe_load(open(a.rooms))["rooms"] if a.rooms else []
xs = list(pose[:, 1]) + [x for r in rooms for x in (r["rect"][0], r["rect"][2])]
ys = list(pose[:, 2]) + [y for r in rooms for y in (r["rect"][1], r["rect"][3])]
x0, x1, y0, y1 = min(xs) - 2.5, max(xs) + 2.5, min(ys) - 2.5, max(ys) + 2.5
MH_T = 720; S = min(MH_T / (y1 - y0), 560 / (x1 - x0)); MW, MH = int((x1 - x0) * S), int((y1 - y0) * S)
cols = ((x0 + (np.arange(MW) + 0.5) / S - ox) / res).astype(int); rows_ = (H - 1 - ((y1 - (np.arange(MH) + 0.5) / S - oy) / res)).astype(int)
okc = (cols >= 0) & (cols < g.shape[1]); okr = (rows_ >= 0) & (rows_ < H)
sub = np.full((MH, MW), 205, np.uint8); sub[np.ix_(okr, okc)] = g[np.ix_(rows_[okr], cols[okc])]
base = np.full((MH, MW, 3), 128, np.uint8); base[sub >= 230] = (255, 255, 255); base[sub <= 50] = (40, 40, 40)
P = lambda x, y: (int((x - x0) * S), int((y1 - y) * S))
for r in rooms:
    xa, ya, xb, yb = r["rect"]; hot = a.target_room and r["name"] == a.target_room
    ov = base.copy(); cv2.rectangle(ov, P(xa, yb), P(xb, ya), (255, 190, 120) if hot else (200, 235, 200), -1); base = cv2.addWeighted(ov, 0.5, base, 0.5, 0)
    cv2.rectangle(base, P(xa, yb), P(xb, ya), (0, 0, 220) if hot else (90, 150, 90), 2)
    cv2.putText(base, r["name"][-1], (P(xa, yb)[0] + 4, P(xa, yb)[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220) if hot else (60, 110, 60), 2)
CW = 720; W = MW + CW; Ht = max(MH, 720)


def put(img, text, org, scale=0.55, col=(20, 20, 20), w=1, wrap=60):
    y = org[1]
    for line in textwrap.wrap(text, wrap) or [""]:
        cv2.putText(img, line, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, w, cv2.LINE_AA); y += int(scale * 36)
    return y


cam_cache = {}
def cam(i):
    if i not in cam_cache:
        cam_cache.clear(); cam_cache[i] = cv2.imread(os.path.join(a.export, "frames", frames[i][1]))
    im = cam_cache[i]
    return cv2.resize(im, (CW - 20, int((CW - 20) * im.shape[0] / im.shape[1])))


vw = cv2.VideoWriter(os.path.join(a.export, "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, Ht))
first_plan = min((p[0] for p in plans if p[0] >= t0), default=None)
step = a.speed / a.fps; T = t0; n = 0
sa = np.array([[s[0], s[1], s[2]] for s in safety]) if safety else np.zeros((0, 3))
while T <= t1 + step:
    k = int(np.searchsorted(pose[:, 0], T, side="right")) - 1; k = max(0, min(k, len(pose) - 1))
    _, x, y, yaw = pose[k]
    img = np.full((Ht, W, 3), 245, np.uint8); mp = base.copy()
    cur = [p for p in plans if t0 - 5 <= p[0] <= T]
    if cur:
        pts = np.array([P(q[0], q[1]) for q in cur[-1][1]], np.int32).reshape(-1, 1, 2); cv2.polylines(mp, [pts], False, (0, 160, 0), 2, cv2.LINE_AA)
    trail = np.array([P(r[1], r[2]) for r in pose[:k + 1]], np.int32).reshape(-1, 1, 2); cv2.polylines(mp, [trail], False, (220, 120, 20), 2, cv2.LINE_AA)
    c, s_ = math.cos(yaw), math.sin(yaw)
    box = np.array([P(x + c * bx - s_ * by, y + s_ * bx + c * by) for bx, by in ((0.5, 0.23), (0.5, -0.23), (-0.5, -0.23), (-0.5, 0.23))], np.int32)
    cv2.fillPoly(mp, [box], (60, 60, 230)); cv2.polylines(mp, [box], True, (0, 0, 120), 1, cv2.LINE_AA)
    cv2.arrowedLine(mp, P(x, y), P(x + 0.9 * c, y + 0.9 * s_), (0, 0, 0), 2, tipLength=0.3)
    img[:MH, :MW] = mp
    i = max(0, int(np.searchsorted(ft, T, side="right")) - 1)
    ci = cam(i); img[10:10 + ci.shape[0], MW + 10:MW + 10 + ci.shape[1]] = ci
    yy = 10 + ci.shape[0] + 26
    yy = put(img, f"sim t = {T - t0:6.1f} s    pose ({x:5.2f}, {y:5.2f})  yaw {math.degrees(yaw):4.0f} deg", (MW + 12, yy), 0.55)
    ci_cmd = cmd[np.searchsorted(cmd[:, 0], T, side="right") - 1] if len(cmd) else None
    gi = int(np.searchsorted(sa[:, 0], T, side="right")) - 1 if len(sa) else -1
    gtxt = f"guard: scale {sa[gi][1]:.2f}, nearest obstacle {sa[gi][2]:.2f} m" if gi >= 0 else "guard: n/a"
    ctxt = f"cmd_vel: vx {ci_cmd[1]:+.2f} vy {ci_cmd[2]:+.2f} wz {ci_cmd[3]:+.2f}" if ci_cmd is not None else ""
    yy = put(img, gtxt + "     " + ctxt, (MW + 12, yy + 2), 0.5, (90, 90, 90))
    yy = put(img, "Task: " + res_json["instruction"], (MW + 12, yy + 10), 0.6, (0, 0, 0), 2)
    if first_plan is None or T < first_plan:
        yy = put(img, "Reasoner: reading the map summary ...", (MW + 12, yy + 6), 0.55, (150, 40, 0))
    else:
        yy = put(img, f"Reasoner -> {call['tool']}({json.dumps(call.get('args'))})" if call else "", (MW + 12, yy + 6), 0.5, (150, 40, 0))
        if T < t1 - 1.0:
            yy = put(img, "Planner: driving to the reported place (green = planned path)", (MW + 12, yy + 4), 0.55, (0, 110, 0))
        else:
            put(img, "Planner: " + res_json["message"].split("planner: ", 1)[-1][:150], (MW + 12, yy + 4), 0.5, (0, 110, 0))
    vw.write(img); T += step; n += 1
vw.release()
print(n, "frames ->", os.path.join(a.export, "demo.mp4"))
