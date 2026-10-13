"""Animated replay of a recorded episode (host; numpy, opencv, pillow, pyyaml): 2D map + rooms + robot footprint + trail + camera frame of the
current agent turn + the tool call and its result. Writes <episode_dir>/replay.mp4 and replay.gif.
python scripts/e2e/replay_episode.py results/e2e/room_C_1 --world aws_hospital [--rooms configs/rooms/aws_hospital.yaml] [--goal x y] [--stride 3]"""
import argparse, glob, json, math, os, textwrap
import cv2, numpy as np, yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("episode"); ap.add_argument("--world", default="aws_hospital"); ap.add_argument("--rooms", default=None)
ap.add_argument("--goal", nargs=2, type=float, default=None, help="world x y to mark (e.g. the room centre)")
ap.add_argument("--stride", type=int, default=3); ap.add_argument("--fps", type=int, default=12); ap.add_argument("--target-room", default=None)
a = ap.parse_args()
ep = a.episode
wd = os.path.join(ROOT, "ros2_ws", "src", "x30_worlds", "worlds", a.world)
m = yaml.safe_load(open(os.path.join(wd, "map.yaml"))); res = m["resolution"]; ox, oy = m["origin"][:2]
g = np.array(Image.open(os.path.join(wd, "map.png")).convert("L")); H = g.shape[0]
traj = np.load(os.path.join(ep, "trajectory.npy"))
start = json.load(open(os.path.join(ep, "result.json")))["start"]
i0 = next((i for i, r in enumerate(traj) if math.hypot(r[1] - start[0], r[2] - start[1]) < 0.5), 0)   # drop samples before the teleport
traj = traj[i0:]
fb = json.load(open(os.path.join(ep, "feedback.json")))
t_end_abs = traj[-1][0]; sim_end = fb[-1]["sim"]
turns = {}
for f in fb:
    turns.setdefault(f["turn"], t_end_abs - sim_end + f["sim"])       # abs sim time when each turn started
tr = [json.loads(l) for l in open(glob.glob(os.path.join(ep, "traces", "*.jsonl"))[0])]
tinfo = {r["turn"]: r for r in tr if r["kind"] == "turn"}
rooms = yaml.safe_load(open(a.rooms))["rooms"] if a.rooms else []
# view: bounding box of the trajectory + rooms + margin
xs = list(traj[:, 1]) + [x for r in rooms for x in (r["rect"][0], r["rect"][2])]
ys = list(traj[:, 2]) + [y for r in rooms for y in (r["rect"][1], r["rect"][3])]
x0, x1, y0, y1 = min(xs) - 2.5, max(xs) + 2.5, min(ys) - 2.5, max(ys) + 2.5
S = min(700 / (y1 - y0), 520 / (x1 - x0))        # px per metre
MW, MH = int((x1 - x0) * S), int((y1 - y0) * S)
base = np.full((MH, MW, 3), 128, np.uint8)
for r in range(MH):
    wy = y1 - (r + 0.5) / S
    cy = H - 1 - int((wy - oy) / res)
    if 0 <= cy < H:
        for c in range(0, MW):
            pass
# vectorised sampling of the map image
cols = ((x0 + (np.arange(MW) + 0.5) / S - ox) / res).astype(int); rows_ = (H - 1 - ((y1 - (np.arange(MH) + 0.5) / S - oy) / res)).astype(int)
ok_c = (cols >= 0) & (cols < g.shape[1]); ok_r = (rows_ >= 0) & (rows_ < H)
sub = np.full((MH, MW), 205, np.uint8); sub[np.ix_(ok_r, ok_c)] = g[np.ix_(rows_[ok_r], cols[ok_c])]
base[sub >= 230] = (255, 255, 255); base[sub <= 50] = (40, 40, 40)
P = lambda x, y: (int((x - x0) * S), int((y1 - y) * S))
for r in rooms:
    xa, ya, xb, yb = r["rect"]; hot = a.target_room and r["name"] == a.target_room
    ov = base.copy(); cv2.rectangle(ov, P(xa, yb), P(xb, ya), (255, 190, 120) if hot else (200, 235, 200), -1); base = cv2.addWeighted(ov, 0.5, base, 0.5, 0)
    cv2.rectangle(base, P(xa, yb), P(xb, ya), (0, 0, 220) if hot else (90, 150, 90), 2)
    cv2.putText(base, r["name"][-1], (P(xa, yb)[0] + 4, P(xa, yb)[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220) if hot else (60, 110, 60), 2)
if a.goal: cv2.drawMarker(base, P(*a.goal), (0, 0, 255), cv2.MARKER_STAR, 18, 2)
W, Ht = MW + 700, max(MH, 720)
frames = []
cam_cache = {}
def cam(turn):
    if turn not in cam_cache:
        p = os.path.join(ep, f"cam_turn{turn:02d}.jpg"); im = cv2.imread(p) if os.path.exists(p) else None
        cam_cache[turn] = cv2.resize(im, (680, 382)) if im is not None else np.zeros((382, 680, 3), np.uint8)
    return cam_cache[turn]
def put(img, text, org, scale=0.5, col=(20, 20, 20), w=1, wrap=62):
    y = org[1]
    for line in textwrap.wrap(text, wrap) or [""]:
        cv2.putText(img, line, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, w, cv2.LINE_AA); y += int(scale * 38)
    return y
idx = list(range(0, len(traj), a.stride)) + [len(traj) - 1]
for k in idx:
    t, x, y, yaw = traj[k]
    turn = max([tn for tn, ts in turns.items() if ts <= t + 1e-6], default=0)
    img = np.full((Ht, W, 3), 245, np.uint8); mp = base.copy()
    pts = np.array([P(r[1], r[2]) for r in traj[:k + 1]], np.int32).reshape(-1, 1, 2)
    cv2.polylines(mp, [pts], False, (220, 120, 20), 2, cv2.LINE_AA)
    c, s = math.cos(yaw), math.sin(yaw)
    box = np.array([P(x + c * bx - s * by, y + s * bx + c * by) for bx, by in ((0.55, 0.32), (0.55, -0.32), (-0.55, -0.32), (-0.55, 0.32))], np.int32)
    cv2.fillPoly(mp, [box], (60, 60, 230)); cv2.polylines(mp, [box], True, (0, 0, 120), 1, cv2.LINE_AA)
    cv2.arrowedLine(mp, P(x, y), P(x + 0.9 * c, y + 0.9 * s), (0, 0, 0), 2, tipLength=0.3)
    img[:MH, :MW] = mp
    img[10:392, MW + 10:MW + 690] = cam(turn)
    yy = put(img, f"sim t = {t - traj[0][0]:6.1f} s   pose ({x:5.2f}, {y:5.2f})  yaw {math.degrees(yaw):4.0f} deg", (MW + 12, 420), 0.55, w=1)
    ti = tinfo.get(turn)
    if ti:
        yy = put(img, f"turn {turn}: {ti.get('tool')} {json.dumps(ti.get('args'))}", (MW + 12, yy + 6), 0.55, (150, 40, 0), 1)
        res_ = (ti.get("result") or {}).get("message", "")
        put(img, "-> " + str(res_)[:330], (MW + 12, yy + 4), 0.48, (20, 20, 20), 1)
    frames.append(img)
h, w = frames[0].shape[:2]
vw = cv2.VideoWriter(os.path.join(ep, "replay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
for f in frames: vw.write(f)
vw.release()
pil = [Image.fromarray(cv2.cvtColor(cv2.resize(f, (w // 2, h // 2)), cv2.COLOR_BGR2RGB)) for f in frames]
pil[0].save(os.path.join(ep, "replay.gif"), save_all=True, append_images=pil[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
print(len(frames), "frames", (w, h), "->", os.path.join(ep, "replay.mp4"), "and replay.gif")
