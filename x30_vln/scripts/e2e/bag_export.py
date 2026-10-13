"""Export what the demo needs from an episode bag (run inside e2e-fw):  python3 /x30/scripts/e2e/bag_export.py <bag dir> <out dir>
Writes <out>/frames/NNNNN.jpg (camera, 640 px wide), <out>/index.json = {"frames": [[sim_t, file], ...], "pose": [[sim_t, x, y, yaw], ...],
"plans": [[sim_t, [[x, y], ...]], ...] (known_map_nav plan events, world frame), "cmd": [[sim_t, vx, vy, wz], ...], "safety": [[sim_t, scale, min_obstacle_m, reason], ...]}.
Plan/other non-stamped messages are placed on the sim clock by interpolating (bag receive time -> camera header stamp)."""
import json, math, os, sys
import cv2, numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, "frames"), exist_ok=True)
r = rosbag2_py.SequentialCompressionReader()      # the bags are zstd message-compressed
r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = {"/camera/color/image_raw", "/gt/pose", "/nav/known_map_info", "/cmd_vel", "/x30/safety", "/gt/person", "/gt/people", "/follow/status", "/teach/reply"}
frames, wall_cam, pose, plans, cmd, safety, person, fstatus, people, replies = [], [], [], [], [], [], [], [], [], []
msgs = {t: get_message(types[t]) for t in want if t in types}
st = lambda h: h.stamp.sec + h.stamp.nanosecond * 1e-9 if hasattr(h.stamp, "nanosecond") else h.stamp.sec + h.stamp.nanosec * 1e-9
while r.has_next():
    topic, data, wall = r.read_next()
    if topic not in msgs: continue
    m = deserialize_message(data, msgs[topic])
    if topic == "/camera/color/image_raw":
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else a
        a = cv2.resize(a, (640, int(640 * m.height / m.width)))
        fn = f"{len(frames):05d}.jpg"; cv2.imwrite(os.path.join(out, "frames", fn), a, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append([st(m.header), fn]); wall_cam.append([wall * 1e-9, st(m.header)])
    elif topic == "/gt/pose":
        q = m.pose.orientation; pose.append([st(m.header), m.pose.position.x, m.pose.position.y, 2.0 * math.atan2(q.z, q.w)])
    elif topic == "/gt/person":
        person.append([st(m.header), m.pose.position.x, m.pose.position.y])
    elif topic == "/gt/people":
        try: people.append(json.loads(m.data))
        except Exception: pass
    elif topic == "/teach/reply":
        try: replies.append([wall * 1e-9, json.loads(m.data)])
        except Exception: pass
    elif topic == "/follow/status":
        try: fstatus.append(json.loads(m.data))
        except Exception: pass
    elif topic == "/nav/known_map_info":
        try: d = json.loads(m.data)
        except Exception: continue
        if d.get("event") in ("plan", "replan_blocked"): plans.append([wall * 1e-9, d["path"]])
    elif topic == "/cmd_vel":
        cmd.append([wall * 1e-9, m.linear.x, m.linear.y, m.angular.z])
    elif topic == "/x30/safety":
        safety.append([st(m.header), m.scale, m.min_obstacle_distance, m.reason])
wc = np.array(wall_cam)
to_sim = (lambda w: float(np.interp(w, wc[:, 0], wc[:, 1]))) if len(wc) > 1 else (lambda w: w)
replies = [[to_sim(w), r] for w, r in replies]; plans = [[to_sim(w), p] for w, p in plans]; cmd = [[to_sim(c[0]), *c[1:]] for c in cmd]
json.dump({"frames": frames, "pose": pose, "plans": plans, "cmd": cmd, "safety": safety, "person": person, "status": fstatus, "people": people, "replies": replies}, open(os.path.join(out, "index.json"), "w"))
print(len(frames), "frames", len(pose), "poses", len(plans), "plans", len(cmd), "cmd", len(safety), "safety")
