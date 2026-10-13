#!/usr/bin/env python3
"""Episodes through the ROS action servers with the REAL model server (nodes started without X30_REASONER_URL) in the fake stack.
Usage: ros_live.py out.json [n_objnav n_follow]. FAKE environment results; they show that the whole ROS path works with the live model."""
import json
import sys
import time

import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor

sys.argv = [sys.argv[0]] + sys.argv[1:]
OUT = sys.argv[1]
N_ON = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N_VF = int(sys.argv[3]) if len(sys.argv) > 3 else 5
sys.argv = [sys.argv[0], "/dev/null"]
import ros_check as RC  # noqa: E402


def main():
    rclpy.init()
    c = RC.Client()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(c)
    threading.Thread(target=ex.spin, daemon=True).start()
    assert c.run.wait_for_server(timeout_sec=30) and c.follow.wait_for_server(timeout_sec=30)
    rows = []
    for mode, cli, n, seed0, extra in (("object_nav", c.run, N_ON, 300, {"max_turns": 25}), ("velocity_follow", c.follow, N_VF, 400, {"max_turns": 40})):
        for i in range(n):
            info = c.call("/fake/reset", {"seed": seed0 + i, "mode": mode, "skill_delay_s": 0.05})
            t0 = time.time()
            h, fb = c.send(cli, info["instruction"], info["target_label"] if mode == "object_nav" else "", mode, extra, 300, f"live_{mode}_{i}")
            r = c.result(h, 600)
            st = c.call("/fake/state", {})
            rows.append({"mode": mode, "seed": seed0 + i, "instruction": info["instruction"], "claimed": bool(r.result.success), "gt_success": bool(r.result.success and st["success"]),
                         "stop_reason": r.result.stop_reason, "turns": r.result.agent_turns, "wall_s": round(time.time() - t0, 1), "feedback_msgs": len(fb)})
            print(rows[-1], flush=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"object_nav gt success {sum(r['gt_success'] for r in rows if r['mode'] == 'object_nav')}/{N_ON}; "
          f"velocity_follow {sum(r['gt_success'] for r in rows if r['mode'] == 'velocity_follow')}/{N_VF}")
    rclpy.shutdown()


main()
