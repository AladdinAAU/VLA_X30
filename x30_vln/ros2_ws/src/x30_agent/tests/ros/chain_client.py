"""Sends one object_nav goal to /x30/navigate (mode_manager) and prints the result; the manager delegates to /x30/agent/run."""
import json
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from x30_msgs.action import Navigate

rclpy.init()
n = Node("chain_client")
threading.Thread(target=lambda: rclpy.spin(n), daemon=True).start()
c = ActionClient(n, Navigate, "/x30/navigate")
assert c.wait_for_server(timeout_sec=30), "no /x30/navigate"
g = Navigate.Goal()
g.mode, g.instruction, g.target_label, g.timeout_s, g.episode_id = "object_nav", "Find the chair and stop next to it.", "chair", 120.0, "chain1"
fb = []
ev = threading.Event()
f = c.send_goal_async(g, feedback_callback=lambda m: fb.append(m.feedback.status))
f.add_done_callback(lambda _f: ev.set())
ev.wait(10)
h = f.result()
print("accepted:", h.accepted, flush=True)
t0 = time.time()
ev2 = threading.Event()
r = h.get_result_async()
r.add_done_callback(lambda _f: ev2.set())
ev2.wait(180)
res = r.result().result
print(json.dumps({"success": res.success, "stop_reason": res.stop_reason, "message": res.message, "path_length_m": round(res.path_length_m, 2),
                  "duration_s": round(res.duration_s, 1), "agent_turns": res.agent_turns, "feedback_msgs": len(fb), "last_feedback": fb[-1] if fb else ""}), flush=True)
rclpy.shutdown()
sys.exit(0 if res.success else 1)
