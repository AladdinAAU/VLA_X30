"""Drive at the east wall of the simulator arena at a given speed and record time series (ground truth x, guard output, clearance).

  python3 approach_trace.py 0.4     -> results/modes/approach_trace_0.4.json   (used for the figure in docs/figures/modes)
"""
import json
import sys
import threading
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor

sys.path.insert(0, str(Path(__file__).parent))
from sim_check import D  # noqa: E402


def main():
    speed = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    rclpy.init()
    d = D()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(d)
    threading.Thread(target=ex.spin, daemon=True).start()
    while d.gt is None or d.cloud is None or d.safety is None:
        time.sleep(0.2)
    out = Twist()
    guard_out: list = []
    d.create_subscription(Twist, "/cmd_vel", lambda m: guard_out.append((time.monotonic(), m.linear.x)), 20)
    d.mode("explore")
    d.teleport(2.0, 0.0, 0.0)
    rows = []
    t0 = time.monotonic()
    m = Twist()
    m.linear.x = speed
    last_move = t0
    prev = d.gtpose().x
    while time.monotonic() - t0 < 40.0:
        d.pub.publish(m)
        time.sleep(0.05)
        if int((time.monotonic() - t0) * 20) % 5 == 0:
            gap, _ = d.lidar_gap()
            rows.append({"t": round(time.monotonic() - t0, 2), "x": round(d.gtpose().x, 3), "vx_out": round(guard_out[-1][1], 3) if guard_out else None,
                         "lidar_gap": round(gap, 3), "guard_clear": round(d.safety.min_obstacle_distance, 3), "scale": round(d.safety.scale, 2),
                         "reason": d.safety.reason})
        x = d.gtpose().x
        if abs(x - prev) > 0.01:
            prev, last_move = x, time.monotonic()
        elif time.monotonic() - last_move > 5.0 and x > 3.0:
            break
    Path("/x30/results/modes").mkdir(parents=True, exist_ok=True)
    Path(f"/x30/results/modes/approach_trace_{speed}.json").write_text(json.dumps(rows))
    for r in rows[:: max(len(rows) // 25, 1)]:
        print(r)
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
