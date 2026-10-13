"""Print person detections (label, score, 3D centroid in the array frame, bearing_only, points) for N seconds.  python3 peek_detections.py [seconds]"""
import sys, time, rclpy
from rclpy.node import Node
from x30_msgs.msg import Detection3DArray
rclpy.init(); n = Node("peek", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)]); t0 = time.time(); k = [0]
def cb(m):
    k[0] += 1
    for d in m.detections:
        if d.label == "person":
            print(f"{m.header.frame_id} person score={d.score:.2f} bearing_only={d.bearing_only} pts={d.point_count} c=({d.centroid.x:.2f},{d.centroid.y:.2f},{d.centroid.z:.2f}) ext=({d.extent.x:.2f},{d.extent.y:.2f},{d.extent.z:.2f})", flush=True)
n.create_subscription(Detection3DArray, "/perception/detections", cb, 10)
while time.time() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 15): rclpy.spin_once(n, timeout_sec=0.2)
print("messages:", k[0])
