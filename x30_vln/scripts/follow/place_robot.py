"""Teleport the robot: python3 place_robot.py x y yaw   (inside e2e-fw)"""
import sys
sys.path.insert(0, "/x30/scripts/e2e")
import rclpy
from run_episode import Rec
rclpy.init(); n = Rec()
for _ in range(30): rclpy.spin_once(n, timeout_sec=0.2)
print(n.call("/sim/set_robot_pose", {"x": float(sys.argv[1]), "y": float(sys.argv[2]), "yaw": float(sys.argv[3])}, 60.0))
