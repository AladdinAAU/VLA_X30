"""Save the current camera frame: python3 grab_frame.py out.jpg   (inside e2e-fw)"""
import sys, cv2, numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
rclpy.init(); n = Node("grab"); box = {}
n.create_subscription(Image, "/camera/color/image_raw", lambda m: box.setdefault("m", m), 1)
for _ in range(100):
    rclpy.spin_once(n, timeout_sec=0.2)
    if "m" in box: break
m = box["m"]; a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
cv2.imwrite(sys.argv[1], cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else a); print("saved", m.width, m.height)
