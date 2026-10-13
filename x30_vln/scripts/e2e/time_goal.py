"""Time how long known_map_nav takes to accept a navigate_to_pose goal (map frame). python3 time_goal.py x y"""
import sys, time, rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
rclpy.init(); n = Node("time_goal", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
c = ActionClient(n, NavigateToPose, "navigate_to_pose"); c.wait_for_server(10)
g = NavigateToPose.Goal(); g.pose.header.frame_id = "map"; g.pose.pose.position.x = float(sys.argv[1]); g.pose.pose.position.y = float(sys.argv[2]); g.pose.pose.orientation.w = 1.0
t = time.time(); f = c.send_goal_async(g); rclpy.spin_until_future_complete(n, f, timeout_sec=60)
print("accept after %.2f s wall, accepted=%s" % (time.time() - t, f.result().accepted if f.result() else None))
if f.result() and f.result().accepted:
    f.result().cancel_goal_async()
