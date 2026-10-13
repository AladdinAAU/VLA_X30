"""Cancel whatever skill the skill server is running (a killed client leaves its skill running and every new skill answers 'Busy').
python3 /x30/scripts/e2e/stop_skill.py   (inside e2e-fw)"""
import json, time, rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from x30_msgs.action import Skill
rclpy.init(); n = Node("stop_skill")
c = ActionClient(n, Skill, "/x30/skill/stop"); c.wait_for_server(15)
f = c.send_goal_async(Skill.Goal(name="stop", args_json="{}", timeout_s=20.0)); rclpy.spin_until_future_complete(n, f, timeout_sec=20)
r = f.result().get_result_async(); rclpy.spin_until_future_complete(n, r, timeout_sec=30)
print(r.result().result.message)
