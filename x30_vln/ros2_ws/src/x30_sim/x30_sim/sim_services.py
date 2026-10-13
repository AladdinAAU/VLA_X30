"""Episode reset services: /sim/set_robot_pose and /sim/reset_world (x30_msgs/srv/Json).

/sim/set_robot_pose  request {"x":.., "y":.., "yaw":.., "z": optional}   world frame
/sim/reset_world     request {}                                           every world model back at its SDF pose, the robot at its spawn pose
Response JSON: {"success": bool, "message": str, "pose": {...}}.
Both reset the odometry origin (/sim/reset_odom). Gazebo is driven through the `gz service` CLI (set_pose / world control), so Gazebo is
never restarted and the simulation clock keeps running. Legged backend: the controller is first held in its standing state
(/sim/legged/hold), the robot is moved, then the controller trots again; the call returns when /x30/ready is true again.
Kinematic backend: teleport, velocity command cleared.
"""
from __future__ import annotations

import json
import math
import subprocess
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from x30_msgs.srv import Json


class SimServices(Node):
    def __init__(self):
        super().__init__('sim_services')
        self.declare_parameter('world_name', 'default_arena')
        self.declare_parameter('locomotion', 'kinematic')
        self.declare_parameter('model_name', 'x30')
        self.declare_parameter('z_stand', 0.51)
        self.declare_parameter('spawn_x', 0.0)
        self.declare_parameter('spawn_y', 0.0)
        self.declare_parameter('spawn_yaw', 0.0)
        self.declare_parameter('settle_timeout_s', 40.0)
        self.world = self.get_parameter('world_name').value
        self.locomotion = self.get_parameter('locomotion').value
        self.model = self.get_parameter('model_name').value
        self.cb = ReentrantCallbackGroup()
        self.ready = False
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Bool, '/x30/ready', self.on_ready, latched, callback_group=self.cb)
        self.hold_pub = self.create_publisher(Bool, '/sim/legged/hold', 5)
        self.reset_odom = self.create_client(Trigger, '/sim/reset_odom', callback_group=self.cb)
        self.create_service(Json, '/sim/set_robot_pose', self.on_set_pose, callback_group=self.cb)
        self.create_service(Json, '/sim/reset_world', self.on_reset_world, callback_group=self.cb)

    def on_ready(self, msg: Bool) -> None:
        self.ready = bool(msg.data)

    # ----------------------------------------------------------------------------------------------- gz helpers
    def gz_service(self, service: str, reqtype: str, reptype: str, req: str, timeout_ms: int = 4000) -> tuple[bool, str]:
        cmd = ['gz', 'service', '-s', service, '--reqtype', reqtype, '--reptype', reptype, '--timeout', str(timeout_ms), '--req', req]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_ms / 1000 + 5)
        except subprocess.TimeoutExpired:
            return False, f'gz service {service} timed out'
        text = (out.stdout + out.stderr).strip()
        return out.returncode == 0 and 'data: true' in text, text

    def wait_for(self, cond, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.1)
        return cond()

    def hold(self, on: bool) -> None:
        self.hold_pub.publish(Bool(data=on))

    def run_motion(self, action) -> tuple[bool, str]:
        """Run `action` (a gz call) with the legged controller parked in its standing state when needed."""
        legged = self.locomotion == 'legged'
        timeout = float(self.get_parameter('settle_timeout_s').value)
        if legged:
            self.hold(True)
            if not self.wait_for(lambda: not self.ready, timeout):
                return False, 'legged controller did not leave trotting'
            time.sleep(1.5)                       # let the standing pose settle
        ok, text = action()
        if legged:
            time.sleep(1.0)
            self.hold(False)
            if not self.wait_for(lambda: self.ready, timeout):
                return False, 'legged controller did not return to trotting'
        if self.reset_odom.wait_for_service(timeout_sec=1.0):
            self.reset_odom.call(Trigger.Request())
        return ok, text

    # ----------------------------------------------------------------------------------------------- services
    def on_set_pose(self, req: Json.Request, resp: Json.Response) -> Json.Response:
        try:
            p = json.loads(req.request or '{}')
            x, y, yaw = float(p['x']), float(p['y']), float(p.get('yaw', 0.0))
            z = float(p['z']) if 'z' in p else float(self.get_parameter('z_stand').value)
        except (ValueError, KeyError, TypeError) as exc:
            resp.ok, resp.response = False, json.dumps({'success': False, 'message': f'bad request: {exc}'})
            return resp
        gz_req = self.pose_request(x, y, z, yaw)
        ok, text = self.run_motion(lambda: self.gz_service(f'/world/{self.world}/set_pose', 'gz.msgs.Pose', 'gz.msgs.Boolean', gz_req))
        resp.ok = ok
        resp.response = json.dumps({'success': ok, 'message': '' if ok else text, 'pose': {'x': x, 'y': y, 'z': z, 'yaw': yaw}})
        return resp

    def pose_request(self, x: float, y: float, z: float, yaw: float) -> str:
        return (f'name: "{self.model}" position {{x: {x} y: {y} z: {z}}} '
                f'orientation {{z: {math.sin(yaw / 2):.9f} w: {math.cos(yaw / 2):.9f}}}')

    def on_reset_world(self, req: Json.Request, resp: Json.Response) -> Json.Response:
        """Models of the world go back to their SDF poses (clock keeps running), then the robot to its spawn pose."""
        def action():
            ok, text = self.gz_service(f'/world/{self.world}/control', 'gz.msgs.WorldControl', 'gz.msgs.Boolean', 'reset {model_only: true}')
            if not ok:
                return ok, text
            return self.gz_service(f'/world/{self.world}/set_pose', 'gz.msgs.Pose', 'gz.msgs.Boolean',
                                   self.pose_request(float(self.get_parameter('spawn_x').value), float(self.get_parameter('spawn_y').value),
                                                     float(self.get_parameter('z_stand').value), float(self.get_parameter('spawn_yaw').value)))
        ok, text = self.run_motion(action)
        resp.ok = ok
        resp.response = json.dumps({'success': ok, 'message': '' if ok else text})
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = SimServices()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
