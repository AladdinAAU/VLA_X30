"""person_mover: scripted walking people for follow_person trials (sim only).

Moves person models of the world along timed paths with the Gazebo `set_pose` service and publishes their ground truth.
Single person (legacy): parameter path_file (+ model, default `person_07`).
Crowd: parameter crowd_file, JSON {"world": ..., "people": [{"name": "target", "model": "person_07", "path": "x.json", "target": true, "outfit": "navy",
"delay_s": 0, "yaw_offset": 0}, {"name": "crowd_red", "uri": "file:///x30/configs/follow/models/ScrubsRed", "path": "y.json", ...}]}.
Entries with `uri` are spawned (static) at the first pose of their path; entries with `model` move an existing model of the world.
Paths come from scripts/follow/make_person_path.py: {"poses": [[t, x, y, yaw], ...]} in sim seconds from the start trigger (+ delay_s).

Topics: /sim/person/start (std_msgs/Empty): everybody starts; /sim/person/reset (Empty): all back to their first pose; /gt/person (PoseStamped, the target);
/gt/people (std_msgs/String JSON {"t": sim time, "people": {name: [x, y, yaw, outfit]}}). Models glide (no limb animation).
Run in the sim container (GZ_PARTITION must match the simulator):
python3 /ws/src/x30_sim/x30_sim/person_mover.py --ros-args -p crowd_file:=/x30/configs/follow/crowd1.json
"""
import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.empty_pb2 import Empty as GzEmpty
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.scene_pb2 import Scene
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node as GzNode
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Empty, String


class PersonMover(Node):
    def __init__(self):
        super().__init__('person_mover')
        self.declare_parameter('world', 'aws_hospital')
        self.declare_parameter('model', 'person_07')
        self.declare_parameter('path_file', '')
        self.declare_parameter('crowd_file', '')
        self.declare_parameter('yaw_offset', 0.0)
        self.declare_parameter('hz', 20.0)
        g = self.get_parameter
        self.world, self.hz = g('world').value, float(g('hz').value)
        self.gz = GzNode()
        crowd = g('crowd_file').value
        if crowd:
            spec = json.load(open(crowd))
            self.world = spec.get('world', self.world)
            base = os.path.dirname(crowd)
            self.people = []
            for p in spec['people']:
                path = p['path'] if os.path.isabs(p['path']) else os.path.join(base, p['path'])
                self.people.append({**p, 'poses': json.load(open(path))['poses'], 'yaw_offset': float(p.get('yaw_offset', 0.0)),
                                    'delay_s': float(p.get('delay_s', 0.0)), 'fail': 0})
        else:
            poses = json.load(open(g('path_file').value))['poses']
            self.people = [{'name': 'target', 'model': g('model').value, 'poses': poses, 'target': True, 'yaw_offset': float(g('yaw_offset').value),
                            'delay_s': 0.0, 'outfit': '', 'fail': 0}]
        self.pub_target = self.create_publisher(PoseStamped, '/gt/person', 10)
        self.pub_people = self.create_publisher(String, '/gt/people', 10)
        self.t_start = None
        self.sim_t = 0.0
        self.last_t = -1e9
        self.create_subscription(Clock, '/clock', self.on_clock, 10)
        self.create_subscription(Empty, '/sim/person/start', self.on_start, 10)
        self.create_subscription(Empty, '/sim/person/reset', self.on_reset, 10)
        for p in self.people:
            if p.get('uri'):
                self.spawn(p)
            self.place(p, p['poses'][0])
        self.remove_leftovers()
        names = self.model_names()
        missing = [self.model_name(q) for q in self.people if self.model_name(q) not in names]
        if missing:
            raise RuntimeError(f'models missing in the scene: {missing}')
        self.get_logger().info(f'person_mover: all {len(self.people)} people verified in the scene of {self.world}: ' + ', '.join(
            f"{p['name']}({p.get('outfit', '')})" for p in self.people) + '; waiting for /sim/person/start')

    def model_name(self, p) -> str:
        return p.get('model') or p['name']

    def model_names(self) -> set:
        ok, rep = self.gz.request(f'/world/{self.world}/scene/info', GzEmpty(), GzEmpty, Scene, 5000)
        return {m.name for m in rep.model} if ok else set()

    def remove_leftovers(self) -> None:
        """People of earlier scenarios (crowd_*) stay in the world as static figures: remove those that are not in this scenario and verify they are gone."""
        keep = {self.model_name(p) for p in self.people}
        for name in sorted(n for n in self.model_names() if n.startswith('crowd_') and n not in keep):
            e = Entity()
            e.name, e.type = name, Entity.MODEL
            self.gz.request(f'/world/{self.world}/remove', e, Entity, Boolean, 3000)
            self.get_logger().info(f'removed leftover model {name} of an earlier scenario')
        for _ in range(20):
            if not [n for n in self.model_names() if n.startswith('crowd_') and n not in keep]:
                return
            time.sleep(0.5)
        raise RuntimeError('leftover crowd_* models could not be removed')

    def spawn(self, p) -> None:
        """Idempotent: a model that already exists is kept (set_pose repositions it); a missing one is created and verified in the scene
        (set_pose answers 'true' even for models that do not exist, and remove-then-create races: both made models silently disappear)."""
        name = self.model_name(p)
        for attempt in range(1, 4):
            if name in self.model_names():
                return
            x, y, yaw = p['poses'][0][1], p['poses'][0][2], p['poses'][0][3] + p['yaw_offset']
            req = EntityFactory()
            req.sdf = (f"<sdf version='1.9'><include><uri>{p['uri']}</uri><name>{name}</name><static>true</static>"
                       f"<pose>{x} {y} 0 0 0 {yaw}</pose></include></sdf>")
            self.gz.request(f'/world/{self.world}/create', req, EntityFactory, Boolean, 10000)
            for _ in range(20):                                   # the model appears in the scene after the next simulation steps
                time.sleep(0.5)
                if name in self.model_names():
                    return
            self.get_logger().warn(f'{name}: not in the scene after create (attempt {attempt})')
        raise RuntimeError(f'spawn of {name} from {p["uri"]} failed: model never appeared in the scene')

    def place(self, p, pose) -> bool:
        req = Pose()
        req.name = self.model_name(p)
        req.position.x, req.position.y, req.position.z = float(pose[1]), float(pose[2]), 0.0
        yaw = float(pose[3]) + p['yaw_offset']
        req.orientation.z, req.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        ok, rep = self.gz.request(f'/world/{self.world}/set_pose', req, Pose, Boolean, 500)
        if not (ok and rep.data):
            p['fail'] += 1
            if p['fail'] in (1, 20):
                self.get_logger().warn(f'set_pose failed for {req.name} ({p["fail"]}x)')
        return bool(ok and rep.data)

    def on_start(self, _):
        self.t_start = self.sim_t
        self.get_logger().info(f'start at sim t={self.sim_t:.1f}')

    def on_reset(self, _):
        self.t_start = None
        for p in self.people:
            self.place(p, p['poses'][0])

    def pose_at(self, p, t):
        i = int(max(0.0, t - p['delay_s']) * self.hz)
        return p['poses'][min(i, len(p['poses']) - 1)]

    def on_clock(self, m: Clock):
        self.sim_t = m.clock.sec + m.clock.nanosec * 1e-9
        if self.t_start is None or self.sim_t - self.last_t < 1.0 / self.hz:
            return
        self.last_t = self.sim_t
        t = self.sim_t - self.t_start
        out, done = {}, True
        for p in self.people:
            pose = self.pose_at(p, t)
            self.place(p, pose)
            out[p['name']] = [round(pose[1], 3), round(pose[2], 3), round(pose[3], 3), p.get('outfit', '')]
            done = done and t - p['delay_s'] >= p['poses'][-1][0]
            if p.get('target'):
                msg = PoseStamped()
                msg.header.stamp, msg.header.frame_id = m.clock, 'world'
                msg.pose.position.x, msg.pose.position.y = float(pose[1]), float(pose[2])
                msg.pose.orientation.z, msg.pose.orientation.w = math.sin(pose[3] / 2.0), math.cos(pose[3] / 2.0)
                self.pub_target.publish(msg)
        self.pub_people.publish(String(data=json.dumps({'t': round(self.sim_t, 3), 'people': out})))
        if done:
            self.t_start = None
            self.get_logger().info('all paths finished, people stay at their last pose')


def main(args=None):
    rclpy.init(args=args)
    node = PersonMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
