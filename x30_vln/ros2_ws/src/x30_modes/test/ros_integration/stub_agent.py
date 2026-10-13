"""Stub of the x30_agent action servers /x30/agent/follow and /x30/agent/run (x30_msgs/action/Navigate).

Runs `duration_s` seconds (from params_json, default 5) counting one agent turn per second and optionally driving forward on
/cmd_vel_nav (params_json {"drive": true}); finishes with success, or is cancelled. Used to test the mode manager's delegation,
feedback relay, cancel-on-mode-switch and watchdogs without the real agent.
"""
from __future__ import annotations

import json
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from x30_msgs.action import Navigate


class StubAgent(Node):
    def __init__(self):
        super().__init__("stub_agent")
        cb = ReentrantCallbackGroup()
        self.pub = self.create_publisher(Twist, "/cmd_vel_nav", 10)
        self.log: list[str] = []
        for name in ("/x30/agent/follow", "/x30/agent/run"):
            ActionServer(self, Navigate, name, execute_callback=lambda gh, n=name: self.run(gh, n),
                         goal_callback=lambda g: GoalResponse.ACCEPT, cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=cb)

    def run(self, gh, name):
        p = json.loads(gh.request.params_json or "{}")
        dur, drive = float(p.get("duration_s", 5.0)), bool(p.get("drive", False))
        freeze = bool(p.get("freeze", False))      # a stuck agent: the turn counter stops advancing
        t0, turn = time.monotonic(), 0
        res = Navigate.Result()
        while time.monotonic() - t0 < dur:
            if gh.is_cancel_requested:
                self.pub.publish(Twist())
                gh.canceled()
                res.success, res.stop_reason, res.message = False, "aborted", f"{name} cancelled after {turn} turns"
                res.agent_turns = turn
                return res
            if drive:
                m = Twist()
                m.linear.x = 0.2
                self.pub.publish(m)
            turn = 1 if freeze else int(time.monotonic() - t0) + 1
            fb = Navigate.Feedback()
            fb.status, fb.current_skill, fb.agent_turn = f"{name} thinking", "move_relative", turn
            gh.publish_feedback(fb)
            time.sleep(0.1)
        res.success, res.stop_reason, res.message, res.agent_turns = True, "finished", f"{name} done ({gh.request.instruction})", turn
        gh.succeed()
        return res


def main():
    rclpy.init()
    node = StubAgent()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
